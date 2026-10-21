"""Bounded local HTTP token completion service; no third-party serving engine."""
import argparse
import json
import queue
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .detok import IncrementalText
from .scheduler import Scheduler, BusyError

# OpenAI /v1/completions subset. ignore_eos and stream_options are what sglang.bench_serving and
# vLLM-style clients send; n and best_of are accepted only as 1. Anything else is rejected.
COMPLETION_FIELDS={'model','prompt','max_tokens','temperature','seed','stream','ignore_eos',
    'stream_options','n','best_of'}

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, scheduler, tokenizer=None, model_name='glm-vx-tiny-random'):
        super().__init__(address, Handler)
        self.scheduler, self.tokenizer, self.model_name = scheduler, tokenizer, model_name
        self.slots = threading.BoundedSemaphore(32)
    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False): request.close(); return
        try: super().process_request(request,address)
        except BaseException: self.slots.release(); raise
    def process_request_thread(self, request, address):
        try: super().process_request_thread(request,address)
        finally: self.slots.release()

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    heartbeat_seconds = 10
    def setup(self):
        super().setup()
        self.connection.settimeout(30)
        # SSE tokens are small latency-sensitive writes. Do not let Nagle wait
        # for an ACK before sending a subsequent token.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    def log_message(self,*args): pass
    def _json(self,code,value):
        data=json.dumps(value,allow_nan=False).encode()
        self.send_response(code)
        if code == 429: self.send_header('Retry-After', '1')
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':
            return self._json(200,{'status':'ok','model':self.server.model_name,
                'backend':self.server.scheduler.model.backend.name,**self.server.scheduler.status()})
        if self.path=='/v1/models':
            return self._json(200,{'object':'list','data':[{'id':self.server.model_name,'object':'model','owned_by':'local'}]})
        self._json(404,{'error':'not found'})
    def _sse(self,value):
        self.wfile.write(('data: '+json.dumps(value,allow_nan=False)+'\n\n').encode()); self.wfile.flush()
    def _disconnected(self):
        # Long-running requests must keep the TCP connection open for the reply.
        if not select.select([self.connection], [], [], 0)[0]:return False
        try:return self.connection.recv(1,socket.MSG_PEEK | socket.MSG_DONTWAIT)==b''
        except BlockingIOError:return False
        except OSError:return True
    def _options(self,body):
        """Validate an OpenAI-compatible completion body; return (prompt, submit kwargs, opts)."""
        unknown=set(body)-COMPLETION_FIELDS
        if unknown: raise ValueError('unsupported fields: '+', '.join(sorted(unknown)))
        if 'model' in body and body['model']!=self.server.model_name:
            raise ValueError(f'unknown model; this server serves {self.server.model_name!r} (see /v1/models)')
        stream=body.get('stream',False)
        if type(stream) is not bool: raise ValueError('stream must be boolean')
        ignore_eos=body.get('ignore_eos',False)
        if type(ignore_eos) is not bool: raise ValueError('ignore_eos must be boolean')
        for key in ('n','best_of'):
            value=body.get(key)
            if value is not None and (type(value) is not int or value!=1): raise ValueError(key+' must be 1')
        options=body.get('stream_options')
        if options is not None:
            if not stream: raise ValueError('stream_options requires stream')
            if not isinstance(options,dict) or set(options)-{'include_usage'} or type(options.get('include_usage',False)) is not bool:
                raise ValueError('stream_options supports only boolean include_usage')
        prompt=body.get('prompt')
        if isinstance(prompt,str):
            if self.server.tokenizer is None: raise ValueError('text needs --tokenizer; tiny demo accepts token IDs')
            prompt=self.server.tokenizer.encode(prompt).ids
        # OpenAI clients always send `model`; they get OpenAI stream framing (text deltas, a
        # text-only terminal chunk). Requests without it keep the glm-vx framing, whose terminal
        # event repeats the complete text and token IDs.
        return prompt,dict(max_tokens=body.get('max_tokens',16),temperature=body.get('temperature',0),
            seed=body.get('seed',0),ignore_eos=ignore_eos),dict(stream=stream,openai='model' in body,
            include_usage=bool(options and options.get('include_usage')))
    def do_POST(self):
        req=None; streaming=False
        try:
            if self.path!='/v1/completions':
                self.close_connection=True
                return self._json(404,{'error':'not found'})
            if self.headers.get('Transfer-Encoding'): raise ValueError('chunked request bodies unsupported')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=1048576: raise ValueError('body must be 1..1048576 bytes')
            raw=self.rfile.read(size)
            if len(raw)!=size: raise ValueError('incomplete body')
            body=json.loads(raw)
            if not isinstance(body,dict): raise ValueError('JSON object required')
            prompt,submit,opts=self._options(body)
            req=self.server.scheduler.submit(prompt,**submit)
            ids=[]; created=int(time.time()); detok=IncrementalText(self.server.tokenizer)
            head={'id':req.id,'object':'text_completion','created':created,'model':self.server.model_name}
            if opts['stream']:
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache'); self.send_header('Connection','close')
                self.end_headers(); self.close_connection=True; streaming=True
            last_heartbeat=time.monotonic()
            while True:
                if self._disconnected():return
                try: event=req.events.get(timeout=.25)
                except queue.Empty:
                    if streaming and time.monotonic()-last_heartbeat>=self.heartbeat_seconds:
                        # An SSE comment is the usual keepalive, but sglang.bench_serving feeds every
                        # nonblank line to json.loads, so OpenAI-framed streams send a blank line.
                        self.wfile.write(b'\n' if opts['openai'] else b': keepalive\n\n'); self.wfile.flush()
                        last_heartbeat=time.monotonic()
                    continue
                if event['type']=='token':
                    ids.append(event['token_id'])
                    if streaming:
                        self._sse({**head,'choices':[{'index':0,'text':detok.push(ids[-1]),'token_ids':[ids[-1]],
                            'logprobs':None,'finish_reason':None}]})
                else:
                    if event['reason'] in ('error', 'backpressure'):
                        if streaming: self._sse({'error':event['error']})
                        else: self._json(500,{'error':event['error']})
                    else:
                        usage={'prompt_tokens':event['prompt_tokens'],'completion_tokens':event['completion_tokens'],
                            'total_tokens':event['prompt_tokens']+event['completion_tokens'],
                            'prompt_tokens_details':{'cached_tokens':event['cached_prompt_tokens']}}
                        if streaming and opts['openai']:
                            # Terminal chunk carries only the held-back text tail, so concatenated
                            # deltas are the completion and no client counts the text twice.
                            self._sse({**head,'choices':[{'index':0,'text':detok.flush(),'logprobs':None,
                                'finish_reason':event['reason']}],'usage':usage,'timings':event['timings']})
                            if opts['include_usage']:self._sse({**head,'choices':[],'usage':usage})
                        else:
                            text=self.server.tokenizer.decode(ids) if self.server.tokenizer else ''
                            result={**head,'choices':[{'index':0,'text':text,'token_ids':ids,'logprobs':None,
                                'finish_reason':event['reason']}],'usage':usage,'timings':event['timings']}
                            if streaming:self._sse(result)
                            else:self._json(200,result)
                    if streaming:self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
                    return
        except BusyError as exc:self._json(429,{'error':str(exc)})
        except (ValueError,TypeError,OverflowError) as exc:
            self.close_connection=True
            if not streaming:self._json(400,{'error':str(exc)})
        except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
        finally:
            if req:self.server.scheduler.cancel(req)

RESIDENT='vx-gpu-resident'

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['vx','vx-gpu',RESIDENT,'numpy-reference'],default='vx')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--tiny',action='store_true',help='random weights, correctness demo only')
    source.add_argument('--checkpoint',help='local safetensors checkpoint directory')
    source.add_argument('--gguf',help='GLM-DSA GGUF first shard or directory; CPU disk-offload PoC')
    p.add_argument('--config',help='HF config.json matching the GGUF; defaults to config.json beside its shards')
    p.add_argument('--decode-threads',type=int,default=1,help='GGUF unpacking workers, 1..32')
    p.add_argument('--decoded-cache-mib',type=int,default=512,help='bounded GGUF decoded-weight cache')
    p.add_argument('--packed-weights',action='store_true',help='opt in to Vx packed GGUF dot products with F32 activations')
    p.add_argument('--tokenizer',help='local tokenizer.json')
    p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8000)
    p.add_argument('--max-sequences',type=int,help=f'concurrent sequences; default 8, and 1 (the only choice) for {RESIDENT}')
    p.add_argument('--token-budget',type=int,default=8192)
    p.add_argument('--max-waiting',type=int,help=f'requests queued for capacity instead of rejected with 429; default 0, and 32 for {RESIDENT}')
    p.add_argument('--max-context',type=int,help=f'{RESIDENT} only: prompt+generated tokens per request, the fixed device cache length; default 8192')
    p.add_argument('--prefix-cache-mib',type=int,default=0,help='CPU prompt snapshot storage budget; disabled by default, includes retained Python storage')
    p.add_argument('--max-pending-events',type=int,default=64,help='bounded unread output events per request; slow consumers are terminated')
    p.add_argument('--batched-prefill',action='store_true',help='opt in to bounded layer-wise CPU prefill; one chunk is indivisible')
    p.add_argument('--prefill-chunk',type=int,default=16)
    p.add_argument('--decode-prefill-tokens',type=int,default=4,help='total prefill tokens per round with active decoders; lower favors decode latency')
    p.add_argument('--gpu-ptx-dir',help='directory containing compiled kernels.ptx')
    p.add_argument('--gpu-device',type=int,default=0)
    p.add_argument('--gpu-weight-cache-mib',type=int,default=512)
    p.add_argument('--gpu-tuning',help='verified shape-specific plan produced by python -m gpu.tune')
    args=p.parse_args(argv)
    if args.packed_weights and (not args.gguf or args.backend != 'vx'):p.error('packed weights requires --gguf and --backend vx')
    if args.prefix_cache_mib<0:p.error('prefix cache must be nonnegative')
    if args.max_pending_events<2:p.error('max pending events must be at least two')
    if args.gpu_weight_cache_mib<0:p.error('GPU weight cache must be nonnegative')
    if args.decoded_cache_mib<0:p.error('decoded cache must be nonnegative')
    if args.max_waiting is not None and args.max_waiting<0:p.error('max waiting must be nonnegative')
    if args.backend==RESIDENT:
        # ResidentGLM holds one sequence's device caches; batched decode is #8.
        if args.max_sequences not in (None,1):p.error(f'{RESIDENT} serves one sequence at a time: --max-sequences must be 1')
        if args.prefix_cache_mib:p.error(f'{RESIDENT} does not support the CPU prefix cache')
        if args.batched_prefill:p.error(f'{RESIDENT} does not support batched CPU prefill')
        if args.gguf:p.error(f'{RESIDENT} loads --checkpoint or --tiny, not --gguf')
        if args.gpu_tuning:p.error(f'{RESIDENT} does not use --gpu-tuning')
        if args.max_context is None:args.max_context=8192
        if args.max_context<1:p.error('max context must be positive')
        args.max_sequences=1
        if args.max_waiting is None:args.max_waiting=32
    else:
        if args.max_context is not None:p.error(f'--max-context applies to --backend {RESIDENT} only')
        if args.max_sequences is None:args.max_sequences=8
        if args.max_waiting is None:args.max_waiting=0
    return args

def build_model(args):
    """(model the scheduler drives, backend, weights, served name)."""
    from .model import Model,tiny_weights
    if args.backend=='vx':
        from kernels.backend import VxBackend
        backend=VxBackend()
    elif args.backend=='vx-gpu':
        from gpu.backend import GPUBackend
        backend=GPUBackend(args.gpu_ptx_dir,args.gpu_device,args.gpu_weight_cache_mib*1024**2,args.gpu_tuning)
    else:
        # For vx-gpu-resident the host model only supplies config, weights and embedding rows.
        from .backend import NumpyBackend
        backend=NumpyBackend()
    if args.tiny:
        from .tiny import tiny_config
        config=tiny_config(); weights=tiny_weights(config,seed=7); name='glm-vx-tiny-random'
        if args.backend=='vx-gpu':
            for value in weights.values():value.flags.writeable=False
    elif args.gguf:
        from pathlib import Path
        from .gguf_checkpoint import GGUFCheckpoint
        directory=Path(args.gguf) if Path(args.gguf).is_dir() else Path(args.gguf).parent
        with open(args.config or directory/'config.json') as f:config=json.load(f)
        weights=GGUFCheckpoint(args.gguf,config,cache_bytes=args.decoded_cache_mib*1024**2,decode_threads=args.decode_threads)
        if not args.tokenizer and (directory/'tokenizer.json').exists():args.tokenizer=str(directory/'tokenizer.json')
        name='glm-gguf-vx-experimental'
    else:
        from .checkpoint import Checkpoint
        from .config import GLMConfig
        with open(args.checkpoint+'/config.json') as f:config=json.load(f)
        validated=GLMConfig.from_dict(config)
        weights=Checkpoint(args.checkpoint,block_size=validated.weight_block_size)
        weights.validate_manifest(validated.tensor_shapes(),inspect_shards=True)
        name='glm-5.3-vx-experimental'
    model=Model(config,weights,backend,packed_weights=args.packed_weights)
    if args.backend==RESIDENT:
        # FP8 / INT4 / BF16 Linears stay packed on the device; weights upload on first use.
        from gpu import resident
        backend=resident.ResidentGLM(model,max_tokens=args.max_context,ptx_dir=args.gpu_ptx_dir,device=args.gpu_device)
        model=resident.ResidentSequenceModel(model,backend)
    return model,backend,weights,name

def main(argv=None):
    args=parse_args(argv)
    model,backend,weights,name=build_model(args)
    tokenizer=None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(args.tokenizer)
    scheduler=Scheduler(model,max_sequences=args.max_sequences,token_budget=args.token_budget,prefill_chunk=args.prefill_chunk,decode_prefill_tokens=args.decode_prefill_tokens,prefix_cache_bytes=args.prefix_cache_mib*1024**2,max_pending_events=args.max_pending_events,batched_prefill=args.batched_prefill,max_waiting=args.max_waiting)
    server=Server((args.host,args.port),scheduler,tokenizer,name)
    print(json.dumps({'listening':f'http://{args.host}:{server.server_port}','model':name,'backend':backend.name}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close(); scheduler.close()
        if hasattr(weights,'close'):weights.close()
        if hasattr(backend,'close'):backend.close()

if __name__=='__main__':main()
