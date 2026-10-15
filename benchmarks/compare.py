"""Cross-engine serving benchmark: the same workload against several OpenAI-compatible servers.

For M5 of the GPU bring-up (issue vx-lang/glm-vx#1): run glm-vx, llama.cpp's `llama-server` and
vLLM on the same pod, same model, same precision, with the same token-ID prompts, and report
prefill tokens/s, decode tokens/s, time to first token and peak memory. Nothing is filtered:
failed requests are recorded with their error, and every per-request number is in the output so
runs where Vx loses are published like any other.

  python benchmarks/compare.py --prompt-lengths 128,2048 --tokens 128 --requests 8 \
      --engine glm-vx=glm-vx@http://127.0.0.1:8000@pid=1234 \
      --engine vllm=vllm@http://127.0.0.1:8001@pid=2345 \
      --engine llama.cpp=llamacpp@http://127.0.0.1:8002@pid=3456 \
      --model-note "GLM-5.3 FP8, A100-80G" --output compare.json

Engine spec: NAME=KIND@URL[@pid=N][@version=TEXT]. KIND is glm-vx | vllm | llamacpp. Servers are
started separately (same prompts, greedy decoding); this script only measures.

Measurement notes (printed into the output): TTFT is client wall-clock to the first streamed token.
Prefill tokens/s is prompt_tokens / TTFT, which includes the first decode step, and is also reported
from server-side timings when the engine provides them (llama.cpp does). Decode tokens/s is
(output_tokens - 1) / (last token time - first token time). Each streamed chunk is counted as one
output token unless the engine reports token IDs or usage, which take precedence. Peak GPU memory
is the maximum of `nvidia-smi` memory.used (whole device, so run one engine at a time) sampled
while requests run; peak host memory is the server pid's VmHWM from /proc.
"""
import argparse
import http.client
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

KINDS = ('glm-vx', 'vllm', 'llamacpp')


def request_body(kind, prompt, tokens):
    body = {'prompt': prompt, 'max_tokens': tokens, 'temperature': 0, 'stream': True}
    if kind == 'vllm':
        body['stream_options'] = {'include_usage': True}
        body['ignore_eos'] = True
    elif kind == 'llamacpp':
        body['ignore_eos'] = True
        body['return_tokens'] = True
        body['cache_prompt'] = False
        body['n_predict'] = tokens
    return body


def completion(kind, url, prompt, tokens, timeout=600):
    """One streamed completion; returns a dict with ok=True/False and every raw number."""
    started = time.perf_counter()
    stamps, chunk_tokens, usage, timings, finish = [], 0, {}, {}, None
    conn = None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('URL must be HTTP(S)')
        cls = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
        conn = cls(parsed.hostname, parsed.port, timeout=timeout)
        conn.request('POST', parsed.path.rstrip('/') + '/v1/completions',
                     json.dumps(request_body(kind, prompt, tokens)), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read(300)!r}')
        done = False
        while not done:
            if time.perf_counter() - started > timeout:
                raise TimeoutError('request deadline exceeded')
            line = response.readline(1 << 20)
            if not line:
                break
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                continue
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(str(event['error']))
            if event.get('usage'):
                usage = event['usage']
            if event.get('timings'):
                timings = event['timings']
            for choice in event.get('choices') or []:
                ids = choice.get('token_ids') or choice.get('tokens')
                piece = choice.get('text')
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
                    continue  # terminal chunks repeat/close; they are not a new token
                n = len(ids) if isinstance(ids, list) and ids else (1 if piece else 0)
                if n:
                    stamps.extend([time.perf_counter()] * n)
                    chunk_tokens += n
        if not done:
            raise RuntimeError('stream ended without [DONE]')
        out = int(usage.get('completion_tokens') or chunk_tokens)
        if out < 1:
            raise RuntimeError('no output tokens')
        return {'ok': True, 'prompt_tokens': len(prompt), 'output_tokens': out, 'finish_reason': finish,
                'streamed_tokens': chunk_tokens, 'ttft_s': stamps[0] - started if stamps else None,
                'decode_s': (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
                'latency_s': time.perf_counter() - started, 'server_timings': timings}
    except Exception as exc:
        return {'ok': False, 'prompt_tokens': len(prompt), 'latency_s': time.perf_counter() - started,
                'error': f'{type(exc).__name__}: {exc}'}
    finally:
        if conn is not None:
            conn.close()


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * q
    lo, hi = int(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(rows, elapsed):
    ok = [r for r in rows if r['ok']]
    decode_rates = [(r['output_tokens'] - 1) / r['decode_s'] for r in ok if r.get('decode_s')]
    prefill_rates = [r['prompt_tokens'] / r['ttft_s'] for r in ok if r.get('ttft_s')]
    server_prefill = [r['server_timings']['prompt_per_second'] for r in ok if r['server_timings'].get('prompt_per_second')]
    server_decode = [r['server_timings']['predicted_per_second'] for r in ok if r['server_timings'].get('predicted_per_second')]
    stat = lambda v: {'p50': percentile(v, .5), 'p95': percentile(v, .95), 'min': min(v) if v else None,
                      'max': max(v) if v else None}
    return {'requests': len(rows), 'successful': len(ok), 'failed': len(rows) - len(ok),
            'errors': sorted({r['error'] for r in rows if not r['ok']})[:5],
            'elapsed_s': elapsed, 'output_tokens': sum(r['output_tokens'] for r in ok),
            'aggregate_output_tokens_per_s': sum(r['output_tokens'] for r in ok) / elapsed if elapsed else None,
            'ttft_s': stat([r['ttft_s'] for r in ok if r.get('ttft_s') is not None]),
            'prefill_tokens_per_s_client': stat(prefill_rates),
            'decode_tokens_per_s_client': stat(decode_rates),
            'prefill_tokens_per_s_server': stat(server_prefill),
            'decode_tokens_per_s_server': stat(server_decode)}


class MemorySampler:
    """Samples whole-device GPU memory.used (nvidia-smi) and the server pid's VmHWM."""

    def __init__(self, pid=None, interval=0.25):
        self.pid, self.interval = pid, interval
        self.gpu_mib_peak, self.gpu_available = None, True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def host_peak_kib(pid):
        try:
            for line in Path(f'/proc/{pid}/status').read_text().splitlines():
                if line.startswith('VmHWM:'):
                    return int(line.split()[1])
        except OSError:
            return None
        return None

    def _gpu_mib(self):
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            return max(int(x) for x in out.split())
        except (OSError, subprocess.SubprocessError, ValueError):
            self.gpu_available = False
            return None

    def _run(self):
        while not self._stop.is_set() and self.gpu_available:
            value = self._gpu_mib()
            if value is not None and (self.gpu_mib_peak is None or value > self.gpu_mib_peak):
                self.gpu_mib_peak = value
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def result(self):
        return {'gpu_memory_used_mib_peak': self.gpu_mib_peak, 'gpu_sampled': self.gpu_available,
                'server_host_rss_peak_kib': self.host_peak_kib(self.pid) if self.pid else None}


def environment():
    def smi(query):
        try:
            return subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    meminfo = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines()[:2]:
            key, value = line.split(':')
            meminfo[key] = int(value.split()[0])
    except OSError:
        pass
    return {'vcpus': os.cpu_count(), 'host_ram_gib': round(meminfo.get('MemTotal', 0) / 1048576, 1) or None,
            'gpu': smi('name,memory.total,driver_version,compute_cap'), 'platform': platform.platform(),
            'python': platform.python_version()}


def parse_engine(spec):
    name, _, rest = spec.partition('=')
    parts = rest.split('@')
    if not name or len(parts) < 2 or parts[0] not in KINDS:
        raise argparse.ArgumentTypeError(f'engine spec must be NAME=KIND@URL[@pid=N][@version=TEXT], KIND in {KINDS}')
    engine = {'name': name, 'kind': parts[0], 'url': parts[1], 'pid': None, 'version': None}
    for extra in parts[2:]:
        key, _, value = extra.partition('=')
        if key == 'pid':
            engine['pid'] = int(value)
        elif key == 'version':
            engine['version'] = value
        else:
            raise argparse.ArgumentTypeError(f'unknown engine option {key!r}')
    return engine


def make_prompt(index, length, vocab):
    return [(index * 17 + j * 3 + 1) % vocab for j in range(length)]


def run_engine(engine, args):
    lengths = [int(n) for n in args.prompt_lengths.split(',')]
    prompts = [make_prompt(i, lengths[i % len(lengths)], args.vocab_size) for i in range(args.requests)]
    warmup = [completion(engine['kind'], engine['url'], make_prompt(10_000 + i, lengths[0], args.vocab_size),
                         min(args.tokens, 8), args.timeout) for i in range(args.warmup)]
    result = {'engine': {k: v for k, v in engine.items()}, 'warmup_ok': all(w['ok'] for w in warmup)}
    if not result['warmup_ok']:
        result.update(status='warmup_failed', warmup=warmup)
        return result
    sampler = MemorySampler(engine['pid'])
    with sampler:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda p: completion(engine['kind'], engine['url'], p, args.tokens, args.timeout),
                                 prompts))
        elapsed = time.perf_counter() - start
    result.update(status='ok' if all(r['ok'] for r in rows) else 'failed', memory=sampler.result(),
                  metrics=summarize(rows, elapsed),
                  by_prompt_length={str(n): summarize([r for r in rows if r['prompt_tokens'] == n], elapsed)
                                    for n in sorted(set(lengths))},
                  requests=rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', action='append', type=parse_engine, required=True)
    p.add_argument('--requests', type=int, default=8)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--prompt-lengths', default='128')
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--vocab-size', type=int, default=154880)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--model-note', default='', help='free text: model, precision, checkpoint hash, flags')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error('output exists; preserve evidence')
    names = [e['name'] for e in args.engine]
    if len(set(names)) != len(names):
        p.error('duplicate engine names')
    results = [run_engine(e, args) for e in args.engine]
    report = {'schema': 'glm-vx-engine-comparison-v1', 'environment': environment(), 'model_note': args.model_note,
              'workload': {k: v for k, v in vars(args).items() if k not in ('engine', 'output')},
              'engines': results, 'measurement': __doc__.split('Measurement notes (printed into the output): ')[1].split('\n\n')[0].replace('\n', ' ')}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for r in results:
        m = r.get('metrics')
        print(r['engine']['name'], r['status'],
              '' if not m else f"decode p50 {m['decode_tokens_per_s_client']['p50']} tok/s, ttft p50 {m['ttft_s']['p50']} s")
    return 0 if all(r['status'] == 'ok' for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
