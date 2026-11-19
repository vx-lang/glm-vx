"""GLM-5.3 itself (the FP8 checkpoint) split over --ranks GPUs (#35, #63): load, greedy text, step time.

One process a rank, rank r on GPU r (every GPU visible: the one-shot sum maps peers' memory through
CUDA IPC), joined by Vx's NCCL collectives (gpu/collective.py). Each rank reads the checkpoint lazily
(glm_vx.checkpoint, memory-mapped shards) and keeps its tensor-parallel slice (gpu/tp.ShardedModel);
the weights go to the GPU on the first step. Then:

  1. greedy decode of --new tokens after the prompt (token by token: forward_many is not split yet),
     every step's top-5 logits and log-softmax, for comparison with SGLang's (validation/sglang_greedy.py);
  2. for each --contexts N: a new sequence of N random tokens (seed 0), then --time steps timed, each
     the whole step: layers (a replayed graph), final norm, lm_head, the logits' gather and download.

  python -m validation.tp_real --checkpoint ~/GLM-5.3 --ranks 8 --one-shot --output run/real.json

--prefill fills the prompt (its tokens after the first and before the last) and each context through
ResidentGLM.prefill_chunk (#63, item 5b) in chunks of its capacity (--option prefill_rows=N), and records
the first decode step's top-5 at each context to compare with a run without it.
"""
import argparse
import ctypes as C
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

OPTIONS = dict(device_experts=True, graph=True, fused_attention=True, mla_kernel='tc', fp8_kernels='cvt',
               matvec_kernels='x4', norm_block=1024, long_topk='ilp_hist', fuse_glue=True, index_kernel='mma',
               fold_shared=True)
PROMPT = ('The history of the transistor begins in 1947 at Bell Labs, where John Bardeen, Walter Brattain and '
          'William Shockley')


def top(logits, k=5):
    ids = np.argsort(-logits, kind='stable')[:k]
    peak = float(logits.max())
    lse = peak + float(np.log(np.exp(logits.astype(np.float64) - peak).sum()))
    return dict(ids=[int(i) for i in ids], logits=[float(logits[i]) for i in ids],
                logprobs=[float(logits[i]) - lse for i in ids])


def rank_main(args, rank, job):
    from glm_vx.backend import NumpyBackend
    from glm_vx.checkpoint import Checkpoint
    from glm_vx.config import GLMConfig
    from glm_vx.model import GlmMoeDsaModel
    from gpu.collective import Collective
    from gpu.resident import ResidentGLM
    from gpu.runtime import CUDAContext
    from gpu.tp import TensorParallel
    job = json.loads(Path(job).read_text())
    started = time.time()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    if args.layers is not None:  # the first layers only (a smoke test): GLM-5.3's own per-layer patterns
        from validation.write_proxy import proxy_config
        config = proxy_config(args.layers, base=config)
    checkpoint = Checkpoint(args.checkpoint, block_size=GLMConfig.from_dict(config).weight_block_size)
    model = GlmMoeDsaModel(config, checkpoint, NumpyBackend())
    context = CUDAContext(rank, primary=True, minimum_compute_capability=(8, 0))
    tp = None
    if args.ranks > 1:
        collective = Collective(C.CDLL(os.environ['VX_DISPATCH_LIB']), context, rank, args.ranks)
        tp = TensorParallel(rank, args.ranks, collective, one_shot=args.one_shot)
    longest = max([len(job['prompt']) + args.new] + [n + args.time for n in args.contexts])
    from gpu.tp_serve import parse_options
    r = ResidentGLM(model, ctx=context, max_tokens=longest + 8, tp=tp, **{**OPTIONS, **parse_options(args.option)})
    result = dict(rank=rank)

    def note(**values):
        result.update(values)
        if rank == 0:
            print(json.dumps(dict(rank=rank, **values))[:400], file=sys.stderr, flush=True)

    # 1. The prompt, then greedy tokens. The first step uploads every weight.
    prompt = job['prompt']
    first = time.time()
    r.forward(prompt[0], logits=False)
    context.synchronize()
    note(setup_s=round(first - started, 1), first_step_s=round(time.time() - first, 1),
         device_bytes=getattr(context, 'allocated_bytes', None))
    def fill(tokens):
        if args.prefill:
            rows = r._prefill_capacity()
            for s in range(0, len(tokens), rows):
                r.prefill_chunk(tokens[s:s + rows])
        else:
            for t in tokens:
                r.forward(t, logits=False)

    fill(prompt[1:-1])
    logits = r.forward(prompt[-1])
    generated, tops = [], []
    for _ in range(args.new):
        tops.append(top(logits))
        generated.append(int(np.argmax(logits)))
        logits = r.forward(generated[-1])
    note(generated=generated, tops=tops if rank == 0 else None)

    # 2. Step time at each context.
    rng = np.random.default_rng(0)
    vocab = int(config['vocab_size'])
    timings = {}
    # With --option 'topk_bitmap="stats"': topk_select_bitmap_stats_at's counts (#63) over each context's fill
    # and over its timed steps: launches, launches whose threshold bin held more than 2048 keys (the rounds
    # read the scores again), the most keys it held, launches whose bin held more than 1024.
    stats = getattr(r, 'topk_stats', None)

    def topk_stats():
        counts = np.frombuffer(stats.download(), np.uint32).tolist()
        stats.upload(np.zeros(4, np.uint32).tobytes())
        return counts

    for n in args.contexts:
        r.reset()
        if stats is not None:
            topk_stats()
        tokens = [int(t) for t in rng.integers(0, vocab, n + args.time)]
        start = time.perf_counter()
        fill(tokens[:n])
        context.synchronize()
        filled = time.perf_counter() - start
        fill_stats = topk_stats() if stats is not None else None
        steps, first = [], None
        for t in tokens[n:n + args.time]:
            start = time.perf_counter()
            logits = r.forward(t)  # returns the downloaded logits: the step is done
            steps.append((time.perf_counter() - start) * 1e6)
            first = top(logits) if first is None else first
        steps = np.array(steps[2:])  # the first steps past a context capture the long graphs
        timings[str(n)] = dict(fill_tokens_per_s=round(n / filled, 1), fill_s=round(filled, 3),
                               us_per_step_median=round(float(np.median(steps)), 1),
                               us_per_step_mean=round(float(steps.mean()), 1), us_per_step_min=round(float(steps.min()), 1),
                               first_top=first if rank == 0 else None)
        if stats is not None:
            timings[str(n)]['topk_stats'] = dict(fill=fill_stats, steps=topk_stats())
        note(**{f'context_{n}': timings[str(n)]})
    result['timings'] = timings
    result['replays'] = r.replays
    print('RESULT ' + json.dumps(result), flush=True)
    r.close()
    if tp is not None:
        tp.collective.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--ranks', type=int, default=8)
    p.add_argument('--layers', type=int, help='only the first LAYERS decoder layers (default: all)')
    p.add_argument('--one-shot', action='store_true', help='TensorParallel(one_shot=True): gpu/spawn/tp_all_reduce')
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--new', type=int, default=32, help='greedy tokens after the prompt')
    p.add_argument('--contexts', type=int, nargs='*', default=[1000, 3001])
    p.add_argument('--time', type=int, default=50, help='steps timed at each context')
    p.add_argument('--output', type=Path)
    p.add_argument('--prefill', action='store_true', help='fill the prompt and the contexts with prefill_chunk (#63, item 5b)')
    p.add_argument('--option', action='append', help='KEY=VALUE (JSON): a ResidentGLM option over OPTIONS, e.g. fuse_tp_norm=8')
    p.add_argument('--role', help=argparse.SUPPRESS)
    p.add_argument('--job', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.role is not None:
        return rank_main(args, int(args.role), args.job)
    if args.output is not None and args.output.exists():
        p.error('output exists; preserve evidence')
    if args.ranks > 1 and 'VX_DISPATCH_LIB' not in os.environ:
        p.error('VX_DISPATCH_LIB is not set (source scripts/env.sh)')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    prompt = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    command = [sys.executable, '-m', 'validation.tp_real', '--checkpoint', str(args.checkpoint), '--ranks', str(args.ranks),
               '--new', str(args.new), '--time', str(args.time), '--contexts', *map(str, args.contexts)]
    command += ['--one-shot'] if args.one_shot else []
    command += ['--prefill'] if args.prefill else []
    for option in args.option or ():
        command += ['--option', option]
    command += ['--layers', str(args.layers)] if args.layers is not None else []
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'job.json').write_text(json.dumps(dict(prompt=prompt)))
        environment = dict(os.environ, VX_NCCL_ID_FILE=str(tmp / 'nccl-id'))
        processes = [subprocess.Popen(command + ['--role', str(r), '--job', str(tmp / 'job.json')], env=environment,
                                      stdout=subprocess.PIPE, text=True) for r in range(args.ranks)]
        outputs = [process.communicate()[0] for process in processes]
    results = []
    for r, (process, output) in enumerate(zip(processes, outputs)):
        lines = [line[7:] for line in output.splitlines() if line.startswith('RESULT ')]
        results.append(json.loads(lines[-1]) if process.returncode == 0 and lines else dict(rank=r, error=f'exit {process.returncode}'))
    ok = all('error' not in r for r in results)
    if ok:  # every rank all-gathers the same logits: the same greedy tokens
        ok = all(r['generated'] == results[0]['generated'] for r in results)
    from gpu.tp_serve import parse_options
    summary = dict(checkpoint=str(args.checkpoint), layers=args.layers, ranks=args.ranks, one_shot=args.one_shot, prefill=args.prefill,
                   options={**OPTIONS, **parse_options(args.option)}, ok=ok,
                   wall_s=round(time.time() - started, 1), prompt=args.prompt, prompt_ids=prompt)
    if 'error' not in results[0]:
        summary.update(generated=results[0]['generated'], text=tokenizer.decode(results[0]['generated']),
                       tops=results[0]['tops'], timings=results[0]['timings'],
                       setup_s=results[0]['setup_s'], first_step_s=results[0]['first_step_s'])
    summary['ranks_results'] = [{k: v for k, v in r.items() if k not in ('tops', 'generated')} for r in results]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'tops'})[:3000])
    print('tp_real:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
