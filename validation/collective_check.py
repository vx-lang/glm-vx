"""Vx's NCCL collectives through gpu/collective.py on real GPUs (#34, Vx PR #2306).

Starts --ranks processes, one per GPU (CUDA_VISIBLE_DEVICES=<rank>), with a fresh
VX_NCCL_ID_FILE. Each rank checks, bit for bit (small integers, so every sum is exact):
  all_reduce_sum f32 and bf16, in place and into another buffer; all_gather f32 and bf16;
  the same all-reduce recorded into a CUDA graph and replayed with new inputs.
Then it times a graph of --calls all-reduces of --hidden elements (wall time over the replay,
divided by --calls): the cost of one tensor-parallel all-reduce of a decode step.

Needs VX_DISPATCH_LIB (scripts/env.sh) from a Vx build with PR #2306, and libnccl.so.2 (or
VX_NCCL_LIB).

  python -m validation.collective_check --ranks 2 --output run/collective.json
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


def bf16(values):
    return (np.asarray(values, 'f4').view('u4') >> 16).astype('u2')


def rank_main(rank, ranks, hidden, calls, replays):
    from gpu.collective import Collective
    from gpu.runtime import CUDAContext
    context = CUDAContext(primary=True)
    stream = context.stream(nonblocking=False)
    collective = Collective(C.CDLL(os.environ['VX_DISPATCH_LIB']), context, rank, ranks)
    up = lambda a: context.tensor(a.shape, str(a.dtype), a.tobytes())
    down = lambda t: np.frombuffer(t.download(), str(t.dtype)).reshape(t.shape)
    failures = []

    def check(name, got, want):
        if not np.array_equal(got, want):
            failures.append(name)
            print(f'rank {rank}: {name} FAILED, first {got.ravel()[:4]} want {want.ravel()[:4]}', flush=True)

    def values(r, seed):  # integers in [-64, 64]: every sum over ranks is exact in f32 and bf16
        return np.random.default_rng(seed * 1000 + r).integers(-64, 65, hidden).astype('f4')

    every = lambda seed: [values(r, seed) for r in range(ranks)]
    mine = values(rank, 1)
    total = np.sum(every(1), axis=0, dtype='f4')
    x = up(mine)
    collective.all_reduce_sum(x, stream=stream)
    stream.synchronize()
    check('all_reduce_sum f32 in place', down(x), total)
    x, out = up(mine), context.tensor((hidden,))
    collective.all_reduce_sum(x, out=out, stream=stream)
    stream.synchronize()
    check('all_reduce_sum f32 into out', down(out), total)
    check('all_reduce_sum f32 leaves in', down(x), mine)
    x = up(bf16(mine))
    collective.all_reduce_sum(x, stream=stream)
    stream.synchronize()
    check('all_reduce_sum bf16', down(x), bf16(total))
    for dtype, convert in (('float32', lambda a: a), ('uint16', bf16)):
        out = context.tensor((ranks * hidden,), dtype)
        collective.all_gather(up(convert(mine)), out, stream=stream)
        stream.synchronize()
        check(f'all_gather {dtype}', down(out), convert(np.concatenate(every(1))))

    x = up(mine)
    with stream.capture() as graph:
        collective.all_reduce_sum(x, stream=stream)
    for i in range(replays):
        x.upload(values(rank, 10 + i).tobytes())
        graph.launch(stream)
        stream.synchronize()
        check(f'graph replay {i}', down(x), np.sum(every(10 + i), axis=0, dtype='f4'))

    timings = {}
    for dtype, convert in (('float32', lambda a: a), ('uint16', bf16)):
        x = up(convert(np.zeros(hidden, 'f4')))
        with stream.capture() as graph:
            for _ in range(calls):
                collective.all_reduce_sum(x, stream=stream)
        graph.launch(stream)
        stream.synchronize()
        start = time.perf_counter()
        graph.launch(stream)
        stream.synchronize()
        timings[dtype] = (time.perf_counter() - start) / calls * 1e6
    collective.finish()
    print(json.dumps(dict(rank=rank, failures=failures, us_per_all_reduce=timings)), flush=True)
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--hidden', type=int, default=6144)
    p.add_argument('--calls', type=int, default=200)
    p.add_argument('--replays', type=int, default=3)
    p.add_argument('--output', type=Path)
    p.add_argument('--rank', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.rank is not None:
        return rank_main(args.rank, args.ranks, args.hidden, args.calls, args.replays)
    if args.output is not None and args.output.exists():
        p.error('output exists; preserve evidence')
    if 'VX_DISPATCH_LIB' not in os.environ:
        p.error('VX_DISPATCH_LIB is not set (source scripts/env.sh)')
    with tempfile.TemporaryDirectory() as tmp:
        environment = dict(os.environ, VX_NCCL_ID_FILE=str(Path(tmp) / 'nccl-id'))
        processes = [subprocess.Popen([sys.executable, '-m', 'validation.collective_check', '--rank', str(r),
                                       '--ranks', str(args.ranks), '--hidden', str(args.hidden),
                                       '--calls', str(args.calls), '--replays', str(args.replays)],
                                      env=dict(environment, CUDA_VISIBLE_DEVICES=str(r)), stdout=subprocess.PIPE,
                                      text=True)
                     for r in range(args.ranks)]
        outputs = [process.communicate()[0] for process in processes]
    results = []
    for r, (process, output) in enumerate(zip(processes, outputs)):
        sys.stdout.write(output)
        lines = [line for line in output.splitlines() if line.startswith('{')]
        results.append(json.loads(lines[-1]) if process.returncode == 0 and lines else
                       dict(rank=r, failures=[f'exit {process.returncode}']))
    ok = all(not result['failures'] for result in results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(ranks=args.ranks, hidden=args.hidden, calls=args.calls,
                                               ok=ok, results=results), indent=2) + '\n')
    print('collective_check:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
