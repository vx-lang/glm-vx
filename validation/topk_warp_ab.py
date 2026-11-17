"""topk_select_at against topk_select_warp_at (#32), each launch alone: CUDA events after an L2 flush
(moe_profile.event_us), the median of --repeats, at GLM-5.3's index_topk of 2048. Both get the step
buffer [position, tokens, ...] and must return np.sort(np.argsort(-scores, kind='stable')[:2048]).

  python -m validation.topk_warp_ab --output run/topk-warp.md
"""
import argparse
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

K = 2048
TOKENS = [2049, 3001, 4096, 8192, 16384, 32768, 65536, 131072, 200000]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--tokens', type=int, nargs='+', default=TOKENS)
    args = p.parse_args()
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    spawn = Path(__file__).resolve().parent.parent / 'gpu' / 'build' / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    kernels = {'topk_select_at': SpawnKernel(ctx, spawn / 'topk_select_at.ptx'),
               'topk_select_warp_at': SpawnKernel(ctx, spawn / 'index_mma' / 'topk_select_warp_at.ptx')}
    rng = np.random.default_rng(1)
    rows = ['| tokens | ' + ' | '.join(f'{n} us' for n in kernels) + ' |', '|---|' + '---|' * len(kernels)]
    for tokens in args.tokens:
        scores = rng.normal(size=tokens).astype('f4')
        want = np.sort(np.argsort(-scores, kind='stable')[:K])
        d_scores = ctx.tensor((tokens,), 'int32', scores.view('i4').tobytes())
        step = ctx.tensor((4,), 'int32', np.array([tokens - 1, tokens, K, 1], 'i4').tobytes())
        times = []
        for name, kernel in kernels.items():
            out = ctx.tensor((K,), 'int32')
            launch = lambda: kernel.launch(stream=stream, out_ids=out, scores=d_scores, step=step, k=K)
            launch()
            stream.synchronize()
            got = np.frombuffer(out.download(), 'i4')
            if not np.array_equal(got, want):
                raise SystemExit(f'{name} at {tokens} tokens: not the stable set')
            times.append(event_us(ctx, stream, spawn, launch, repeats=args.repeats))
        rows.append(f'| {tokens} | ' + ' | '.join(f'{t:.1f}' for t in times) + ' |')
        print(rows[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
