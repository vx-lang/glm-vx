"""The indexer's stable top-k kernels against context length (#32).

gpu/topk.vx's radix select (topk_stable, one block, 256 histogram rows in device memory cleared
and summed four times), gpu/spawn's topk_sort (one shared-memory bitonic sort, up to 2048
tokens) and, past 2048 tokens, topk_chunks then topk_merge rounds (a block sorts each 2048-token
chunk, then rows merge pairwise), at GLM-5.3's index_topk of 2048. The chunk path's time is that
of all its launches, back to back on one stream. Each launch is timed with CUDA events after a 512 MB
copy (moe_profile.l2_flush), the median of --repeats; every id list is checked against
np.argsort(-scores, kind='stable').

  python -m validation.topk_profile --output run/topk.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

K = 2048
TOKENS = [64, 256, 1024, 2048, 8192, 32768, 131072]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    import ctypes as C
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    from gpu.topk import scratch_ints
    build = Path(__file__).resolve().parent.parent / 'gpu' / 'build'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    rng = np.random.default_rng(1)
    radix = ctx.load_ptx(build / 'topk.ptx').kernel('glm_vx_gpu_topk_stable')
    sort = SpawnKernel(ctx, build / 'spawn' / 'topk_sort.ptx')
    chunk_sort = SpawnKernel(ctx, build / 'spawn' / 'topk_chunks.ptx')
    merge = SpawnKernel(ctx, build / 'spawn' / 'topk_merge.ptx')
    results = []
    for tokens in TOKENS:
        scores = rng.normal(size=tokens).astype('f4')
        want = np.argsort(-scores, kind='stable')[:min(K, tokens)]
        d_scores = ctx.tensor((tokens,), 'int32', scores.view('i4').tobytes())
        scratch = ctx.tensor((scratch_ints(tokens, K),), 'int32')
        outs = {name: ctx.tensor((min(K, tokens),), 'int32') for name in (('radix', 'sort') if tokens <= 2048 else ('radix', 'merge'))}
        chunks = -(-tokens // 2048)
        lists = [ctx.tensor((chunks, 2048), 'int64') for _ in range(2)]

        def chunks_and_merge(out):
            chunk_sort.launch(stream=stream, lists=lists[0], scores=d_scores, tokens=tokens)
            count, r = chunks, 0
            while count > 1:
                merge.launch(stream=stream, dst=lists[(r + 1) % 2], src=lists[r % 2], out_ids=out.view((1, K)), lists=count,
                             tokens=tokens, k=K)
                count, r = -(-count // 2), r + 1

        runs = {
            'radix': lambda out: radix.launch((1,), (256,), [out, d_scores, scratch, C.c_int32(tokens), C.c_int32(K)],
                                              stream=stream),
            'sort': lambda out: sort.launch(stream=stream, out_ids=out, scores=d_scores, tokens=tokens, k=K),
            'merge': chunks_and_merge}
        for name, out in outs.items():
            runs[name](out)
            stream.synchronize()
            if not np.array_equal(np.frombuffer(out.download(), 'i4'), want):
                raise SystemExit(f'{name} at {tokens} tokens: not the stable argsort')
            us = event_us(ctx, stream, build / 'spawn', lambda: runs[name](out), args.repeats)
            results.append(dict(kernel=name, tokens=tokens, k=K, us=us))
    ctx.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(k=K, results=results), indent=2) + '\n')
    by = {(r['kernel'], r['tokens']): r['us'] for r in results}
    lines = [f'# Stable top-k, k = {K}. CUDA events, L2 flushed, median of {args.repeats}.', '',
             '| tokens | topk_stable (radix) us | topk_sort us | topk_chunks + topk_merge us |', '|---|---|---|---|']
    cell = lambda name, t: f"{by[(name, t)]:.1f}" if (name, t) in by else '-'
    lines += [f"| {t} | {cell('radix', t)} | {cell('sort', t)} | {cell('merge', t)} |" for t in TOKENS]
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
