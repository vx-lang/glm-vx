"""The fused MLA decode kernels at GLM-5.3's attention shape (#33).

gpu/spawn's mla_decode_at (one token at a time per block), mla_chunks_at (32 tokens a chunk) and
mla_heads_at (mla_chunks_at with four heads a block, each latent row read once for them),
64 heads, rank 512, rope 64, 4096 cached tokens, for 256, 1024 and 2048 selected tokens with
splits as ResidentGLM picks them (splits_max 16). Each launch is timed with CUDA events after a
512 MB copy (moe_profile.l2_flush), the median of --repeats; mla_chunks_at must agree with
mla_decode_at to 1e-4, mla_heads_at must equal mla_chunks_at, and the arrival counts return to zero.

  python -m validation.mla_profile --output run/mla.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

HEADS, RANK, ROPE, TOKENS, SPLITS_MAX = 64, 512, 64, 4096, 16
KERNELS = ('mla_decode_at', 'mla_chunks_at', 'mla_heads_at')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    spawn = Path(__file__).resolve().parent.parent / 'gpu' / 'build' / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    rng = np.random.default_rng(1)
    up = lambda a: ctx.tensor(a.shape, str(a.dtype), a.tobytes())
    kernels = {name: SpawnKernel(ctx, spawn / f'{name}.ptx') for name in KERNELS}
    common = dict(q_latent=up(rng.normal(size=HEADS * RANK).astype('f4')),
                  q_rope=up(rng.normal(size=HEADS * ROPE).astype('f4')),
                  cache_latent=up(rng.normal(size=TOKENS * RANK).astype('f4')),
                  cache_rope=up(rng.normal(size=TOKENS * ROPE).astype('f4')),
                  heads=HEADS, rank=RANK, rope_dim=ROPE, selection_stride=0, splits_max=SPLITS_MAX,
                  scale=float(np.float32(RANK ** -0.5)), threads=((RANK + 31) // 32) * 32)
    results = []
    for count in (256, 1024, 2048):
        splits = min(SPLITS_MAX, max(1, (count + 255) // 256))
        selected = up(rng.permutation(TOKENS)[:count].astype('i4'))
        step = up(np.array([TOKENS - 1, TOKENS, count, splits], 'i4'))
        outs = {}
        for name, kernel in kernels.items():
            # mla_heads_at takes the buffers per group of four heads.
            g = 4 if name == 'mla_heads_at' else 1
            values = dict(common, out=ctx.tensor((HEADS // g, g * RANK)),
                          numerator=ctx.tensor((HEADS // g, SPLITS_MAX, g * RANK)),
                          stats=ctx.tensor((HEADS // g, SPLITS_MAX, 2 * g)), selected=selected, step=step,
                          arrivals=up(np.zeros(HEADS // g, 'i4')))
            kernel.launch(stream=stream, **values)
            stream.synchronize()
            outs[name] = np.frombuffer(values['out'].download(), 'f4')
            us = event_us(ctx, stream, spawn, lambda: kernel.launch(stream=stream, **values), args.repeats)
            if np.frombuffer(values['arrivals'].download(), 'i4').any():
                raise SystemExit(f'{name} at {count}: arrival counts not back at zero')
            results.append(dict(kernel=name, count=count, splits=splits, us=us))
        if not np.allclose(outs['mla_chunks_at'], outs['mla_decode_at'], rtol=1e-4, atol=1e-5):
            raise SystemExit(f'count {count}: mla_chunks_at disagrees with mla_decode_at')
        if not np.array_equal(outs['mla_heads_at'].view('u4'), outs['mla_chunks_at'].view('u4')):
            raise SystemExit(f'count {count}: mla_heads_at is not mla_chunks_at bit for bit')
    ctx.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(heads=HEADS, rank=RANK, rope=ROPE, tokens=TOKENS, results=results), indent=2) + '\n')
    by = {(r['kernel'], r['count']): r for r in results}
    lines = [f'# Fused MLA decode, {HEADS} heads, rank {RANK}, rope {ROPE}. CUDA events, L2 flushed, median of '
             f'{args.repeats}.', '', '| selected (splits) | mla_decode_at us | mla_chunks_at us | mla_heads_at us |',
             '|---|---|---|---|']
    lines += [f"| {c} ({by[('mla_decode_at', c)]['splits']}) | {by[('mla_decode_at', c)]['us']:.1f} | "
              f"{by[('mla_chunks_at', c)]['us']:.1f} | {by[('mla_heads_at', c)]['us']:.1f} |" for c in (256, 1024, 2048)]
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
