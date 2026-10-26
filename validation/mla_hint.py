"""mla_decode_at as built against the same PTX with `.minnctapersm 1`, at GLM-5.3's shapes.

This is a Vx codegen question (from vx-21), not a model measurement. On sm_90, ptxas gives
mla_decode_at's 1024-thread block 32 registers so that 2 blocks fit on an SM, and the kernel
spills 8 bytes. With `.minnctapersm 1` after `.maxntid` it gets 64 registers and no spill, but
then only 1 block fits per SM. Vx emits the hint only if it wins on hardware.

Both variants run on the same inputs. The outputs and the arrival counts must match bit for bit,
and each launch is timed with CUDA events (median of 20, L2 flushed before each):
- 64 heads, rank 512, rope 64, which is a full-width layer;
- selected counts 256, 1024 and 2048 (the config's index_topk), with splits as ResidentGLM picks
  them, and splits_max 16.

  python -m validation.mla_hint --output run/mla-hint.json   # the hinted PTX lands in run/mla-hint.ptx-variant/
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

HEADS, RANK, ROPE, TOKENS, SPLITS_MAX = 64, 512, 64, 4096, 16


def hinted(source, directory):
    """A copy of source (.ptx and its .json manifest) with `.minnctapersm 1` after `.maxntid`."""
    ptx = source.read_text()
    out, n = re.subn(r'^(\.maxntid 1024, 1, 1)$', r'\1\n.minnctapersm 1', ptx, flags=re.M)
    if n != 1:
        raise SystemExit(f'{source}: expected one `.maxntid 1024, 1, 1`, found {n}')
    target = directory / source.name
    target.write_text(out)
    shutil.copy(source.with_suffix('.json'), target.with_suffix('.json'))
    return target


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
    variant = args.output.with_suffix('.ptx-variant')  # kept with the evidence
    variant.mkdir(parents=True)
    kernels = {'as built': SpawnKernel(ctx, spawn / 'mla_decode_at.ptx'),
               'minnctapersm 1': SpawnKernel(ctx, hinted(spawn / 'mla_decode_at.ptx', variant))}
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
        outputs = {}
        for name, kernel in kernels.items():
            values = dict(common, out=ctx.tensor((HEADS, RANK)), numerator=ctx.tensor((HEADS, SPLITS_MAX, RANK)),
                          stats=ctx.tensor((HEADS, SPLITS_MAX, 2)), selected=selected, step=step,
                          arrivals=up(np.zeros(HEADS, 'i4')))
            kernel.launch(stream=stream, **values)
            stream.synchronize()
            outputs[name] = (values['out'].download(), values['arrivals'].download())
            us = event_us(ctx, stream, spawn, lambda: kernel.launch(stream=stream, **values), args.repeats)
            results.append(dict(variant=name, count=count, splits=splits, us=us))
        (a_out, a_arr), (b_out, b_arr) = outputs.values()
        if a_out != b_out or a_arr != b_arr or np.frombuffer(a_arr, 'i4').any():
            raise SystemExit(f'count {count}: the variants differ, or arrivals are not back at zero')
    ctx.close()
    args.output.write_text(json.dumps(dict(heads=HEADS, rank=RANK, rope=ROPE, tokens=TOKENS, results=results), indent=2) + '\n')
    lines = ['| Variant | selected | splits | GPU us (median) |', '|---|---|---|---|']
    lines += [f"| {r['variant']} | {r['count']} | {r['splits']} | {r['us']:.1f} |" for r in results]
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
