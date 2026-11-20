"""z60 (glm-vx docs/benchmarks/2026-11-20-h200-moe-down; from z56's vxmm_time.py): the MoE down projection of one decode step at one TP8 rank's shapes (6144 rows,
256 columns, 257 experts, k = 9; validation/moe_mma_check.run's random inputs, seed 1), four kernels built by the same
vxc: the inline-PTX fp8_moe_mma_down_at (act e4m3) and fp8_moe_mma16_down_at (act f16), and their plain-Vx-matmul
rewrites fp8_moe_vxmm_down_at / fp8_moe_vxmm16_down_at (the weights as a f8e4m3 [16, 256] tile in HBM, act through
shared memory). Each against moe_mma_check.model (e4m3 or f16) and against its inline-PTX counterpart (relative to
max |out|); GPU time by CUDA events, the median of --repeats, L2 flushed before each launch (validation/moe_profile).
--only NAME runs one kernel once (for ncu)."""
import argparse
import json
from pathlib import Path

import numpy as np

KERNELS = {'mma16': 'fp8_moe_mma16_down_at', 'vxmm16': 'fp8_moe_vxmm16_down_at', 'mma': 'fp8_moe_mma_down_at'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ptx', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--only')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    from validation.moe_mma_check import model
    from validation.moe_profile import event_us
    spawn = Path('gpu/build/spawn').resolve()
    ctx = CUDAContext(minimum_compute_capability=(9, 0))
    stream = ctx.stream()
    rows, cols, experts, k = 6144, 256, 257, 9
    rng = np.random.default_rng(args.seed)
    codes = rng.integers(0, 256, size=(experts, rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = rng.uniform(0.5e-3, 2e-3, size=(experts, -(-rows // 128) * (cols // 128))).astype(np.float32)
    act = (0.05 * rng.standard_normal(k * cols)).astype(np.float32)
    ids = np.concatenate([rng.choice(experts - 1, size=k - 1, replace=False), [experts - 1]]).astype(np.int32)
    weights = np.concatenate([rng.uniform(0.01, 0.3, size=k - 1), [1.0]]).astype(np.float32)
    t = lambda a, dtype: ctx.tensor((a.size,), dtype, np.ascontiguousarray(a).tobytes())
    common = dict(scales=t(scales.ravel(), 'float32'), act=t(act, 'float32'), weights=t(weights, 'float32'),
                  ids=t(ids, 'int32'), k=k, experts=experts, rows=rows, cols=cols)
    flat = ctx.tensor((codes.size // 4,), 'uint32', codes.tobytes())
    tiles = ctx.tensor((experts * rows // 16, 16, cols), 'uint8', codes.tobytes())
    names = [args.only] if args.only else list(KERNELS)
    kernels, outs = {}, {}
    for name in names:
        kernels[name] = SpawnKernel(ctx, args.ptx / f'{KERNELS[name]}.ptx')
        out = ctx.tensor((rows,), 'float32', np.full(rows, np.nan, np.float32).tobytes())
        down = tiles if name.startswith('vxmm') else flat
        kernels[name].launch(stream=stream, out=out, down=down, **common)
        stream.synchronize()
        outs[name] = (np.frombuffer(out.download(), np.float32).astype(np.float64), out, down)
    if args.only:
        ctx.close()
        return
    times = {n: [] for n in names}
    for rnd in range(args.rounds):  # every kernel once a round, the order reversed every other round
        for name in (names if rnd % 2 == 0 else names[::-1]):
            _, out, down = outs[name]
            kern = kernels[name]
            times[name].append(event_us(ctx, stream, spawn, lambda: kern.launch(stream=stream, out=out, down=down, **common),
                                        args.repeats))
    ref = {'e4m3': model(codes, scales, act, weights, ids, rows, cols), 'f16': model(codes, scales, act, weights, ids, rows, cols, 'f16')}
    peak = float(np.abs(ref['e4m3']).max())
    rel = lambda a, b: float(np.abs(a - b).max() / peak)
    o = {n: outs[n][0] for n in names}
    weight_bytes = k * rows * cols
    res = dict(rows=rows, cols=cols, experts=experts, k=k, weight_mb=weight_bytes / 1e6, max_abs_out=peak,
               finite={n: bool(np.isfinite(o[n]).all()) for n in names},
               vs_model={n: rel(o[n], ref['f16' if n.endswith('16') else 'e4m3']) for n in names},
               vxmm16_vs_mma16=rel(o['vxmm16'], o['mma16']),
               us_rounds=times, us={n: float(np.median(v)) for n, v in times.items()})
    res['tb_per_s'] = {n: weight_bytes / us / 1e6 for n, us in res['us'].items()}
    ctx.close()
    print(json.dumps(res, indent=1))
    if args.output:
        args.output.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
