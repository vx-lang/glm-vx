"""gpu/spawn/moe_mma/fp8_moe_mma_down_at on a GPU (#63, docs/TP-SERVING-PLAN.md item 4): the routed experts'
down projection on the FP8 tensor cores, against

  1. a NumPy model of its arithmetic: act quantized to E4M3 in groups of 128 as the kernel does (scale
     max(amax, 1e-10) / 448, act * (1 / scale) rounded to the nearest code, gpu/fp8_quant.e4m3_encode), the dots
     in float64, times the weight and act scales, the router weights, the experts in ascending id order. Only
     the tensor cores' f32 sums differ, so the error must be small (--tolerance, relative to max |out|);
  1b. the same for fp8_moe_mma16_down_at, act in f16 (a power of two a group, then f16 rounding);
  2. fp8_moe_cvt_down2, the kernel they replace (act in f32 on the CUDA cores): the difference is what quantizing
     act to E4M3 costs, reported, not checked;

then the GPU time of each (CUDA events, the median of --repeats, L2 flushed before each launch).
Random FP8 codes (no NaN), scales and act; the shapes default to one rank of 8 on GLM-5.3 (6144 rows, 256
columns, 256 routed experts + the shared one, k = 9), --cols 2048 is one GPU's.

  python -m validation.moe_mma_check --output run/moe-mma.json [--cols 256 2048]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gpu.fp8_quant import e4m3_encode, e4m3_values

GROUP = 128


def act_codes(x, mode):
    """act's groups [tiles, 128] as the kernels round them: (values float64, scale float64 [tiles])."""
    amax = np.abs(x).max(-1)
    if mode == 'f16':  # a power of two puts the max in [2^14, 2^15): exact, then f16 (fp8_moe_mma16_down_at)
        m = np.maximum(amax, np.float32(2.0 ** -100)).astype(np.float32)
        scale = ((m.view(np.uint32) & np.uint32(0x7F800000)) - np.uint32(14 << 23)).view(np.float32)
        inv = (np.uint32(0x7F000000) - scale.view(np.uint32)).view(np.float32)
        return (x * inv[:, None]).astype(np.float32).astype(np.float16).astype(np.float64), scale.astype(np.float64)
    scale = (np.maximum(amax, np.float32(1e-10)) / np.float32(448)).astype(np.float32)
    inv = (np.float32(1) / scale).astype(np.float32)
    return e4m3_values()[e4m3_encode((x * inv[:, None]).astype(np.float32))].astype(np.float64), scale.astype(np.float64)


def model(codes, scales, act, weights, ids, rows, cols, mode='e4m3'):
    """The kernel's arithmetic in float64: out [rows]; mode 'e4m3' (fp8_moe_mma_down_at) or 'f16' (mma16)."""
    table = e4m3_values()
    order = np.argsort(ids, kind='stable')  # place s -> slot
    srows, tiles = -(-rows // 128), cols // 128
    out = np.zeros(rows, np.float64)
    for s, slot in enumerate(order):
        e = int(ids[slot])
        x = act[s * cols:(s + 1) * cols].astype(np.float32).reshape(tiles, GROUP)
        q, scale = act_codes(x, mode)  # [tiles, 128], [tiles]
        w = table[codes[e]].astype(np.float64).reshape(rows, tiles, GROUP)
        dots = np.einsum('rtk,tk->rt', w, q)  # [rows, tiles]
        ws = scales[e].reshape(srows, tiles)[np.arange(rows) // 128]  # [rows, tiles]
        out += float(weights[slot]) * (dots * ws * scale[None, :]).sum(1)
    return out


def run(ctx, stream, spawn, rng, rows, cols, experts, k, repeats, kernel=None):
    from gpu.spawn.launch import SpawnKernel
    from validation.moe_profile import event_us
    codes = rng.integers(0, 256, size=(experts, rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = rng.uniform(0.5e-3, 2e-3, size=(experts, -(-rows // 128) * (cols // 128))).astype(np.float32)
    act = (0.05 * rng.standard_normal(k * cols)).astype(np.float32)
    ids = np.concatenate([rng.choice(experts - 1, size=k - 1, replace=False), [experts - 1]]).astype(np.int32)
    weights = np.concatenate([rng.uniform(0.01, 0.3, size=k - 1), [1.0]]).astype(np.float32)
    t = lambda a, dtype: ctx.tensor(a.shape if a.ndim == 1 else (a.size,), dtype, np.ascontiguousarray(a).tobytes())
    args = dict(down=ctx.tensor((codes.size // 4,), 'uint32', codes.tobytes()), scales=t(scales.ravel(), 'float32'),
                act=t(act, 'float32'), weights=t(weights, 'float32'), ids=t(ids, 'int32'), k=k, experts=experts,
                rows=rows, cols=cols)
    outs = {}
    kernels = {'mma': SpawnKernel(ctx, Path(kernel or spawn / 'moe_mma' / 'fp8_moe_mma_down_at.ptx')),
               'mma16': SpawnKernel(ctx, spawn / 'moe_mma' / 'fp8_moe_mma16_down_at.ptx'),
               'down2': SpawnKernel(ctx, spawn / 'fp8_moe_cvt_down2.ptx')}
    times = {}
    for name, kernel in kernels.items():
        out = ctx.tensor((rows,), 'float32', np.full(rows, np.nan, np.float32).tobytes())
        kernel.launch(stream=stream, out=out, **args)
        stream.synchronize()
        outs[name] = np.frombuffer(out.download(), np.float32).astype(np.float64)
        times[name] = event_us(ctx, stream, spawn, lambda: kernel.launch(stream=stream, out=out, **args), repeats)
    ref = model(codes, scales, act, weights, ids, rows, cols)
    ref16 = model(codes, scales, act, weights, ids, rows, cols, 'f16')
    peak = float(np.abs(ref).max())
    rel = lambda a, b: float(np.abs(a - b).max() / peak)
    return dict(rows=rows, cols=cols, experts=experts, k=k, max_abs_out=peak,
                finite=bool(np.isfinite(outs['mma']).all() and np.isfinite(outs['mma16']).all()),
                mma_vs_model=rel(outs['mma'], ref), mma16_vs_model=rel(outs['mma16'], ref16),
                down2_vs_model=rel(outs['down2'], ref),
                mma_vs_down2=rel(outs['mma'], outs['down2']), mma16_vs_down2=rel(outs['mma16'], outs['down2']),
                weight_mb=k * rows * cols / 1e6, us=times,
                tb_per_s={n: k * rows * cols / us / 1e6 for n, us in times.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, default=6144)
    p.add_argument('--cols', type=int, nargs='+', default=[256, 2048])
    p.add_argument('--experts', type=int, default=257)
    p.add_argument('--k', type=int, default=9)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--tolerance', type=float, default=2e-5, help='max |mma - model| / max |model|')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--kernel', type=Path, help="another build's fp8_moe_mma_down_at.ptx (e.g. a register budget)")
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('output exists; preserve evidence')
    from gpu.runtime import CUDAContext
    spawn = Path(__file__).resolve().parents[1] / 'gpu' / 'build' / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 9))
    stream = ctx.stream()
    rng = np.random.default_rng(args.seed)
    results = [run(ctx, stream, spawn, rng, args.rows, c, args.experts, args.k, args.repeats, args.kernel) for c in args.cols]
    ctx.close()
    ok = all(r['finite'] and r['mma_vs_model'] <= args.tolerance and r['mma16_vs_model'] <= args.tolerance for r in results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(ok=ok, tolerance=args.tolerance, kernel=str(args.kernel or 'moe_mma/fp8_moe_mma_down_at.ptx'),
                                           results=results), indent=2) + '\n')
    for r in results:
        print(json.dumps(r))
    print('moe_mma_check:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
