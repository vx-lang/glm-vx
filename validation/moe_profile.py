"""The fused MoE expert kernels at GLM-5.3's real shapes, under Nsight Compute, against the bytes
each launch must read.

gpu/spawn's moe_gate_up (gate, up and SwiGLU of the k routed experts) and moe_down (their down
projections, summed with the router weights) in every weight format, on all 256 experts' weights
stacked as ResidentGLM uploads them (hidden 6144, expert width 2048), for k = 1, 2, 4 and 8
routed experts (8 is GLM-5.3's). A decode token only needs its k experts' weights, so a launch's
DRAM traffic should be k experts' bytes: the report gives that expectation, what ncu measured,
the GPU time, and the share of peak DRAM bandwidth and of peak SM throughput, which tell a
bandwidth-bound kernel from a compute-bound one.

Each kernel runs alone with cold caches (ncu), at the GPU's own clocks: a kernel measurement,
not a step time. Random weights and ids; timing does not depend on their values.

  python -m validation.moe_profile --output run/moe.json [--formats f32 bf16 fp8 int4] [--k 1 2 4 8]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from validation.layer_profile import METRICS, number, parse, peak_bandwidth

HIDDEN, INTER, EXPERTS = 6144, 2048, 256
PREFIX = {'f32': '', 'bf16': 'bf16_', 'fp8': 'fp8_', 'int4': 'int4_'}
ELEMENT = {'f32': 4.0, 'bf16': 2.0, 'fp8': 1.0, 'int4': 0.5}


def scale_bytes(fmt, rows, cols):
    """Bytes of one expert matrix's scales: fp8 one f32 per 128x128 block, int4 one 16-bit value
    per 128 columns of a row."""
    if fmt == 'fp8':
        return 4 * -(-rows // 128) * -(-cols // 128)
    if fmt == 'int4':
        return 2 * rows * -(-cols // 128)
    return 0


def expected_mb(fmt, kernel, k):
    """The weight bytes a launch must read for k experts (activations are kilobytes)."""
    if kernel == 'moe_gate_up':
        per = 2 * (INTER * HIDDEN * ELEMENT[fmt] + scale_bytes(fmt, INTER, HIDDEN))
    else:
        per = HIDDEN * INTER * ELEMENT[fmt] + scale_bytes(fmt, HIDDEN, INTER)
    return k * per / 1e6


def weights(ctx, rng, fmt, rows, cols):
    """One projection of all experts on the device, named as the expert kernels name it."""
    n = EXPERTS * rows * cols
    if fmt == 'f32':
        return dict(weight=ctx.tensor((n,), 'float32', rng.standard_normal(n, dtype=np.float32).tobytes()))
    if fmt == 'bf16':
        bits = (rng.standard_normal(n, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
        return dict(weight=ctx.tensor((n,), 'uint16', bits.tobytes()))
    if fmt == 'fp8':
        codes = rng.integers(0, 256, size=n, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
        scales = rng.uniform(0.001, 0.1, size=EXPERTS * -(-rows // 128) * -(-cols // 128)).astype(np.float32)
        return dict(weight=ctx.tensor((n,), 'uint8', codes.tobytes()), scales=ctx.tensor(scales.shape, 'float32', scales.tobytes()))
    packed = rng.integers(0, 256, size=EXPERTS * rows * (cols // 2), dtype=np.uint8)
    scales = (rng.uniform(0.001, 0.1, size=EXPERTS * rows * -(-cols // 128)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return dict(packed=ctx.tensor(packed.shape, 'uint8', packed.tobytes()),
                scales=ctx.tensor(scales.shape, 'uint16', scales.tobytes()), scale_kind=1)


def child(args):
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    spawn = Path(__file__).resolve().parent.parent / 'gpu' / 'build' / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    rng = np.random.default_rng(args.seed)
    stream = ctx.stream()
    plan, measured = [], []
    x = ctx.tensor((HIDDEN,), 'float32', rng.standard_normal(HIDDEN, dtype=np.float32).tobytes())
    out = ctx.tensor((HIDDEN,), 'float32')
    for fmt in args.formats:
        gate, up, down = (weights(ctx, rng, fmt, r, c) for r, c in ((INTER, HIDDEN), (INTER, HIDDEN), (HIDDEN, INTER)))
        extra = {'scale_kind': gate.pop('scale_kind')} if fmt == 'int4' else {}
        up.pop('scale_kind', None)
        down.pop('scale_kind', None)
        g = {('gate' if n in ('weight', 'packed') else 'gate_scales'): v for n, v in gate.items()}
        g.update({('up' if n in ('weight', 'packed') else 'up_scales'): v for n, v in up.items()})
        d = {('down' if n in ('weight', 'packed') else 'scales'): v for n, v in down.items()}
        gate_up = SpawnKernel(ctx, spawn / f'{PREFIX[fmt]}moe_gate_up.ptx')
        down_k = SpawnKernel(ctx, spawn / f'{PREFIX[fmt]}moe_down.ptx')
        for k in args.k:
            ids = rng.choice(EXPERTS, size=k, replace=False).astype(np.int32)
            ids_t = ctx.tensor((k,), 'int32', ids.tobytes())
            weights_t = ctx.tensor((k,), 'float32', rng.uniform(0.01, 1, size=k).astype(np.float32).tobytes())
            act = ctx.tensor((k * INTER,), 'float32')
            launches = [
                (gate_up, dict(act=act, x=x, ids=ids_t, k=k, experts=EXPERTS, inter=INTER, up_row=0, stride=INTER,
                               cols=HIDDEN, **g, **extra), 'moe_gate_up'),
                (down_k, dict(out=out, act=act, weights=weights_t, ids=ids_t, k=k, experts=EXPERTS, rows=HIDDEN,
                              cols=INTER, **d, **extra), 'moe_down')]
            for kernel, values, name in launches:
                kernel.launch(stream=stream, **values)  # warm: the module is loaded, the first launch paid
                plan.append((kernel, values))
                measured.append(dict(format=fmt, kernel=name, k=k, expected_mb=expected_mb(fmt, name, k)))
        stream.synchronize()
    with ctx._activate():
        ctx.driver.call('cuProfilerStart')
    for kernel, values in plan:
        kernel.launch(stream=stream, **values)
    stream.synchronize()
    with ctx._activate():
        ctx.driver.call('cuProfilerStop')
    Path(args.launches).write_text(json.dumps(measured))
    ctx.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formats', nargs='+', choices=list(PREFIX), default=['f32', 'bf16', 'fp8', 'int4'])
    p.add_argument('--k', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--launches', help=argparse.SUPPRESS)  # set when running under ncu
    args = p.parse_args()
    if args.launches:
        return child(args)
    if args.output.exists():
        p.error('output exists; preserve evidence')
    ncu = os.environ.get('NCU') or shutil.which('ncu') or '/usr/local/cuda/bin/ncu'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    launches, raw = args.output.with_suffix('.launches.json'), args.output.with_suffix('.ncu.csv')
    subprocess.run([ncu, '--profile-from-start', 'off', '--clock-control', 'none', '--csv', '--page', 'raw',
                    '--metrics', ','.join(METRICS), '--log-file', str(raw),
                    sys.executable, '-m', 'validation.moe_profile', '--output', str(args.output),
                    '--formats', *args.formats, '--k', *map(str, args.k), '--seed', str(args.seed),
                    '--launches', str(launches)], check=True)
    plan, rows = json.loads(launches.read_text()), parse(raw.read_text())
    if len(plan) != len(rows):
        raise SystemExit(f'ncu profiled {len(rows)} kernels, the plan launched {len(plan)}')
    peak = peak_bandwidth()
    results = []
    for m, row in zip(plan, rows):
        us = number(row['gpu__time_duration.sum']) / 1e3
        moved = (number(row['dram__bytes_read.sum']) + number(row['dram__bytes_write.sum'])) / 1e6
        results.append(dict(m, us=us, dram_mb=moved, gbs=moved / us * 1e3 if us else 0.0,
                            pct_of_peak_bandwidth=100 * moved / us * 1e3 / peak if us else 0.0,
                            us_at_peak_bandwidth=m['expected_mb'] / peak * 1e3,
                            dram_pct=number(row['dram__throughput.avg.pct_of_peak_sustained_elapsed']),
                            sm_pct=number(row['sm__throughput.avg.pct_of_peak_sustained_elapsed']),
                            grid=row['launch__grid_size'], block=row['launch__block_size'],
                            registers=row['launch__registers_per_thread']))
    args.output.write_text(json.dumps(dict(hidden=HIDDEN, inter=INTER, experts=EXPERTS, peak_dram_gbs=peak,
                                           results=results), indent=2) + '\n')
    lines = [f'# Fused MoE expert kernels: hidden {HIDDEN}, expert width {INTER}, {EXPERTS} experts', '',
             f'ncu, each kernel alone with cold caches. Peak DRAM {peak:.0f} GB/s.', '',
             '| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['format']} | {r['kernel']} | {r['k']} | {r['expected_mb']:.1f} | {r['dram_mb']:.1f} | {r['us']:.1f} | "
                     f"{r['us_at_peak_bandwidth']:.1f} | {r['gbs']:.0f} | {r['pct_of_peak_bandwidth']:.0f} | {r['sm_pct']:.0f} | "
                     f"{r['grid']} x {r['block']} | {r['registers']} |")
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
