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

Where ncu cannot read the GPU's counters (a container without profiling permission,
ERR_NVGPUCTRPERM) or is missing, or with --timer events, each launch is instead timed with CUDA
events, the median of 5, each after a 512 MB copy that evicts it from L2 and keeps the GPU busy while
the launch is queued. That gives GPU time and
the bandwidth the expected bytes imply, but no measured DRAM bytes or SM throughput.

--inter 256 gives one rank's shapes under --tp 8 (each expert's slice of 2048), and --k 9 the 8 routed experts
with the shared one folded in (fold_shared); fp8cvt2 is fp8cvt with fp8_moe_cvt_down2 (moe_down2), as served
in #63. --detail adds the counters that tell what holds a kernel back: instructions, issue activity, the
warps resident, and the stall reasons a warp waits on most.

  python -m validation.moe_profile --output run/moe.json [--formats f32 bf16 fp8 int4] [--k 1 2 4 8]
  python -m validation.moe_profile --output run/moe-tp8.json --formats fp8cvt2 --inter 256 --k 9 --detail
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
# fp8x4: the same FP8 weights through fp8_moe4_* (sixteen codes a lane step, #31); fp8fma through
# fp8_moe_fma_* (one fma a code, x read once for gate and up, two down rows a block).
KERNELS = {'f32': ('moe_gate_up', 'moe_down'), 'bf16': ('bf16_moe_gate_up', 'bf16_moe_down'),
           'fp8': ('fp8_moe_gate_up', 'fp8_moe_down'), 'fp8x4': ('fp8_moe4_gate_up', 'fp8_moe4_down'),
           'fp8fma': ('fp8_moe_fma_gate_up', 'fp8_moe_fma_down'),
           'fp8cvt': ('fp8_moe_cvt_gate_up', 'fp8_moe_cvt_down'),  # sm_90 (Hopper's conversion, glm-vx #38)
           'fp8cvt2': ('fp8_moe_cvt_gate_up', 'fp8_moe_cvt_down2'),  # moe_down2: short tensor-parallel rows
           'fp8mma': ('fp8_moe_cvt_gate_up', 'moe_mma/fp8_moe_mma_down_at'),  # moe_mma: down on the FP8 tensor cores
           'fp8mma16': ('fp8_moe_cvt_gate_up', 'moe_mma/fp8_moe_mma16_down_at'),  # moe_mma='f16': act in f16
           'fp8mma16gu': ('moe_mma/fp8_moe_mma16_gate_up_at', 'moe_mma/fp8_moe_mma16_down_at'),  # both on the tensor cores
           'fp8mma16pgu': ('moe_mma/fp8_moe_mma16p_gate_up_at', 'moe_mma/fp8_moe_mma16_down_at'),  # gate_up pipelined
           'int4': ('int4_moe_gate_up', 'int4_moe_down')}
ELEMENT = {'f32': 4.0, 'bf16': 2.0, 'fp8': 1.0, 'fp8x4': 1.0, 'fp8fma': 1.0, 'fp8cvt': 1.0, 'fp8cvt2': 1.0, 'fp8mma': 1.0, 'fp8mma16': 1.0, 'fp8mma16gu': 1.0, 'fp8mma16pgu': 1.0,
           'int4': 0.5}
FP8 = ('fp8', 'fp8x4', 'fp8fma', 'fp8cvt', 'fp8cvt2', 'fp8mma', 'fp8mma16', 'fp8mma16gu', 'fp8mma16pgu')  # one copy of the FP8 weights; the others view the codes as u32
DETAIL = ['sm__inst_executed.sum', 'smsp__issue_active.avg.pct_of_peak_sustained_active',
          'sm__warps_active.avg.pct_of_peak_sustained_active',
          'smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio',
          'smsp__average_warp_latency_issue_stalled_lg_throttle.ratio',
          'smsp__average_warp_latency_issue_stalled_wait.ratio',
          'smsp__average_warp_latency_issue_stalled_barrier.ratio',
          'smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio',
          'smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio',
          'smsp__average_warp_latency_issue_stalled_not_selected.ratio',
          'smsp__average_warp_latency_issue_stalled_selected.ratio']


def shapes(args):
    """The module's shapes from the arguments (child and parent agree)."""
    global INTER, EXPERTS
    INTER, EXPERTS = args.inter, args.experts


def scale_bytes(fmt, rows, cols):
    """Bytes of one expert matrix's scales: fp8 one f32 per 128x128 block, int4 one 16-bit value
    per 128 columns of a row."""
    if fmt in FP8:
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
    if fmt in FP8:
        codes = rng.integers(0, 256, size=n, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
        scales = rng.uniform(0.001, 0.1, size=EXPERTS * -(-rows // 128) * -(-cols // 128)).astype(np.float32)
        weight = ctx.tensor((n,), 'uint8', codes.tobytes()) if fmt == 'fp8' else ctx.tensor((n // 4,), 'uint32', codes.tobytes())
        return dict(weight=weight, scales=ctx.tensor(scales.shape, 'float32', scales.tobytes()))
    packed = rng.integers(0, 256, size=EXPERTS * rows * (cols // 2), dtype=np.uint8)
    scales = (rng.uniform(0.001, 0.1, size=EXPERTS * rows * -(-cols // 128)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return dict(packed=ctx.tensor(packed.shape, 'uint8', packed.tobytes()),
                scales=ctx.tensor(scales.shape, 'uint16', scales.tobytes()), scale_kind=1)


def l2_flush(ctx, stream, spawn):
    """Copy 512 MB to 512 MB on stream: none of the next kernel's data is left in L2 (40 MB on an
    A100, 50 MB on an H100), and the GPU is busy for about 0.3 ms (H100) to 0.7 ms (A100), longer
    than the host takes to queue the events and the launch that follow. Without it, a start event
    on an idle GPU also times the host preparing the launch."""
    from gpu.spawn.launch import SpawnKernel
    if getattr(l2_flush, 'ctx', None) is not ctx:
        n = 128 << 20
        l2_flush.ctx = ctx
        l2_flush.state = (SpawnKernel(ctx, spawn / 'copy.ptx'), ctx.tensor((n,), 'float32'), ctx.tensor((n,), 'float32'), n)
    copy, a, b, n = l2_flush.state
    copy.launch(stream=stream, out=a, x=b, n=n)


def event_us(ctx, stream, spawn, launch, repeats=5):
    """Median GPU microseconds of launch() between two events, each after l2_flush."""
    times = []
    for _ in range(repeats):
        l2_flush(ctx, stream, spawn)
        start, end = ctx.event().record(stream), ctx.event()
        launch()
        end.record(stream)
        times.append(start.elapsed_ms(end) * 1e3)
    return float(np.median(times))


def child(args):
    shapes(args)
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    spawn = Path(__file__).resolve().parent.parent / 'gpu' / 'build' / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    rng = np.random.default_rng(args.seed)
    stream = ctx.stream()
    plan, measured = [], []
    x = ctx.tensor((HIDDEN,), 'float32', rng.standard_normal(HIDDEN, dtype=np.float32).tobytes())
    out = ctx.tensor((HIDDEN,), 'float32')
    fp8 = None  # the FP8 formats share one copy of the FP8 weights (fp8x4, fp8fma: a u32 view): all six
    for fmt in args.formats:  # formats at once would not fit in 80 GB
        if fmt in ('fp8cvt', 'fp8cvt2', 'fp8mma', 'fp8mma16', 'fp8mma16gu', 'fp8mma16pgu') and tuple(ctx.compute_capability) < (8, 9):
            continue  # sm_90 PTX (Hopper's e4m3x2 -> f16x2 conversion)
        if fmt in FP8 and fp8 is not None:
            codes = EXPERTS * INTER * HIDDEN  # each projection's code bytes
            gate, up, down = ({**w, 'weight': w['weight'].view((codes // 4,), 'uint32') if fmt != 'fp8'
                               else w['weight'].view((codes,), 'uint8')} for w in fp8)
        else:
            gate, up, down = (weights(ctx, rng, fmt, r, c) for r, c in ((INTER, HIDDEN), (INTER, HIDDEN), (HIDDEN, INTER)))
            if fmt in FP8:
                fp8 = (dict(gate), dict(up), dict(down))
        extra = {'scale_kind': gate.pop('scale_kind')} if fmt == 'int4' else {}
        up.pop('scale_kind', None)
        down.pop('scale_kind', None)
        g = {('gate' if n in ('weight', 'packed') else 'gate_scales'): v for n, v in gate.items()}
        g.update({('up' if n in ('weight', 'packed') else 'up_scales'): v for n, v in up.items()})
        d = {('down' if n in ('weight', 'packed') else 'scales'): v for n, v in down.items()}
        gate_up, down_k = (SpawnKernel(ctx, spawn / f'{name}.ptx') for name in KERNELS[fmt])
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
    if args.timer == 'events':
        for m, (kernel, values) in zip(measured, plan):
            m['us'] = event_us(ctx, stream, spawn, lambda: kernel.launch(stream=stream, **values))
    else:
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
    p.add_argument('--formats', nargs='+', choices=list(KERNELS), default=['f32', 'bf16', 'fp8', 'fp8x4', 'fp8fma', 'fp8cvt', 'int4'])
    p.add_argument('--k', nargs='+', type=int, default=[1, 2, 4, 8])
    p.add_argument('--inter', type=int, default=INTER, help="each expert's width: 2048, or 256 for one rank of 8")
    p.add_argument('--experts', type=int, default=EXPERTS)
    p.add_argument('--detail', action='store_true', help='ncu: also the instruction, issue, occupancy and stall counters')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--timer', choices=['ncu', 'events'], default='ncu', help='events: no ncu (see above)')
    p.add_argument('--launches', help=argparse.SUPPRESS)  # set when running under ncu
    args = p.parse_args()
    shapes(args)
    if args.launches:
        return child(args)
    if args.output.exists():
        p.error('output exists; preserve evidence')
    ncu = os.environ.get('NCU') or shutil.which('ncu') or '/usr/local/cuda/bin/ncu'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    launches, raw = args.output.with_suffix('.launches.json'), args.output.with_suffix('.ncu.csv')
    if args.timer == 'ncu' and not counters(ncu, args, launches, raw):
        print('ncu gave no counters; timing with CUDA events instead', file=sys.stderr)
        args.timer = 'events'
    if args.timer == 'events':
        return events_report(args, launches)
    plan, rows = json.loads(launches.read_text()), parse(raw.read_text())
    if len(plan) != len(rows):
        raise SystemExit(f'ncu profiled {len(rows)} kernels, the plan launched {len(plan)}')
    return write(args, [dict(m, timer='ncu', **measure(m, row), **({k: number(row[k]) for k in DETAIL if k in row}
                                                                    if args.detail else {}))
                        for m, row in zip(plan, rows)])


def counters(ncu, args, launches, raw):
    """Run the plan under ncu; False when ncu is missing or read no counters."""
    if not (os.path.exists(ncu) or shutil.which(ncu)):
        return False
    done = subprocess.run([ncu, '--profile-from-start', 'off', '--clock-control', 'none', '--csv', '--page', 'raw',
                           '--print-units', 'base', '--metrics', ','.join(METRICS + (DETAIL if args.detail else [])),
                           '--log-file', str(raw), sys.executable, '-m', 'validation.moe_profile', '--output', str(args.output),
                           '--formats', *args.formats, '--k', *map(str, args.k), '--seed', str(args.seed),
                           '--inter', str(args.inter), '--experts', str(args.experts),
                           '--launches', str(launches)])
    text = raw.read_text() if raw.exists() else ''
    if done.returncode != 0 or 'ERR_NVGPUCTRPERM' in text or '"ID"' not in text:
        if raw.exists():
            raw.rename(raw.with_suffix('.failed.txt'))  # kept: why ncu could not measure
        return False
    return True


def measure(m, row):
    us = number(row['gpu__time_duration.sum']) / 1e3  # ns
    moved = (number(row['dram__bytes_read.sum']) + number(row['dram__bytes_write.sum'])) / 1e6
    return dict(us=us, dram_mb=moved, gbs=moved / us * 1e3 if us else 0.0,
                dram_pct=number(row['dram__throughput.avg.pct_of_peak_sustained_elapsed']),
                sm_pct=number(row['sm__throughput.avg.pct_of_peak_sustained_elapsed']),
                grid=row['launch__grid_size'], block=row['launch__block_size'],
                registers=row['launch__registers_per_thread'])


def events_report(args, launches):
    subprocess.run([sys.executable, '-m', 'validation.moe_profile', '--output', str(args.output),
                    '--formats', *args.formats, '--k', *map(str, args.k), '--seed', str(args.seed),
                    '--inter', str(args.inter), '--experts', str(args.experts), '--timer', 'events', '--launches', str(launches)], check=True)
    plan = json.loads(launches.read_text())
    # Without counters the GB/s is what the expected bytes imply in the measured time.
    return write(args, [dict(m, timer='events', dram_mb=None, dram_pct=None, sm_pct=None, grid='', block='',
                             registers='', gbs=m['expected_mb'] / m['us'] * 1e3 if m['us'] else 0.0) for m in plan])


def write(args, results):
    peak = peak_bandwidth()
    for r in results:
        r['pct_of_peak_bandwidth'] = 100 * r['gbs'] / peak if peak else 0.0
        r['us_at_peak_bandwidth'] = r['expected_mb'] / peak * 1e3
    timer = results[0]['timer'] if results else args.timer
    args.output.write_text(json.dumps(dict(hidden=HIDDEN, inter=INTER, experts=EXPERTS, peak_dram_gbs=peak,
                                           results=results), indent=2) + '\n')
    lines = [f'# Fused MoE expert kernels: hidden {HIDDEN}, expert width {INTER}, {EXPERTS} experts', '',
             (f'ncu, each kernel alone with cold caches. Peak DRAM {peak:.0f} GB/s.' if timer == 'ncu' else
              f'CUDA events (no ncu counters): median of 5, L2 flushed before each; GB/s is expected MB over '
              f'GPU time. Peak DRAM {peak:.0f} GB/s.'), '',
             '| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        opt = lambda v, f: '-' if v is None else format(v, f)
        lines.append(f"| {r['format']} | {r['kernel']} | {r['k']} | {r['expected_mb']:.1f} | {opt(r['dram_mb'], '.1f')} | {r['us']:.1f} | "
                     f"{r['us_at_peak_bandwidth']:.1f} | {r['gbs']:.0f} | {r['pct_of_peak_bandwidth']:.0f} | {opt(r['sm_pct'], '.0f')} | "
                     f"{r['grid']} x {r['block']} | {r['registers']} |")
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
