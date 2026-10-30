"""Where one full-width decode layer spends its GPU time, kernel by kernel, under Nsight Compute.

One decode step of ResidentGLM on the full GLM-5.3 width cut to 4 layers (layers 0-2 dense with
the full indexer, layer 3 MoE with the shared indexer), after --context tokens so the indexer
selects sparsely. Every kernel of that step is measured by ncu: GPU time, DRAM bytes read and
written, and the share of peak DRAM bandwidth and of peak SM throughput. Each launch is labelled
with its layer and what it computes (the Linear's weight for a projection), and the report groups
them per layer: layer 0 is a dense layer, layer 3 a MoE layer.

This is for finding bottlenecks in one layer, not a speed claim: ncu runs each kernel alone with
cold caches, at the GPU's own clocks (--clock-control none), so the sum is not a step time.

  python -m validation.layer_profile --format fp8 --output run/layer-fp8.json

It runs itself under ncu (NCU, default ncu on PATH or /usr/local/cuda/bin/ncu) and writes the
report as JSON and as a Markdown table beside it (.md). Where ncu cannot read the GPU's counters
(a container without profiling permission, ERR_NVGPUCTRPERM) or is missing, or with --timer
events, each launch of the step is timed between two CUDA events instead, after a 512 MB copy
that evicts L2 and keeps the GPU busy while the launch is queued (so the time is the kernel's, not
the host's): GPU time with cold caches as under ncu, but no DRAM bytes or SM throughput.
"""
import argparse
import csv
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

METRICS = ['gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
           'dram__throughput.avg.pct_of_peak_sustained_elapsed', 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
           'launch__grid_size', 'launch__block_size', 'launch__registers_per_thread']


def label(frames):
    """(layer, what) for a launch from inside ResidentGLM: the nearest `layer`, and the Linear's
    prefix for a projection or else the resident method that launched it."""
    layer, what = None, None
    for frame in frames:
        local = frame.frame.f_locals
        if what is None and frame.function in ('_project', '_matvec') and 'prefix' in local:
            what = local['prefix']
        if what is None and frame.function == '_device_moe':
            what = local['prefix'] + '.experts'
        if layer is None and 'layer' in local and isinstance(local['layer'], int):
            layer = local['layer']
    if what is None:
        # A block's own launch by the block; a helper (_rmsnorm, _rope, _copy, _add) with the block
        # that called it.
        names = [f.function.lstrip('_') for f in frames[:2]]
        blocks = ('attention', 'index', 'feed_forward', 'device_moe', 'layers', 'forward')
        what = names[0] if not names or names[0] in blocks or len(names) == 1 else f'{names[1]}.{names[0]}'
    if what.startswith(f'model.layers.{layer}.'):
        what = what[len(f'model.layers.{layer}.'):]
    return layer, what


def child(args):
    """Run the warm steps, then the profiled step between cuProfilerStart and cuProfilerStop,
    and write each launch's (kernel, layer, what) in order."""
    from glm_vx.backend import NumpyBackend
    from glm_vx.model import GlmMoeDsaModel
    from gpu.resident import ResidentGLM
    from validation.gpu_vs_official import full_geometry, random_weights
    from validation.moe_profile import l2_flush
    from validation.resident_compare import bf16_source, fp8_source, int4_source
    config = full_geometry(args.layers)
    weights = random_weights(config, args.seed)
    source = weights if args.format == 'f32' else {'int4': int4_source, 'fp8': fp8_source, 'bf16': bf16_source}[args.format](weights)
    model = GlmMoeDsaModel(config, source, NumpyBackend())
    tokens = [int(t) for t in np.random.default_rng(args.seed).integers(0, config['vocab_size'], args.context + 1)]
    resident = ResidentGLM(model, max_tokens=len(tokens) + 1, device_experts=True, fp8_kernels=args.fp8_kernels,
                           matvec_kernels=args.matvec_kernels)
    for t in tokens[:-1]:
        resident.forward(t, logits=False)
    resident.forward(tokens[-1])  # every kernel loaded and weight uploaded before the profiled step
    resident.reset()
    for t in tokens[:-1]:
        resident.forward(t, logits=False)
    launches, events = [], []
    launch, spawn = resident._launch, resident._spawn
    ctx = resident.ctx

    def timed(run, name, *a, **k):
        launches.append((name, *label(inspect.stack(0)[2:])))
        if args.timer == 'events':
            l2_flush(ctx, resident.stream, resident._ptx_dir / 'spawn')
            start = ctx.event().record(resident.stream)
            run(name, *a, **k)
            events.append((start, ctx.event().record(resident.stream)))
        else:
            run(name, *a, **k)

    resident._launch = lambda name, *a, **k: timed(launch, name, *a, **k)
    resident._spawn = lambda name, **k: timed(spawn, name, **k)
    ctx.synchronize()
    if args.timer == 'events':
        resident.forward(tokens[-1])
        ctx.synchronize()
    else:
        with ctx._activate():
            ctx.driver.call('cuProfilerStart')
        resident.forward(tokens[-1])
        ctx.synchronize()
        with ctx._activate():
            ctx.driver.call('cuProfilerStop')
    Path(args.launches).write_text(json.dumps(dict(
        launches=launches, format=args.format, layers=args.layers, context=args.context, hidden=config['hidden_size'],
        weights_gib=resident.weight_bytes / 1024 ** 3, timer=args.timer, fp8_kernels=args.fp8_kernels, matvec_kernels=args.matvec_kernels,
        us=[start.elapsed_ms(end) * 1e3 for start, end in events] or None)))
    resident.close()


def parse(csv_text):
    """ncu --csv --page raw: one row per kernel, after a units row."""
    rows = list(csv.DictReader(io.StringIO(csv_text[csv_text.index('"ID"'):])))
    return [r for r in rows[1:] if r.get('Kernel Name')]


def number(value):
    return float(str(value).replace(',', '')) if value not in (None, '', 'n/a') else 0.0


def report(info, rows, peak_gbs):
    """rows: ncu's, one per launch; None when info['us'] has event times instead."""
    launches, counted = info['launches'], rows is not None
    if not counted:
        rows = [{'gpu__time_duration.sum': us * 1e3, 'launch__grid_size': '', 'launch__block_size': '',
                 'launch__registers_per_thread': '', 'Kernel Name': name} for us, (name, *_) in zip(info['us'], launches)]
    if len(rows) != len(launches):
        raise SystemExit(f'ncu profiled {len(rows)} kernels, the step launched {len(launches)}')
    kernels = []
    for (name, layer, what), row in zip(launches, rows):
        us = number(row['gpu__time_duration.sum']) / 1e3  # ns
        kernel = dict(kernel=name, layer=layer, what=what, us=us, dram_mb=None, gbs=None, dram_pct=None, sm_pct=None,
                      grid=row['launch__grid_size'], block=row['launch__block_size'],
                      registers=row['launch__registers_per_thread'], ncu_name=row['Kernel Name'])
        if counted:
            moved = number(row['dram__bytes_read.sum']) + number(row['dram__bytes_write.sum'])
            kernel.update(dram_mb=moved / 1e6, gbs=moved / (us * 1e3) if us else 0.0,
                          dram_pct=number(row['dram__throughput.avg.pct_of_peak_sustained_elapsed']),
                          sm_pct=number(row['sm__throughput.avg.pct_of_peak_sustained_elapsed']))
        kernels.append(kernel)
    groups = {}
    for k in kernels:
        group = groups.setdefault('head' if k['layer'] is None else f"layer {k['layer']}", dict(us=0.0, dram_mb=0.0, kernels=0))
        group['us'] += k['us']
        group['dram_mb'] += k['dram_mb'] or 0.0
        group['kernels'] += 1
    for group in groups.values():
        if not counted:
            group.update(dram_mb=None, gbs=None, pct_of_peak_bandwidth=None, bandwidth_bound_us=None)
            continue
        group['gbs'] = group['dram_mb'] / group['us'] * 1e3 if group['us'] else 0.0  # MB/us is TB/s
        group['pct_of_peak_bandwidth'] = 100 * group['gbs'] / peak_gbs if peak_gbs else None
        group['bandwidth_bound_us'] = group['dram_mb'] * 1e6 / (peak_gbs * 1e9) * 1e6 if peak_gbs else None
    return dict({k: v for k, v in info.items() if k not in ('launches', 'us')}, kernels=kernels, groups=groups,
                peak_dram_gbs=peak_gbs)


def _cell(value, spec):
    return '-' if value is None else format(value, spec)


def markdown(rep):
    lines = [f"# One decode step, kernel by kernel: {rep['format']}, full width, {rep['layers']} layers, "
             f"context {rep['context']}", '',
             ('ncu, each kernel alone with cold caches; not a step time.' if rep.get('timer', 'ncu') == 'ncu' else
              'CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold '
              'caches, no DRAM bytes.'), '',
             '| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |', '|---|---|---|---|---|---|---|']
    for name, g in rep['groups'].items():
        lines.append(f"| {name} | {g['kernels']} | {g['us']:.0f} | {_cell(g['dram_mb'], '.1f')} | {_cell(g['gbs'], '.0f')} | "
                     f"{_cell(g['pct_of_peak_bandwidth'], '.0f')} | {_cell(g['bandwidth_bound_us'], '.0f')} |")
    ids = sorted({k['layer'] for k in rep['kernels'] if k['layer'] is not None})
    for layer in sorted({ids[0], ids[-1]}):  # the first layer (dense) and the last (MoE)
        lines += ['', f'## Layer {layer}: kernels by GPU time', '',
                  '| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |', '|---|---|---|---|---|---|---|---|---|']
        for k in sorted((k for k in rep['kernels'] if k['layer'] == layer), key=lambda k: -k['us']):
            lines.append(f"| {k['kernel']} | {k['what']} | {k['us']:.1f} | {_cell(k['dram_mb'], '.2f')} | {_cell(k['gbs'], '.0f')} | "
                         f"{_cell(k['dram_pct'], '.0f')} | {_cell(k['sm_pct'], '.0f')} | {k['grid']} x {k['block']} | {k['registers']} |")
    return '\n'.join(lines) + '\n'


def peak_bandwidth():
    """Peak DRAM GB/s from the device's memory clock and bus width."""
    from gpu.runtime import CUDAContext
    ctx = CUDAContext()
    try:
        clock_khz = ctx._attribute(36)  # CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
        bus_bits = ctx._attribute(37)   # CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
        return 2 * clock_khz * 1e3 * bus_bits / 8 / 1e9
    finally:
        ctx.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--format', choices=['f32', 'int4', 'fp8', 'bf16'], default='f32')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--context', type=int, default=64, help='tokens before the profiled step (above index_topk=16)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--timer', choices=['ncu', 'events'], default='ncu', help='events: no ncu (see above)')
    p.add_argument('--fp8-kernels', choices=['hand', 'x4'], default='hand', help="ResidentGLM's fp8_kernels (#30, #31)")
    p.add_argument('--matvec-kernels', choices=['hand', 'spawn'], default='hand',
                   help="ResidentGLM's matvec_kernels: INT4 and BF16 Linears through gpu/spawn")
    p.add_argument('--launches', help=argparse.SUPPRESS)  # set when running under ncu
    args = p.parse_args()
    if args.launches:
        return child(args)
    if args.output.exists():
        p.error('output exists; preserve evidence')
    ncu = os.environ.get('NCU') or shutil.which('ncu') or '/usr/local/cuda/bin/ncu'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    launches = args.output.with_suffix('.launches.json')
    raw = args.output.with_suffix('.ncu.csv')
    command = [ncu, '--profile-from-start', 'off', '--clock-control', 'none', '--csv', '--page', 'raw', '--print-units', 'base',
               '--metrics', ','.join(METRICS), '--log-file', str(raw),
               sys.executable, '-m', 'validation.layer_profile', '--output', str(args.output), '--format', args.format,
               '--layers', str(args.layers), '--context', str(args.context), '--seed', str(args.seed),
               '--fp8-kernels', args.fp8_kernels, '--matvec-kernels', args.matvec_kernels,
               '--launches', str(launches)]
    rows = None
    if args.timer == 'ncu' and (os.path.exists(ncu) or shutil.which(ncu)):
        done = subprocess.run(command)
        text = raw.read_text() if raw.exists() else ''
        if done.returncode == 0 and 'ERR_NVGPUCTRPERM' not in text and '"ID"' in text:
            rows = parse(text)
        elif raw.exists():
            raw.rename(raw.with_suffix('.failed.txt'))  # kept: why ncu could not measure
    if rows is None:
        print('timing with CUDA events (no ncu counters)', file=sys.stderr)
        subprocess.run([sys.executable, *command[command.index('-m'):-2], '--timer', 'events', '--launches', str(launches)],
                       check=True)
    rep = report(json.loads(launches.read_text()), rows, peak_bandwidth())
    args.output.write_text(json.dumps(rep, indent=2) + '\n')
    args.output.with_suffix('.md').write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
