"""The per-layer timing gate (#29, item 7): layer_profile reports against a baseline run.

  python -m validation.layer_gate --baseline evidence/h100-20261026 --output run/layer-gate.md \
      run/layer-profile-*.json

Each report (validation/layer_profile.py) is one decode step at full width, kernel by kernel. The
gate groups its launches by layer and by what they compute (the Linear's name for a projection,
the block for the rest: `self_attn.o_proj`, `mlp.experts`, `index`, `attention`), so a run with
other kernels (fp8_matvec4 for fp8_matvec) still lines up with its baseline, and compares:

  - each run with the baseline directory's report of the same format (layer-events-<format>.json,
    else layer-profile-<format>.json): the run before, on another pod;
  - a run with other kernels (--fp8-kernels x4, --matvec-kernels spawn/x4) also with this run's
    report of the same format with the hand kernels: the same pod, clocks and driver.

A group slower than its reference by more than --tolerance (and by more than --floor-us, for the
small kernels whose time is mostly launch noise) is a regression, and the exit code is 1 if there is
one. The targets of #30, #31 and #32 are reported as met or not, but do not fail the gate: they are
goals, not regressions.

Event timing has no DRAM bytes, so the bytes a projection must read are its weights as the device
holds them, from GLM-5.3's shapes and the kernel's format: FP8 is one byte a weight plus a float32
scale per 128x128 block, INT4 half a byte plus a 16-bit scale per row and 128 columns, BF16 two
bytes, the rest (matvec, moe_*) float32. The routed experts read num_experts_per_tok experts (8):
gate and up in *_gate_up, down in *_down. % of peak is those bytes over the time, against the run's
peak DRAM bandwidth; it is how #29 counts o_proj's 224 us against 30 us at peak.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# (issue, format, layer kind, group, target): the projection groups that #30 and #31 time.
BANDWIDTH_TARGETS = [
    ('#30', 'fp8', 'dense', 'self_attn.o_proj', 60.0),
    ('#30', 'fp8', 'dense', 'mlp.gate_proj', 60.0),
    ('#30', 'fp8', 'dense', 'mlp.up_proj', 60.0),
    ('#30', 'fp8', 'dense', 'mlp.down_proj', 60.0),
    ('#30', 'fp8', 'dense', 'self_attn.q_a_proj', 60.0),
    ('#31', 'fp8', 'sparse', 'mlp.experts', 60.0),
]
TOPK_TARGET_US = 20.0  # #32, at the profile's context (64 tokens)


def bytes_per_weight(kernel, rows, cols):
    """Bytes the kernel reads for a (rows, cols) weight, scales included."""
    if kernel.startswith('fp8'):
        return rows * cols + 4 * -(-rows // 128) * -(-cols // 128)
    if kernel.startswith('int4'):
        return rows * cols // 2 + 2 * rows * -(-cols // 128)
    if kernel.startswith('bf16'):
        return 2 * rows * cols
    return 4 * rows * cols


def weight_bytes(shapes, experts_per_token, layer, what, kernel):
    """The weight bytes one launch must read, or None for a launch that reads no Linear."""
    if layer is None:
        return None
    prefix = f'model.layers.{layer}.'
    if what == 'mlp.experts':
        expert = prefix + 'mlp.experts.0.'
        names = (['down_proj'] if 'down' in kernel else ['gate_proj', 'up_proj'] if 'gate_up' in kernel else [])
        return sum(experts_per_token * bytes_per_weight(kernel, *shapes[expert + n + '.weight']) for n in names) or None
    shape = shapes.get(prefix + what + '.weight')
    if shape is None or len(shape) != 2:
        return None
    return bytes_per_weight(kernel, *shape)


def config_of(report):
    from glm_vx.config import GLMConfig
    from validation.gpu_vs_official import full_geometry
    return GLMConfig.from_dict(full_geometry(report['layers']))


def kernels_of(report):
    """The run's kernel choice: 'hand', or the fp8_kernels / matvec_kernels it set."""
    for key in ('fp8_kernels', 'matvec_kernels'):
        if report.get(key, 'hand') != 'hand':
            return report[key]
    return 'hand'


def name_of(report):
    kernels = kernels_of(report)
    return report['format'] if kernels == 'hand' else f"{report['format']}-{kernels}"


def groups(report, config=None):
    """{(layer, what): {us, kernels, bytes}}, in launch order."""
    config = config or config_of(report)
    shapes, k = config.tensor_shapes(), config.num_experts_per_tok
    out = {}
    for launch in report['kernels']:
        key = (launch['layer'], launch['what'])
        g = out.setdefault(key, dict(us=0.0, kernels=[], bytes=0, counted=True))
        g['us'] += launch['us']
        g['kernels'].append(launch['kernel'])
        b = weight_bytes(shapes, k, launch['layer'], launch['what'], launch['kernel'])
        if b is None:
            g['counted'] = False
        else:
            g['bytes'] += b
    for g in out.values():
        if not g['counted'] or not g['bytes']:
            g['bytes'] = None
    return out


def pct_of_peak(group, peak_gbs):
    if group['bytes'] is None or not group['us'] or not peak_gbs:
        return None
    return 100.0 * group['bytes'] / (group['us'] * 1e3) / peak_gbs  # B/ns is GB/s


def layer_kind(config, layer):
    return None if layer is None else config.mlp_layer_types[layer]


def compare(run, reference, tolerance, floor_us):
    """Rows (layer, what, kernels, ref us, run us, change, regression) for each group of run."""
    ref_groups = groups(reference)
    rows = []
    for key, g in groups(run).items():
        r = ref_groups.get(key)
        ref_us = r['us'] if r else None
        change = (g['us'] / ref_us - 1.0) if ref_us else None
        regression = ref_us is not None and g['us'] > ref_us * (1 + tolerance) and g['us'] - ref_us > floor_us
        rows.append(dict(layer=key[0], what=key[1], kernels=g['kernels'], ref_kernels=r['kernels'] if r else None,
                         ref_us=ref_us, us=g['us'], change=change, regression=regression))
    return rows


def targets(run):
    """The #30, #31 and #32 targets this run measures, as rows (issue, layer, what, value, target, met)."""
    config = config_of(run)
    found, peak = groups(run, config), run.get('peak_dram_gbs')
    rows = []
    for issue, fmt, kind, what, target in BANDWIDTH_TARGETS:
        if run['format'] != fmt:
            continue
        for (layer, w), g in found.items():
            if w == what and layer_kind(config, layer) == kind:
                pct = pct_of_peak(g, peak)
                rows.append(dict(issue=issue, layer=layer, what=what, value=pct, unit='% of peak', target=target,
                                 met=pct is not None and pct >= target, us=g['us'],
                                 us_at_peak=g['bytes'] / peak / 1e3 if g['bytes'] and peak else None))
                break  # the first layer of its kind: the dense layers are the same
    topk = [l for l in run['kernels'] if l['kernel'].startswith('topk')]
    if topk:
        first = min(l['layer'] for l in topk if l['layer'] is not None)
        us = sum(l['us'] for l in topk if l['layer'] == first)
        rows.append(dict(issue='#32', layer=first, what='topk at %d tokens' % run['context'], value=us, unit='us',
                         target=TOPK_TARGET_US, met=us < TOPK_TARGET_US, us=us, us_at_peak=None))
    return rows


def baseline_for(directory, fmt):
    # layer-events-* first: in h100-20261026 the layer-profile-* reports are superseded (no L2 flush).
    for stem in (f'layer-events-{fmt}', f'layer-profile-{fmt}'):
        path = Path(directory) / f'{stem}.json'
        if path.exists():
            return path
    return None


def _us(v):
    return '-' if v is None else f'{v:.1f}'


def _change(v):
    return '-' if v is None else f'{100 * v:+.0f}%'


def gate(runs, baseline_dir, tolerance=0.10, floor_us=2.0):
    """(markdown, summary dict) for the run reports `runs` ({path: report})."""
    lines = ['# Layer timing gate', '',
             f'Baseline: `{baseline_dir}`. A group is a regression when it is more than {100 * tolerance:.0f}% '
             f'and {floor_us:g} us slower than its reference. Targets are reported, not gated.', '']
    hand = {r['format']: r for r in runs.values() if kernels_of(r) == 'hand'}
    summary = dict(regressions=[], targets=[], compared=[])
    for path, run in runs.items():
        name = name_of(run)
        refs = []
        base = baseline_for(baseline_dir, run['format']) if baseline_dir else None
        if base:
            refs.append((f'baseline {base.name}', json.loads(base.read_text())))
        if kernels_of(run) != 'hand' and run['format'] in hand:
            refs.append((f'this run, {run["format"]} hand kernels', hand[run['format']]))
        lines += [f'## {name} (`{Path(path).name}`)', '']
        if not refs:
            lines += ['No reference: nothing to compare.', '']
        for title, ref in refs:
            rows = compare(run, ref, tolerance, floor_us)
            config = config_of(run)
            total_ref = sum(r['ref_us'] or 0.0 for r in rows)
            total = sum(r['us'] for r in rows)
            regressions = [r for r in rows if r['regression']]
            summary['compared'].append(dict(run=name, reference=title, us=total, ref_us=total_ref))
            summary['regressions'] += [dict(run=name, reference=title, layer=r['layer'], what=r['what'],
                                            us=r['us'], ref_us=r['ref_us']) for r in regressions]
            lines += [f'### Against {title}: step {total_ref:.0f} -> {total:.0f} us '
                      f'({_change(total / total_ref - 1 if total_ref else None)}), '
                      f'{len(regressions)} regression{"s" if len(regressions) != 1 else ""}', '']
            per_layer = {}
            for r in rows:
                p = per_layer.setdefault(r['layer'], [0.0, 0.0])
                p[0] += r['ref_us'] or 0.0
                p[1] += r['us']
            lines += ['| Layer | Reference us | Run us | Change |', '|---|---|---|---|']
            for layer, (a, b) in per_layer.items():
                lines.append(f"| {'head' if layer is None else layer} | {a:.0f} | {b:.0f} | {_change(b / a - 1 if a else None)} |")
            ids = sorted(l for l in per_layer if l is not None)
            shown = sorted({ids[0], ids[-1]}) if ids else []
            for layer in shown:
                lines += ['', f'Layer {layer} ({layer_kind(config, layer)}), by run time:', '',
                          '| What | Kernels | Reference us | Run us | Change | |', '|---|---|---|---|---|---|']
                for r in sorted((r for r in rows if r['layer'] == layer), key=lambda r: -r['us']):
                    names = ', '.join(sorted(set(r['kernels'])))
                    if r['ref_kernels'] and sorted(set(r['ref_kernels'])) != sorted(set(r['kernels'])):
                        names += ' (was ' + ', '.join(sorted(set(r['ref_kernels']))) + ')'
                    lines.append(f"| {r['what']} | {names} | {_us(r['ref_us'])} | {_us(r['us'])} | {_change(r['change'])} | "
                                 f"{'REGRESSION' if r['regression'] else ''} |")
            others = [r for r in regressions if r['layer'] not in shown]
            if others:
                lines += ['', 'Regressions in other layers:', '']
                lines += [f"- layer {r['layer']} {r['what']}: {_us(r['ref_us'])} -> {_us(r['us'])} us" for r in others]
            lines.append('')
        rows = targets(run)
        if rows:
            lines += ['### Targets', '', '| Issue | Layer | What | Run us | us at peak | Value | Target | Met |',
                      '|---|---|---|---|---|---|---|---|']
            for t in rows:
                value = '-' if t['value'] is None else (f"{t['value']:.0f}%" if t['unit'] == '% of peak' else f"{t['value']:.1f} us")
                target = f">= {t['target']:.0f}%" if t['unit'] == '% of peak' else f"< {t['target']:.0f} us"
                lines.append(f"| {t['issue']} | {t['layer']} | {t['what']} | {_us(t['us'])} | {_us(t['us_at_peak'])} | "
                             f"{value} | {target} | {'yes' if t['met'] else 'no'} |")
            lines.append('')
            summary['targets'] += [dict(t, run=name) for t in rows]
    n = len(summary['regressions'])
    lines += [f'**{n} regression{"s" if n != 1 else ""}.**' if n else '**No regressions.**', '']
    return '\n'.join(lines), summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path, help='layer_profile reports (.json)')
    p.add_argument('--baseline', type=Path, help='a directory of earlier reports (e.g. evidence/h100-20261026)')
    p.add_argument('--output', type=Path, help='write the Markdown here, and the summary beside it as .json')
    p.add_argument('--tolerance', type=float, default=0.10, help='allowed slowdown, as a fraction (default 0.10)')
    p.add_argument('--floor-us', type=float, default=2.0, help='ignore slowdowns smaller than this (default 2 us)')
    args = p.parse_args(argv)
    runs = {}
    for path in args.runs:
        report = json.loads(path.read_text())
        if 'kernels' not in report or 'format' not in report:
            p.error(f'{path} is not a layer_profile report')
        runs[str(path)] = report
    text, summary = gate(runs, args.baseline, args.tolerance, args.floor_us)
    if args.output:
        if args.output.exists():
            p.error('output exists; preserve evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(text)
    return 1 if summary['regressions'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
