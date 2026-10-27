"""The FP8 dense matvec kernels at GLM-5.3's projection shapes, timed against the bytes they read.

#30: gpu/fp8.vx's hand kernel, gpu/spawn's fp8_matvec (the same order through Vx's own path) and
gpu/spawn's fp8_matvec4 (four codes a load, table decode). Each launch is timed with CUDA events
after a 512 MB copy that evicts L2 and keeps the GPU busy while it is queued (moe_profile.l2_flush),
the median of --repeats. The report gives GPU time, GB/s for the weight bytes and scales, and the
share of the device's peak DRAM bandwidth. The three outputs are checked to agree to 1e-4
relative (their summation orders differ); each kernel's exact order is pinned by the tests.

  python -m validation.matvec_profile --output run/matvec.json
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

from validation.layer_profile import peak_bandwidth
from validation.moe_profile import event_us

# (name, rows, cols) of GLM-5.3's FP8 projections in one decode step.
SHAPES = [('q_a_proj', 2048, 6144), ('q_b_proj', 16384, 2048), ('kv_a_proj', 576, 6144),
          ('o_proj', 6144, 16384), ('mlp.gate/up', 12288, 6144), ('mlp.down', 6144, 12288),
          ('shared_expert.gate/up', 2048, 6144), ('shared_expert.down', 6144, 2048)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    build = Path(__file__).resolve().parent.parent / 'gpu' / 'build'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    rng = np.random.default_rng(1)
    up = lambda a: ctx.tensor(a.shape, str(a.dtype), a.tobytes())
    hand = ctx.load_ptx(build / 'fp8.ptx').kernel('glm_vx_gpu_fp8_matvec')
    spawn = SpawnKernel(ctx, build / 'spawn' / 'fp8_matvec.ptx')
    four = SpawnKernel(ctx, build / 'spawn' / 'fp8_matvec4.ptx')
    peak = peak_bandwidth()
    results = []
    for name, rows, cols in SHAPES:
        codes = rng.integers(0, 256, size=rows * cols, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = rng.uniform(0.001, 0.1, size=-(-rows // 128) * -(-cols // 128)).astype('f4')
        weight, d_scales = up(codes), up(scales)
        words = weight.view((rows * cols // 4,), 'uint32')  # the same bytes, four codes an element
        x = up(rng.normal(size=cols).astype('f4'))
        outs = {k: ctx.tensor((rows,)) for k in ('hand', 'spawn', 'matvec4')}
        launches = {
            'hand': lambda: hand.launch((-(-rows // 8),), (256,), [outs['hand'], weight, d_scales, x, C.c_int32(rows),
                                                                C.c_int32(cols)], stream=stream),
            'spawn': lambda: spawn.launch(stream=stream, out=outs['spawn'], weight=weight, scales=d_scales, x=x,
                                          rows=rows, cols=cols),
            'matvec4': lambda: four.launch(stream=stream, out=outs['matvec4'], weight=words, scales=d_scales, x=x,
                                           rows=rows, cols=cols)}
        for launch in launches.values():
            launch()
        stream.synchronize()
        got = {k: np.frombuffer(v.download(), 'f4') for k, v in outs.items()}
        for k in ('spawn', 'matvec4'):
            if not np.allclose(got[k], got['hand'], rtol=1e-4, atol=1e-4 * float(np.abs(got['hand']).max())):
                raise SystemExit(f'{name}: {k} disagrees with the hand kernel')
        mb = (codes.nbytes + scales.nbytes) / 1e6
        for kernel, launch in launches.items():
            us = event_us(ctx, stream, build / 'spawn', launch, args.repeats)
            results.append(dict(shape=name, rows=rows, cols=cols, kernel=kernel, us=us, mb=mb, gbs=mb / us * 1e3,
                                pct_of_peak_bandwidth=100 * mb / us * 1e3 / peak, us_at_peak_bandwidth=mb / peak * 1e3))
    ctx.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(peak_dram_gbs=peak, results=results), indent=2) + '\n')
    lines = [f'# FP8 dense matvec at GLM-5.3 shapes. Peak DRAM {peak:.0f} GB/s; CUDA events, L2 flushed, '
             f'median of {args.repeats}.', '',
             '| Projection | rows x cols | MB | us at peak | hand us | spawn us | matvec4 us | matvec4 % of peak |',
             '|---|---|---|---|---|---|---|---|']
    for i in range(0, len(results), 3):
        h, s, f = results[i:i + 3]
        lines.append(f"| {h['shape']} | {h['rows']} x {h['cols']} | {h['mb']:.1f} | {h['us_at_peak_bandwidth']:.1f} | "
                     f"{h['us']:.1f} | {s['us']:.1f} | {f['us']:.1f} | {f['pct_of_peak_bandwidth']:.0f} |")
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
