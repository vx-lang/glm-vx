"""The dense matvec kernels at GLM-5.3's projection shapes, timed against the bytes they read.

#30, for --format fp8 (the default), bf16 or int4: the hand kernel (gpu/fp8.vx, bf16.vx, int4.vx),
gpu/spawn's kernel of the same order through Vx's own path (fp8_matvec, bf16_matvec, int4_matvec), and
gpu/spawn's kernel that reads four u32 words a lane step (fp8_matvec4 with its table decode,
bf16_matvec4, int4_matvec4). Each launch is timed with CUDA events after a 512 MB copy that evicts L2
and keeps the GPU busy while it is queued (moe_profile.l2_flush), the median of --repeats. The report
gives GPU time, GB/s for the weight bytes and scales, and the share of the device's peak DRAM
bandwidth. The three outputs are checked to agree to 1e-4 relative (their summation orders differ);
each kernel's exact order is pinned by the tests.

  python -m validation.matvec_profile --format bf16 --output run/matvec-bf16.json
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np

from validation.layer_profile import peak_bandwidth
from validation.moe_profile import event_us

# (name, rows, cols) of GLM-5.3's dense projections in one decode step.
SHAPES = [('q_a_proj', 2048, 6144), ('q_b_proj', 16384, 2048), ('kv_a_proj', 576, 6144),
          ('o_proj', 6144, 16384), ('mlp.gate/up', 12288, 6144), ('mlp.down', 6144, 12288),
          ('shared_expert.gate/up', 2048, 6144), ('shared_expert.down', 6144, 2048)]


def kernels(fmt, ctx, build, rng, rows, cols, stream):
    """(weight and scale bytes, {kernel: launch}, {kernel: out}) for one shape of format fmt."""
    from gpu.spawn.launch import SpawnKernel
    up = lambda a: ctx.tensor(a.shape, str(a.dtype), a.tobytes())
    i32 = C.c_int32
    outs = {k: ctx.tensor((rows,)) for k in ('hand', 'spawn', 'matvec4')}
    x = up(rng.normal(size=cols).astype('f4'))
    if fmt == 'fp8':
        codes = rng.integers(0, 256, size=rows * cols, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = rng.uniform(0.001, 0.1, size=-(-rows // 128) * -(-cols // 128)).astype('f4')
        weight, d_scales = up(codes), up(scales)
        words = weight.view((rows * cols // 4,), 'uint32')  # the same bytes, four codes an element
        hand = ctx.load_ptx(build / 'fp8.ptx').kernel('glm_vx_gpu_fp8_matvec')
        spawn, four = (SpawnKernel(ctx, build / 'spawn' / f'{n}.ptx') for n in ('fp8_matvec', 'fp8_matvec4'))
        return codes.nbytes + scales.nbytes, {
            'hand': lambda: hand.launch((-(-rows // 8),), (256,), [outs['hand'], weight, d_scales, x, i32(rows), i32(cols)],
                                        stream=stream),
            'spawn': lambda: spawn.launch(stream=stream, out=outs['spawn'], weight=weight, scales=d_scales, x=x,
                                          rows=rows, cols=cols),
            'matvec4': lambda: four.launch(stream=stream, out=outs['matvec4'], weight=words, scales=d_scales, x=x,
                                           rows=rows, cols=cols)}, outs
    if fmt == 'bf16':
        bits = (rng.normal(size=rows * cols).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
        weight = up(bits)
        words = weight.view((rows * cols // 2,), 'uint32')
        hand = ctx.load_ptx(build / 'bf16.ptx').kernel('glm_vx_gpu_bf16_matvec')
        spawn, four = (SpawnKernel(ctx, build / 'spawn' / f'{n}.ptx') for n in ('bf16_matvec', 'bf16_matvec4'))
        return bits.nbytes, {
            'hand': lambda: hand.launch((-(-rows // 4),), (128,), [outs['hand'], weight, x, i32(rows), i32(cols)],
                                        stream=stream),
            'spawn': lambda: spawn.launch(stream=stream, out=outs['spawn'], weight=weight, x=x, rows=rows, cols=cols),
            'matvec4': lambda: four.launch(stream=stream, out=outs['matvec4'], weight=words, x=x, rows=rows,
                                           cols=cols)}, outs
    packed = rng.integers(0, 256, size=rows * cols // 2, dtype=np.uint8)
    scales = (rng.uniform(0.001, 0.1, size=rows * cols // 128).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    d_packed, d_scales = up(packed), up(scales)
    words = d_packed.view((rows * cols // 8,), 'uint32')
    hand = ctx.load_ptx(build / 'int4.ptx').kernel('glm_vx_gpu_int4_matvec')
    spawn, four = (SpawnKernel(ctx, build / 'spawn' / f'{n}.ptx') for n in ('int4_matvec', 'int4_matvec4'))
    return packed.nbytes + scales.nbytes, {
        'hand': lambda: hand.launch((-(-rows // 4),), (128,), [outs['hand'], d_packed, d_scales, x, i32(rows), i32(cols),
                                                              i32(1)], stream=stream),
        'spawn': lambda: spawn.launch(stream=stream, out=outs['spawn'], packed=d_packed, scales=d_scales, x=x,
                                      rows=rows, cols=cols, scale_kind=1),
        'matvec4': lambda: four.launch(stream=stream, out=outs['matvec4'], packed=words, scales=d_scales, x=x,
                                       rows=rows, cols=cols, scale_kind=1)}, outs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--format', choices=['fp8', 'bf16', 'int4'], default='fp8')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    from gpu.runtime import CUDAContext
    build = Path(__file__).resolve().parent.parent / 'gpu' / 'build'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    rng = np.random.default_rng(1)
    peak = peak_bandwidth()
    results = []
    for name, rows, cols in SHAPES:
        nbytes, launches, outs = kernels(args.format, ctx, build, rng, rows, cols, stream)
        for launch in launches.values():
            launch()
        stream.synchronize()
        got = {k: np.frombuffer(v.download(), 'f4') for k, v in outs.items()}
        for k in ('spawn', 'matvec4'):
            if not np.allclose(got[k], got['hand'], rtol=1e-4, atol=1e-4 * float(np.abs(got['hand']).max())):
                raise SystemExit(f'{name}: {k} disagrees with the hand kernel')
        mb = nbytes / 1e6
        for kernel, launch in launches.items():
            us = event_us(ctx, stream, build / 'spawn', launch, args.repeats)
            results.append(dict(shape=name, rows=rows, cols=cols, kernel=kernel, us=us, mb=mb, gbs=mb / us * 1e3,
                                pct_of_peak_bandwidth=100 * mb / us * 1e3 / peak, us_at_peak_bandwidth=mb / peak * 1e3))
    ctx.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(format=args.format, peak_dram_gbs=peak, results=results), indent=2) + '\n')
    lines = [f'# {args.format.upper()} dense matvec at GLM-5.3 shapes. Peak DRAM {peak:.0f} GB/s; CUDA events, '
             f'L2 flushed, median of {args.repeats}.', '',
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
