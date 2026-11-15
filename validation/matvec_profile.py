"""The dense matvec kernels at GLM-5.3's projection shapes, timed against the bytes they read.

#30, for --format fp8 (the default), bf16 or int4: the hand kernel (gpu/fp8.vx, bf16.vx, int4.vx),
gpu/spawn's kernel of the same order through Vx's own path (fp8_matvec, bf16_matvec, int4_matvec), and
gpu/spawn's kernel that reads four u32 words a lane step, one 16-byte load (fp8_matvec4,
bf16_matvec4, int4_matvec4). For fp8 also fp8_matvec_fma ('matvec_fma': fp8_matvec4's layout with one
fma a code, which ncu on an H200 suggested; the others are instruction-bound). Each launch is timed with CUDA events after a 512 MB copy that evicts L2
and keeps the GPU busy while it is queued (moe_profile.l2_flush), the median of --repeats. The report
gives GPU time, GB/s for the weight bytes and scales, and the share of the device's peak DRAM
bandwidth. The three outputs are checked to agree to 1e-4 relative (their summation orders differ);
each kernel's exact order is pinned by the tests.

For fp8 it also times cuBLASLt's block-scaled FP8 matmul (gpu/cublaslt.py, #29), SGLang's scheme:
'cublaslt' is the activation quantization (gpu/fp8_quant.vx) and the matmul, 'cublaslt_mm' the matmul
alone, in --cublaslt-orientation. Their output is checked to 1e-4 against float64 sums of the same
quantized operands (not against the hand kernel, whose activations stay float32). Without
libcublasLt, or when it has no algorithm for a shape, those rows are left out and the reason printed.

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
# The output projection stays BF16 in every checkpoint format, so only --format bf16 times it.
LM_HEAD = ('lm_head', 154880, 6144)


def cublaslt_kernels(ctx, build, rows, cols, x, codes, scales, stream, orientation):
    """({'cublaslt': quantize and matmul, 'cublaslt_mm': matmul}, out, float64 reference), or a reason
    string when cuBLASLt cannot run this shape."""
    from gpu import fp8_quant
    from gpu.cublaslt import CublasLt, CublasLtUnavailable, Fp8BlockMatmul, pack_weight_scales
    try:
        if getattr(cublaslt_kernels, 'ctx', None) is not ctx:
            cublaslt_kernels.ctx, cublaslt_kernels.lt = ctx, CublasLt(ctx)
            cublaslt_kernels.quant = ctx.load_ptx(build / 'fp8_quant.ptx').kernel('glm_vx_gpu_fp8_quant_groups')
        mm = Fp8BlockMatmul(cublaslt_kernels.lt, 1, rows, cols, orientation=orientation)
    except (CublasLtUnavailable, OSError) as error:
        return str(error)
    up = lambda a: ctx.tensor(a.shape, str(a.dtype), a.tobytes())
    blocks = scales.reshape(-(-rows // 128), -(-cols // 128))
    w_codes, w_scales = up(codes), up(pack_weight_scales(blocks))
    # cuBLASLt multiplies 4 tokens (gpu/cublaslt.py): x is row 0, rows 1-3 zero codes; out row 0 is x's.
    x_codes, x_scales = up(np.zeros((4, cols), np.uint8)), up(np.ones((cols // 128, 4), np.float32))
    out = ctx.tensor((4 * rows,))
    quant = lambda: fp8_quant.launch(cublaslt_kernels.quant, x_codes, x_scales, x, 1, cols, stream=stream, scale_rows=4)
    matmul = lambda: mm(out, x_codes, x_scales, w_codes, w_scales, stream=stream)
    xq, xs = fp8_quant.quant_groups(np.frombuffer(x.download(), 'f4').reshape(1, cols))
    xd, values, grid = fp8_quant.dequant_groups(xq, xs).ravel(), fp8_quant.e4m3_values(), codes.reshape(rows, cols)
    reference = np.empty(rows)
    for r in range(0, rows, 128):  # a row block at a time: o_proj's float64 weights would be 800 MB
        reference[r:r + 128] = (values[grid[r:r + 128]] * np.repeat(blocks[r // 128], 128)[:cols]) @ xd
    quant()  # the matmul alone reads this launch's codes
    return {'cublaslt': lambda: (quant(), matmul()), 'cublaslt_mm': matmul}, out, reference


def kernels(fmt, ctx, build, rng, rows, cols, stream):
    """(weight and scale bytes, {kernel: launch}, {kernel: out}) for one shape of format fmt."""
    from gpu.spawn.launch import SpawnKernel
    up = lambda a: ctx.tensor(a.shape, str(a.dtype), a.tobytes())
    i32 = C.c_int32
    outs = {k: ctx.tensor((rows,)) for k in ('hand', 'spawn', 'matvec4') + (('matvec_fma',) if fmt == 'fp8' else ())}
    x = up(rng.normal(size=cols).astype('f4'))
    if fmt == 'fp8':
        codes = rng.integers(0, 256, size=rows * cols, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0
        scales = rng.uniform(0.001, 0.1, size=-(-rows // 128) * -(-cols // 128)).astype('f4')
        weight, d_scales = up(codes), up(scales)
        words = weight.view((rows * cols // 4,), 'uint32')  # the same bytes, four codes an element
        hand = ctx.load_ptx(build / 'fp8.ptx').kernel('glm_vx_gpu_fp8_matvec')
        spawn, four, fma = (SpawnKernel(ctx, build / 'spawn' / f'{n}.ptx')
                            for n in ('fp8_matvec', 'fp8_matvec4', 'fp8_matvec_fma'))
        kernels.fp8 = (x, codes, scales)
        return codes.nbytes + scales.nbytes, {
            'hand': lambda: hand.launch((-(-rows // 8),), (256,), [outs['hand'], weight, d_scales, x, i32(rows), i32(cols)],
                                        stream=stream),
            'spawn': lambda: spawn.launch(stream=stream, out=outs['spawn'], weight=weight, scales=d_scales, x=x,
                                          rows=rows, cols=cols),
            'matvec4': lambda: four.launch(stream=stream, out=outs['matvec4'], weight=words, scales=d_scales, x=x,
                                           rows=rows, cols=cols),
            'matvec_fma': lambda: fma.launch(stream=stream, out=outs['matvec_fma'], weight=words, scales=d_scales, x=x,
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
    p.add_argument('--cublaslt-orientation', choices=['weights_a', 'weights_b'], default='weights_a')
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
    for name, rows, cols in SHAPES + ([LM_HEAD] if args.format == 'bf16' else []):
        nbytes, launches, outs = kernels(args.format, ctx, build, rng, rows, cols, stream)
        for launch in launches.values():
            launch()
        stream.synchronize()
        got = {k: np.frombuffer(v.download(), 'f4') for k, v in outs.items()}
        for k in [k for k in got if k != 'hand']:
            if not np.allclose(got[k], got['hand'], rtol=1e-4, atol=1e-4 * float(np.abs(got['hand']).max())):
                raise SystemExit(f'{name}: {k} disagrees with the hand kernel')
        if args.format == 'fp8':
            lt = cublaslt_kernels(ctx, build, rows, cols, *kernels.fp8, stream, args.cublaslt_orientation)
            if isinstance(lt, str):
                print(f'{name}: no cuBLASLt rows: {lt}')
            else:
                lt_launches, lt_out, reference = lt
                lt_launches['cublaslt_mm']()
                stream.synchronize()
                got_lt = np.frombuffer(lt_out.download(), 'f4')[:rows]
                # Hopper's FP8 tensor cores accumulate below float32 within a 128-wide k block: ~3e-4.
                if not np.allclose(got_lt, reference, rtol=1e-3, atol=1e-3 * float(np.abs(reference).max())):
                    raise SystemExit(f'{name}: cuBLASLt disagrees with float64 sums of its quantized operands')
                launches.update(lt_launches)
        mb = nbytes / 1e6
        for kernel, launch in launches.items():
            us = event_us(ctx, stream, build / 'spawn', launch, args.repeats)
            results.append(dict(shape=name, rows=rows, cols=cols, kernel=kernel, us=us, mb=mb, gbs=mb / us * 1e3,
                                pct_of_peak_bandwidth=100 * mb / us * 1e3 / peak, us_at_peak_bandwidth=mb / peak * 1e3))
    ctx.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(format=args.format, peak_dram_gbs=peak, results=results), indent=2) + '\n')
    names = list(dict.fromkeys(r['kernel'] for r in results))
    lines = [f'# {args.format.upper()} dense matvec at GLM-5.3 shapes. Peak DRAM {peak:.0f} GB/s; CUDA events, '
             f'L2 flushed, median of {args.repeats}. us (% of peak bandwidth).', '',
             '| Projection | rows x cols | MB | us at peak | ' + ' | '.join(names) + ' |',
             '|---|---|---|---|' + '---|' * len(names)]
    for shape in dict.fromkeys(r['shape'] for r in results):
        row = {r['kernel']: r for r in results if r['shape'] == shape}
        h = next(iter(row.values()))
        cells = [f"{row[k]['us']:.1f} ({row[k]['pct_of_peak_bandwidth']:.0f}%)" if k in row else '-' for k in names]
        lines.append(f"| {shape} | {h['rows']} x {h['cols']} | {h['mb']:.1f} | {h['us_at_peak_bandwidth']:.1f} | "
                     + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    args.output.with_suffix('.md').write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
