"""Kernels built through Vx's own GPU path (gpu/spawn) against the hand-pipeline kernels on a
real device. Skipped when no supported CUDA GPU exists; build with gpu/spawn/build.sh first."""
from pathlib import Path

import ctypes as C
import numpy as np
import pytest

from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.spawn.launch import SpawnKernel

BUILD = Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def context():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as e:
        pytest.skip('real GPU unavailable: ' + str(e))
    try:
        yield ctx
    finally:
        ctx.close()


def upload(context, array):
    return context.tensor(array.shape, str(array.dtype), array.tobytes())


def download(tensor):
    return np.frombuffer(tensor.download(), dtype=tensor.dtype).reshape(tensor.shape)


# Kernels with exp (softmax_rows, swiglu, router, mla_partial, mla_merge) use f32::fast_exp,
# libdevice's expf, where the hand kernels use ex2.approx.ftz(x * log2 e): the two differ in the
# last bits, and below about -87.3 where ftz flushes. Exact parity comes back when an `across`
# loop takes an `mlir!` block that declares it writes nothing (Vx #2076 D1-D3).
EXP = dict(rtol=1e-5, atol=1e-37)
# mla_partial's numerators and mla_merge's output are sums of exp-weighted terms that cancel to near
# zero in some elements: there the last-bit exp difference is large relative to the value (A100,
# 2026-10-25: 3e-4 relative on a numerator of 2e-4, 1.6e-6 absolute on outputs of order 1), so
# those sums also get an absolute floor.
SUMS = dict(rtol=1e-5, atol=1e-5)


# Sizes cover a tail, one element, and more elements than threads (grid-stride loop).
@pytest.mark.parametrize('n,grid,block', [(1, 1, 32), (1000, 4, 128), (6144, 3, 256), (100003, 7, 96)])
def test_spawn_add_matches_hand_kernel(context, n, grid, block):
    rng = np.random.default_rng(n)
    a, b = rng.normal(size=n).astype('f4'), rng.normal(size=n).astype('f4')
    da, db = upload(context, a), upload(context, b)
    spawn_out, hand_out = context.tensor((n,)), context.tensor((n,))
    SpawnKernel(context, BUILD / 'spawn/add.ptx').launch(n=n, a=da, b=db, out=spawn_out)
    hand = context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_add')
    hand.launch(grid, block, [hand_out, da, db, C.c_int32(n)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), a + b)
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


def test_spawn_add_in_place(context):
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=4097).astype('f4'), rng.normal(size=4097).astype('f4')
    da, db = upload(context, a), upload(context, b)
    SpawnKernel(context, BUILD / 'spawn/add.ptx').launch(n=a.size, a=da, b=db, out=da)
    context.synchronize()
    np.testing.assert_array_equal(download(da), a + b)


def test_spawn_add_rejects_wrong_arguments(context):
    kernel = SpawnKernel(context, BUILD / 'spawn/add.ptx')
    x = context.tensor((4,))
    with pytest.raises(TypeError):
        kernel.launch(n=4, a=x, out=x)
    with pytest.raises(TypeError):
        kernel.launch(n=4, a=x, b=context.tensor((4,), 'int32'), out=x)


@pytest.mark.parametrize('rows,cols,block', [(7, 131, 64), (64, 6144, 256), (1, 1, 32)])
def test_spawn_matvec_matches_hand_kernel(context, rows, cols, block):
    rng = np.random.default_rng(rows * cols)
    w, x = rng.normal(size=(rows, cols)).astype('f4'), rng.normal(size=cols).astype('f4')
    dw, dx = upload(context, w.reshape(-1)), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // (block // 32))
    SpawnKernel(context, BUILD / 'spawn/matvec.ptx').launch(rows=rows, cols=cols, weight=dw, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_matvec')
    hand.launch(grid, block, [hand_out, dw, dx, C.c_int32(rows), C.c_int32(cols)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))
    np.testing.assert_allclose(download(spawn_out), w @ x, rtol=1e-4, atol=1e-4)


def test_spawn_bf16_decode_is_exact_for_every_encoding(context):
    # One column per row: out[r] = decode(r) * 1.0, so every bf16 code passes through the decode.
    codes = np.arange(65536, dtype=np.uint16)
    dw, dx, out = upload(context, codes), upload(context, np.ones(1, 'f4')), context.tensor((65536,))
    SpawnKernel(context, BUILD / 'spawn/bf16_matvec.ptx').launch(rows=65536, cols=1, weight=dw, x=dx, out=out)
    context.synchronize()
    # The row sum starts at +0.0, so the code of -0.0 comes out as +0.0.
    with np.errstate(invalid='ignore'):
        expected = (codes.astype(np.uint32) << 16).view(np.float32) + np.float32(0)
    np.testing.assert_array_equal(download(out).view(np.uint32)[~np.isnan(expected)],
                                  expected.view(np.uint32)[~np.isnan(expected)])
    assert np.isnan(download(out)[np.isnan(expected)]).all()


@pytest.mark.parametrize('rows,cols', [(5, 131), (96, 6144)])
def test_spawn_bf16_matvec_matches_hand_kernel(context, rows, cols):
    rng = np.random.default_rng(rows + cols)
    w = (rng.normal(size=(rows, cols)).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=cols).astype('f4')
    dw, dx = upload(context, w.reshape(-1)), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // 8)
    SpawnKernel(context, BUILD / 'spawn/bf16_matvec.ptx').launch(rows=rows, cols=cols, weight=dw, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'bf16.ptx').kernel('glm_vx_gpu_bf16_matvec')
    hand.launch(grid, 256, [hand_out, dw, dx, C.c_int32(rows), C.c_int32(cols)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


@pytest.mark.parametrize('rows,cols,scale_kind', [(5, 131, 0), (64, 6144, 1), (3, 257, 1)])
def test_spawn_int4_matvec_matches_hand_kernel(context, rows, cols, scale_kind):
    rng = np.random.default_rng(rows * 7 + cols)
    groups, row_bytes = -(-cols // 128), -(-cols // 2)
    packed = rng.integers(0, 256, size=rows * row_bytes, dtype=np.uint8)
    if cols % 2:  # odd tail: the padding nibble is zero
        packed.reshape(rows, row_bytes)[:, -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=rows * groups).astype(np.float32)
    if scale_kind:
        scales = (magnitudes.view(np.uint32) >> 16).astype(np.uint16)
    else:
        scales = magnitudes.astype(np.float16).view(np.uint16)
    x = rng.normal(size=cols).astype('f4')
    dp, ds, dx = upload(context, packed), upload(context, scales), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // 8)
    SpawnKernel(context, BUILD / 'spawn/int4_matvec.ptx').launch(
        rows=rows, cols=cols, scales=ds, scale_kind=scale_kind, packed=dp, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'int4.ptx').kernel('glm_vx_gpu_int4_matvec')
    hand.launch(grid, 256, [hand_out, dp, ds, dx, C.c_int32(rows), C.c_int32(cols), C.c_int32(scale_kind)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


@pytest.mark.parametrize('rows,cols', [(5, 131), (300, 6144)])
def test_spawn_fp8_matvec_matches_hand_kernel(context, rows, cols):
    rng = np.random.default_rng(rows * 11 + cols)
    weight = rng.integers(0, 256, size=rows * cols, dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = rng.uniform(0.001, 0.1, size=-(-rows // 128) * -(-cols // 128)).astype('f4')
    x = rng.normal(size=cols).astype('f4')
    dw, ds, dx = upload(context, weight), upload(context, scales), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // 8)
    SpawnKernel(context, BUILD / 'spawn/fp8_matvec.ptx').launch(
        rows=rows, cols=cols, scales=ds, weight=dw, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'fp8.ptx').kernel('glm_vx_gpu_fp8_matvec')
    hand.launch(grid, 256, [hand_out, dw, ds, dx, C.c_int32(rows), C.c_int32(cols)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


# mla_chunks_at (#33) against mla_decode_at on the device, at a full-width layer's shape.
@pytest.mark.skipif(not (BUILD / 'spawn/mla_chunks_at.ptx').exists(), reason='mla_chunks_at not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('count,splits', [(5, 1), (256, 1), (2048, 8), (700, 3)])
def test_spawn_mla_chunks_at_matches_mla_decode_at(context, count, splits):
    heads, rank, rope, tokens, splits_max = 64, 512, 64, 4096, 16
    rng = np.random.default_rng(count + splits)
    q, qr = rng.normal(size=heads * rank).astype('f4'), rng.normal(size=heads * rope).astype('f4')
    kv, kr = rng.normal(size=tokens * rank).astype('f4'), rng.normal(size=tokens * rope).astype('f4')
    common = dict(heads=heads, rank=rank, rope_dim=rope, selection_stride=0, cache_latent=upload(context, kv),
                  q_latent=upload(context, q), q_rope=upload(context, qr), cache_rope=upload(context, kr),
                  scale=float(np.float32(rank ** -0.5)), threads=512, splits_max=splits_max,
                  selected=upload(context, rng.permutation(tokens)[:count].astype('i4')),
                  step=upload(context, np.array([tokens - 1, tokens, count, splits], 'i4')))
    outs = {}
    for name in ('mla_decode_at', 'mla_chunks_at'):
        out, arrivals = context.tensor((heads, rank)), upload(context, np.zeros(heads, 'i4'))
        SpawnKernel(context, BUILD / f'spawn/{name}.ptx').launch(
            out=out, numerator=context.tensor((heads, splits_max, rank)), stats=context.tensor((heads, splits_max, 2)),
            arrivals=arrivals, **common)
        context.synchronize()
        np.testing.assert_array_equal(download(arrivals), 0)
        outs[name] = download(out)
    np.testing.assert_allclose(outs['mla_chunks_at'], outs['mla_decode_at'], rtol=1e-4, atol=1e-5)


# fp8_matvec4 (sixteen codes a lane step, one 16-byte load, #30) against the NumPy model of its order, bit for
# bit: the kernel has no fused multiply-adds, so the device rounds as NumPy does.
@pytest.mark.skipif(not (BUILD / 'spawn/fp8_matvec4.ptx').exists(), reason='fp8_matvec4 not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('rows,cols', [(5, 144), (300, 1008), (6144, 16384), (2048, 6144)])
def test_spawn_fp8_matvec4_matches_its_order(context, rows, cols):
    from gpu.test_spawn_vx_host import fp8_matvec4_case
    weight, scales, x, expected = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    out = context.tensor((rows,))
    SpawnKernel(context, BUILD / 'spawn/fp8_matvec4.ptx').launch(
        rows=rows, cols=cols, weight=upload(context, weight.view(np.uint32).reshape(-1)),
        scales=upload(context, scales.reshape(-1)), x=upload(context, x), out=out)
    context.synchronize()
    np.testing.assert_array_equal(download(out).view(np.uint32), np.asarray(expected, 'f4').view(np.uint32))


# fp8_matvec_fma and fp8_matvec_fma_tiles (#30, one fma a code) on the device: within gpu/test_spawn_vx_host's FMA_TOLERANCE of
# the float64 sums at GLM-5.3's widths, and for tiny activations.
@pytest.mark.skipif(not (BUILD / 'spawn/fp8_matvec_fma_tiles.ptx').exists(), reason='fp8_matvec_fma* not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('kernel', ['fp8_matvec_fma', 'fp8_matvec_fma_tiles'])
@pytest.mark.parametrize('rows,cols,x_scale', [(5, 144, 1.0), (300, 1008, 1.0), (6144, 16384, 1.0), (2048, 6144, 1.0),
                                               (130, 6144, 1e-9)])
def test_spawn_fp8_matvec_fma_within_its_tolerance(context, kernel, rows, cols, x_scale):
    from gpu.test_spawn_vx_host import assert_fma_close, fp8_matvec4_case
    weight, scales, x, _ = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    x = (x * np.float32(x_scale)).astype('f4')
    out = context.tensor((rows,))
    SpawnKernel(context, BUILD / f'spawn/{kernel}.ptx').launch(
        rows=rows, cols=cols, weight=upload(context, weight.view(np.uint32).reshape(-1)),
        scales=upload(context, scales.reshape(-1)), x=upload(context, x), out=out)
    context.synchronize()
    assert_fma_close(download(out), weight, scales, x)


# fp8_moe_fma_gate_up and fp8_moe_fma_down (#31) on the device at GLM-5.3's expert shapes (top-8 of 256,
# 6144 x 2048): within gpu/test_spawn_vx_host's FMA_TOLERANCE of float64 sums, as on the host.
@pytest.mark.skipif(not (BUILD / 'spawn/fp8_moe_fma_down.ptx').exists(), reason='fp8_moe_fma_* not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('experts,k,inter,hidden', [(256, 8, 2048, 6144), (12, 8, 144, 640)])
def test_spawn_fp8_moe_fma_within_its_tolerance(context, experts, k, inter, hidden):
    from gpu.test_spawn_vx_host import FMA_TOLERANCE, dequantized, expert_ids, moe_weights
    rng = np.random.default_rng(experts + inter)
    (gate, _), (up, _), (down, _) = (moe_weights(rng, 'fp8', experts, r, c) for r, c in
                                     ((inter, hidden), (inter, hidden), (hidden, inter)))
    x, ids = rng.normal(size=hidden).astype('f4'), expert_ids(rng, experts, k)
    weights = rng.uniform(0.01, 1, size=k).astype('f4')
    words = lambda w: upload(context, w['weight'].view(np.uint32).reshape(-1))
    act, out = context.tensor((k * inter,)), context.tensor((hidden,))
    ids_t = upload(context, ids)
    SpawnKernel(context, BUILD / 'spawn/fp8_moe_fma_gate_up.ptx').launch(
        act=act, gate=words(gate), gate_scales=upload(context, gate['scales'].reshape(-1)), up=words(up),
        up_scales=upload(context, up['scales'].reshape(-1)), x=upload(context, x), ids=ids_t, k=k, experts=experts,
        inter=inter, up_row=0, stride=inter, cols=hidden)
    SpawnKernel(context, BUILD / 'spawn/fp8_moe_fma_down.ptx').launch(
        out=out, down=words(down), scales=upload(context, down['scales'].reshape(-1)), act=act,
        weights=upload(context, weights), ids=ids_t, k=k, experts=experts, rows=hidden, cols=inter)
    context.synchronize()
    got_act, got = download(act), download(out)
    want_act, want = np.zeros(k * inter), np.zeros(hidden)
    for s, slot in enumerate(np.argsort(ids, kind='stable')):
        e = int(ids[slot])
        g, u = dequantized(gate, e) @ x.astype(np.float64), dequantized(up, e) @ x.astype(np.float64)
        want_act[s * inter:(s + 1) * inter] = g * (0.5 + 0.5 * np.tanh(g / 2)) * u
        want += float(weights[slot]) * (dequantized(down, e) @ got_act[s * inter:(s + 1) * inter].astype(np.float64))
    assert np.abs(got_act - want_act).max() <= FMA_TOLERANCE * np.abs(want_act).max()
    assert np.abs(got - want).max() <= FMA_TOLERANCE * np.abs(want).max()


# The FP8 conversion kernels (glm-vx #38): each sm_90 build (Hopper's e4m3x2 -> f16x2 conversion) must
# give its _ref build's bits (the exact integer decode), which the host tests check against float64.
CVT_PAIRS = {
    'fp8_matvec_cvt_tiles': lambda rng: _cvt_matvec_args(rng, 2048, 6144),
    'fp8_matvec_cvt_tiles3': lambda rng: _cvt_tiles3_args(rng),
    'fp8_moe_cvt_gate_up': lambda rng: _cvt_moe_args(rng, 'gate_up'),
    'fp8_moe_cvt_down': lambda rng: _cvt_moe_args(rng, 'down'),
}


def _codes(rng, n, nan=False):
    codes = rng.integers(0, 256, size=n, dtype=np.uint8)
    if not nan:
        codes[(codes & 0x7F) == 0x7F] = 0
    return codes


def _cvt_matvec_args(rng, rows, cols):
    return dict(rows=rows, cols=cols, out=np.zeros(rows, 'f4'), weight=_codes(rng, rows * cols).view(np.uint32),
                scales=rng.uniform(0.001, 0.1, size=-(-rows // 128) * -(-cols // 128)).astype('f4'),
                x=rng.normal(size=cols).astype('f4'))


def _cvt_tiles3_args(rng):
    cols, rows = 6144, (2048, 576, 128)
    args = dict(cols=cols, x=rng.normal(size=cols).astype('f4'), out=np.zeros(2048 + 576 + 128, 'f4'))
    for tag, r in zip('abc', rows):
        args.update({f'weight_{tag}': _codes(rng, r * cols).view(np.uint32), f'rows_{tag}': r,
                     f'scales_{tag}': rng.uniform(0.001, 0.1, size=-(-r // 128) * 48).astype('f4')})
    return args


def _cvt_moe_args(rng, half):
    experts, k, inter, hidden = 256, 8, 2048, 6144
    ids = rng.choice(experts, size=k, replace=False).astype(np.int32)
    if half == 'gate_up':
        n = experts * inter * hidden
        return dict(act=np.zeros(k * inter, 'f4'), gate=_codes(rng, n).view(np.uint32), up=_codes(rng, n).view(np.uint32),
                    gate_scales=rng.uniform(0.001, 0.1, size=experts * 16 * 48).astype('f4'),
                    up_scales=rng.uniform(0.001, 0.1, size=experts * 16 * 48).astype('f4'),
                    x=rng.normal(size=hidden).astype('f4'), ids=ids, k=k, experts=experts, inter=inter, up_row=0,
                    stride=inter, cols=hidden)
    return dict(out=np.zeros(hidden, 'f4'), down=_codes(rng, experts * hidden * inter).view(np.uint32),
                scales=rng.uniform(0.001, 0.1, size=experts * 48 * 16).astype('f4'),
                act=rng.normal(size=k * inter).astype('f4'), weights=rng.uniform(0.01, 1, size=k).astype('f4'), ids=ids,
                k=k, experts=experts, rows=hidden, cols=inter)


@pytest.mark.parametrize('name', sorted(CVT_PAIRS))
def test_spawn_cvt_sm90_gives_the_ref_bits(context, name):
    if not (BUILD / f'spawn/{name}.ptx').exists():
        pytest.skip(f'{name} not built (gpu/spawn/build.sh)')
    if tuple(context.compute_capability) < (8, 9):
        pytest.skip(f'{name} is sm_90 PTX (Hopper conversion); this GPU is {context.compute_capability}')
    args = CVT_PAIRS[name](np.random.default_rng(len(name)))
    out_name = next(k for k in ('out', 'act') if k in args)
    results = []
    for kernel in (name, name + '_ref'):
        device = {k: upload(context, v) if isinstance(v, np.ndarray) else v for k, v in args.items()}
        SpawnKernel(context, BUILD / f'spawn/{kernel}.ptx').launch(**device)
        context.synchronize()
        results.append(download(device[out_name]))
    np.testing.assert_array_equal(results[0].view(np.uint32), results[1].view(np.uint32))
    assert np.isfinite(results[0]).all() and np.abs(results[0]).max() > 0


# fp8_matvec4_rows (MTP verification, #16): each of m vectors gets fp8_matvec4's bits on the device,
# at GLM-5.3's widest projections, so verifying k + 1 tokens at once gives decoding's logits.
@pytest.mark.skipif(not (BUILD / 'spawn/fp8_matvec4_rows.ptx').exists(), reason='fp8_matvec4_rows not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('rows,cols', [(5, 144), (6144, 16384), (2048, 6144)])
@pytest.mark.parametrize('m', [1, 4])
def test_spawn_fp8_matvec4_rows_is_fp8_matvec4_per_vector(context, rows, cols, m):
    from gpu.test_spawn_vx_host import fp8_matvec4_case
    weight, scales, _, _ = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    xs = np.random.default_rng(rows + m).normal(size=(4, cols)).astype('f4')
    w, s = upload(context, weight.view(np.uint32).reshape(-1)), upload(context, scales.reshape(-1))
    out = context.tensor((4, rows))
    SpawnKernel(context, BUILD / 'spawn/fp8_matvec4_rows.ptx').launch(
        rows=rows, cols=cols, m=m, weight=w, scales=s, x=upload(context, xs.reshape(-1)), out=out)
    single = SpawnKernel(context, BUILD / 'spawn/fp8_matvec4.ptx')
    context.synchronize()
    got = download(out)
    for v in range(m):
        one = context.tensor((rows,))
        single.launch(rows=rows, cols=cols, weight=w, scales=s, x=upload(context, xs[v]), out=one)
        context.synchronize()
        np.testing.assert_array_equal(got[v].view(np.uint32), download(one).view(np.uint32))


# argmax_rows (#16): greedy decoding's token on the device, numpy.argmax at GLM-5.3's vocabulary,
# ties to the first index, whatever the block size.
@pytest.mark.skipif(not (BUILD / 'spawn/argmax_rows.ptx').exists(), reason='argmax_rows not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('threads', [256, 1024])
@pytest.mark.parametrize('ties', [False, True])
def test_spawn_argmax_rows_is_numpys_argmax(context, threads, ties):
    rng = np.random.default_rng(threads + ties)
    x = (rng.integers(-3, 4, size=(4, 154880)) if ties else rng.normal(size=(4, 154880)) * 5).astype('f4')
    out = context.tensor((4,), 'int32')
    SpawnKernel(context, BUILD / 'spawn/argmax_rows.ptx').launch(rows=4, dim=154880, threads=threads,
                                                                  x=upload(context, x), out=out)
    context.synchronize()
    np.testing.assert_array_equal(np.frombuffer(out.download(), np.int32), np.argmax(x, axis=1))


# bf16_matvec4 and int4_matvec4 (four u32 words a lane step, #30) against the NumPy model of
# their order, bit for bit, at small and GLM-5.3 shapes.
@pytest.mark.skipif(not (BUILD / 'spawn/bf16_matvec4.ptx').exists(), reason='bf16_matvec4 not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('rows,cols', [(5, 136), (7, 520), (2048, 6144), (6144, 16384)])
def test_spawn_bf16_matvec4_matches_its_order(context, rows, cols):
    from gpu.test_spawn_vx_host import bf16_matvec4_case
    words, x, expected = bf16_matvec4_case(rows, cols, rows + cols + 4)
    out = context.tensor((rows,))
    SpawnKernel(context, BUILD / 'spawn/bf16_matvec4.ptx').launch(
        rows=rows, cols=cols, weight=upload(context, words), x=upload(context, x), out=out)
    context.synchronize()
    np.testing.assert_array_equal(download(out).view(np.uint32), np.asarray(expected, 'f4').view(np.uint32))


@pytest.mark.skipif(not (BUILD / 'spawn/int4_matvec4.ptx').exists(), reason='int4_matvec4 not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('rows,cols,scale_kind', [(5, 128, 0), (9, 1152, 1), (2048, 6144, 1), (6144, 16384, 0)])
def test_spawn_int4_matvec4_matches_its_order(context, rows, cols, scale_kind):
    from gpu.test_spawn_vx_host import int4_matvec4_case
    packed, scales, x, expected = int4_matvec4_case(rows, cols, scale_kind, rows * 5 + cols)
    out = context.tensor((rows,))
    SpawnKernel(context, BUILD / 'spawn/int4_matvec4.ptx').launch(
        rows=rows, cols=cols, scale_kind=scale_kind, packed=upload(context, packed.view(np.uint32).reshape(-1)),
        scales=upload(context, scales.reshape(-1)), x=upload(context, x), out=out)
    context.synchronize()
    np.testing.assert_array_equal(download(out).view(np.uint32), np.asarray(expected, 'f4').view(np.uint32))


# mla_heads_at (#33): four heads a block, each head's arithmetic mla_chunks_at's, so equal bit for bit
# on the device too (their only fused multiply-adds are inside libdevice's expf), at a full layer's shape.
@pytest.mark.skipif(not (BUILD / 'spawn/mla_heads_at.ptx').exists(), reason='mla_heads_at not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('count,splits', [(5, 1), (256, 1), (2048, 8), (700, 3)])
def test_spawn_mla_heads_at_equals_mla_chunks_at(context, count, splits):
    heads, rank, rope, tokens, splits_max = 64, 512, 64, 4096, 16
    groups = heads // 4
    rng = np.random.default_rng(count * 3 + splits)
    q, qr = rng.normal(size=heads * rank).astype('f4'), rng.normal(size=heads * rope).astype('f4')
    kv, kr = rng.normal(size=tokens * rank).astype('f4'), rng.normal(size=tokens * rope).astype('f4')
    common = dict(heads=heads, rank=rank, rope_dim=rope, selection_stride=0, cache_latent=upload(context, kv),
                  q_latent=upload(context, q), q_rope=upload(context, qr), cache_rope=upload(context, kr),
                  scale=float(np.float32(rank ** -0.5)), threads=512, splits_max=splits_max,
                  selected=upload(context, rng.permutation(tokens)[:count].astype('i4')),
                  step=upload(context, np.array([tokens - 1, tokens, count, splits], 'i4')))
    chunks, heads_out = context.tensor((heads, rank)), context.tensor((groups, 4 * rank))
    arrivals = [upload(context, np.zeros(heads, 'i4')), upload(context, np.zeros(groups, 'i4'))]
    SpawnKernel(context, BUILD / 'spawn/mla_chunks_at.ptx').launch(
        out=chunks, numerator=context.tensor((heads, splits_max, rank)), stats=context.tensor((heads, splits_max, 2)),
        arrivals=arrivals[0], **common)
    SpawnKernel(context, BUILD / 'spawn/mla_heads_at.ptx').launch(
        out=heads_out, numerator=context.tensor((groups, splits_max, 4 * rank)), stats=context.tensor((groups, splits_max, 8)),
        arrivals=arrivals[1], **common)
    context.synchronize()
    assert not download(arrivals[0]).any() and not download(arrivals[1]).any()
    np.testing.assert_array_equal(download(heads_out).reshape(heads, rank).view('u4'), download(chunks).view('u4'))


# topk_chunks then topk_merge rounds (#32): the selection past 2048 tokens, against the host's rule.
@pytest.mark.skipif(not (BUILD / 'spawn/topk_merge.ptx').exists(), reason='topk_merge not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('tokens,k,ties', [(2049, 2048, True), (8192, 2048, False), (70000, 2048, True), (1 << 20, 2048, False),
                                           (9000, 16, True)])
def test_spawn_topk_chunks_and_merge_match_stable_argsort(context, tokens, k, ties):
    rng = np.random.default_rng(tokens * 7 + k)
    scores = rng.integers(-50, 50, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    scores[[3, 5, 7, 9]] = np.nan, -np.inf, -0.0, np.inf
    chunks = -(-tokens // 2048)
    lists = [context.tensor((chunks, 2048), 'int64') for _ in range(2)]
    out = context.tensor((1, k), 'int32')
    d_scores = upload(context, scores.view(np.int32))
    SpawnKernel(context, BUILD / 'spawn/topk_chunks.ptx').launch(lists=lists[0], scores=d_scores, tokens=tokens)
    merge, count, r = SpawnKernel(context, BUILD / 'spawn/topk_merge.ptx'), chunks, 0
    while count > 1:
        merge.launch(dst=lists[(r + 1) % 2], src=lists[r % 2], out_ids=out, lists=count, tokens=tokens, k=k)
        count, r = -(-count // 2), r + 1
    context.synchronize()
    np.testing.assert_array_equal(download(out)[0], np.argsort(-scores, kind='stable')[:k])


# topk_radix rounds 0 .. 8 (#32): the selection past 2048 tokens by radix select with atomic
# counts, against the host's rule, twice on the same buffers: the gather's order depends on timing,
# the ids must not, and each call leaves hist, count and the arrival count at zero for the next.
@pytest.mark.skipif(not (BUILD / 'spawn/topk_radix.ptx').exists(), reason='topk_radix not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('tokens,k,ties', [(2049, 2048, True), (8192, 2048, False), (70000, 2048, True), (1 << 20, 2048, False),
                                           (9000, 16, True), (100000, 2048, 'equal')])
def test_spawn_topk_radix_matches_stable_argsort(context, tokens, k, ties):
    rng = np.random.default_rng(tokens * 7 + k)
    if ties == 'equal':
        scores = np.zeros(tokens, 'f4')
    else:
        scores = rng.integers(-50, 50, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    scores[[3, 5, 7, 9]] = np.nan, -np.inf, -0.0, np.inf
    hist, state = upload(context, np.zeros(256, 'u4')), context.tensor((5,), 'int64')
    vals, count, out = context.tensor((2048,), 'int64'), upload(context, np.zeros(1, 'u4')), context.tensor((k,), 'int32')
    arrivals, d_scores = upload(context, np.zeros(1, 'i4')), upload(context, scores.view(np.int32))
    radix = SpawnKernel(context, BUILD / 'spawn/topk_radix.ptx')
    want = np.argsort(-scores, kind='stable')[:k]
    for _ in range(2):
        for r in range(9):
            radix.launch(out_ids=out, hist=hist, state=state, vals=vals, count=count, scores=d_scores, tokens=tokens, k=k,
                         round=r, arrivals=arrivals)
        context.synchronize()
        np.testing.assert_array_equal(download(out), want)
        assert not download(hist).any() and not download(count).any() and not download(arrivals).any()


# topk_select (#32): the stable top-k's set in position order, by one block's radix select in
# shared memory (atomic counts, whose order depends on timing; the ids must not), twice each.
@pytest.mark.skipif(not (BUILD / 'spawn/topk_select.ptx').exists(), reason='topk_select not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('tokens,k,ties', [(2049, 2048, True), (3001, 2048, False), (8192, 2048, False), (70000, 2048, True),
                                           (1 << 20, 2048, False), (9000, 16, True), (100000, 2048, 'equal'),
                                           (5000, 2048, 'few')])
def test_spawn_topk_select_is_the_stable_set_in_position_order(context, tokens, k, ties):
    rng = np.random.default_rng(tokens * 7 + k)
    if ties == 'equal':
        scores = np.zeros(tokens, 'f4')
    elif ties == 'few':
        scores = rng.integers(0, 2, size=tokens).astype('f4')
    else:
        scores = rng.integers(-50, 50, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    scores[[3, 5, 7, 9]] = np.nan, -np.inf, -0.0, np.inf
    out, d_scores = context.tensor((k,), 'int32'), upload(context, scores.view(np.int32))
    kernel = SpawnKernel(context, BUILD / 'spawn/topk_select.ptx')
    want = np.sort(np.argsort(-scores, kind='stable')[:k])
    for _ in range(2):
        kernel.launch(out_ids=out, scores=d_scores, tokens=tokens, k=k)
        context.synchronize()
        np.testing.assert_array_equal(download(out), want)


# topk_sort (#32): one shared-memory bitonic sort for up to 2048 tokens, against the host's rule.
@pytest.mark.skipif(not (BUILD / 'spawn/topk_sort.ptx').exists(), reason='topk_sort not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('tokens,k,ties', [(1, 1, False), (64, 2048, False), (300, 64, True), (2048, 2048, True),
                                           (2048, 16, False)])
def test_spawn_topk_sort_matches_stable_argsort(context, tokens, k, ties):
    rng = np.random.default_rng(tokens * 5 + k)
    scores = rng.integers(-4, 4, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    if tokens > 10:
        scores[3], scores[7] = np.nan, -0.0
    kk = min(k, tokens)
    out = upload(context, np.full(kk, -7, 'i4'))
    SpawnKernel(context, BUILD / 'spawn/topk_sort.ptx').launch(out_ids=out, scores=upload(context, scores.view('i4')),
                                                               tokens=tokens, k=k)
    context.synchronize()
    np.testing.assert_array_equal(download(out), np.argsort(-scores, kind='stable')[:kk])


@pytest.mark.parametrize('tokens,k,ties', [(1, 1, False), (300, 64, True), (5000, 2048, True), (70000, 2048, False)])
def test_spawn_topk_matches_stable_argsort_and_hand_kernel(context, tokens, k, ties):
    from gpu.topk import TopK, spawn_buffers
    rng = np.random.default_rng(tokens + k)
    scores = rng.integers(-4, 4, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    if tokens > 10:
        scores[3], scores[7] = np.nan, -0.0  # NaN sorts last; -0.0 ties +0.0
    kk = min(k, tokens)
    d_scores = upload(context, scores)
    spawn_ids = context.tensor((kk,), 'int32')
    scratch, order_a, order_b = (context.tensor(s, t) for s, t in spawn_buffers(tokens, k))
    SpawnKernel(context, BUILD / 'spawn/topk.ptx').launch(
        tokens=tokens, k=k, scratch=scratch, order_a=order_a, order_b=order_b,
        scores=d_scores.view((tokens,), 'int32'), out_ids=spawn_ids)
    hand_ids = TopK.from_ptx(context, BUILD / 'topk.ptx').topk_stable(d_scores, k)
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_ids), np.argsort(-scores, kind='stable')[:kk])
    np.testing.assert_array_equal(download(spawn_ids), download(hand_ids))


def hand_kernel(context, module, name):
    return context.load_ptx(BUILD / module).kernel(name)


def compare(context, spawn_name, hand, grid, block, spawn_args, hand_args, outs):
    """Launch the spawn kernel (with its manifest's shape) and the hand kernel (with grid, block)
    on separate outputs, then require bitwise equality. outs maps each spawn output name to its
    (spawn tensor, hand tensor) pair."""
    SpawnKernel(context, BUILD / f'spawn/{spawn_name}.ptx').launch(**spawn_args)
    hand.launch(grid, block, hand_args)
    context.synchronize()
    for name, (spawn_out, hand_out) in outs.items():
        np.testing.assert_array_equal(download(spawn_out), download(hand_out), err_msg=name)


@pytest.mark.parametrize('n,grid,block', [(1, 1, 32), (1000, 4, 128), (100003, 7, 96)])
@pytest.mark.parametrize('kernel', ['copy', 'scale'])
def test_spawn_elementwise_matches_hand_kernel(context, kernel, n, grid, block):
    x = np.random.default_rng(n).normal(size=n).astype('f4')
    dx = upload(context, x)
    s, h = context.tensor((n,)), context.tensor((n,))
    alpha = np.float32(0.37)
    if kernel == 'copy':
        compare(context, 'copy', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_copy'), grid, block,
                dict(n=n, x=dx, out=s), [h, dx, C.c_int32(n)], {'out': (s, h)})
        np.testing.assert_array_equal(download(s), x)
    else:
        compare(context, 'scale', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_scale'), grid, block,
                dict(n=n, x=dx, alpha=float(alpha), out=s), [h, dx, C.c_float(alpha), C.c_int32(n)], {'out': (s, h)})
        np.testing.assert_array_equal(download(s), x * alpha)


@pytest.mark.parametrize('n,grid,block,slot', [(1, 1, 32, 0), (6144, 3, 256, 5), (100003, 7, 96, 7)])
def test_spawn_scaled_add_matches_hand_kernel(context, n, grid, block, slot):
    rng = np.random.default_rng(n + slot)
    base, x, weights = (rng.normal(size=s).astype('f4') for s in (n, n, 8))
    dx, dw = upload(context, x), upload(context, weights)
    s, h = upload(context, base), upload(context, base)  # accumulated in place
    compare(context, 'scaled_add', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_scaled_add'), grid, block,
            dict(slot=slot, weights=dw, n=n, x=dx, out=s), [h, dx, dw, C.c_int32(slot), C.c_int32(n)], {'out': (s, h)})
    np.testing.assert_array_equal(download(s), base + x * weights[slot])


@pytest.mark.parametrize('heads,dim,grid,block', [(1, 2, 1, 32), (16, 64, 2, 256), (64, 64, 3, 96)])
def test_spawn_rope_glm_matches_hand_kernel(context, heads, dim, grid, block):
    rng = np.random.default_rng(heads * dim)
    x = rng.normal(size=heads * dim).astype('f4')
    angles = rng.uniform(0, 6.3, size=dim // 2)
    cos, sin = np.cos(angles).astype('f4'), np.sin(angles).astype('f4')
    dx, dc, ds = upload(context, x), upload(context, cos), upload(context, sin)
    s, h = context.tensor((heads * dim,)), context.tensor((heads * dim,))
    compare(context, 'rope_glm', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_rope_glm'), grid, block,
            dict(heads=heads, dim=dim, x=dx, cosine=dc, sine=ds, out=s.view((heads, dim))),
            [h, dx, dc, ds, C.c_int32(heads), C.c_int32(dim)], {'out': (s, h)})


@pytest.mark.parametrize('heads,dim,x_stride,out_stride,grid,block', [(1, 2, 2, 2, 1, 32), (64, 64, 256, 64, 3, 96), (16, 64, 192, 128, 2, 256)])
def test_spawn_rope_strided_matches_hand_kernel(context, heads, dim, x_stride, out_stride, grid, block):
    rng = np.random.default_rng(heads + dim + x_stride)
    x = rng.normal(size=heads * x_stride).astype('f4')
    angles = rng.uniform(0, 6.3, size=dim // 2)
    cos, sin = np.cos(angles).astype('f4'), np.sin(angles).astype('f4')
    dx, dc, ds = upload(context, x), upload(context, cos), upload(context, sin)
    # Untouched gaps between strided heads must stay as they were in both outputs.
    fill = rng.normal(size=heads * out_stride).astype('f4')
    s, h = upload(context, fill), upload(context, fill)
    compare(context, 'rope_strided', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_rope_strided'), grid, block,
            dict(heads=heads, dim=dim, x_stride=x_stride, out_stride=out_stride, x=dx, cosine=dc, sine=ds, out=s.view((heads, out_stride))),
            [h, dx, dc, ds, C.c_int32(heads), C.c_int32(dim), C.c_int32(x_stride), C.c_int32(out_stride)], {'out': (s, h)})


@pytest.mark.parametrize('tokens,heads,dim', [(1, 1, 1), (37, 16, 128), (1000, 32, 128)])
def test_spawn_index_scores_matches_hand_kernel(context, tokens, heads, dim):
    rng = np.random.default_rng(tokens * heads + dim)
    q, keys = rng.normal(size=heads * dim).astype('f4'), rng.normal(size=tokens * dim).astype('f4')
    weights = (rng.normal(size=heads) * heads ** -0.5).astype('f4')
    scale = np.float32(dim ** -0.5)
    dq, dk, dw = upload(context, q), upload(context, keys), upload(context, weights)
    s, h = context.tensor((tokens,)), context.tensor((tokens,))
    compare(context, 'index_scores', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_index_scores'), (tokens + 3) // 4, 128,
            dict(tokens=tokens, heads=heads, dim=dim, q=dq, keys=dk, scale=float(scale), weights=dw, out=s),
            [h, dq, dk, dw, C.c_int32(tokens), C.c_int32(heads), C.c_int32(dim), C.c_float(scale)], {'out': (s, h)})


@pytest.mark.parametrize('heads,rows,cols,w_stride,x_stride,out_stride', [
    (1, 1, 1, 1, 1, 1), (16, 64, 192, 64 * 192, 256, 64), (8, 37, 131, 37 * 131 + 5, 140, 40)])
def test_spawn_matvec_heads_matches_hand_kernel(context, heads, rows, cols, w_stride, x_stride, out_stride):
    rng = np.random.default_rng(heads * rows + cols)
    w = rng.normal(size=heads * w_stride).astype('f4')
    x = rng.normal(size=heads * x_stride).astype('f4')
    dw, dx = upload(context, w), upload(context, x)
    fill = rng.normal(size=heads * out_stride).astype('f4')
    s, h = upload(context, fill), upload(context, fill)
    grid = -(-(heads * rows) // 4)
    compare(context, 'matvec_heads', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_matvec_heads'), grid, 128,
            dict(heads=heads, rows=rows, cols=cols, w_stride=w_stride, x_stride=x_stride, weight=dw, x=dx,
                 out_stride=out_stride, out=s.view((heads, out_stride))),
            [h, dw, dx, C.c_int32(heads), C.c_int32(rows), C.c_int32(cols), C.c_int32(w_stride),
             C.c_int32(x_stride), C.c_int32(out_stride)], {'out': (s, h)})


@pytest.mark.parametrize('m,n,k', [(1, 1, 0), (1, 1, 1), (7, 9, 131), (33, 65, 256)])
def test_spawn_gemm_f32_matches_hand_kernel(context, m, n, k):
    rng = np.random.default_rng(m * n + k)
    a, b = rng.normal(size=max(m * k, 1)).astype('f4'), rng.normal(size=max(k * n, 1)).astype('f4')
    da, db = upload(context, a), upload(context, b)
    s, h = upload(context, np.full(m * n, 7, 'f4')), upload(context, np.full(m * n, 7, 'f4'))
    compare(context, 'gemm_f32', hand_kernel(context, 'gemm.ptx', 'glm_vx_gpu_gemm_f32'), -(-(m * n) // 128), 128,
            dict(m=m, n=n, k=k, a=da, b=db, out=s.view((m, n))), [h, da, db, C.c_int32(m), C.c_int32(n), C.c_int32(k)], {'out': (s, h)})


# Batched and grouped matmuls: experts == 0 (dense, expert_ids unread) and experts > 0 with an
# out-of-range id (-1 and E) in the routing, which must write zeros.
MATMUL_CASES = [(1, 1, 1, 0), (5, 37, 131, 0), (6, 300, 257, 3)]


def routing(context, rng, m, experts):
    if not experts:
        return upload(context, np.zeros(1, 'i4')), C.c_uint64(0)
    ids = rng.integers(0, experts, size=m).astype('i4')
    ids[0], ids[-1] = -1, experts
    d = upload(context, ids)
    return d, d


@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_spawn_bf16_matmul_matches_hand_kernel(context, m, n, k, experts):
    rng = np.random.default_rng(m + n + k + experts)
    e = max(experts, 1)
    w = (rng.normal(size=e * n * k).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=m * k).astype('f4')
    dw, dx = upload(context, w), upload(context, x)
    spawn_ids, hand_ids = routing(context, rng, m, experts)
    s, h = context.tensor((m * n,)), context.tensor((m * n,))
    compare(context, 'bf16_matmul', hand_kernel(context, 'bf16.ptx', 'glm_vx_gpu_bf16_matmul'), -(-(m * n) // 8), 256,
            dict(rows_m=m, rows_n=n, cols=k, experts=experts, expert_ids=spawn_ids, weight=dw, x=dx, out=s.view((m, n))),
            [h, dw, dx, hand_ids, C.c_int32(m), C.c_int32(n), C.c_int32(k), C.c_int32(experts)], {'out': (s, h)})


@pytest.mark.parametrize('scale_kind', [0, 1])
@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_spawn_int4_matmul_matches_hand_kernel(context, m, n, k, experts, scale_kind):
    rng = np.random.default_rng(m * 3 + n + k + experts + scale_kind)
    e, groups, row_bytes = max(experts, 1), -(-k // 128), -(-k // 2)
    packed = rng.integers(0, 256, size=(e * n, row_bytes), dtype=np.uint8)
    if k % 2:
        packed[:, -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=e * n * groups).astype(np.float32)
    scales = ((magnitudes.view(np.uint32) >> 16).astype(np.uint16) if scale_kind
              else magnitudes.astype(np.float16).view(np.uint16))
    x = rng.normal(size=m * k).astype('f4')
    dp, ds, dx = upload(context, packed.reshape(-1)), upload(context, scales), upload(context, x)
    spawn_ids, hand_ids = routing(context, rng, m, experts)
    s, h = context.tensor((m * n,)), context.tensor((m * n,))
    compare(context, 'int4_matmul', hand_kernel(context, 'int4.ptx', 'glm_vx_gpu_int4_matmul'), -(-(m * n) // 8), 256,
            dict(rows_m=m, rows_n=n, cols=k, experts=experts, expert_ids=spawn_ids, scales=ds, scale_kind=scale_kind,
                 packed=dp, x=dx, out=s.view((m, n))),
            [h, dp, ds, dx, hand_ids, C.c_int32(m), C.c_int32(n), C.c_int32(k), C.c_int32(experts),
             C.c_int32(scale_kind)], {'out': (s, h)})


@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_spawn_fp8_matmul_matches_hand_kernel(context, m, n, k, experts):
    rng = np.random.default_rng(m * 5 + n + k + experts)
    e = max(experts, 1)
    weight = rng.integers(0, 256, size=e * n * k, dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = rng.uniform(0.001, 0.1, size=e * -(-n // 128) * -(-k // 128)).astype('f4')
    x = rng.normal(size=m * k).astype('f4')
    dw, ds, dx = upload(context, weight), upload(context, scales), upload(context, x)
    spawn_ids, hand_ids = routing(context, rng, m, experts)
    s, h = context.tensor((m * n,)), context.tensor((m * n,))
    compare(context, 'fp8_matmul', hand_kernel(context, 'fp8_batched.ptx', 'glm_vx_gpu_fp8_matmul'), -(-(m * n) // 8), 256,
            dict(rows_m=m, rows_n=n, cols=k, experts=experts, expert_ids=spawn_ids, weight=dw, x=dx, scales=ds, out=s.view((m, n))),
            [h, dw, ds, dx, hand_ids, C.c_int32(m), C.c_int32(n), C.c_int32(k), C.c_int32(experts)], {'out': (s, h)})


# Norms need a correctly rounded sqrt in the device image: Vx #1944 (PR #1955) or later.
@pytest.mark.parametrize('rows,dim,block', [(1, 6144, 256), (5, 131, 64), (3, 1, 32), (4, 2048, 1024)])
def test_spawn_rmsnorm_rows_matches_hand_kernel(context, rows, dim, block):
    rng = np.random.default_rng(rows * dim)
    x, w = rng.normal(size=rows * dim).astype('f4'), rng.normal(size=dim).astype('f4')
    dx, dw = upload(context, x), upload(context, w)
    spawn_out, hand_out, single_out = (context.tensor((rows * dim,)) for _ in range(3))
    SpawnKernel(context, BUILD / 'spawn/rmsnorm_rows.ptx').launch(rows=rows, dim=dim, eps=1e-5, x=dx, weight=dw, threads=block,
                                                                    out=spawn_out.view((rows, dim)))
    hand = context.load_ptx(BUILD / 'kernels.ptx')
    hand.kernel('glm_vx_gpu_rmsnorm_rows').launch(rows, block, [hand_out, dx, dw, C.c_int32(rows), C.c_int32(dim), C.c_float(1e-5)])
    if rows == 1:
        hand.kernel('glm_vx_gpu_rmsnorm').launch(1, block, [single_out, dx, dw, C.c_int32(dim), C.c_float(1e-5)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))
    if rows == 1:
        np.testing.assert_array_equal(download(spawn_out), download(single_out))


@pytest.mark.parametrize('rows,dim,block', [(1, 6144, 256), (7, 333, 96)])
def test_spawn_residual_rmsnorm_matches_hand_kernel(context, rows, dim, block):
    rng = np.random.default_rng(rows + dim)
    x, r, w = (rng.normal(size=n).astype('f4') for n in (rows * dim, rows * dim, dim))
    dx, dr, dw = upload(context, x), upload(context, r), upload(context, w)
    outs = [context.tensor((rows * dim,)) for _ in range(4)]
    SpawnKernel(context, BUILD / 'spawn/residual_rmsnorm.ptx').launch(
        rows=rows, dim=dim, eps=1e-5, x=dx, residual=dr, residual_out=outs[1].view((rows, dim)), weight=dw,
        threads=block, out=outs[0].view((rows, dim)))
    context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_residual_rmsnorm').launch(
        rows, block, [outs[2], outs[3], dx, dr, dw, C.c_int32(rows), C.c_int32(dim), C.c_float(1e-5)])
    context.synchronize()
    np.testing.assert_array_equal(download(outs[1]), x + r)
    np.testing.assert_array_equal(download(outs[1]), download(outs[3]))
    np.testing.assert_array_equal(download(outs[0]), download(outs[2]))


@pytest.mark.parametrize('rows,dim,block', [(1, 128, 128), (9, 333, 64)])
def test_spawn_layernorm_matches_hand_kernel(context, rows, dim, block):
    rng = np.random.default_rng(rows * 3 + dim)
    x, w, b = (rng.normal(size=n).astype('f4') for n in (rows * dim, dim, dim))
    dx, dw, db = upload(context, x + 3.0), upload(context, w), upload(context, b)
    spawn_out, hand_out = context.tensor((rows * dim,)), context.tensor((rows * dim,))
    SpawnKernel(context, BUILD / 'spawn/layernorm.ptx').launch(rows=rows, dim=dim, eps=1e-6, x=dx, weight=dw, bias=db,
                                                                 threads=block, out=spawn_out.view((rows, dim)))
    context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_layernorm').launch(
        rows, block, [hand_out, dx, dw, db, C.c_int32(rows), C.c_int32(dim), C.c_float(1e-6)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


@pytest.mark.parametrize('rows,dim,block', [(1, 154880, 256), (5, 131, 64), (3, 1, 32), (4, 2048, 1024)])
def test_spawn_softmax_rows_matches_hand_kernel(context, rows, dim, block):
    rng = np.random.default_rng(rows * dim + 5)
    x = (rng.normal(size=rows * dim) * 3).astype('f4')
    dx = upload(context, x)
    spawn_out, hand_out, single_out = (context.tensor((rows * dim,)) for _ in range(3))
    SpawnKernel(context, BUILD / 'spawn/softmax_rows.ptx').launch(rows=rows, dim=dim, x=dx, threads=block,
                                                                    out=spawn_out.view((rows, dim)))
    hand = context.load_ptx(BUILD / 'kernels.ptx')
    hand.kernel('glm_vx_gpu_softmax_rows').launch(rows, block, [hand_out, dx, C.c_int32(rows), C.c_int32(dim)])
    if rows == 1:
        hand.kernel('glm_vx_gpu_softmax').launch(1, block, [single_out, dx, C.c_int32(dim)])
    context.synchronize()
    np.testing.assert_allclose(download(spawn_out), download(hand_out), **EXP)
    if rows == 1:
        np.testing.assert_allclose(download(spawn_out), download(single_out), **EXP)


@pytest.mark.xfail(reason='exp is libdevice expf until Vx #2076 D1-D3 (EXP above)', strict=False)
def test_spawn_softmax_flushes_subnormal_exp_like_hand_kernel(context):
    # x - max reaches -100: exp of anything below about -87.3 is subnormal, which ftz flushes.
    x = np.linspace(-100.0, 0.0, 4096).astype('f4')
    dx = upload(context, x)
    spawn_out, hand_out = context.tensor((4096,)), context.tensor((4096,))
    SpawnKernel(context, BUILD / 'spawn/softmax_rows.ptx').launch(rows=1, dim=4096, x=dx, threads=256,
                                                                  out=spawn_out.view((1, 4096)))
    context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_softmax_rows').launch(1, 256, [hand_out, dx, C.c_int32(1), C.c_int32(4096)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))
    assert (download(spawn_out)[x < -88] == 0).all()


@pytest.mark.parametrize('n,grid,block', [(1, 1, 32), (1000, 4, 128), (12288, 48, 256), (100003, 7, 96)])
def test_spawn_swiglu_matches_hand_kernel(context, n, grid, block):
    rng = np.random.default_rng(n + 9)
    gate, up = (rng.normal(size=n) * 6).astype('f4'), rng.normal(size=n).astype('f4')
    gate[:min(n, 4)] = [0.0, -0.0, 20.0, -20.0][:min(n, 4)]
    dg, du = upload(context, gate), upload(context, up)
    spawn_out, hand_out = context.tensor((n,)), context.tensor((n,))
    SpawnKernel(context, BUILD / 'spawn/swiglu.ptx').launch(n=n, gate=dg, up=du, out=spawn_out)
    context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_swiglu').launch(grid, block, [hand_out, dg, du, C.c_int32(n)])
    context.synchronize()
    np.testing.assert_allclose(download(spawn_out), download(hand_out), **EXP)


# Ties: integer logits plus a bias that repeats, so equal scores must go to the lower id.
@pytest.mark.parametrize('experts,k,ties', [(256, 8, False), (256, 8, True), (7, 3, True), (1, 1, False), (33, 8, True)])
def test_spawn_router_matches_hand_kernel(context, experts, k, ties):
    rng = np.random.default_rng(experts * 10 + k)
    if ties:
        logits = rng.integers(-3, 3, size=experts).astype('f4')
        bias = np.tile(np.array([0.0, 0.5], 'f4'), experts)[:experts].copy()
    else:
        logits, bias = rng.normal(size=experts).astype('f4'), (rng.normal(size=experts) * 0.1).astype('f4')
    dl, db = upload(context, logits), upload(context, bias)
    ids = [context.tensor((k,), 'int32') for _ in range(2)]
    weights = [context.tensor((k,)) for _ in range(2)]
    SpawnKernel(context, BUILD / 'spawn/router.ptx').launch(
        experts=experts, k=k, indices=ids[0], logits=dl, bias=db, weights=weights[0], scale=2.5)
    context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_router').launch(
        1, 32, [ids[1], weights[1], dl, db, C.c_int32(experts), C.c_int32(k), C.c_float(2.5)])
    context.synchronize()
    np.testing.assert_array_equal(download(ids[0]), download(ids[1]))
    np.testing.assert_allclose(download(weights[0]), download(weights[1]), **EXP)
    if ties:  # integer logits keep distinct scores far apart, so a float64 oracle orders them the same
        score = 1.0 / (1.0 + np.exp(-logits.astype('f8'))) + bias
        np.testing.assert_array_equal(download(ids[0]), np.lexsort((np.arange(experts), -score))[:k])


# selected mixes padding (-1) and out-of-range ids; selection_stride 0 shares one list across
# heads, > 0 gives each head its own. splits > count leaves some partitions empty (l = 0).
@pytest.mark.parametrize('heads,rank,rope,tokens,count,splits,stride,block', [
    (2, 48, 4, 7, 5, 3, 0, 64), (2, 48, 4, 7, 5, 8, 0, 64), (2, 48, 4, 7, 0, 3, 0, 64),
    (64, 512, 64, 4096, 2048, 16, 0, 512), (4, 64, 64, 300, 37, 4, 37, 64)])
def test_spawn_mla_matches_hand_kernels(context, heads, rank, rope, tokens, count, splits, stride, block):
    rng = np.random.default_rng(heads + rank + count + splits)
    q, qr = rng.normal(size=heads * rank).astype('f4'), rng.normal(size=heads * rope).astype('f4')
    kv, kr = rng.normal(size=tokens * rank).astype('f4'), rng.normal(size=tokens * rope).astype('f4')
    lists = heads if stride else 1
    selected = rng.integers(0, tokens, size=max(1, lists * max(count, 1))).astype('i4')
    if count:
        selected[1 % selected.size] = -1
        selected[-1] = tokens + 5
    dq, dqr, dkv, dkr, dsel = (upload(context, a) for a in (q, qr, kv, kr, selected))
    scale = rank ** -0.5
    num = [context.tensor((heads * splits * rank,)) for _ in range(2)]
    stats = [context.tensor((heads * splits * 2,)) for _ in range(2)]
    out = [context.tensor((heads * rank,)) for _ in range(2)]
    SpawnKernel(context, BUILD / 'spawn/mla_partial.ptx').launch(
        heads=heads, rank=rank, rope_dim=rope, tokens=tokens, count=count,
        selection_stride=stride, splits=splits, selected=dsel, cache_latent=dkv, q_latent=dq, q_rope=dqr,
        cache_rope=dkr, scale=scale, threads=block, numerator=num[0].view((heads * splits, rank)),
        stats=stats[0].view((heads * splits, 2)))
    SpawnKernel(context, BUILD / 'spawn/mla_merge.ptx').launch(
        heads=heads, rank=rank, splits=splits, stats=stats[0], numerator=num[0], out=out[0].view((heads, rank)))
    hand = context.load_ptx(BUILD / 'kernels.ptx')
    I = C.c_int32
    hand.kernel('glm_vx_gpu_mla_partial').launch(heads * splits, block, [
        num[1], stats[1], dq, dqr, dkv, dkr, dsel, I(heads), I(rank), I(rope), I(tokens), I(count), I(stride),
        I(splits), C.c_float(scale)])
    hand.kernel('glm_vx_gpu_mla_merge').launch(heads, 256, [out[1], num[1], stats[1], I(heads), I(rank), I(splits)])
    context.synchronize()
    np.testing.assert_allclose(download(stats[0]), download(stats[1]), **EXP)
    np.testing.assert_allclose(download(num[0]), download(num[1]), **SUMS)
    np.testing.assert_allclose(download(out[0]), download(out[1]), **SUMS)


# mla_decode: mla_partial and mla_merge in one launch, the block of each head's last split merging
# that head (Vx #2187). Launched three times on one arrivals buffer: each launch must leave the
# counts at zero for the next (Vx #2188) and give the spawn partial-then-merge result bit for bit.
@pytest.mark.skipif(not (BUILD / 'spawn/mla_decode.ptx').exists(), reason='mla_decode not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('heads,rank,rope,tokens,count,splits,stride,block', [
    (2, 48, 4, 7, 5, 3, 0, 64), (2, 48, 4, 7, 5, 8, 0, 64), (3, 48, 4, 7, 5, 1, 0, 64),
    (64, 512, 64, 4096, 2048, 16, 0, 512), (4, 64, 64, 300, 37, 4, 37, 64)])
def test_spawn_mla_decode_equals_partial_then_merge(context, heads, rank, rope, tokens, count, splits, stride, block):
    rng = np.random.default_rng(heads * 3 + rank + count + splits)
    q, qr = rng.normal(size=heads * rank).astype('f4'), rng.normal(size=heads * rope).astype('f4')
    kv, kr = rng.normal(size=tokens * rank).astype('f4'), rng.normal(size=tokens * rope).astype('f4')
    selected = rng.integers(0, tokens, size=max(1, (heads if stride else 1) * max(count, 1))).astype('i4')
    if count:
        selected[1 % selected.size] = -1
        selected[-1] = tokens + 5
    dq, dqr, dkv, dkr, dsel = (upload(context, a) for a in (q, qr, kv, kr, selected))
    common = dict(heads=heads, rank=rank, rope_dim=rope, tokens=tokens, count=count, selection_stride=stride,
                  splits=splits, selected=dsel, cache_latent=dkv, q_latent=dq, q_rope=dqr, cache_rope=dkr,
                  scale=rank ** -0.5, threads=block)
    num, stats, out = context.tensor((heads * splits * rank,)), context.tensor((heads * splits * 2,)), context.tensor((heads * rank,))
    SpawnKernel(context, BUILD / 'spawn/mla_partial.ptx').launch(
        numerator=num.view((heads * splits, rank)), stats=stats.view((heads * splits, 2)), **common)
    SpawnKernel(context, BUILD / 'spawn/mla_merge.ptx').launch(
        heads=heads, rank=rank, splits=splits, stats=stats, numerator=num, out=out.view((heads, rank)))
    arrivals = upload(context, np.zeros(heads, 'i4'))
    decode = SpawnKernel(context, BUILD / 'spawn/mla_decode.ptx')
    for _ in range(3):
        d_num, d_stats = context.tensor((heads, splits, rank)), context.tensor((heads, splits, 2))
        d_out = upload(context, np.full(heads * rank, 7, 'f4'))
        decode.launch(out=d_out.view((heads, rank)), numerator=d_num, stats=d_stats,
                      arrivals=arrivals, **common)
        context.synchronize()
        np.testing.assert_array_equal(download(arrivals).view('i4'), 0)
        np.testing.assert_array_equal(download(d_num).view('u4').ravel(), download(num).view('u4'))
        np.testing.assert_array_equal(download(d_stats).view('u4').ravel(), download(stats).view('u4'))
        np.testing.assert_array_equal(download(d_out).view('u4'), download(out).view('u4'))


# mla_decode_at (glm-vx #27): tokens, count and splits from the step buffer, splits_max splits per
# head with the ones past splits idle but arriving. Launched as a replayed graph would be: splits
# 1, 2, 3 on one arrivals buffer, each equal to mla_decode with those values bit for bit, the counts
# back at zero after each.
@pytest.mark.skipif(not (BUILD / 'spawn/mla_decode_at.ptx').exists(), reason='mla_decode_at not built (gpu/spawn/build.sh)')
@pytest.mark.parametrize('heads,rank,rope,tokens,block,splits_max', [(2, 48, 4, 700, 64, 3), (64, 512, 64, 4096, 512, 16)])
def test_spawn_mla_decode_at_equals_mla_decode_for_each_splits(context, heads, rank, rope, tokens, block, splits_max):
    rng = np.random.default_rng(heads + rank + splits_max)
    q, qr = rng.normal(size=heads * rank).astype('f4'), rng.normal(size=heads * rope).astype('f4')
    kv, kr = rng.normal(size=tokens * rank).astype('f4'), rng.normal(size=tokens * rope).astype('f4')
    dq, dqr, dkv, dkr = (upload(context, a) for a in (q, qr, kv, kr))
    common = dict(heads=heads, rank=rank, rope_dim=rope, selection_stride=0, cache_latent=dkv, q_latent=dq,
                  q_rope=dqr, cache_rope=dkr, scale=rank ** -0.5, threads=block)
    decode = SpawnKernel(context, BUILD / 'spawn/mla_decode.ptx')
    at = SpawnKernel(context, BUILD / 'spawn/mla_decode_at.ptx')
    arrivals = upload(context, np.zeros(heads, 'i4'))
    for count, splits in ((5, 1), (300, 2), (640, 3)):
        dsel = upload(context, rng.permutation(tokens)[:count].astype('i4'))
        num, stats = context.tensor((heads, splits, rank)), context.tensor((heads, splits, 2))
        out = upload(context, np.full(heads * rank, 7, 'f4'))
        decode.launch(out=out.view((heads, rank)), numerator=num, stats=stats, selected=dsel, tokens=tokens,
                      count=count, splits=splits, arrivals=upload(context, np.zeros(heads, 'i4')), **common)
        a_num, a_stats = context.tensor((heads, splits_max, rank)), context.tensor((heads, splits_max, 2))
        a_out = upload(context, np.full(heads * rank, 7, 'f4'))
        at.launch(out=a_out.view((heads, rank)), numerator=a_num, stats=a_stats, selected=dsel, splits_max=splits_max,
                  step=upload(context, np.array([tokens - 1, tokens, count, splits], 'i4')), arrivals=arrivals, **common)
        context.synchronize()
        np.testing.assert_array_equal(download(arrivals).view('i4'), 0)
        got_num = np.frombuffer(download(a_num), 'u4').reshape(heads, splits_max, rank)[:, :splits]
        got_stats = np.frombuffer(download(a_stats), 'u4').reshape(heads, splits_max, 2)[:, :splits]
        np.testing.assert_array_equal(got_num.ravel(), np.frombuffer(download(num), 'u4'))
        np.testing.assert_array_equal(got_stats.ravel(), np.frombuffer(download(stats), 'u4'))
        np.testing.assert_array_equal(np.frombuffer(download(a_out), 'u4'), np.frombuffer(download(out), 'u4'))


# Glue fusions (#39, ResidentGLM(fuse_glue=True)): on the GPU each gives the bits of the launches it
# replaces, at GLM-5.3's widths.
def test_glue_fusions_give_the_bits_of_the_launches_they_replace(context):
    rng = np.random.default_rng(39)
    hand = context.load_ptx(BUILD / 'kernels.ptx')
    f32 = lambda *a: [C.c_float(v) for v in a]
    i32 = lambda *a: [C.c_int32(v) for v in a]
    grid = lambda n: (n + 255) // 256

    def launch(name, g, block, args):
        hand.kernel('glm_vx_gpu_' + name).launch((g,), (block,), args)

    # residual_rmsnorm8 against residual_rmsnorm, and against add then rmsnorm, hidden 6144.
    dim = 6144
    x, r, w = (upload(context, rng.normal(size=dim).astype('f4')) for _ in range(3))
    outs = [context.tensor((dim,)) for _ in range(6)]
    launch('residual_rmsnorm', 1, 1024, [outs[0], outs[1], x, r, w] + i32(1, dim) + f32(1e-6))
    launch('add', grid(dim), 256, [outs[2], x, r] + i32(dim))
    launch('rmsnorm', 1, 1024, [outs[3], outs[2], w] + i32(dim) + f32(1e-6))
    launch('residual_rmsnorm8', 1, 1024, [outs[4], outs[5], x, r, w] + i32(dim) + f32(1e-6))
    # rmsnorm_pair against two rmsnorms: q_a (2048) and the latent (512).
    xa, wa = (upload(context, rng.normal(size=2048).astype('f4')) for _ in range(2))
    xb, wb = (upload(context, rng.normal(size=512).astype('f4')) for _ in range(2))
    pair = [context.tensor((2048,)), context.tensor((512,)), context.tensor((2048,)), context.tensor((512,))]
    launch('rmsnorm', 1, 1024, [pair[0], xa, wa] + i32(2048) + f32(1e-6))
    launch('rmsnorm', 1, 1024, [pair[1], xb, wb] + i32(512) + f32(1e-6))
    launch('rmsnorm_pair', 2, 1024, [pair[2], xa, wa] + i32(2048) + [pair[3], xb, wb] + i32(512) + f32(1e-6))
    # index_key_at against layernorm, copy, rope_strided and store_row: index_dim 128, rot 64.
    angle = rng.normal(size=32).astype('f4')
    cosine, sine = upload(context, np.cos(angle)), upload(context, np.sin(angle))
    step = upload(context, np.array([3, 4, 4, 1], np.int32))
    ik, kw, kb = (upload(context, rng.normal(size=128).astype('f4')) for _ in range(3))
    normed, row = context.tensor((128,)), context.tensor((128,))
    keys = [upload(context, np.full((6, 128), np.nan, 'f4')) for _ in range(2)]
    launch('layernorm', 1, 256, [normed, ik, kw, kb] + i32(1, 128) + f32(1e-6))
    launch('copy', 1, 256, [row, normed] + i32(128))
    launch('rope_strided', 1, 256, [row, normed, cosine, sine] + i32(1, 64, 64, 64))
    launch('store_row', 1, 256, [keys[0], row, step] + i32(128))
    launch('index_key_at', 1, 256, [keys[1], ik, kw, kb, cosine, sine, step] + i32(128, 64) + f32(1e-6))
    # kv_rows_bf16_at against store_row_bf16_at, rope_strided and store_row_bf16_at: rank 512, rot 64.
    latent, rope_x = upload(context, rng.normal(size=512).astype('f4')), upload(context, rng.normal(size=64).astype('f4'))
    rope_row = context.tensor((64,))
    caches = [upload(context, np.zeros((6, n), np.uint16)) for n in (512, 64, 512, 64)]
    store = SpawnKernel(context, BUILD / 'spawn/store_row_bf16_at.ptx')
    store.launch(cache=caches[0], row=latent, step=step, width=512)
    launch('rope_strided', 1, 256, [rope_row, rope_x, cosine, sine] + i32(1, 64, 64, 64))
    store.launch(cache=caches[1], row=rope_row, step=step, width=64)
    SpawnKernel(context, BUILD / 'spawn/kv_rows_bf16_at.ptx').launch(
        latents=caches[2], rope_keys=caches[3], latent_row=latent, rope_x=rope_x, cosine=cosine, sine=sine, step=step,
        rank=512, rot=64)
    context.synchronize()
    bits = lambda t: download(t).view(np.uint32) if download(t).dtype == np.float32 else download(t)
    for a, b in ((outs[1], outs[5]), (outs[0], outs[4]), (outs[2], outs[5]), (outs[3], outs[4]), (pair[0], pair[2]),
                 (pair[1], pair[3]), (keys[0], keys[1]), (caches[0], caches[2]), (caches[1], caches[3])):
        np.testing.assert_array_equal(bits(a), bits(b))
    assert not np.isnan(download(keys[1])[3]).any() and np.isnan(download(keys[1])[[0, 1, 2, 4, 5]]).all()
