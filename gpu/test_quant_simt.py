"""INT4 and batched/grouped FP8 kernels: the compiled Vx kernels under the CPU SIMT emulator
(NOT GPU validation) against the glm_vx/quant.py oracles. Rebuild with gpu/testing/build_simt.sh."""
import ctypes as C
import math
import os

import numpy as np
import pytest

from glm_vx.checkpoint import FP8_E4M3FN_TABLE
from glm_vx.quant import (bf16_to_f32, compressed_int4_pack, compressed_int4_to_packed, compressed_int4_unpack,
                          fp8_matvec_blocked, int4_dequantize, int4_matvec, int4_quantize, int4_unpack)
from gpu.test_simt import Simt


@pytest.fixture(scope='module')
def simt():
    return Simt()


def launch_dims(tasks, block):
    return (tasks + block // 32 - 1) // (block // 32), block


def close(actual, expected, scale):
    # fp32 warp-order accumulation vs float64 oracle: bound by the sum of magnitudes.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5 * max(float(np.max(scale)), 1.0))


def int4_data(n, k, seed, experts=None):
    rng = np.random.default_rng(seed + n * 131 + k)
    shape = (n, k) if experts is None else (experts, n, k)
    w = rng.normal(size=shape).astype('f4')
    flat = w.reshape(-1, k)
    packed, scales = int4_quantize(flat)
    tail = ((k + 1) // 2,), ((k + 127) // 128,)
    lead = () if experts is None else (experts,)
    return packed.reshape(*lead, n, tail[0][0]), scales.reshape(*lead, n, tail[1][0])


def run_int4(simt, packed, scales, x, ids, experts, block=64, kind=0):
    # kind 0: scales are float16; kind 1: scales are raw bfloat16 bits (uint16).
    m, k = x.shape
    n = packed.shape[-2]
    out = np.full((m, n), -7.0, 'f4')
    dummy = np.zeros(1, 'i4') if ids is None else ids
    grid, block = launch_dims(m * n, block)
    simt.run('glm_vx_gpu_int4_matmul', out, packed, scales.view(np.uint16), x, dummy, m, n, k, experts, kind, grid=grid, block=block)
    return out


ODD = [(1, 1), (3, 31), (4, 128), (5, 129), (7, 255), (9, 257), (130, 300)]


def test_half_decode_all_65536(simt):
    f = simt.lib.glm_vx_i4_half
    f.argtypes, f.restype = [C.c_int32], C.c_float
    bits = np.arange(65536, dtype=np.uint16)
    expected = bits.view(np.float16).astype('f4')
    actual = np.array([f(int(b)) for b in bits], dtype='f4')
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    assert np.array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))  # incl. -0, inf


@pytest.mark.parametrize('n,k', ODD)
@pytest.mark.parametrize('block', [32, 128])
def test_int4_matvec(simt, n, k, block):
    packed, scales = int4_data(n, k, 1)
    x = np.random.default_rng(5).normal(size=k).astype('f4')
    out = np.full(n, -7.0, 'f4')
    grid, block = launch_dims(n, block)
    simt.run('glm_vx_gpu_int4_matvec', out, packed, scales.view(np.uint16), x, n, k, 0, grid=grid, block=block)
    close(out, int4_matvec(packed, scales, x), np.abs(int4_dequantize(packed, scales, k)).astype('f8') @ np.abs(x).astype('f8'))


def test_int4_all_codes_and_extreme_scales(simt):
    # Arbitrary packed bytes cover every nibble (-8 included); scales span fp16 subnormal..large.
    rng = np.random.default_rng(3)
    n, k = 16, 256
    packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.choice([2.0**-24, 2.0**-14, 0.37, 1.0, 1024.0, 60000.0], size=(n, 2)).astype(np.float16)
    x = rng.normal(size=k).astype('f4')
    out = run_int4(simt, packed, scales, x[None], None, 0)[0]
    close(out, int4_matvec(packed, scales, x), np.abs(int4_dequantize(packed, scales, k)).astype('f8') @ np.abs(x).astype('f8'))


@pytest.mark.parametrize('m', [1, 3, 5])
@pytest.mark.parametrize('n,k', [(3, 31), (5, 129), (130, 257)])
def test_int4_batched(simt, m, n, k):
    packed, scales = int4_data(n, k, 2)
    x = np.random.default_rng(6).normal(size=(m, k)).astype('f4')
    out = run_int4(simt, packed, scales, x, None, 0)
    expected = np.stack([int4_matvec(packed, scales, row) for row in x])
    close(out, expected, np.abs(int4_dequantize(packed, scales, k)).sum() * np.abs(x).max())


def test_int4_grouped_experts_and_unrouted(simt):
    e, n, k, m = 4, 6, 200, 7
    packed, scales = int4_data(n, k, 4, experts=e)
    x = np.random.default_rng(8).normal(size=(m, k)).astype('f4')
    ids = np.array([2, 0, 3, -1, 1, 4, 2], dtype='i4')  # -1 and 4 (== E) are unrouted
    out = run_int4(simt, packed, scales, x, ids, e)
    for row, i in enumerate(ids):
        if 0 <= i < e:
            close(out[row], int4_matvec(packed[i], scales[i], x[row]), np.abs(int4_dequantize(packed[i], scales[i], k)).sum() * np.abs(x[row]).max())
        else:
            assert np.array_equal(out[row], np.zeros(n, 'f4'))


def test_bf16_decode_all_65536(simt):
    f = simt.lib.glm_vx_i4_bf16
    f.argtypes, f.restype = [C.c_int32], C.c_float
    bits = np.arange(65536, dtype=np.uint16)
    expected = bf16_to_f32(bits)
    actual = np.array([f(int(b)) for b in bits], dtype='f4')
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    finite = ~np.isnan(expected)
    assert np.array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))  # incl. -0, inf, subnormals


def compressed_data(n, k, seed, experts=None, extreme=False):
    # A compressed-tensors checkpoint as stored (int32 words, bf16 scale bits), and the int4.vx
    # repack of it. Codes span all 16 values; extreme scales cover bf16 subnormal..2^100.
    rng = np.random.default_rng(seed + n * 59 + k)
    lead = () if experts is None else (experts,)
    codes = rng.integers(-8, 8, (*lead, n, k)).astype(np.int8)
    groups = (k + 127) // 128
    if extreme:
        values = rng.choice([2.0**-133, 2.0**-126, 2.0**-20, 0.0078125, 1.0, 2.0**100], size=(*lead, n, groups))
        bits = (values.astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    else:
        bits = (rng.uniform(1e-3, 0.1, (*lead, n, groups)).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    words = np.stack([compressed_int4_pack(c) for c in codes.reshape(-1, n, k)]).reshape(*lead, n, (k + 7) // 8)
    packed = np.stack([compressed_int4_to_packed(w, k) for w in words.reshape(-1, n, (k + 7) // 8)]).reshape(*lead, n, (k + 1) // 2)
    return codes, words, bits, packed


def compressed_reference(words, bits, x, k):
    # Independent of int4.vx's layout: codes from the int32 words by shifts, scales from bf16 bits.
    codes = compressed_int4_unpack(words, k).astype('f8')
    scale = np.repeat(bf16_to_f32(bits).astype('f8'), 128, axis=1)[:, :k]
    w = codes * scale
    return w @ x.astype('f8'), np.abs(w) @ np.abs(x).astype('f8')


@pytest.mark.parametrize('n,k', ODD + [(16, 1024)])
@pytest.mark.parametrize('extreme', [False, True])
def test_int4_bf16_scales_compressed_tensors(simt, n, k, extreme):
    codes, words, bits, packed = compressed_data(n, k, 11, extreme=extreme)
    assert np.array_equal(int4_unpack(packed, k), codes)
    x = np.random.default_rng(12).normal(size=k).astype('f4')
    expected, bound = compressed_reference(words, bits, x, k)
    out = np.full(n, -7.0, 'f4')
    grid, block = launch_dims(n, 64)
    simt.run('glm_vx_gpu_int4_matvec', out, packed, bits, x, n, k, 1, grid=grid, block=block)
    close(out, expected, bound)
    batched = run_int4(simt, packed, bits, np.stack([x, -x]), None, 0, kind=1)
    close(batched[0], expected, bound)
    close(batched[1], -expected, bound)


def test_int4_scale_kind_selects_the_decoder(simt):
    # 0x3F80 is 1.0 as bf16 and 1.875 as fp16; the same bits must give different, exact results.
    n, k = 2, 128
    packed = np.full((n, k // 2), 0x11, np.uint8)  # every code 1
    bits = np.full((n, 1), 0x3F80, np.uint16)
    x = np.ones((1, k), 'f4')
    assert np.array_equal(run_int4(simt, packed, bits, x, None, 0, kind=1), np.full((1, n), 128.0, 'f4'))
    assert np.array_equal(run_int4(simt, packed, bits, x, None, 0, kind=0), np.full((1, n), 240.0, 'f4'))


def test_int4_bf16_grouped_experts(simt):
    e, n, k, m = 3, 5, 384, 6
    codes, words, bits, packed = compressed_data(n, k, 13, experts=e)
    x = np.random.default_rng(14).normal(size=(m, k)).astype('f4')
    ids = np.array([1, 2, -1, 0, 3, 1], dtype='i4')
    out = run_int4(simt, packed, bits, x, ids, e, kind=1)
    for row, i in enumerate(ids):
        if 0 <= i < e:
            close(out[row], *compressed_reference(words[i], bits[i], x[row], k))
        else:
            assert np.array_equal(out[row], np.zeros(n, 'f4'))


REAL = os.environ.get('GLM_VX_INT4_CHECKPOINT')


@pytest.mark.skipif(not REAL, reason='set GLM_VX_INT4_CHECKPOINT to a validation.fetch_tensors directory')
def test_int4_real_checkpoint_linears(simt):
    # Real GLM W4A16 tensors: loader repack + bf16 kernel vs the float64 reference built from the
    # stored int32 words. Rows are capped to keep the emulator quick; every column is used.
    from glm_vx.checkpoint import SafetensorsCheckpoint
    with SafetensorsCheckpoint(REAL) as ckpt:
        prefixes = sorted(n[:-len('.weight_packed')] for n in ckpt.names if n.endswith('.weight_packed'))
        assert prefixes
        for prefix in prefixes:
            packed, bits, scale_format, (n, k) = ckpt.compressed_int4(prefix)
            assert scale_format == 'bfloat16'
            rows = min(n, 512)
            words = ckpt.raw(prefix + '.weight_packed', rows=slice(0, rows))
            x = np.random.default_rng(k).normal(size=(2, k)).astype('f4')
            out = run_int4(simt, np.ascontiguousarray(packed[:rows]), np.ascontiguousarray(bits[:rows]), x, None, 0, kind=1)
            for r in range(2):
                close(out[r], *compressed_reference(words, bits[:rows], x[r], k))


def fp8_data(n, k, seed, experts=None, finite=True):
    rng = np.random.default_rng(seed + n * 977 + k)
    lead = () if experts is None else (experts,)
    w = rng.integers(0, 256, (*lead, n, k), dtype=np.uint8)
    if finite:
        w[w == 127] = 126
        w[w == 255] = 254
    scales = rng.uniform(0.01, 0.7, (*lead, (n + 127) // 128, (k + 127) // 128)).astype('f4')
    return w, scales


def run_fp8(simt, w, scales, x, ids, experts, block=64):
    m, k = x.shape
    n = w.shape[-2]
    out = np.full((m, n), -7.0, 'f4')
    dummy = np.zeros(1, 'i4') if ids is None else ids
    grid, block = launch_dims(m * n, block)
    simt.run('glm_vx_gpu_fp8_matmul', out, w, scales, x, dummy, m, n, k, experts, grid=grid, block=block)
    return out


def fp8_scale(w, x):
    return float(np.nansum(np.abs(FP8_E4M3FN_TABLE[w])) * 0.7 * np.abs(x).max())


@pytest.mark.parametrize('m', [1, 4])
@pytest.mark.parametrize('n,k', [(1, 1), (3, 31), (4, 128), (5, 129), (127, 127), (129, 129), (130, 257)])
@pytest.mark.parametrize('block', [32, 128])
def test_fp8_batched(simt, m, n, k, block):
    w, scales = fp8_data(n, k, 1)
    x = np.random.default_rng(9).uniform(-1, 1, (m, k)).astype('f4')
    out = run_fp8(simt, w, scales, x, None, 0, block)
    expected = np.stack([fp8_matvec_blocked(w, scales, row) for row in x])
    close(out, expected, fp8_scale(w, x))


def test_fp8_all_256_codes_including_nan_and_signed_zero(simt):
    w = np.arange(256, dtype=np.uint8).reshape(256, 1)
    out = run_fp8(simt, w, np.ones((2, 1), 'f4'), np.ones((1, 1), 'f4'), None, 0)[0]
    expected = FP8_E4M3FN_TABLE[np.arange(256)].astype('f4')
    assert np.array_equal(np.isnan(out), np.isnan(expected)) and np.isnan(expected).sum() == 2
    ok = ~np.isnan(expected)
    assert np.array_equal(out[ok], expected[ok])
    assert out[0x00] == 0 and out[0x80] == 0  # signed zeros contribute nothing
    assert out[0x01] == 2.0**-9 and out[0x81] == -(2.0**-9) and out[0x07] == 7 * 2.0**-9  # subnormals
    assert out[0x7E] == 448 and out[0xFE] == -448


def test_fp8_nan_stays_in_its_rows_and_columns_with_zero_x_still_propagate(simt):
    w, scales = fp8_data(3, 130, 5)
    w[1, 129] = 0x7F
    x = np.random.default_rng(1).uniform(-1, 1, (2, 130)).astype('f4')
    out = run_fp8(simt, w, scales, x, None, 0)
    assert np.isnan(out[:, 1]).all() and np.isfinite(out[:, [0, 2]]).all()
    x[:, 129] = 0  # NaN * 0 is NaN, as in the dense oracle
    assert np.isnan(run_fp8(simt, w, scales, x, None, 0)[:, 1]).all()


def test_fp8_grouped_experts_and_unrouted(simt):
    e, n, k, m = 3, 130, 257, 6
    w, scales = fp8_data(n, k, 7, experts=e)
    x = np.random.default_rng(2).uniform(-1, 1, (m, k)).astype('f4')
    ids = np.array([1, 1, 0, 2, -5, 3], dtype='i4')
    out = run_fp8(simt, w, scales, x, ids, e)
    for row, i in enumerate(ids):
        if 0 <= i < e:
            close(out[row], fp8_matvec_blocked(w[i], scales[i], x[row]), fp8_scale(w[i], x[row:row + 1]))
        else:
            assert np.array_equal(out[row], np.zeros(n, 'f4'))


def test_degenerate_dimensions_write_nothing(simt):
    packed, scales = int4_data(2, 1, 1)
    out = np.full((1, 2), -7.0, 'f4')
    # K=0 writes zeros (empty sum); M=0 or N=0 must not write.
    simt.run('glm_vx_gpu_int4_matmul', out, packed, scales.view(np.uint16), np.zeros(1, 'f4'), np.zeros(1, 'i4'), 0, 2, 1, 0, 0, grid=1, block=32)
    assert (out == -7.0).all()
    w, s = fp8_data(2, 1, 1)
    simt.run('glm_vx_gpu_fp8_matmul', out, w, s, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 1, 0, 1, 0, grid=1, block=32)
    assert (out == -7.0).all()
    simt.run('glm_vx_gpu_fp8_matmul', out, w, s, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 1, 2, 0, 0, grid=1, block=64)
    assert (out == 0).all()


# ---- BF16 weights (bf16.vx): raw uint16 bits, float32 activations/accumulation ----
def bf16_weights(shape, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    f = (rng.normal(size=shape) * scale).astype('f4')
    return (f.view(np.uint32) >> 16).astype(np.uint16)  # truncation is fine: any bf16 bits will do


def bf16_close(actual, w, x):
    """float64 reference over exact bf16->f32 widening; fp32 warp-order error bounded by the
    sum of |terms|. NaN and inf must match exactly (same positions, same sign)."""
    with np.errstate(invalid='ignore'):
        wf = bf16_to_f32(w).astype('f8')
        expected = x.astype('f8') @ wf.T
        abssum = np.abs(x).astype('f8') @ np.abs(wf).T
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(np.isinf(actual), np.isinf(expected)) and (np.sign(actual[np.isinf(actual)]) == np.sign(expected[np.isinf(expected)])).all()
    fin = np.isfinite(expected)
    np.testing.assert_allclose(actual[fin], expected[fin], rtol=1e-5, atol=4e-6 * float(np.max(abssum[fin], initial=1e-30)))


def run_bf16(simt, w, x, ids, experts, block=64):
    m, k = x.shape
    n = w.shape[-2]
    out = np.full((m, n), -7.0, 'f4')
    dummy = np.zeros(1, 'i4') if ids is None else ids
    grid, block = launch_dims(m * n, block)
    simt.run('glm_vx_gpu_bf16_matmul', out, w, x, dummy, m, n, k, experts, grid=grid, block=block)
    return out


def test_bf16_weight_decode_all_65536(simt):
    f = simt.lib.glm_vx_bf16_decode
    f.argtypes, f.restype = [C.c_int32], C.c_float
    bits = np.arange(65536, dtype=np.uint16)
    expected = bf16_to_f32(bits)
    actual = np.array([f(int(b)) for b in bits], dtype='f4')
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    ok = ~np.isnan(expected)
    assert np.array_equal(actual[ok].view(np.uint32), expected[ok].view(np.uint32))  # -0, subnormals, inf exact


@pytest.mark.parametrize('n,k', [(1, 1), (3, 31), (5, 33), (4, 128), (7, 129), (9, 257), (130, 300)])
@pytest.mark.parametrize('block', [32, 128])
def test_bf16_matvec(simt, n, k, block):
    w = bf16_weights((n, k), 1)
    x = np.random.default_rng(5).normal(size=k).astype('f4')
    out = np.full(n, -7.0, 'f4')
    grid, block = launch_dims(n, block)
    simt.run('glm_vx_gpu_bf16_matvec', out, w, x, n, k, grid=grid, block=block)
    bf16_close(out[None], w, x[None])


@pytest.mark.parametrize('m', [1, 3, 5])
@pytest.mark.parametrize('n,k', [(3, 31), (5, 129), (130, 257)])
@pytest.mark.parametrize('block', [32, 128])
def test_bf16_batched(simt, m, n, k, block):
    w = bf16_weights((n, k), 2)
    x = np.random.default_rng(6).normal(size=(m, k)).astype('f4')
    bf16_close(run_bf16(simt, w, x, None, 0, block), w, x)


def test_bf16_matmul_matches_matvec_bitwise(simt):
    # Same lane order and reduction, so the M=1 matmul and the matvec must agree exactly.
    w = bf16_weights((37, 301), 3)
    x = np.random.default_rng(7).normal(size=(1, 301)).astype('f4')
    out = np.empty(37, 'f4')
    simt.run('glm_vx_gpu_bf16_matvec', out, w, x[0].copy(), 37, 301, grid=10, block=128)
    assert np.array_equal(run_bf16(simt, w, x, None, 0)[0].view(np.uint32), out.view(np.uint32))


def test_bf16_grouped_experts_and_unrouted(simt):
    e, n, k, m = 4, 6, 200, 7
    w = bf16_weights((e, n, k), 4)
    x = np.random.default_rng(8).normal(size=(m, k)).astype('f4')
    ids = np.array([2, 0, 3, -1, 1, 4, 2], dtype='i4')  # -1 and 4 (== E) are unrouted
    out = run_bf16(simt, w, x, ids, e)
    for row, i in enumerate(ids):
        if 0 <= i < e:
            bf16_close(out[row:row + 1], w[i], x[row:row + 1])
        else:
            assert np.array_equal(out[row], np.zeros(n, 'f4'))


def test_bf16_special_weights_extreme_subnormal_inf_nan(simt):
    k = 40
    cases = {'min_subnormal': 0x0001, 'neg_min_subnormal': 0x8001, 'max_subnormal': 0x007F, 'min_normal': 0x0080,
             'max_finite': 0x7F7F, 'neg_max_finite': 0xFF7F, 'pos_zero': 0x0000, 'neg_zero': 0x8000,
             'pos_inf': 0x7F80, 'neg_inf': 0xFF80, 'nan': 0x7FC0, 'neg_nan': 0xFFC1, 'nan_payload_low': 0x7F81}
    w = np.zeros((len(cases), k), np.uint16)
    for row, bits in enumerate(cases.values()):
        w[row, 5] = bits
        w[row, 37] = 0x3F80  # 1.0 so the row is not only the special value
    x = np.ones((2, k), 'f4')
    x[1, 5] = 0.0  # inf * 0 and nan * 0 are NaN; finite * 0 is 0
    out = run_bf16(simt, w, x, None, 0)
    bf16_close(out, w, x)
    names = list(cases)
    row = {n: i for i, n in enumerate(names)}
    assert out[0, row['pos_inf']] == np.inf and out[0, row['neg_inf']] == -np.inf
    assert np.isnan(out[1, row['pos_inf']]) and np.isnan(out[0, row['nan']]) and np.isnan(out[1, row['nan']])
    assert out[0, row['max_finite']] == bf16_to_f32(np.array([0x7F7F], np.uint16))[0]  # +1.0 is absorbed
    assert out[0, row['min_subnormal']] == 1.0 and out[1, row['max_finite']] == 1.0  # tiny/zeroed terms vs 1.0
    assert out[0, row['neg_zero']] == 1.0


def test_bf16_degenerate_dimensions(simt):
    w = bf16_weights((2, 1), 1)
    out = np.full((1, 2), -7.0, 'f4')
    simt.run('glm_vx_gpu_bf16_matmul', out, w, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 0, 2, 1, 0, grid=1, block=32)
    simt.run('glm_vx_gpu_bf16_matmul', out, w, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 1, 0, 1, 0, grid=1, block=32)
    assert (out == -7.0).all()  # M=0 and N=0 write nothing
    simt.run('glm_vx_gpu_bf16_matmul', out, w, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 1, 2, 0, 0, grid=1, block=64)
    assert (out == 0).all()  # K=0 is an empty sum
