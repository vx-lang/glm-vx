"""INT4 and batched/grouped FP8 kernels: the compiled Vx kernels under the CPU SIMT emulator
(NOT GPU validation) against the glm_vx/quant.py oracles. Rebuild with gpu/testing/build_simt.sh."""
import ctypes as C
import math

import numpy as np
import pytest

from glm_vx.checkpoint import FP8_E4M3FN_TABLE
from glm_vx.quant import fp8_matvec_blocked, int4_dequantize, int4_matvec, int4_quantize, int4_unpack
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


def run_int4(simt, packed, scales, x, ids, experts, block=64):
    m, k = x.shape
    n = packed.shape[-2]
    out = np.full((m, n), -7.0, 'f4')
    dummy = np.zeros(1, 'i4') if ids is None else ids
    grid, block = launch_dims(m * n, block)
    simt.run('glm_vx_gpu_int4_matmul', out, packed, scales.view(np.uint16), x, dummy, m, n, k, experts, grid=grid, block=block)
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
    simt.run('glm_vx_gpu_int4_matvec', out, packed, scales.view(np.uint16), x, n, k, grid=grid, block=block)
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
    simt.run('glm_vx_gpu_int4_matmul', out, packed, scales.view(np.uint16), np.zeros(1, 'f4'), np.zeros(1, 'i4'), 0, 2, 1, 0, grid=1, block=32)
    assert (out == -7.0).all()
    w, s = fp8_data(2, 1, 1)
    simt.run('glm_vx_gpu_fp8_matmul', out, w, s, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 1, 0, 1, 0, grid=1, block=32)
    assert (out == -7.0).all()
    simt.run('glm_vx_gpu_fp8_matmul', out, w, s, np.zeros(1, 'f4'), np.zeros(1, 'i4'), 1, 2, 0, 0, grid=1, block=64)
    assert (out == 0).all()
