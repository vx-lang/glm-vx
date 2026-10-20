"""Real GPU gate for the INT4 and batched/grouped FP8 kernels. Skipped only when no SM80 device is
present; PTX/JIT/launch/arithmetic problems are failures. Oracles are glm_vx/quant.py."""
from pathlib import Path

import numpy as np
import pytest

from glm_vx.checkpoint import FP8_E4M3FN_TABLE
from glm_vx.quant import fp8_matvec_blocked, int4_dequantize, int4_matvec
from gpu.quant_ops import QuantKernels
from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.test_quant_simt import (REAL, bf16_close, bf16_weights, compressed_data, compressed_reference, fp8_data,
                                 fp8_scale, int4_data)

BUILD = Path(__file__).parent / 'build'


@pytest.fixture
def device():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        yield ctx, QuantKernels.from_ptx(ctx, BUILD / 'int4.ptx', BUILD / 'fp8_batched.ptx', BUILD / 'bf16.ptx')
    finally:
        ctx.close()


def download(tensor, shape):
    return np.frombuffer(tensor.download(), 'f4').reshape(shape)


def close(actual, expected, scale):
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5 * max(float(np.max(scale)), 1.0))


@pytest.mark.parametrize('n,k', [(1, 1), (3, 31), (5, 129), (130, 257), (256, 1024)])
def test_int4_matvec_device(device, n, k):
    ctx, q = device
    packed, scales = int4_data(n, k, 1)
    x = np.random.default_rng(5).normal(size=k).astype('f4')
    out = q.int4_matvec(ctx.tensor(packed.shape, 'uint8', packed), ctx.tensor(scales.shape, 'float16', scales), ctx.tensor((k,), data=x))
    ctx.synchronize()
    close(download(out, n), int4_matvec(packed, scales, x), np.abs(int4_dequantize(packed, scales, k)).astype('f8') @ np.abs(x))


@pytest.mark.parametrize('m', [1, 5, 33])
@pytest.mark.parametrize('n,k', [(3, 31), (130, 257)])
def test_int4_matmul_device(device, m, n, k):
    ctx, q = device
    packed, scales = int4_data(n, k, 2)
    x = np.random.default_rng(6).normal(size=(m, k)).astype('f4')
    out = q.int4_matmul(ctx.tensor(packed.shape, 'uint8', packed), ctx.tensor(scales.shape, 'float16', scales), ctx.tensor(x.shape, data=x))
    ctx.synchronize()
    expected = np.stack([int4_matvec(packed, scales, r) for r in x])
    close(download(out, (m, n)), expected, np.abs(int4_dequantize(packed, scales, k)).sum() * np.abs(x).max())


@pytest.mark.parametrize('extreme', [False, True])
def test_int4_bf16_compressed_tensors_device(device, extreme):
    ctx, q = device
    e, n, k, m = 4, 130, 1024, 9
    codes, words, bits, packed = compressed_data(n, k, 21, experts=e, extreme=extreme)
    x = np.random.default_rng(22).normal(size=(m, k)).astype('f4')
    ids = np.array([3, 0, -1, 1, 2, 4, 2, 0, 1], dtype='i4')
    out = q.int4_matmul(ctx.tensor(packed.shape, 'uint8', packed), ctx.tensor(bits.shape, 'uint16', bits), ctx.tensor(x.shape, data=x),
                        expert_ids=ctx.tensor(ids.shape, 'int32', ids), scale_format='bfloat16')
    vec = q.int4_matvec(ctx.tensor(packed[1].shape, 'uint8', packed[1]), ctx.tensor(bits[1].shape, 'uint16', bits[1]),
                        ctx.tensor((k,), data=x[0]), scale_format='bfloat16')
    ctx.synchronize()
    got = download(out, (m, n))
    for row, i in enumerate(ids):
        if 0 <= i < e:
            close(got[row], *compressed_reference(words[i], bits[i], x[row], k))
        else:
            assert (got[row] == 0).all()
    close(download(vec, n), *compressed_reference(words[1], bits[1], x[0], k))


@pytest.mark.skipif(not REAL, reason='set GLM_VX_INT4_CHECKPOINT to a validation.fetch_tensors directory')
def test_int4_real_checkpoint_linears_device(device):
    from glm_vx.checkpoint import SafetensorsCheckpoint
    ctx, q = device
    with SafetensorsCheckpoint(REAL) as ckpt:
        for prefix in sorted(n[:-len('.weight_packed')] for n in ckpt.names if n.endswith('.weight_packed')):
            packed, bits, scale_format, (n, k) = ckpt.compressed_int4(prefix)
            words = ckpt.raw(prefix + '.weight_packed')
            x = np.random.default_rng(k).normal(size=(3, k)).astype('f4')
            out = q.int4_matmul(ctx.tensor(packed.shape, 'uint8', packed), ctx.tensor(bits.shape, 'uint16', bits),
                                ctx.tensor(x.shape, data=x), scale_format=scale_format)
            ctx.synchronize()
            got = download(out, (3, n))
            for r in range(3):
                close(got[r], *compressed_reference(words, bits, x[r], k))


def test_int4_grouped_device(device):
    ctx, q = device
    e, n, k, m = 8, 40, 300, 11
    packed, scales = int4_data(n, k, 4, experts=e)
    x = np.random.default_rng(8).normal(size=(m, k)).astype('f4')
    ids = np.array([7, 0, 3, -1, 1, 8, 2, 2, 5, 6, 4], dtype='i4')
    out = q.int4_matmul(ctx.tensor(packed.shape, 'uint8', packed), ctx.tensor(scales.shape, 'float16', scales), ctx.tensor(x.shape, data=x),
                        expert_ids=ctx.tensor(ids.shape, 'int32', ids))
    ctx.synchronize()
    got = download(out, (m, n))
    for row, i in enumerate(ids):
        if 0 <= i < e:
            close(got[row], int4_matvec(packed[i], scales[i], x[row]), np.abs(int4_dequantize(packed[i], scales[i], k)).sum() * np.abs(x[row]).max())
        else:
            assert (got[row] == 0).all()


@pytest.mark.parametrize('m', [1, 5, 33])
@pytest.mark.parametrize('n,k', [(3, 31), (129, 129), (130, 257)])
def test_fp8_matmul_device(device, m, n, k):
    ctx, q = device
    w, scales = fp8_data(n, k, 1)
    x = np.random.default_rng(9).uniform(-1, 1, (m, k)).astype('f4')
    out = q.fp8_matmul(ctx.tensor(w.shape, 'uint8', w), ctx.tensor(scales.shape, data=scales), ctx.tensor(x.shape, data=x))
    ctx.synchronize()
    close(download(out, (m, n)), np.stack([fp8_matvec_blocked(w, scales, r) for r in x]), fp8_scale(w, x))


def test_fp8_all_codes_device(device):
    ctx, q = device
    w = np.arange(256, dtype=np.uint8).reshape(256, 1)
    scales = np.ones((2, 1), 'f4')
    out = download(q.fp8_matmul(ctx.tensor(w.shape, 'uint8', w), ctx.tensor(scales.shape, data=scales), ctx.tensor((1, 1), data=np.ones((1, 1), 'f4'))), (256,))
    expected = FP8_E4M3FN_TABLE[np.arange(256)].astype('f4')
    assert np.array_equal(np.isnan(out), np.isnan(expected))
    assert np.array_equal(out[~np.isnan(expected)], expected[~np.isnan(expected)])


def test_fp8_grouped_device(device):
    ctx, q = device
    e, n, k, m = 5, 130, 257, 9
    w, scales = fp8_data(n, k, 7, experts=e)
    x = np.random.default_rng(2).uniform(-1, 1, (m, k)).astype('f4')
    ids = np.array([4, 1, 1, 0, 2, -5, 3, 5, 4], dtype='i4')
    out = q.fp8_matmul(ctx.tensor(w.shape, 'uint8', w), ctx.tensor(scales.shape, data=scales), ctx.tensor(x.shape, data=x),
                       expert_ids=ctx.tensor(ids.shape, 'int32', ids))
    ctx.synchronize()
    got = download(out, (m, n))
    for row, i in enumerate(ids):
        if 0 <= i < e:
            close(got[row], fp8_matvec_blocked(w[i], scales[i], x[row]), fp8_scale(w[i], x[row:row + 1]))
        else:
            assert (got[row] == 0).all()


def test_wrapper_rejects_bad_shapes(device):
    ctx, q = device
    packed, scales = int4_data(4, 130, 1)
    pt, st = ctx.tensor(packed.shape, 'uint8', packed), ctx.tensor(scales.shape, 'float16', scales)
    with pytest.raises(ValueError):
        q.int4_matmul(pt, st, ctx.tensor((2, 200), data=np.zeros((2, 200), 'f4')))
    with pytest.raises(ValueError):
        q.int4_matmul(pt, st, ctx.tensor((2, 130), data=np.zeros((2, 130), 'f4')), expert_ids=ctx.tensor((2,), 'int32', np.zeros(2, 'i4')))


@pytest.mark.parametrize('n,k', [(1, 1), (5, 33), (130, 300), (256, 1024)])
def test_bf16_matvec_device(device, n, k):
    ctx, q = device
    w = bf16_weights((n, k), 1)
    x = np.random.default_rng(5).normal(size=k).astype('f4')
    out = q.bf16_matvec(ctx.tensor(w.shape, 'uint16', w), ctx.tensor((k,), data=x))
    ctx.synchronize()
    bf16_close(download(out, (1, n)), w, x[None])


@pytest.mark.parametrize('m', [1, 5, 33])
@pytest.mark.parametrize('n,k', [(3, 31), (130, 257)])
def test_bf16_matmul_device(device, m, n, k):
    ctx, q = device
    w = bf16_weights((n, k), 2)
    x = np.random.default_rng(6).normal(size=(m, k)).astype('f4')
    out = q.bf16_matmul(ctx.tensor(w.shape, 'uint16', w), ctx.tensor(x.shape, data=x))
    ctx.synchronize()
    bf16_close(download(out, (m, n)), w, x)


def test_bf16_grouped_and_special_device(device):
    ctx, q = device
    e, n, k, m = 5, 9, 70, 8
    w = bf16_weights((e, n, k), 4)
    w[1, 2, 3] = 0x7F80  # +inf
    w[2, 0, 4] = 0x7FC0  # NaN
    w[3, 1, 5] = 0x0001  # smallest subnormal
    x = np.random.default_rng(8).uniform(0.5, 1.5, (m, k)).astype('f4')
    ids = np.array([1, 2, 3, 0, -1, 5, 4, 1], dtype='i4')
    out = q.bf16_matmul(ctx.tensor(w.shape, 'uint16', w), ctx.tensor(x.shape, data=x), expert_ids=ctx.tensor(ids.shape, 'int32', ids))
    ctx.synchronize()
    got = download(out, (m, n))
    for row, i in enumerate(ids):
        if 0 <= i < e:
            bf16_close(got[row:row + 1], w[i], x[row:row + 1])
        else:
            assert (got[row] == 0).all()
    assert got[0, 2] == np.inf and np.isnan(got[1, 0])


def test_bf16_wrapper_rejects_bad_shapes(device):
    ctx, q = device
    w = bf16_weights((4, 10), 1)
    wt = ctx.tensor(w.shape, 'uint16', w)
    with pytest.raises(ValueError):
        q.bf16_matmul(wt, ctx.tensor((2, 11), data=np.zeros((2, 11), 'f4')))
    with pytest.raises(ValueError):
        q.bf16_matvec(wt, ctx.tensor((11,), data=np.zeros(11, 'f4')))
    with pytest.raises((ValueError, TypeError)):
        q.bf16_matmul(ctx.tensor(w.shape, 'float32', w.astype('f4')), ctx.tensor((2, 10), data=np.zeros((2, 10), 'f4')))
