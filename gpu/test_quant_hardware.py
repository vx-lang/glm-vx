"""Real GPU gate for the INT4 and batched/grouped FP8 kernels. Skipped only when no SM80 device is
present; PTX/JIT/launch/arithmetic problems are failures. Oracles are glm_vx/quant.py."""
from pathlib import Path

import numpy as np
import pytest

from glm_vx.checkpoint import FP8_E4M3FN_TABLE
from glm_vx.quant import fp8_matvec_blocked, int4_dequantize, int4_matvec
from gpu.quant_ops import QuantKernels
from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.test_quant_simt import fp8_data, fp8_scale, int4_data

BUILD = Path(__file__).parent / 'build'


@pytest.fixture
def device():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        yield ctx, QuantKernels.from_ptx(ctx, BUILD / 'int4.ptx', BUILD / 'fp8_batched.ptx')
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
