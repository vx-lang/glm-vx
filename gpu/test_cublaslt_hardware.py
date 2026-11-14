"""cuBLASLt's block-scaled FP8 matmul (gpu/cublaslt.py) and the activation quantizer that feeds it
(gpu/fp8_quant.vx) on a real GPU (#29). Unavailable hardware or library is a skip, not validation success.

The quantizer must give gpu/fp8_quant.py's codes and scales bit for bit. The matmul is checked against
float64 sums of the same dequantized operands, with a distinct scale in every weight block and every
activation group, so a wrong scale layout or orientation fails. Each orientation and output type that
cuBLASLt refuses (no algorithm) is a skip with the shapes in its reason: that is the finding to read.
"""
from pathlib import Path

import numpy as np
import pytest

from gpu import fp8_quant
from gpu.runtime import CUDAContext, CUDAUnavailable

BUILD = Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def context():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 9))
    except CUDAUnavailable as error:
        pytest.skip('real FP8-capable GPU (sm_89+) unavailable: ' + str(error))
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture(scope='module')
def lt(context):
    from gpu.cublaslt import CublasLt, CublasLtUnavailable
    try:
        handle = CublasLt(context)
    except CublasLtUnavailable as error:
        pytest.skip(str(error))
    yield handle
    handle.close()


def _up(ctx, a):
    return ctx.tensor(a.shape, str(a.dtype), a.tobytes())


def _quantize(context, x):
    rows, cols = x.shape
    kernel = context.load_ptx(BUILD / 'fp8_quant.ptx').kernel('glm_vx_gpu_fp8_quant_groups')
    codes, scales = context.tensor((rows, cols), 'uint8'), context.tensor((cols // 128, rows))
    fp8_quant.launch(kernel, codes, scales, _up(context, x), rows, cols)
    context.synchronize()
    return codes, scales


@pytest.mark.parametrize('rows,cols', [(1, 6144), (4, 2048), (3, 128)])
def test_quant_groups_matches_the_model_bitwise(context, rows, cols):
    rng = np.random.default_rng(rows * cols)
    x = (rng.standard_normal((rows, cols)) * rng.choice([1e-6, 1e-2, 1, 30], size=(rows, cols))).astype('f4')
    codes, scales = _quantize(context, x)
    want_codes, want_scales = fp8_quant.quant_groups(x)
    np.testing.assert_array_equal(np.frombuffer(scales.download(), 'u4'), want_scales.view('u4').ravel())
    np.testing.assert_array_equal(np.frombuffer(codes.download(), 'u1'), want_codes.ravel())


def _weights(rng, n, k):
    codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    blocks = rng.uniform(0.25, 4.0, size=(-(-n // 128), k // 128)).astype('f4')
    values = fp8_quant.e4m3_values()[codes] * np.repeat(np.repeat(blocks, 128, 0), 128, 1)[:n, :k]
    return codes, blocks, values


@pytest.mark.parametrize('orientation', ['weights_a', 'weights_b'])
@pytest.mark.parametrize('out_dtype', ['float32', 'uint16'])
@pytest.mark.parametrize('m,n,k', [(1, 576, 6144), (1, 2048, 6144), (1, 6144, 2048), (4, 2048, 6144),
                                   (1, 200, 384)])
def test_fp8_block_matmul(context, lt, orientation, out_dtype, m, n, k):
    from gpu.cublaslt import CublasLtUnavailable, Fp8BlockMatmul, pack_weight_scales
    rng = np.random.default_rng(n + k + m)
    x = (rng.standard_normal((m, k)) * rng.uniform(0.1, 10, size=(m, k // 128)).repeat(128, 1)).astype('f4')
    x_codes, x_scales = _quantize(context, x)
    w_codes, blocks, w_values = _weights(rng, n, k)
    try:
        mm = Fp8BlockMatmul(lt, m, n, k, out_dtype=out_dtype, orientation=orientation)
    except CublasLtUnavailable as error:
        pytest.skip(str(error))
    out = context.tensor((m, n), out_dtype)
    mm(out, x_codes, x_scales, _up(context, w_codes), _up(context, pack_weight_scales(blocks)))
    context.synchronize()
    got = np.frombuffer(out.download(), 'f4' if out_dtype == 'float32' else 'u2').reshape(m, n)
    if out_dtype == 'uint16':
        got = (got.astype(np.uint32) << 16).view('f4')
    xq, xs = fp8_quant.quant_groups(x)
    want = fp8_quant.dequant_groups(xq, xs) @ w_values.T
    tolerance = 1e-5 if out_dtype == 'float32' else 1e-2  # float32 sums; bfloat16 has 8 bits
    np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance * float(np.abs(want).max()))
    mm.close()
