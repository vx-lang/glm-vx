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


def _quantize(context, x, scale_rows=None):
    rows, cols = x.shape
    kernel = context.load_ptx(BUILD / 'fp8_quant.ptx').kernel('glm_vx_gpu_fp8_quant_groups')
    codes, scales = context.tensor((rows, cols), 'uint8'), context.tensor((cols // 128, scale_rows or rows))
    fp8_quant.launch(kernel, codes, scales, _up(context, x), rows, cols, scale_rows=scale_rows)
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


def test_one_tokens_scales_four_apart(context):
    # gpu/resident.py's cuBLASLt operand: one token, scales at stride 4 (scale_rows 4).
    x = np.random.default_rng(5).standard_normal((1, 6144)).astype('f4')
    codes, scales = _quantize(context, x, scale_rows=4)
    want_codes, want_scales = fp8_quant.quant_groups(x)
    got = np.frombuffer(scales.download(), 'u4').reshape(48, 4)[:, 0]
    np.testing.assert_array_equal(got, want_scales.view('u4').ravel())
    np.testing.assert_array_equal(np.frombuffer(codes.download(), 'u1'), want_codes.ravel())


def _weights(rng, n, k):
    codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    blocks = rng.uniform(0.25, 4.0, size=(-(-n // 128), k // 128)).astype('f4')
    values = fp8_quant.e4m3_values()[codes] * np.repeat(np.repeat(blocks, 128, 0), 128, 1)[:n, :k]
    return codes, blocks, values


@pytest.mark.parametrize('orientation', ['weights_a', 'weights_b'])
@pytest.mark.parametrize('out_dtype', ['float32', 'uint16'])
@pytest.mark.parametrize('m,n,k', [(1, 576, 6144), (1, 2048, 6144), (1, 6144, 2048), (2, 2048, 6144),
                                   (4, 2048, 6144), (1, 200, 384)])
def test_fp8_block_matmul(context, lt, orientation, out_dtype, m, n, k):
    from gpu.cublaslt import CublasLtUnavailable, Fp8BlockMatmul, pack_weight_scales
    rng = np.random.default_rng(n + k + m)
    x = (rng.standard_normal((m, k)) * rng.uniform(0.1, 10, size=(m, k // 128)).repeat(128, 1)).astype('f4')
    mp = -(-m // 4) * 4  # cuBLASLt's token count; the padding rows are zeros
    x_codes, x_scales = _quantize(context, np.concatenate([x, np.zeros((mp - m, k), 'f4')]))
    w_codes, blocks, w_values = _weights(rng, n, k)
    try:
        mm = Fp8BlockMatmul(lt, m, n, k, out_dtype=out_dtype, orientation=orientation)
    except CublasLtUnavailable as error:
        pytest.skip(str(error))
    assert mm.padded_m == mp
    out = context.tensor((mp * n,), out_dtype)
    mm(out, x_codes, x_scales, _up(context, w_codes), _up(context, pack_weight_scales(blocks)))
    context.synchronize()
    got = np.frombuffer(out.download(), 'f4' if out_dtype == 'float32' else 'u2')
    if out_dtype == 'uint16':
        got = (got.astype(np.uint32) << 16).view('f4')
    got = (got.reshape(mp, n) if orientation == 'weights_a' else got.reshape(n, mp).T)[:m]
    xq, xs = fp8_quant.quant_groups(x)
    want = fp8_quant.dequant_groups(xq, xs) @ w_values.T
    # Hopper's FP8 tensor cores accumulate below float32 within a 128-wide k block (~3e-4 measured on an
    # H100 NVL, cuBLASLt 13.10); bfloat16 output has 8 bits.
    tolerance = 1e-3 if out_dtype == 'float32' else 1e-2
    np.testing.assert_allclose(got, want, rtol=tolerance, atol=tolerance * float(np.abs(want).max()))
    mm.close()


def _bf16(rng, shape, scale=1.0):
    """Random values that are bfloat16: (their bits uint16, the values float64)."""
    bits = (((rng.standard_normal(shape) * scale).astype('f4').view(np.uint32) + 0x8000) >> 16).astype(np.uint16)
    return bits, (bits.astype(np.uint32) << 16).view('f4').astype(np.float64)


@pytest.mark.parametrize('m,n,k', [(1000, 256, 6144), (2048, 32, 6144), (1, 256, 6144), (3, 40, 136)])
def test_bf16_matmul(context, lt, m, n, k):
    # The router's gate (256 experts) and the indexer's weights_proj (32 heads) over prefill rows: bfloat16
    # products are exact in float32, so only the float32 sums' order separates cuBLASLt from float64.
    from gpu.cublaslt import Bf16Matmul, CublasLtUnavailable
    rng = np.random.default_rng(m + n + k)
    try:
        mm = Bf16Matmul(lt, m, n, k)
    except CublasLtUnavailable as error:
        pytest.skip(str(error))
    mp = mm.padded_m
    x_bits, x = _bf16(rng, (mp, k))
    w_bits, w = _bf16(rng, (n, k), 0.02)
    out = context.tensor((mp * n,), 'float32')
    mm(out, _up(context, x_bits), _up(context, w_bits))
    context.synchronize()
    got = np.frombuffer(out.download(), 'f4').reshape(mp, n)
    want = x @ w.T
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5 * float(np.abs(want).max()))
    mm.close()


def test_bf16_matmul_batched_over_heads(context, lt):
    # _prefill_mla_sparse's absorb: 8 heads, each head's 192 columns of q [rows, heads, 256] times W_UK^T [512, 192],
    # into q_latent [rows, heads, 512]; one strided-batch cuBLASLt call.
    from gpu.cublaslt import Bf16Matmul, CublasLtUnavailable
    rng = np.random.default_rng(8)
    m, heads, width, n, k = 64, 8, 256, 512, 192
    try:
        mm = Bf16Matmul(lt, m, n, k, batch=heads, x_ld=heads * width, x_stride=width, w_stride=n * k,
                        out_ld=heads * n, out_stride=n)
    except CublasLtUnavailable as error:
        pytest.skip(str(error))
    x_bits, x = _bf16(rng, (m, heads, width))
    w_bits, w = _bf16(rng, (heads, n, k), 0.05)
    out = context.tensor((m * heads * n,), 'float32')
    mm(out, _up(context, x_bits.reshape(-1)), _up(context, w_bits.reshape(-1)))
    context.synchronize()
    got = np.frombuffer(out.download(), 'f4').reshape(m, heads, n)
    want = np.einsum('mhk,hnk->mhn', x[:, :, :k], w)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5 * float(np.abs(want).max()))
    mm.close()
