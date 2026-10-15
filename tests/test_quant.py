import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.quant import (fp8_matvec_blocked, int4_dequantize, int4_matvec, int4_quantize, int4_unpack,
                          quantize_weights)
from glm_vx.tiny import tiny_config
from validation.write_checkpoint import fp8_dequantize_reference, fp8_quantize

SHAPES = [(8, 128), (5, 129), (3, 1), (17, 300), (1, 257), (64, 64)]


@pytest.mark.parametrize('shape', SHAPES)
def test_int4_pack_unpack_is_lossless(shape):
    rng = np.random.default_rng(shape[0] * 7 + shape[1])
    w = rng.normal(size=shape).astype(np.float32)
    packed, scales = int4_quantize(w)
    assert packed.shape == (shape[0], -(-shape[1] // 2)) and packed.dtype == np.uint8
    codes = int4_unpack(packed, shape[1])
    assert codes.min() >= -8 and codes.max() <= 7
    # Re-pack the unpacked codes by hand and compare bytes: pins the low-nibble-first layout.
    nib = (codes & 0xF).astype(np.uint8)
    if shape[1] % 2:
        nib = np.concatenate([nib, np.zeros((shape[0], 1), np.uint8)], axis=1)
    np.testing.assert_array_equal(nib[:, 0::2] | (nib[:, 1::2] << 4), packed)


def test_int4_nibble_layout_and_sign_extension():
    w = np.array([[-8.0, 7.0, 0.0, -1.0, 3.0]], np.float32)
    packed, scales = int4_quantize(w, group=128)
    # amax=8 -> scale 8/7: codes are round(w/scale)
    codes = int4_unpack(packed, 5)[0]
    np.testing.assert_array_equal(codes, np.clip(np.rint(w[0] / np.float32(scales[0, 0])), -8, 7))
    assert packed[0, 0] & 0xF == codes[0] & 0xF and packed[0, 0] >> 4 == codes[1] & 0xF  # element 0 low nibble
    assert packed[0, 2] >> 4 == 0                                                       # odd tail is zero-padded


@pytest.mark.parametrize('shape', SHAPES)
def test_int4_error_is_within_half_a_step(shape):
    rng = np.random.default_rng(shape[1])
    w = (rng.normal(size=shape) * rng.uniform(.1, 4, size=(shape[0], 1))).astype(np.float32)
    packed, scales = int4_quantize(w)
    back = int4_dequantize(packed, scales, shape[1])
    step = np.repeat(scales.astype(np.float32), 128, axis=1)[:, :shape[1]]
    assert np.all(np.abs(back - w) <= step * 0.5 * 1.001 + 1e-6)


def test_int4_zero_rows_and_saturation():
    w = np.zeros((2, 130), np.float32)
    w[1, :] = 1e4
    packed, scales = int4_quantize(w)
    back = int4_dequantize(packed, scales, 130)
    assert (back[0] == 0).all()
    np.testing.assert_allclose(back[1], 1e4, rtol=2e-3)  # fp16 scale rounding only


@pytest.mark.parametrize('shape', SHAPES)
def test_int4_matvec_matches_dequantize_then_multiply(shape):
    rng = np.random.default_rng(3 + shape[0])
    w = rng.normal(size=shape).astype(np.float32)
    x = rng.normal(size=shape[1]).astype(np.float32)
    packed, scales = int4_quantize(w)
    want = int4_dequantize(packed, scales, shape[1]).astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(int4_matvec(packed, scales, x), want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('shape', [(8, 128), (130, 129), (1, 300), (257, 5), (256, 256)])
def test_fp8_blocked_matvec_matches_dequantize_then_multiply(shape):
    rng = np.random.default_rng(shape[0])
    w = rng.normal(size=shape).astype(np.float32)
    x = rng.normal(size=shape[1]).astype(np.float32)
    codes, scales = fp8_quantize(w)
    want = fp8_dequantize_reference(codes, scales).astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(fp8_matvec_blocked(codes, scales, x), want, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        fp8_matvec_blocked(codes, scales, x[:-1])


def test_quantization_noise_on_a_model():
    """Weight-level error is the principled bound; logit error of a tiny random model is not.

    A 4-layer random model with top-2-of-4 routing and top-4 key selection amplifies weight
    noise through discrete decisions (measured 0.2-0.45 relative logit error for FP8 across
    seeds, 0.35-0.7 for INT4), so the logit check only asserts the ordering and sanity.
    """
    config = tiny_config()
    for seed in range(3):
        rng = np.random.default_rng(seed)
        weights = tiny_weights(config, seed=seed)
        for name, w in weights.items():
            weights[name] = (rng.normal(0, 1 / np.sqrt(w.shape[-1]), w.shape).astype(np.float32)
                             if 'norm' not in name else rng.uniform(.7, 1.3, w.shape).astype(np.float32))
        tokens = [3, 17, 5, 9, 200, 1, 44, 8]
        exact = GlmMoeDsaModel(config, weights, NumpyBackend()).prefill(tokens)
        error = {}
        for fmt, weight_bound in (('fp8', 0.035), ('int4', 0.20)):
            q = quantize_weights(weights, fmt)
            changed = [n for n in weights if not np.array_equal(q[n], weights[n])]
            assert changed and all(weights[n].ndim == 2 for n in changed)
            worst = max(np.linalg.norm(q[n] - weights[n]) / np.linalg.norm(weights[n]) for n in changed)
            assert worst < weight_bound, (fmt, worst)          # E4M3 ~2.9% RMS, int4 group-128 well under 20%
            out = GlmMoeDsaModel(config, q, NumpyBackend()).prefill(tokens)
            assert np.all(np.isfinite(out))
            error[fmt] = float(np.linalg.norm(out - exact) / np.linalg.norm(exact))
        assert error['fp8'] < error['int4']
