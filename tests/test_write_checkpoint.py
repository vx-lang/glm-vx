"""Production loader end to end on the real checkpoint formats (BF16, block-scaled FP8, sharded)."""
import json

import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.checkpoint import Checkpoint
from glm_vx.config import GLMConfig
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.tiny import tiny_config
from validation.write_checkpoint import (export_checkpoint, from_bf16_bits, fp8_dequantize_reference,
                                         fp8_quantize, to_bf16_bits)

TOKENS = [3, 17, 5, 5, 9, 200, 1, 44, 8]


@pytest.fixture(scope='module')
def model_inputs():
    config = tiny_config()
    rng = np.random.default_rng(11)
    weights = tiny_weights(config, seed=5)
    for name, w in weights.items():  # make values non-degenerate (tiny_weights norms are all ones)
        weights[name] = (rng.normal(0, 1 / np.sqrt(w.shape[-1]), w.shape).astype(np.float32)
                         if 'norm' not in name else rng.uniform(.7, 1.3, w.shape).astype(np.float32))
    return config, weights


def test_bf16_rounding_is_nearest_even_and_keeps_specials():
    x = np.array([1.0, 1.00390625, 1.005859375, -2.5, 0.0, -0.0, np.inf, -np.inf, np.nan, 3.38e38], np.float32)
    back = from_bf16_bits(to_bf16_bits(x))
    np.testing.assert_array_equal(back[:2], [1.0, 1.0])       # tie rounds to even
    assert back[2] == np.float32(1.0078125)                    # above the tie rounds up
    assert np.isnan(back[8]) and np.isposinf(back[6]) and np.isneginf(back[7])
    assert np.signbit(back[5])
    finite = np.isfinite(x)
    assert np.all(np.abs(back[finite] - x[finite]) <= np.abs(x[finite]) * 2.0 ** -8)


@pytest.mark.parametrize('shape', [(16, 24), (130, 129), (1, 300), (257, 5)])
def test_fp8_quantization_error_is_bounded_per_block(shape):
    rng = np.random.default_rng(shape[0])
    w = rng.normal(size=shape).astype(np.float32) * rng.uniform(.01, 5, size=(shape[0], 1)).astype(np.float32)
    codes, scales = fp8_quantize(w)
    assert scales.shape == (-(-shape[0] // 128), -(-shape[1] // 128))
    back = fp8_dequantize_reference(codes, scales)
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            tile = (slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
            amax = np.abs(w[tile]).max()
            assert np.abs(back[tile] - w[tile]).max() <= 0.04 * amax + 1e-12  # half the widest E4M3 step is 3.6% of amax
    assert np.all(codes != 0x7F) and np.all(codes != 0xFF)  # never emits the NaN codes


def test_fp8_zero_block_and_saturation():
    w = np.zeros((4, 4), np.float32)
    codes, scales = fp8_quantize(w)
    assert (scales == 1).all() and (fp8_dequantize_reference(codes, scales) == 0).all()
    w = np.full((2, 2), 1e30, np.float32)
    back = fp8_dequantize_reference(*fp8_quantize(w))
    np.testing.assert_allclose(back, w, rtol=1e-6)


@pytest.mark.parametrize('fmt,shards', [('f32', 1), ('bf16', 1), ('bf16', 4), ('fp8', 1), ('fp8', 3)])
def test_loader_reproduces_written_tensors_exactly(tmp_path, model_inputs, fmt, shards):
    config, weights = model_inputs
    expected = export_checkpoint(config, weights, tmp_path, fmt, shards)
    validated = GLMConfig.from_file(tmp_path)
    with Checkpoint(tmp_path, block_size=validated.weight_block_size) as ck:
        ck.validate_manifest(validated.tensor_shapes(), inspect_shards=True)
        for name, want in expected.items():
            np.testing.assert_array_equal(ck.tensor(name), want, err_msg=name)
    if shards > 1:
        index = json.loads((tmp_path / 'model.safetensors.index.json').read_text())
        assert len(set(index['weight_map'].values())) == shards


def test_fp8_checkpoint_has_the_official_layout(tmp_path, model_inputs):
    config, weights = model_inputs
    export_checkpoint(config, weights, tmp_path, 'fp8')
    with Checkpoint(tmp_path, block_size=(128, 128)) as ck:
        kinds = {n: ck.info(n).dtype for n in ck.names}
    assert kinds['model.layers.0.self_attn.q_a_proj.weight'] == 'F8_E4M3'
    assert kinds['model.layers.0.self_attn.q_a_proj.weight_scale_inv'] == 'F32'
    for kept in ('model.embed_tokens.weight', 'lm_head.weight', 'model.layers.1.mlp.gate.weight',
                 'model.layers.0.input_layernorm.weight', 'model.layers.1.mlp.gate.e_score_correction_bias'):
        assert kinds[kept] == 'BF16', kept


@pytest.mark.parametrize('fmt,shards', [('bf16', 2), ('fp8', 2)])
def test_model_on_loaded_checkpoint_is_bit_identical_to_reference_weights(tmp_path, model_inputs, fmt, shards):
    config, weights = model_inputs
    expected = export_checkpoint(config, weights, tmp_path, fmt, shards)
    for w in expected.values():
        w.flags.writeable = False
    validated = GLMConfig.from_file(tmp_path)
    with Checkpoint(tmp_path, block_size=validated.weight_block_size) as ck:
        from_disk = GlmMoeDsaModel(config, ck, NumpyBackend())
        want = GlmMoeDsaModel(config, expected, NumpyBackend())
        got = from_disk.prefill(TOKENS)
        np.testing.assert_array_equal(got, want.prefill(TOKENS))
    assert np.all(np.isfinite(got))
