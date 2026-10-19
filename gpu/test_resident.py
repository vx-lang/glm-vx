"""ResidentGLM under the CPU SIMT emulator against the NumPy reference model.

Test-only lanes run the same compiled Vx kernels; this checks the orchestration (buffers, views,
caches, selection reuse, expert order) and that nothing is silently recomputed on the host. It is
not GPU validation: validation/resident_compare.py does that on hardware."""
import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.tiny import tiny_config
from gpu.resident import ResidentGLM
from gpu.testing.context import Context


def build(seed=3, **overrides):
    config = tiny_config()
    config['index_n_heads'] = 16  # few indexer heads give exact score ties that argsort orders arbitrarily
    config.update(overrides)
    weights = tiny_weights(config, seed=seed)
    for w in weights.values():
        w.flags.writeable = False
    return GlmMoeDsaModel(config, weights, NumpyBackend())


TOKENS = [3, 17, 5, 5, 9, 200, 1, 44, 8, 2, 31, 77]  # longer than index_topk=4: sparse selection


def reference(model, tokens):
    cache = model.new_cache()
    return np.stack([model.forward(t, cache) for t in tokens])


def test_logits_match_reference_over_a_sparse_sequence():
    model = build()
    expected = reference(model, TOKENS)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        got = resident.prefill(TOKENS)
    finally:
        resident.close()
    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-5)
    assert (got.argmax(-1) == expected.argmax(-1)).all()


def test_reset_replays_identically_and_cache_rows_are_overwritten():
    model = build(seed=5)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32)
    first = resident.prefill(TOKENS[:8])
    resident.reset()
    np.testing.assert_array_equal(resident.prefill(TOKENS[:8]), first)


def test_prefill_without_logits_matches_the_full_path():
    model = build(seed=7)
    a = ResidentGLM(model, ctx=Context(), max_tokens=32)
    b = ResidentGLM(model, ctx=Context(), max_tokens=32)
    for t in TOKENS[:-1]:
        a.forward(t, logits=False)
        b.forward(t)
    np.testing.assert_array_equal(a.forward(TOKENS[-1]), b.forward(TOKENS[-1]))


def test_fused_expert_layout_matches_per_expert_layout():
    model = build(seed=9)
    expected = reference(model, TOKENS[:6])
    weights = dict(model.weights)
    prefixes = sorted({n.rsplit('.experts.', 1)[0] for n in weights if '.experts.' in n and n.endswith('gate_proj.weight')})
    for prefix in prefixes:
        n_experts = int(model.config['n_routed_experts'])
        gate = np.stack([weights.pop(f'{prefix}.experts.{e}.gate_proj.weight') for e in range(n_experts)])
        up = np.stack([weights.pop(f'{prefix}.experts.{e}.up_proj.weight') for e in range(n_experts)])
        down = np.stack([weights.pop(f'{prefix}.experts.{e}.down_proj.weight') for e in range(n_experts)])
        weights[prefix + '.experts.gate_up_proj'] = np.concatenate([gate, up], axis=1)
        weights[prefix + '.experts.down_proj'] = down
    fused = GlmMoeDsaModel(model.config, weights, NumpyBackend())
    resident = ResidentGLM(fused, ctx=Context(), max_tokens=32)
    np.testing.assert_allclose(resident.prefill(TOKENS[:6]), expected, rtol=2e-5, atol=2e-5)


def test_limits_and_unsupported_modes_are_explicit():
    model = build()
    resident = ResidentGLM(model, ctx=Context(), max_tokens=3)
    resident.prefill(TOKENS[:3])
    with pytest.raises(ValueError, match='max_tokens'):
        resident.forward(1)
    with pytest.raises(ValueError, match='vocabulary'):
        ResidentGLM(model, ctx=Context(), max_tokens=3).forward(10 ** 6)
    with pytest.raises(MemoryError):
        ResidentGLM(model, ctx=Context(), max_tokens=4, max_weight_bytes=1024).forward(1)


# Linears the published GLM W4A16 checkpoint quantizes; the indexer, router, norms, embeddings
# and lm_head stay in the model dtype there. 'all' additionally quantizes the indexer projections.
GLM_INT4 = ('.q_a_proj.', '.q_b_proj.', '.kv_a_proj_with_mqa.', '.kv_b_proj.', '.o_proj.',
            '.gate_proj.', '.up_proj.', '.down_proj.')
INDEXER_INT4 = ('.indexer.wq_b.', '.indexer.wk.')


def int4_model(seed, patterns, **overrides):
    from glm_vx.quant import CompressedInt4Weights, compressed_int4_encode
    base = build(seed=seed, **overrides)
    linears = {}
    for name, w in base.weights.items():
        if w.ndim == 2 and name.endswith('.weight') and any(p in name for p in patterns):
            words, bits = compressed_int4_encode(w)
            linears[name[:-len('.weight')]] = (words, bits, w.shape)
    rest = {n: w for n, w in base.weights.items() if n[:-len('.weight')] not in linears}
    return GlmMoeDsaModel(base.config, CompressedInt4Weights(rest, linears), NumpyBackend()), linears


@pytest.mark.parametrize('patterns,overrides', [
    (GLM_INT4, {}),
    (GLM_INT4 + INDEXER_INT4, {}),
    (GLM_INT4, {'hidden_size': 136}),  # K > 128: two groups, the second partial
])
def test_int4_linears_match_the_dequantized_reference(patterns, overrides):
    model, linears = int4_model(11, patterns, **overrides)
    expected = reference(model, TOKENS)  # NumPy on the exact float32 dequantization
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        got = resident.prefill(TOKENS)
        launches = dict(resident.launches)
        uploaded = set(resident._weights)
    finally:
        resident.close()
    # Only the summation order differs from the reference (per-group code sums, then the scale).
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert (got.argmax(-1) == expected.argmax(-1)).all()
    assert launches.get('int4_matvec', 0) > 0
    # Every quantized Linear except kv_b_proj stayed packed: no float32 copy was uploaded.
    for prefix in linears:
        if prefix.endswith('.kv_b_proj'):
            assert prefix + '.weight' in uploaded and prefix + ':int4' not in uploaded
        elif prefix + ':int4' in uploaded:
            assert prefix + '.weight' not in uploaded


def test_int4_resident_matches_float32_resident_on_the_same_weights():
    # The same exact dequantized weights through the float32 kernels: isolates the INT4 kernel.
    model, _ = int4_model(13, GLM_INT4)
    dense = GlmMoeDsaModel(model.config, model._weight, NumpyBackend())
    a = ResidentGLM(model, ctx=Context(), max_tokens=32)
    b = ResidentGLM(dense, ctx=Context(), max_tokens=32)
    np.testing.assert_allclose(a.prefill(TOKENS[:6]), b.prefill(TOKENS[:6]), rtol=1e-4, atol=1e-4)
    assert 'int4_matvec' in a.launches and 'int4_matvec' not in b.launches


def test_int4_rejects_a_quantization_config_the_kernels_do_not_implement():
    model, _ = int4_model(3, GLM_INT4)
    config = dict(model.config, quantization_config={'quant_method': 'awq'})
    with pytest.raises(ValueError, match='compressed-tensors'):
        ResidentGLM(GlmMoeDsaModel(config, model.weights, NumpyBackend()), ctx=Context(), max_tokens=4)


# The official GLM FP8 checkpoint quantizes every Linear except embeddings, lm_head, the router,
# norms and the indexer weights_proj (validation.write_checkpoint.UNQUANTIZED).
def fp8_model(seed, **overrides):
    from glm_vx.quant import BlockFP8Weights
    from validation.write_checkpoint import UNQUANTIZED, fp8_quantize
    base = build(seed=seed, **overrides)
    linears = {}
    for name, w in base.weights.items():
        if w.ndim == 2 and name.endswith('.weight') and not any(u in name for u in UNQUANTIZED):
            linears[name[:-len('.weight')]] = fp8_quantize(w)
    rest = {n: w for n, w in base.weights.items() if n[:-len('.weight')] not in linears}
    return GlmMoeDsaModel(base.config, BlockFP8Weights(rest, linears), NumpyBackend()), linears


@pytest.mark.parametrize('overrides', [
    {},
    {'hidden_size': 136},  # K > 128: two scale columns, the second block partial
    {'hidden_size': 260, 'moe_intermediate_size': 136},  # partial row and column blocks in the MLPs
])
def test_fp8_linears_match_the_dequantized_reference(overrides):
    model, linears = fp8_model(17, **overrides)
    expected = reference(model, TOKENS)  # NumPy on the exact float32 dequantization
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        got = resident.prefill(TOKENS)
        launches = dict(resident.launches)
        uploaded = set(resident._weights)
    finally:
        resident.close()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert (got.argmax(-1) == expected.argmax(-1)).all()
    assert launches.get('fp8_matvec', 0) > 0 and 'int4_matvec' not in launches
    assert any(p.endswith('.indexer.wq_b') for p in linears) and any(p + ':fp8' in uploaded for p in linears if '.indexer.' in p)
    for prefix in linears:
        if prefix.endswith('.kv_b_proj'):
            assert prefix + '.weight' in uploaded and prefix + ':fp8' not in uploaded
        elif prefix + ':fp8' in uploaded:
            assert prefix + '.weight' not in uploaded


def test_fp8_resident_matches_float32_resident_on_the_same_weights():
    # The same exact dequantized weights through the float32 kernels: isolates the FP8 kernel.
    model, _ = fp8_model(19)
    dense = GlmMoeDsaModel(model.config, model._weight, NumpyBackend())
    a = ResidentGLM(model, ctx=Context(), max_tokens=32)
    b = ResidentGLM(dense, ctx=Context(), max_tokens=32)
    np.testing.assert_allclose(a.prefill(TOKENS[:6]), b.prefill(TOKENS[:6]), rtol=1e-4, atol=1e-4)
    assert 'fp8_matvec' in a.launches and 'fp8_matvec' not in b.launches
    assert a.weight_bytes < b.weight_bytes


def test_fp8_checkpoint_on_disk_runs_packed(tmp_path):
    # The production path: an official-layout FP8 safetensors directory read by SafetensorsCheckpoint.
    import json
    from glm_vx.checkpoint import SafetensorsCheckpoint
    from validation.write_checkpoint import export_checkpoint
    base = build(seed=23)
    export_checkpoint(base.config, base.weights, tmp_path, 'fp8', shards=2)
    config = json.loads((tmp_path / 'config.json').read_text())
    with SafetensorsCheckpoint(tmp_path) as ckpt:
        model = GlmMoeDsaModel(config, ckpt, NumpyBackend())
        expected = reference(model, TOKENS[:8])
        resident = ResidentGLM(model, ctx=Context(), max_tokens=16)
        try:
            got = resident.prefill(TOKENS[:8])
            launches = dict(resident.launches)
        finally:
            resident.close()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert (got.argmax(-1) == expected.argmax(-1)).all()
    assert launches.get('fp8_matvec', 0) > 0


@pytest.mark.parametrize('change,match', [
    ({'weight_block_size': [64, 64]}, '128x128'),
    ({'activation_scheme': 'static'}, 'dynamic'),
    ({'fmt': 'e5m2'}, 'e4m3'),
])
def test_fp8_rejects_a_quantization_config_the_kernels_do_not_implement(change, match):
    model, _ = fp8_model(3)
    q = dict({'quant_method': 'fp8', 'fmt': 'e4m3', 'activation_scheme': 'dynamic', 'weight_block_size': [128, 128]}, **change)
    config = dict(model.config, quantization_config=q)
    with pytest.raises(ValueError, match=match):
        ResidentGLM(GlmMoeDsaModel(config, model.weights, NumpyBackend()), ctx=Context(), max_tokens=4)
