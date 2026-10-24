"""ResidentGLM under the CPU SIMT emulator against the NumPy reference model.

Test-only lanes run the same compiled Vx kernels; this checks the orchestration (buffers, views,
caches, selection reuse, expert order) and that nothing is silently recomputed on the host. It is
not GPU validation: validation/resident_compare.py does that on hardware."""
from pathlib import Path

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


def fused_layout(model):
    """The same model with its routed experts in the fused checkpoint layout."""
    weights = dict(model.weights)
    prefixes = sorted({n.rsplit('.experts.', 1)[0] for n in weights if '.experts.' in n and n.endswith('gate_proj.weight')})
    for prefix in prefixes:
        n_experts = int(model.config['n_routed_experts'])
        gate = np.stack([weights.pop(f'{prefix}.experts.{e}.gate_proj.weight') for e in range(n_experts)])
        up = np.stack([weights.pop(f'{prefix}.experts.{e}.up_proj.weight') for e in range(n_experts)])
        down = np.stack([weights.pop(f'{prefix}.experts.{e}.down_proj.weight') for e in range(n_experts)])
        weights[prefix + '.experts.gate_up_proj'] = np.concatenate([gate, up], axis=1)
        weights[prefix + '.experts.down_proj'] = down
    return GlmMoeDsaModel(model.config, weights, NumpyBackend())


def test_fused_expert_layout_matches_per_expert_layout():
    model = build(seed=9)
    expected = reference(model, TOKENS[:6])
    resident = ResidentGLM(fused_layout(model), ctx=Context(), max_tokens=32)
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
    assert launches.get('bf16_matvec', 0) > 0  # the unconverted BF16 router, weights_proj and lm_head


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


def bf16_model(seed, **overrides):
    """Every 2D Linear weight rounded to bfloat16 and stored as bits; norms and embeddings float32."""
    from glm_vx.quant import BF16Weights
    from validation.write_checkpoint import to_bf16_bits
    base = build(seed=seed, **overrides)
    linears = {name[:-len('.weight')]: to_bf16_bits(w) for name, w in base.weights.items()
               if w.ndim == 2 and name.endswith('.weight') and 'embed_tokens' not in name}
    rest = {n: w for n, w in base.weights.items() if n[:-len('.weight')] not in linears}
    return GlmMoeDsaModel(base.config, BF16Weights(rest, linears), NumpyBackend()), linears


@pytest.mark.parametrize('overrides', [{}, {'hidden_size': 136}])
def test_bf16_resident_equals_float32_resident_on_the_same_values(overrides):
    # bf16_matvec decodes exactly and sums in matvec's order: bit-identical logits, half the bytes.
    model, linears = bf16_model(29, **overrides)
    dense = GlmMoeDsaModel(model.config, model._weight, NumpyBackend())
    a = ResidentGLM(model, ctx=Context(), max_tokens=32)
    b = ResidentGLM(dense, ctx=Context(), max_tokens=32)
    got, want = a.prefill(TOKENS[:6]), b.prefill(TOKENS[:6])
    np.testing.assert_array_equal(got, want)
    np.testing.assert_allclose(got, reference(model, TOKENS[:6]), rtol=2e-5, atol=2e-5)
    assert a.launches.get('bf16_matvec', 0) > 0 and 'bf16_matvec' not in b.launches
    assert a.launches.get('matvec', 0) == 0  # every projection, router and lm_head included, ran in BF16
    assert {'lm_head:bf16', 'model.layers.1.mlp.gate:bf16'} <= set(a._weights)
    kv_b = [p for p in linears if p.endswith('.kv_b_proj')]
    assert kv_b and all(p + '.weight' in a._weights and p + ':bf16' not in a._weights for p in kv_b)
    assert a.weight_bytes < b.weight_bytes


def test_bf16_checkpoint_on_disk_runs_as_bf16(tmp_path):
    import json
    from glm_vx.checkpoint import SafetensorsCheckpoint
    from validation.write_checkpoint import export_checkpoint
    base = build(seed=31)
    expected_weights = export_checkpoint(base.config, base.weights, tmp_path, 'bf16', shards=2)
    config = json.loads((tmp_path / 'config.json').read_text())
    want = ResidentGLM(GlmMoeDsaModel(config, expected_weights, NumpyBackend()), ctx=Context(), max_tokens=16)
    with SafetensorsCheckpoint(tmp_path) as ckpt:
        resident = ResidentGLM(GlmMoeDsaModel(config, ckpt, NumpyBackend()), ctx=Context(), max_tokens=16)
        np.testing.assert_array_equal(resident.prefill(TOKENS[:8]), want.prefill(TOKENS[:8]))
        assert resident.launches.get('bf16_matvec', 0) > 0 and resident.launches.get('matvec', 0) == 0


@pytest.mark.parametrize('overrides', [{}, {'index_n_heads': 1}])  # one head: exact score ties
def test_device_topk_selects_exactly_what_the_host_argsort_selects(overrides):
    model = build(seed=37, **overrides)
    device = ResidentGLM(model, ctx=Context(), max_tokens=32)
    host = ResidentGLM(model, ctx=Context(), max_tokens=32, device_topk=False)
    for t in TOKENS:
        a, b = device.forward(t), host.forward(t)
        np.testing.assert_array_equal(device.selected.data[:device.selected_count],
                                      host.selected.data[:host.selected_count])
        np.testing.assert_array_equal(a, b)
    assert device.launches['topk_stable'] > 0 and 'topk_stable' not in host.launches


# --- Routed experts chosen on the device (device_experts=True) -----------------------------
# The *_expert_matvec spawn kernels run on Vx's host path here (gpu/spawn/host.py), the rest on
# the SIMT emulator; the host never reads the router's ids.

SPAWN = Path(__file__).parent / 'build' / 'spawn'
EXPERT_KERNELS = ('expert_matvec', 'bf16_expert_matvec', 'int4_expert_matvec', 'fp8_expert_matvec', 'expert_scaled_add')
needs_spawn = pytest.mark.skipif(not all((SPAWN / f'lib{k}.so').exists() for k in EXPERT_KERNELS),
                                 reason='gpu/spawn expert kernels not built (gpu/spawn/build.sh with VX_DISPATCH_LIB)')


def expert_models(kind):
    if kind == 'float32':
        return build(seed=41)
    if kind == 'fused':
        return fused_layout(build(seed=43))
    if kind == 'int4':
        return int4_model(47, GLM_INT4)[0]
    if kind == 'fp8':
        return fp8_model(53, hidden_size=136)[0]
    return bf16_model(59)[0]


def refuse_id_reads(resident):
    def refuse():
        raise AssertionError('the router ids were read on the host')
    resident.route_ids.download = refuse


@needs_spawn
@pytest.mark.parametrize('kind', ['float32', 'fused', 'int4', 'fp8', 'bf16'])
def test_device_experts_equal_the_host_loop(kind):
    model = expert_models(kind)
    device = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True)
    host = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        refuse_id_reads(device)
        got, want = device.prefill(TOKENS), host.prefill(TOKENS)
    finally:
        device.close()
        host.close()
    np.testing.assert_array_equal(got, want)
    assert device.launches['expert_scaled_add'] > 0 and 'scaled_add' not in device.launches
    matvec = {'float32': 'expert_matvec', 'fused': 'expert_matvec'}.get(kind, kind + '_expert_matvec')
    assert device.launches[matvec] > 0


@needs_spawn
def test_device_experts_preload_uploads_each_projection_once_stacked():
    model, _ = int4_model(61, GLM_INT4)
    device = ResidentGLM(model, ctx=Context(), max_tokens=8, device_experts=True)
    host = ResidentGLM(model, ctx=Context(), max_tokens=8)
    try:
        device.preload()
        host.preload()
        assert device.weight_bytes == host.weight_bytes  # the same bytes, laid out back to back
        assert not any('.experts.0.' in key for key in device._weights)
        stacked = [key for key in device._weights if key.endswith(':stacked')]
        sparse = sum(t == 'sparse' for t in model.mlp_types)
        assert len(stacked) == 3 * sparse
        refuse_id_reads(device)
        np.testing.assert_array_equal(device.prefill(TOKENS[:4]), host.prefill(TOKENS[:4]))
    finally:
        device.close()
        host.close()
