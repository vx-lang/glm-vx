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
