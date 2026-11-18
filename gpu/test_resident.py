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


def test_matvec_rows_gives_the_hand_matvecs_logits(monkeypatch):
    # The tiny model's float32 Linears all have few rows: matvec_rows (#29) runs every one of them,
    # and the logits are the hand matvec's to the bit.
    import gpu.resident as resident_module
    model = build(seed=11)
    a = ResidentGLM(model, ctx=Context(), max_tokens=32)
    got = a.prefill(TOKENS[:6])
    monkeypatch.setattr(resident_module, 'MATVEC_ROWS_MAX', 0)
    b = ResidentGLM(model, ctx=Context(), max_tokens=32)
    want = b.prefill(TOKENS[:6])
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
    assert a.launches.get('matvec_rows', 0) > 0 and a.launches.get('matvec', 0) == 0
    assert b.launches.get('matvec', 0) > 0 and 'matvec_rows' not in b.launches


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


@pytest.mark.parametrize('graph', [False, True])
def test_fp8_x4_kernels_match_the_hand_kernels(graph):
    # fp8_kernels='x4' (#30, #31): fp8_matvec4 for the Linears and fp8_moe4_* for the routed experts,
    # another summation order, so close to the default path rather than equal; the same argmax.
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    runs = {}
    for kernels in ('hand', 'x4'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, graph=graph, fp8_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:8]]), dict(resident.launches))
        finally:
            resident.close()
    (hand, hand_launches), (x4, x4_launches) = runs['hand'], runs['x4']
    np.testing.assert_allclose(x4, hand, rtol=1e-4, atol=1e-4)
    assert (x4.argmax(-1) == hand.argmax(-1)).all()
    assert x4_launches.get('fp8_matvec4', 0) > 0 and x4_launches.get('fp8_moe4_gate_up', 0) > 0
    assert 'fp8_matvec4' not in hand_launches and 'fp8_moe4_down' not in hand_launches
    with pytest.raises(ValueError, match='fp8_kernels'):
        ResidentGLM(model, ctx=Context(), fp8_kernels='fast')


def test_fp8_fma_kernels_match_the_hand_kernels():
    # fp8_kernels='fma' (#30, #31): one fma a code for the dense Linears and the routed experts, so
    # close to the default path rather than equal; the same argmax.
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    runs = {}
    for kernels in ('hand', 'fma'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, fp8_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:8]]), dict(resident.launches))
        finally:
            resident.close()
    (hand, _), (fma, launches) = runs['hand'], runs['fma']
    np.testing.assert_allclose(fma, hand, rtol=1e-4, atol=1e-4)
    assert (fma.argmax(-1) == hand.argmax(-1)).all()
    # hidden_size 144 is below FMA_LONG_ROWS, so the dense Linears take fp8_matvec_fma_tiles.
    assert launches.get('fp8_matvec_fma_tiles', 0) > 0 and launches.get('fp8_moe_fma_gate_up', 0) > 0
    assert launches.get('fp8_moe_fma_down', 0) > 0
    assert 'fp8_matvec4' not in launches and 'fp8_moe4_gate_up' not in launches
    # Dense and shared-expert gate/up run as one fp8_moe_fma_gate_up with the activation in it, and
    # q_a, kv_a and the indexer's wk as one fp8_matvec_fma_tiles3.
    assert 'swiglu' not in launches
    assert launches.get('fp8_matvec_fma_tiles3', 0) > 0


def test_fp8_cvt_kernels_match_the_hand_kernels():
    # fp8_kernels='cvt' (glm-vx #38): the codes at their true values. The test context has no compute
    # capability, so the _ref builds run (the sm_90 builds give their bits on Hopper).
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    runs = {}
    for kernels in ('hand', 'cvt'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, fp8_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:8]]), dict(resident.launches))
        finally:
            resident.close()
    (hand, _), (cvt, launches) = runs['hand'], runs['cvt']
    np.testing.assert_allclose(cvt, hand, rtol=1e-4, atol=1e-4)
    assert (cvt.argmax(-1) == hand.argmax(-1)).all()
    for name in ('fp8_matvec_cvt_tiles_ref', 'fp8_matvec_cvt_tiles3_ref', 'fp8_moe_cvt_gate_up_ref', 'fp8_moe_cvt_down_ref'):
        assert launches.get(name, 0) > 0, name
    assert not any('fma' in k or 'matvec4' in k or k == 'swiglu' for k in launches), launches


def test_fp8_cvt_absorbs_read_kv_b_proj_codes():
    # fp8_kernels='cvt' with GLM-5.3's MLA widths, nope 192 and value 256 (#42): kv_b_proj's 128-row blocks
    # straddle heads. The absorbs run fp8_matvec_cols_cvt and fp8_matvec_heads_cvt on its FP8 codes as
    # stored, no float32 copy of it is uploaded, and the logits stay near the hand kernels'.
    model, _ = fp8_model(37, hidden_size=144, moe_intermediate_size=144, kv_lora_rank=128, qk_nope_head_dim=192,
                         v_head_dim=256, qk_head_dim=196)
    runs = {}
    for kernels in ('hand', 'cvt'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, fp8_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:6]]), dict(resident.launches),
                             set(resident._weights))
        finally:
            resident.close()
    (hand, _, _), (cvt, launches, weights) = runs['hand'], runs['cvt']
    np.testing.assert_allclose(cvt, hand, rtol=1e-4, atol=1e-4)
    assert (cvt.argmax(-1) == hand.argmax(-1)).all()
    assert launches.get('fp8_matvec_cols_cvt_ref', 0) == launches.get('fp8_matvec_heads_cvt_ref', 0) == model.n_layers * 6
    assert 'matvec_heads' not in launches
    assert not any(k.endswith('kv_b_proj.weight') or k.endswith(':nope_t') for k in weights), weights


def test_fold_shared_gives_the_same_bits_in_fewer_launches():
    # fold_shared: the shared expert as the last of k + 1 experts in fp8_moe_cvt_gate_up and _down (id
    # n_routed_experts, weight 1.0), so its gate_up, down and the add are not launches of their own.
    # The routed kernels' rows are fp8_moe_cvt_gate_up's (k = 1) and fp8_matvec_cvt_tiles' and ascending
    # ids put it last, where the add put it: the same logits, bit for bit, one token at a time and under
    # graph replay. Its Linears are stacked, never uploaded alone.
    model, _ = fp8_model(41, hidden_size=272, moe_intermediate_size=144)
    runs = {}
    for fold in (False, True):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, fp8_kernels='cvt', fold_shared=fold)
        try:
            one = np.stack([resident.forward(t) for t in TOKENS[:6]])
            launches = dict(resident.launches)
            many = resident.forward_many(TOKENS[6:9])[0]
            runs[fold] = (one, launches, set(resident._weights), many)
        finally:
            resident.close()
    (plain, before, _, plain_many), (folded, after, weights, folded_many) = runs[False], runs[True]
    np.testing.assert_array_equal(folded.view(np.uint32), plain.view(np.uint32))
    # forward_many with fp8_kernels='cvt' is not forward()'s bits even unfolded (its shared expert runs
    # gate, up and swiglu apart; test_resident_many checks 'hand' and 'x4' only), so near, not equal.
    np.testing.assert_allclose(folded_many, plain_many, rtol=1e-5, atol=1e-5)
    moe_layers = sum(model.mlp_type(i) != 'dense' for i in range(model.n_layers))
    assert moe_layers > 0
    tokens = 6
    assert after['fp8_moe_cvt_gate_up_ref'] == before['fp8_moe_cvt_gate_up_ref'] - moe_layers * tokens
    assert sum(after.values()) == sum(before.values()) - 3 * moe_layers * tokens
    assert not any('.shared_experts.' in k for k in weights), weights
    assert any(k.endswith(':stacked:shared') for k in weights)
    # Under graph replay: the ninth route slot is written once, before capture, and replays keep it.
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, fp8_kernels='cvt', fold_shared=True,
                           graph=True)
    try:
        replayed = np.stack([resident.forward(t) for t in TOKENS[:6]])
        assert resident.replays > 0
    finally:
        resident.close()
    np.testing.assert_array_equal(replayed.view(np.uint32), plain.view(np.uint32))
    with pytest.raises(ValueError, match='fold_shared'):
        ResidentGLM(model, ctx=Context(), fold_shared=True)


def test_moe_down2_gives_fp8_moe_cvt_down_bits():
    # moe_down2: the routed experts' down projection through fp8_moe_cvt_down2 (two rows and two experts a
    # subgroup, for tensor parallelism's short expert rows): each expert's dot and the ascending-order sum
    # are fp8_moe_cvt_down's, so the logits are the same bits, alone, with the folded shared expert (an odd
    # k + 1) and under graph replay; every MoE layer launches it in place of fp8_moe_cvt_down.
    model, _ = fp8_model(43, hidden_size=272, moe_intermediate_size=144)
    moe_layers = sum(model.mlp_type(i) != 'dense' for i in range(model.n_layers))
    assert moe_layers > 0
    for options in (dict(), dict(fold_shared=True), dict(fold_shared=True, graph=True)):
        runs = {}
        for down2 in (False, True):
            resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, fp8_kernels='cvt',
                                   moe_down2=down2, **options)
            try:
                runs[down2] = (np.stack([resident.forward(t) for t in TOKENS[:6]]), dict(resident.launches))
            finally:
                resident.close()
        (old, before), (new, after) = runs[False], runs[True]
        np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32))
        if not options.get('graph'):
            assert after.get('fp8_moe_cvt_down2_ref') == before.get('fp8_moe_cvt_down_ref') == moe_layers * 6, (before, after)
            assert 'fp8_moe_cvt_down_ref' not in after
    with pytest.raises(ValueError, match='moe_down2'):
        ResidentGLM(model, ctx=Context(), moe_down2=True)


def test_moe_mma_needs_cvt_and_falls_back_without_fp8_tensor_cores():
    # moe_mma's kernel is inline PTX for sm_89+ (gpu/test_moe_mma_hardware.py). The emulator has no compute
    # capability, so the cvt kernels are the _ref builds and the down projection stays down2's.
    model, _ = fp8_model(43, hidden_size=272, moe_intermediate_size=256)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=8, device_experts=True, fp8_kernels='cvt',
                           moe_down2=True, moe_mma=True)
    try:
        resident.forward(TOKENS[0])
        assert resident.launches.get('fp8_moe_cvt_down2_ref') and 'moe_mma/fp8_moe_mma_down_at' not in resident.launches
    finally:
        resident.close()
    with pytest.raises(ValueError, match='moe_mma'):
        ResidentGLM(model, ctx=Context(), moe_mma=True)


def fp8_bf16_router_model(seed, **overrides):
    """fp8_model with each MoE router's gate in bfloat16, as the official FP8 checkpoint stores it."""
    from glm_vx.quant import BF16Weights, BlockFP8Weights
    from validation.write_checkpoint import UNQUANTIZED, fp8_quantize, to_bf16_bits
    base = build(seed=seed, **overrides)
    linears, routers = {}, {}
    for name, w in base.weights.items():
        if w.ndim == 2 and name.endswith('.mlp.gate.weight'):
            routers[name[:-len('.weight')]] = to_bf16_bits(w)
        elif w.ndim == 2 and name.endswith('.weight') and not any(u in name for u in UNQUANTIZED):
            linears[name[:-len('.weight')]] = fp8_quantize(w)
    rest = {n: w for n, w in base.weights.items() if n[:-len('.weight')] not in linears and n[:-len('.weight')] not in routers}
    return GlmMoeDsaModel(base.config, BlockFP8Weights(BF16Weights(rest, routers), linears), NumpyBackend())


def test_fuse_router_gives_the_same_bits_in_one_launch():
    # fuse_router: router_bf16 sums the BF16 gate's rows in the hand bf16_matvec's order, then picks the
    # router's experts in the last block to finish: the same ids and weights in one launch instead of two
    # (on the emulator, where both sigmoids are libm's), so the same logits bit for bit, one token at a
    # time, with fold_shared (the ninth slot is not the router's), and under graph replay (the arrival
    # count goes back to zero each launch). Under matvec_kernels='x4' the unfused projection is
    # bf16_matvec4, another order: near, not equal.
    model = fp8_bf16_router_model(43, hidden_size=272, moe_intermediate_size=144)
    options = dict(max_tokens=32, device_experts=True, fp8_kernels='cvt')
    runs = {}
    for fuse, fold, graph, kernels in ((False, False, False, 'hand'), (True, False, False, 'hand'),
                                       (True, True, False, 'hand'), (True, True, True, 'hand'),
                                       (False, False, False, 'x4'), (True, False, False, 'x4')):
        resident = ResidentGLM(model, ctx=Context(), fuse_router=fuse, fold_shared=fold, graph=graph,
                               matvec_kernels=kernels, **options)
        try:
            runs[fuse, fold, graph, kernels] = (np.stack([resident.forward(t) for t in TOKENS[:6]]),
                                                dict(resident.launches), resident.replays if graph else None)
        finally:
            resident.close()
    plain, before, _ = runs[False, False, False, 'hand']
    moe_layers = sum(model.mlp_type(i) != 'dense' for i in range(model.n_layers))
    assert moe_layers > 0
    for key in ((True, False, False, 'hand'), (True, True, False, 'hand'), (True, True, True, 'hand')):
        np.testing.assert_array_equal(runs[key][0].view(np.uint32), plain.view(np.uint32), err_msg=str(key))
    np.testing.assert_allclose(runs[True, False, False, 'x4'][0], runs[False, False, False, 'x4'][0], rtol=1e-5, atol=1e-5)
    _, after, _ = runs[True, False, False, 'hand']
    tokens = 6
    assert before['router'] == moe_layers * tokens and 'router' not in after
    assert after['router_bf16'] == moe_layers * tokens
    assert sum(after.values()) == sum(before.values()) - moe_layers * tokens
    assert runs[True, True, True, 'hand'][2] > 0


def test_norm_block_1024_is_close_to_256_and_other_sizes_are_refused():
    # norm_block (#29): RMSNorm with 1024 threads reads a row in one batch of loads; each thread sums
    # other squares, so the logits are near the 256-thread ones (which the hybrid backend matches
    # bit for bit), not equal.
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    runs = {}
    for block in (256, 1024):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, norm_block=block)
        try:
            runs[block] = np.stack([resident.forward(t) for t in TOKENS[:4]])
        finally:
            resident.close()
    np.testing.assert_allclose(runs[1024], runs[256], rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match='norm_block'):
        ResidentGLM(model, ctx=Context(), norm_block=300)


def test_fp8_cublaslt_projection_is_its_quantized_sum():
    # fp8_kernels='cublaslt' (#29): x to e4m3 per 128 columns (gpu/fp8_quant.vx, run here), then the
    # block-scaled FP8 matmul (here the test context's float64 stand-in for cuBLASLt). The output is
    # the float64 sum of the quantized operands, rounded once.
    from gpu import fp8_quant
    model, linears = fp8_model(23, hidden_size=128)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32, fp8_kernels='cublaslt')
    try:
        prefix = next(p for p in linears if p.endswith('mlp.gate_proj'))
        codes, scales, block, (rows, cols) = model.weights.block_fp8(prefix)
        assert cols == 128
        x = np.random.default_rng(5).normal(size=cols).astype('f4')
        out = resident._project(prefix, resident._tensor((cols,), data=x.tobytes()), resident._tensor((rows,)))
        got = np.frombuffer(out.download(), 'f4')
        xq, xs = fp8_quant.quant_groups(x[None])
        w = fp8_quant.e4m3_values()[codes] * np.repeat(np.repeat(scales, 128, 0), 128, 1)[:rows, :cols]
        np.testing.assert_array_equal(got, (fp8_quant.dequant_groups(xq, xs) @ w.T).astype('f4').ravel())
        assert resident.launches == {'fp8_quant_groups': 1, 'cublaslt_fp8': 1}
    finally:
        resident.close()


@pytest.mark.parametrize('graph', [False, True])
def test_fp8_cublaslt_logits_are_near_the_float32_activation_kernels(graph):
    # Quantized activations move the logits by about the e4m3 step; the hidden-width Linears take
    # cuBLASLt, the narrower ones the hand kernel, the routed experts the x4 kernels.
    model, _ = fp8_model(29, hidden_size=128, moe_intermediate_size=128)
    runs = {}
    for kernels in ('hand', 'cublaslt'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, graph=graph, fp8_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:6]]), dict(resident.launches))
        finally:
            resident.close()
    (hand, _), (lt, lt_launches) = runs['hand'], runs['cublaslt']
    scale = float(np.abs(hand).max())
    np.testing.assert_allclose(lt, hand, rtol=0.1, atol=0.05 * scale)
    assert np.abs(lt - hand).max() > 0  # the activations were quantized
    assert lt_launches.get('cublaslt_fp8', 0) > 0 and lt_launches['fp8_quant_groups'] == lt_launches['cublaslt_fp8']
    assert lt_launches.get('fp8_moe4_gate_up', 0) > 0


@pytest.mark.parametrize('fmt', ['int4', 'bf16'])
def test_spawn_matvec_kernels_equal_the_hand_kernels(fmt):
    # matvec_kernels='spawn': gpu/spawn's int4_matvec and bf16_matvec for the Linears, the hand
    # kernels' arithmetic in the same order, so the logits are equal bit for bit.
    model = (int4_model(37, GLM_INT4) if fmt == 'int4' else bf16_model(37))[0]
    runs = {}
    for kernels in ('hand', 'spawn'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, matvec_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:6]]), set(resident._spawn_kernels))
        finally:
            resident.close()
    (hand, hand_spawned), (spawn, spawned) = runs['hand'], runs['spawn']
    assert hand.tobytes() == spawn.tobytes()
    assert f'{fmt}_matvec' in spawned and f'{fmt}_matvec' not in hand_spawned
    with pytest.raises(ValueError, match='matvec_kernels'):
        ResidentGLM(model, ctx=Context(), matvec_kernels='fast')


@pytest.mark.parametrize('fmt', ['int4', 'bf16'])
def test_x4_matvec_kernels_match_the_hand_kernels(fmt):
    # matvec_kernels='x4' (#30): int4_matvec4 and bf16_matvec4, four u32 words a lane step, another
    # summation order, so close to the hand kernels rather than equal; the same argmax. Hidden and
    # MLP widths of 256 give INT4 Linears whose width is a multiple of 128 (two groups); narrower
    # ones keep the hand kernel.
    sizes = dict(hidden_size=256, intermediate_size=256)
    model = (int4_model(41, GLM_INT4, **sizes) if fmt == 'int4' else bf16_model(41, **sizes))[0]
    runs = {}
    for kernels in ('hand', 'x4'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, matvec_kernels=kernels)
        try:
            runs[kernels] = (np.stack([resident.forward(t) for t in TOKENS[:6]]), set(resident._spawn_kernels))
        finally:
            resident.close()
    (hand, hand_spawned), (x4, spawned) = runs['hand'], runs['x4']
    np.testing.assert_allclose(x4, hand, rtol=1e-4, atol=1e-4)
    assert (x4.argmax(-1) == hand.argmax(-1)).all()
    assert f'{fmt}_matvec4' in spawned and f'{fmt}_matvec4' not in hand_spawned


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


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('short', [2048, 5])  # 5: positions 6 on take the radix select (long_at under replay)
def test_topk_sort_selects_what_the_radix_select_selects(monkeypatch, graph, short):
    # gpu/spawn's topk_sort (#32) returns the same ids as topk_stable, so the logits are equal bit
    # for bit; past TOPK_SORT_TOKENS the radix select takes over (both captured under graph replay).
    # Both are long_topk='merge''s, the score order.
    import gpu.resident as resident_module
    monkeypatch.setattr(resident_module, 'TOPK_SORT_TOKENS', short)
    model = build(seed=41, index_n_heads=1)  # one indexer head: exact score ties
    runs = {}
    for topk in ('sort', 'radix'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=graph, graph=graph, topk=topk,
                               long_topk='merge')
        try:
            runs[topk] = (np.stack([resident.forward(t) for t in TOKENS]), dict(resident.launches))
        finally:
            resident.close()
    np.testing.assert_array_equal(runs['sort'][0], runs['radix'][0])
    sort, radix = runs['sort'][1], runs['radix'][1]
    assert sort.get('topk_sort_at' if graph else 'topk_sort', 0) > 0
    assert 'topk_sort' not in radix and 'topk_sort_at' not in radix
    if graph:
        assert sort.get('topk_stable_long_at', 0) > 0 and 'topk_stable_at' not in sort
    else:
        assert (sort.get('topk_stable', 0) > 0) == (short < len(TOKENS))


@pytest.mark.parametrize('index_topk', [4, 2048])
def test_long_contexts_select_through_chunk_sorts_and_merges(index_topk):
    # Past TOPK_SORT_TOKENS, topk='sort' selects through topk_chunks and topk_merge rounds (#32): the
    # radix select's ids and the host argsort's. Scores are written straight into the score buffer,
    # since a forward pass per position would take minutes here.
    model = build(seed=43, index_topk=index_topk, max_position_embeddings=8192)
    rng = np.random.default_rng(index_topk)
    resident = {topk: ResidentGLM(model, ctx=Context(), max_tokens=7000, topk=topk, long_topk='merge')
                for topk in ('sort', 'radix')}
    # Under graph replay the launches are those captured for max_tokens, reading tokens from the step buffer.
    resident['graph'] = ResidentGLM(model, ctx=Context(), max_tokens=7000, topk='sort', long_topk='merge', graph=True,
                                    device_experts=True)
    # long_topk='select': topk_radix rounds in place of the chunk sorts and merges.
    resident['select'] = ResidentGLM(model, ctx=Context(), max_tokens=7000, topk='sort', long_topk='select')
    resident['graph_select'] = ResidentGLM(model, ctx=Context(), max_tokens=7000, topk='sort', long_topk='select',
                                           graph=True, device_experts=True)
    # long_topk='position' (the default): one topk_select launch, the same set in position order.
    resident['position'] = ResidentGLM(model, ctx=Context(), max_tokens=7000)
    resident['graph_position'] = ResidentGLM(model, ctx=Context(), max_tokens=7000, graph=True, device_experts=True)
    try:
        for tokens in (2049, 4096, 7000):
            scores = rng.integers(-20, 20, size=tokens).astype(np.float32)  # ties
            scores[[3, 7, 9]] = np.nan, -0.0, np.inf
            want = np.argsort(-scores, kind='stable')[:min(index_topk, tokens)]
            for name in ('graph', 'graph_select', 'graph_position'):
                resident[name].step.upload(np.array([tokens - 1, tokens, min(index_topk, tokens), 1], np.int32))
            for name, r in resident.items():
                r.scores.view((tokens,)).upload(scores)
                r._select(tokens)
                np.testing.assert_array_equal(r.selected.data[:r.selected_count],
                                              np.sort(want) if name.endswith('position') else want)
        sort, radix, graph = resident['sort'].launches, resident['radix'].launches, resident['graph'].launches
        assert sort['topk_chunks'] == 3 and sort['topk_merge'] == 1 + 1 + 2 and 'topk_stable' not in sort
        assert radix['topk_stable'] == 3 and 'topk_chunks' not in radix
        assert graph['topk_chunks_at'] == 3 and graph['topk_merge_at'] == 3 * 2 and 'topk_stable_long_at' not in graph
        select, graph_select = resident['select'].launches, resident['graph_select'].launches
        assert select['topk_radix'] == 3 * 9 and 'topk_chunks' not in select
        assert graph_select['topk_radix_at'] == 3 * 9
        assert 'topk_chunks_at' not in graph_select and 'topk_stable_long_at' not in graph_select
        position, graph_position = resident['position'].launches, resident['graph_position'].launches
        assert position['topk_select'] == 3 and not any(n.startswith('topk_') and n != 'topk_select' for n in position)
        assert graph_position['topk_select_at'] == 3
        assert not any(n.startswith('topk_') and n != 'topk_select_at' for n in graph_position)
    finally:
        for r in resident.values():
            r.close()


@pytest.mark.parametrize('graph', [False, True])
def test_short_contexts_skip_the_indexer_queries(graph):
    # While the context fits in index_topk (4 here) every cached token is selected and nothing is
    # scored (#43), so wq_b and weights_proj do not run: their outputs keep a NaN fill until the
    # first step past index_topk, and the logits are still the reference's.
    model = build(seed=47)
    expected = reference(model, TOKENS[:6])
    r = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=graph, graph=graph)
    try:
        for buf in (r.iq, r.head_weights_raw):
            buf.upload(np.full(buf.shape, np.nan, np.float32))
        got = [r.forward(t) for t in TOKENS[:4]]
        assert np.isnan(r.iq.data).all() and np.isnan(r.head_weights_raw.data).all()
        got += [r.forward(t) for t in TOKENS[4:6]]
        assert np.isfinite(r.iq.data).all() and np.isfinite(r.head_weights_raw.data).all()
    finally:
        r.close()
    np.testing.assert_allclose(np.stack(got), expected, rtol=2e-5, atol=2e-5)


def test_position_order_topk_gives_the_reference_logits():
    # long_topk='position' (#32, the default): past index_topk (4) the stable top-k set in position
    # order, the reference's order. The host argsort (sorted), the eager kernel and the captured one
    # give the same ids, so the same bits. long_topk='merge' takes the set in score order: only the
    # order of the attention sums differs.
    model = build(seed=53, index_n_heads=1)  # one indexer head: exact score ties
    expected = reference(model, TOKENS)
    runs = {}
    for name, kw in (('eager', {}), ('host', dict(device_topk=False)), ('graph', dict(graph=True, device_experts=True)),
                     ('merge', dict(long_topk='merge'))):
        r = ResidentGLM(model, ctx=Context(), max_tokens=32, **kw)
        try:
            runs[name] = (np.stack([r.forward(t) for t in TOKENS]), dict(r.launches))
        finally:
            r.close()
    np.testing.assert_allclose(runs['eager'][0], expected, rtol=2e-5, atol=2e-5)
    assert (runs['eager'][0].argmax(-1) == expected.argmax(-1)).all()
    np.testing.assert_array_equal(runs['host'][0], runs['eager'][0])
    np.testing.assert_array_equal(runs['graph'][0], runs['eager'][0])
    assert runs['eager'][1].get('topk_select', 0) > 0 and runs['graph'][1].get('topk_select_at', 0) > 0
    assert 'topk_select' not in runs['host'][1]
    np.testing.assert_allclose(runs['merge'][0], expected, rtol=2e-5, atol=2e-5)
    assert (runs['merge'][0].argmax(-1) == expected.argmax(-1)).all()
    assert 'topk_select' not in runs['merge'][1]


@pytest.mark.parametrize('overrides', [{}, {'index_n_heads': 1}])  # one head: exact score ties
def test_device_topk_selects_exactly_what_the_host_argsort_selects(overrides):
    # Score order (long_topk='merge'): position order's equivalent is test_position_order_topk_gives_the_reference_logits.
    model = build(seed=37, **overrides)
    device = ResidentGLM(model, ctx=Context(), max_tokens=32, long_topk='merge')
    host = ResidentGLM(model, ctx=Context(), max_tokens=32, device_topk=False, long_topk='merge')
    for t in TOKENS:
        a, b = device.forward(t), host.forward(t)
        np.testing.assert_array_equal(device.selected.data[:device.selected_count],
                                      host.selected.data[:host.selected_count])
        np.testing.assert_array_equal(a, b)
    selections = device.launches.get('topk_stable', 0) + device.launches.get('topk_sort', 0)  # topk='auto' (#32)
    assert selections > 0 and not {'topk_stable', 'topk_sort'} & set(host.launches)


# --- Routed experts chosen on the device (device_experts=True) -----------------------------
# The fused MoE spawn kernels (*moe_gate_up, *moe_down) run on Vx's host path here
# (gpu/spawn/host.py), the rest on the SIMT emulator; the host never reads the router's ids.

SPAWN = Path(__file__).parent / 'build' / 'spawn'
EXPERT_KERNELS = tuple(f'{p}moe_{half}' for p in ('', 'bf16_', 'int4_', 'fp8_') for half in ('gate_up', 'down'))
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
    prefix = '' if kind in ('float32', 'fused') else kind + '_'
    sparse = sum(t == 'sparse' for t in model.mlp_types) * len(TOKENS)
    assert device.launches[prefix + 'moe_gate_up'] == device.launches[prefix + 'moe_down'] == sparse
    assert 'scaled_add' not in device.launches


needs_mla_decode = pytest.mark.skipif(not (SPAWN / 'libmla_decode.so').exists(),
                                      reason='gpu/spawn mla_decode not built (gpu/spawn/build.sh)')


@needs_mla_decode
def test_fused_attention_matches_partial_then_merge():
    """mla_decode in one launch against the hand mla_partial and mla_merge: the same algorithm,
    exp aside (libm expf on the host against ex2.approx). This model selects at most 4 tokens, so
    one split; test_spawn_vx_host.py checks several splits against the spawn partial and merge."""
    model = build(seed=67)
    fused = ResidentGLM(model, ctx=Context(), max_tokens=32, fused_attention=True)
    plain = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        got, want = fused.prefill(TOKENS), plain.prefill(TOKENS)
        assert fused.launches['mla_decode'] == model.n_layers * len(TOKENS)
        assert 'mla_partial' not in fused.launches and 'mla_merge' not in fused.launches
    finally:
        fused.close()
        plain.close()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


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


@needs_spawn
@pytest.mark.parametrize('kind,graph,options', [('float32', False, {}), ('float32', True, {}), ('fused', True, {}),
                                                ('fp8', True, dict(fp8_kernels='cvt')), ('bf16', False, {})])
def test_fused_glue_gives_the_same_bits(kind, graph, options):
    # fuse_glue (#39): residual_rmsnorm8 after the attention and at each layer boundary (with the next
    # layer's input RMSNorm), rmsnorm_pair for q_a and kv_a, and under graph replay index_key_at for
    # the indexer's key row; each has the bits of the launches it replaces, so the logits are equal.
    # fp8_kernels='cvt' stacks q_a, kv_a and the indexer's wk in one launch before the norms.
    model = expert_models(kind)
    runs = {}
    for fuse in (False, True):
        r = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, graph=graph, fuse_glue=fuse, **options)
        try:
            runs[fuse] = (np.stack([r.forward(t) for t in TOKENS]), dict(r.launches))
        finally:
            r.close()
    (plain, before), (fused, after) = runs[False], runs[True]
    np.testing.assert_array_equal(fused, plain)
    n = model.n_layers
    steps = before['residual_rmsnorm'] // n  # the steps that launched (eagerly or under capture)
    assert 'residual_rmsnorm' not in after and after['residual_rmsnorm8'] == steps * (2 * n - 1)
    assert after['rmsnorm_pair'] == steps * n
    # Each step's input norms past the first layer and its q_a and kv_a norms are gone.
    assert after['rmsnorm'] == before['rmsnorm'] - steps * (n - 1) - 2 * steps * n
    if graph:
        assert 'layernorm' not in after and after['index_key_at'] == before['layernorm'] > 0
    else:
        assert after['layernorm'] == before['layernorm'] and 'index_key_at' not in after
    moe = sum(model.mlp_type(i) != 'dense' for i in range(n))
    # moe_sum is read in place, and under graph replay the indexer's key row is not copied either.
    assert after.get('copy', 0) == before.get('copy', 0) - steps * moe - (before['layernorm'] if graph else 0)
