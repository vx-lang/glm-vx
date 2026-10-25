"""ResidentGLM graph=True (#25, #26) under the CPU SIMT emulator: the step's layers captured once
and replayed every token, against the per-launch path. The test context records a capture's
launches and runs them again on replay, and refuses a copy or an allocation during the capture,
as gpu/runtime.py does. Not GPU validation: gpu/test_resident_graph_hardware.py is."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import TOKENS, build, expert_models, needs_spawn, refuse_id_reads
from gpu.test_simt import Simt
from gpu.testing.context import Context


@pytest.fixture(scope='module')
def simt():
    return Simt()


# --- The *_at kernels against their plain entries ----------------------------------------------

def step(position, tokens, count, splits):
    return np.array([position, tokens, count, splits], np.int32)


def test_index_scores_at_reads_tokens_and_exits_past_them(simt):
    rng = np.random.default_rng(1)
    heads, dim, cap, tokens = 4, 40, 64, 23
    q, keys, w = (rng.normal(size=n).astype('f4') for n in (heads * dim, cap * dim, heads))
    want, got = np.zeros(cap, 'f4'), np.full(cap, 7.0, 'f4')
    simt.run('glm_vx_gpu_index_scores', want, q, keys, w, tokens, heads, dim, 0.25, grid=(tokens + 3) // 4, block=128)
    simt.run('glm_vx_gpu_index_scores_at', got, q, keys, w, step(0, tokens, 0, 0), heads, dim, 0.25,
             grid=(cap + 3) // 4, block=128)
    np.testing.assert_array_equal(got[:tokens], want[:tokens])
    assert (got[tokens:] == 7.0).all()


@pytest.mark.parametrize('count,splits', [(5, 1), (300, 2), (700, 3)])
def test_mla_partial_and_merge_at_read_count_and_splits(simt, count, splits):
    rng = np.random.default_rng(count)
    heads, rank, rot, tokens, most = 3, 64, 16, 900, 16
    q, qr = rng.normal(size=heads * rank).astype('f4'), rng.normal(size=heads * rot).astype('f4')
    latent, rope = rng.normal(size=tokens * rank).astype('f4'), rng.normal(size=tokens * rot).astype('f4')
    selected = rng.permutation(tokens)[:count].astype(np.int32)
    out = {}
    for name in ('plain', 'at'):
        num, stats, merged = np.zeros(heads * most * rank, 'f4'), np.zeros(heads * most * 2, 'f4'), np.zeros(heads * rank, 'f4')
        if name == 'plain':
            simt.run('glm_vx_gpu_mla_partial', num, stats, q, qr, latent, rope, selected, heads, rank, rot, tokens,
                     count, 0, splits, 0.125, grid=heads * splits, block=64)
            simt.run('glm_vx_gpu_mla_merge', merged, num, stats, heads, rank, splits, grid=heads, block=256)
        else:
            meta = step(0, tokens, count, splits)
            simt.run('glm_vx_gpu_mla_partial_at', num, stats, q, qr, latent, rope, selected, meta, heads, rank, rot,
                     0, 0.125, grid=heads * most, block=64)
            simt.run('glm_vx_gpu_mla_merge_at', merged, num, stats, meta, heads, rank, grid=heads, block=256)
        out[name] = num[:heads * splits * rank], stats[:heads * splits * 2], merged
    for a, b in zip(out['plain'], out['at']):
        np.testing.assert_array_equal(a, b)


def test_store_row_writes_the_row_at_the_position(simt):
    cache, row = np.zeros((6, 37), 'f4'), np.arange(37, dtype='f4')
    simt.run('glm_vx_gpu_store_row', cache, row, step(4, 5, 0, 0), 37, grid=1, block=256)
    np.testing.assert_array_equal(cache[4], row)
    assert not cache[:4].any() and not cache[5].any()


def test_topk_stable_at_reads_tokens(simt):
    from gpu.topk import scratch_ints
    rng = np.random.default_rng(3)
    cap, k, tokens = 300, 32, 211
    scores = rng.normal(size=cap).astype('f4')
    scores[5] = scores[9] = scores[17]  # ties keep position order
    ids = np.full(k, -1, np.int32)
    simt.run('glm_vx_gpu_topk_stable_at', ids, scores.view(np.int32), np.zeros(scratch_ints(cap, k), np.int32),
             step(0, tokens, 0, 0), k, grid=1, block=256)
    np.testing.assert_array_equal(ids, np.argsort(-scores[:tokens], kind='stable')[:k])


# --- ResidentGLM graph=True --------------------------------------------------------------------

def run(model, tokens, **kwargs):
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, **kwargs)
    try:
        refuse_id_reads(resident)
        return resident.prefill(tokens), resident
    finally:
        resident.close()


@needs_spawn
@pytest.mark.parametrize('kind', ['float32', 'fused', 'int4', 'fp8', 'bf16'])
def test_graph_replay_equals_the_per_launch_path(kind):
    model = expert_models(kind)
    got, graph = run(model, TOKENS, graph=True)
    want, _ = run(model, TOKENS)
    np.testing.assert_array_equal(got, want)
    # The first step eagerly, the second captured, then every step a replay of that graph.
    assert graph.replays == len(TOKENS) - 1
    full = sum(t == 'full' for t in model.indexer_types)
    assert graph.launches['index_scores_at'] == graph.launches['topk_stable_at'] == 2 * full
    assert graph.launches['mla_partial_at'] == graph.launches['mla_merge_at'] == 2 * model.n_layers
    assert graph.launches['store_row'] == 2 * (2 * model.n_layers + full)
    assert not {'index_scores', 'topk_stable', 'mla_partial', 'mla_merge'} & set(graph.launches)


@needs_spawn
def test_graph_steps_without_logits_and_after_reset_replay_the_same_graph():
    model = build(seed=73)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True, graph=True)
    plain = ResidentGLM(model, ctx=Context(), max_tokens=32, device_experts=True)
    try:
        for t in TOKENS[:5]:
            assert resident.forward(t, logits=False) is None
            plain.forward(t, logits=False)
        np.testing.assert_array_equal(resident.forward(TOKENS[5]), plain.forward(TOKENS[5]))
        graph = resident._step_graph
        resident.reset()
        plain.reset()
        np.testing.assert_array_equal(resident.prefill(TOKENS[:7]), plain.prefill(TOKENS[:7]))
        assert resident._step_graph is graph and resident.replays == 5 + 7
    finally:
        resident.close()
        plain.close()


def test_graph_needs_device_reads_only():
    model = build()
    with pytest.raises(ValueError, match='device_topk and device_experts'):
        ResidentGLM(model, ctx=Context(), graph=True)
    with pytest.raises(ValueError, match='device_topk and device_experts'):
        ResidentGLM(model, ctx=Context(), graph=True, device_experts=True, device_topk=False)
    with pytest.raises(NotImplementedError, match='#27'):
        ResidentGLM(model, ctx=Context(), graph=True, device_experts=True, fused_attention=True)


def test_the_test_context_refuses_copies_and_allocations_during_a_capture():
    ctx = Context()
    stream, x = ctx.stream(), ctx.tensor((4,))
    with pytest.raises(RuntimeError, match='captures'):
        with stream.capture():
            x.upload(np.zeros(4, 'f4'))
    with pytest.raises(RuntimeError, match='captures'):
        with stream.capture():
            x.view((2,)).download()
    with pytest.raises(RuntimeError, match='captures'):
        with stream.capture():
            ctx.tensor((2,))
    x.upload(np.ones(4, 'f4'))  # allowed again once the capture ended
