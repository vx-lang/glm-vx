"""MTP speculative decoding on a ResidentGLM (gpu/speculative.py, #16) under the CPU SIMT emulator:
the MTP layer on the device against glm_vx.speculative.MTPDrafter, and generate() against plain
greedy decoding on the same resident model, tokens and cache rows bit for bit."""
import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.speculative import MTPDrafter
from glm_vx.tiny import tiny_config
from gpu import speculative
from gpu.resident import ResidentGLM
from gpu.test_resident_many import device_rows
from gpu.testing.context import Context

PROMPT = [3, 17, 5, 5, 9, 200]  # past index_topk = 4: sparse selections


def mtp_model(seed=3, **overrides):
    config = dict(tiny_config(), index_n_heads=16, num_nextn_predict_layers=1, **overrides)
    weights = tiny_weights(config, seed=seed, mtp=True)
    for w in weights.values():
        w.flags.writeable = False
    return GlmMoeDsaModel(config, weights, NumpyBackend())


@pytest.fixture(scope='module')
def model():
    return mtp_model()


def resident_of(model, **options):
    return ResidentGLM(model, ctx=Context(), max_tokens=32, mtp=True, **options)


def test_the_mtp_layer_on_the_device_is_the_cpu_ones(model):
    cache, hiddens = model.new_cache(), []
    for t in PROMPT:
        hiddens.append(model.forward_hidden(t, cache)[1])
    drafter = MTPDrafter(model, index_share=False)
    resident = resident_of(model)
    try:
        for p in range(len(PROMPT) - 1):
            want_logits, want_hidden = drafter.step(PROMPT[p + 1], hiddens[p], p)
            got_logits, got_hidden = resident.mtp_step(PROMPT[p + 1], hiddens[p], p)
            np.testing.assert_allclose(got_hidden, want_hidden, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(got_logits, want_logits, rtol=2e-5, atol=2e-5)
    finally:
        resident.close()


GRAPH = dict(graph=True, device_topk=True, device_experts=True)


def test_the_mtp_layer_under_graph_is_the_eager_ones(model):
    # Each graph (full indexer, reused selection) runs eagerly, is captured, then replayed.
    steps = [(0, False), (1, False), (2, True), (3, False), (4, True), (5, True), (6, False), (7, True)]
    rng = np.random.default_rng(7)
    hiddens = [rng.standard_normal(model.hidden).astype(np.float32) for _ in steps]
    results = []
    for options in (dict(device_experts=True), GRAPH):
        resident = resident_of(model, **options)
        try:
            results.append([resident.mtp_step(PROMPT[p % len(PROMPT)], h, p, reuse_selection=reuse)
                            for (p, reuse), h in zip(steps, hiddens)])
            assert resident.replays == (6 if resident.graph else 0)  # each graph: captured, then 2 replays
        finally:
            resident.close()
    for (want_logits, want_hidden), (logits, hidden) in zip(*results):
        np.testing.assert_array_equal(logits.view(np.uint32), want_logits.view(np.uint32))
        np.testing.assert_array_equal(hidden.view(np.uint32), want_hidden.view(np.uint32))


@pytest.mark.parametrize('index_share', [False, True])
def test_generation_under_graph_is_greedy_decoding(model, index_share):
    want, want_rows, want_position = plain(model, PROMPT, 10, device_experts=True)
    got, rows, position, stats = speculated(model, PROMPT, 10, draft_tokens=3, index_share=index_share, **GRAPH)
    assert got == want and position == want_position and stats.rounds >= 3
    assert_rows_equal({key: v for key, v in rows.items() if key[1] < model.n_layers}, want_rows)


def plain(model, prompt, new, eos=(), **options):
    resident = resident_of(model, **options)
    try:
        tokens = speculative.greedy(resident, prompt, new, eos=eos)
        return tokens, device_rows(resident, resident.position), resident.position
    finally:
        resident.close()


def speculated(model, prompt, new, *, graph=False, device_topk=True, device_experts=False, **options):
    resident = resident_of(model, graph=graph, device_topk=device_topk, device_experts=device_experts)
    try:
        tokens, stats = speculative.generate(resident, prompt, new, **options)
        return tokens, device_rows(resident, resident.position), resident.position, stats
    finally:
        resident.close()


def assert_rows_equal(got, want):
    assert got.keys() == want.keys()
    for key in want:
        np.testing.assert_array_equal(got[key].view(np.uint32), want[key].view(np.uint32), err_msg=str(key))


@pytest.mark.parametrize('k', [1, 3])
def test_greedy_generation_is_the_residents_greedy_decoding(model, k):
    want, want_rows, want_position = plain(model, PROMPT, 8)
    got, rows, position, stats = speculated(model, PROMPT, 8, draft_tokens=k)
    assert got == want and position == want_position
    assert_rows_equal({key: v for key, v in rows.items() if key[1] < model.n_layers}, want_rows)
    assert stats.rounds >= 1


@pytest.mark.parametrize('wrong_from', [0, 1, 3])
def test_forced_drafts_take_every_acceptance_path(model, wrong_from):
    want, want_rows, want_position = plain(model, PROMPT, 9)

    def choose(logits, position, step):
        i = position - len(PROMPT)
        if i >= len(want):
            return 0
        return want[i] if step < wrong_from else (want[i] + 1) % 256

    got, rows, position, stats = speculated(model, PROMPT, 9, draft_tokens=3, choose_draft=choose)
    assert got == want and position == want_position
    assert all(a == min(wrong_from, 3) for a in stats.accepted_per_round[:-1])
    assert_rows_equal({key: v for key, v in rows.items() if key[1] < model.n_layers}, want_rows)


def test_a_stop_token_inside_accepted_drafts_ends_like_greedy(model):
    free, _, _ = plain(model, PROMPT, 8)
    assert len(set(free[:4])) == 4
    want, want_rows, want_position = plain(model, PROMPT, 8, eos=[free[2]])

    def choose(logits, position, step):
        i = position - len(PROMPT)
        return free[i] if i < len(free) else 0

    got, rows, position, _ = speculated(model, PROMPT, 8, draft_tokens=3, eos=[free[2]], choose_draft=choose)
    assert got == want and got[-1] == free[2] and position == want_position
    assert_rows_equal({key: v for key, v in rows.items() if key[1] < model.n_layers}, want_rows)


def test_sampling_is_reproducible_and_in_the_vocabulary(model):
    runs = [speculated(model, PROMPT, 6, draft_tokens=2, temperature=1.0, rng=np.random.default_rng(4))[0]
            for _ in range(2)]
    assert runs[0] == runs[1] and all(0 <= t < 256 for t in runs[0])


def test_limits(model):
    resident = resident_of(model)
    try:
        with pytest.raises(ValueError, match='draft_tokens'):
            speculative.generate(resident, PROMPT, 4, draft_tokens=4)
        with pytest.raises(ValueError, match='rng'):
            speculative.generate(resident, PROMPT, 4, temperature=1.0)
    finally:
        resident.close()
    plain_resident = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        with pytest.raises(ValueError, match='mtp'):
            speculative.generate(plain_resident, PROMPT, 4)
        with pytest.raises(ValueError, match='mtp'):
            plain_resident.mtp_step(1, np.zeros(model.hidden, np.float32), 0)
    finally:
        plain_resident.close()
    with pytest.raises(ValueError, match='num_nextn_predict_layers'):
        no_mtp = dict(tiny_config(), index_n_heads=16)
        ResidentGLM(GlmMoeDsaModel(no_mtp, tiny_weights(no_mtp), NumpyBackend()), ctx=Context(), mtp=True)
