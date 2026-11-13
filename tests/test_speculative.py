"""MTP speculative decoding (glm_vx/speculative.py): the model's greedy output, and its caches."""
import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.speculative import MTPDrafter, generate, greedy
from glm_vx.tiny import tiny_config


def model(seed=3, **config):
    c = dict(tiny_config(), num_nextn_predict_layers=1, **config)
    return GlmMoeDsaModel(c, tiny_weights(c, seed=seed, mtp=True), NumpyBackend())


def prompt(seed, length):
    return [int(t) for t in np.random.default_rng(seed).integers(1, 256, length)]


def assert_same_cache(got, want):
    assert got.position == want.position
    for a, b in zip(got.layers, want.layers):
        for name in ('latents', 'rope_keys', 'index_keys'):
            np.testing.assert_array_equal(getattr(a, name).array, getattr(b, name).array)
        if b.selected_indices is None:
            assert a.selected_indices is None
        else:
            np.testing.assert_array_equal(a.selected_indices, b.selected_indices)


def scratch_mtp(m, tokens):
    """The MTP cache built position by position from the true tokens and the model's hidden states."""
    cache, hiddens = m.new_cache(), []
    for t in tokens[:-1]:
        hiddens.append(m.forward_hidden(t, cache)[1])
    drafter = MTPDrafter(m)
    for p in range(len(tokens) - 2):
        drafter.step(tokens[p + 1], hiddens[p], p)
    return drafter


def assert_same_mtp(got, want):
    for name in ('latents', 'rope_keys', 'index_keys'):
        np.testing.assert_array_equal(getattr(got.state, name).array, getattr(want.state, name).array)


def test_tiny_weights_add_the_mtp_layer_after_the_others():
    c = dict(tiny_config(), num_nextn_predict_layers=1)
    plain, mtp = tiny_weights(c, seed=1), tiny_weights(c, seed=1, mtp=True)
    extra = set(mtp) - set(plain)
    assert {'model.layers.4.eh_proj.weight', 'model.layers.4.enorm.weight', 'model.layers.4.hnorm.weight',
            'model.layers.4.shared_head.norm.weight', 'model.layers.4.self_attn.indexer.wk.weight',
            'model.layers.4.mlp.gate.weight', 'model.layers.4.mlp.experts.3.down_proj.weight'} <= extra
    assert mtp['model.layers.4.eh_proj.weight'].shape == (32, 64)
    for name, value in plain.items():
        np.testing.assert_array_equal(mtp[name], value)


@pytest.mark.parametrize('k', [1, 2, 3, 5])
@pytest.mark.parametrize('seed', [0, 1])
def test_mtp_drafts_give_the_greedy_tokens_and_caches(k, seed):
    m = model()
    tokens = prompt(seed, 6)
    want, want_cache = greedy(m, tokens, 12)
    got, cache, drafter, stats = generate(m, tokens, 12, draft_tokens=k)
    assert got == want
    assert_same_cache(cache, want_cache)
    assert_same_mtp(drafter, scratch_mtp(m, tokens + got))
    assert stats.rounds >= 1 and stats.drafted >= stats.accepted


def oracle(want, offset, wrong_from=None):
    """choose_draft that drafts the greedy tokens (then wrong ones from draft index wrong_from on)."""
    def choose(logits, position, step):
        i = position - offset
        if wrong_from is not None and i >= wrong_from:
            return (want[i] + 1) % 256 if i < len(want) else 0
        return want[i] if i < len(want) else 0
    return choose


@pytest.mark.parametrize('k', [1, 3, 4])
def test_accepted_drafts_rebuild_the_mtp_cache_from_the_models_hidden_states(k):
    m = model()
    tokens = prompt(2, 5)
    want, want_cache = greedy(m, tokens, 13)
    # out[i] is the token at position len(tokens) + i.
    got, cache, drafter, stats = generate(m, tokens, 13, draft_tokens=k, choose_draft=oracle(want, len(tokens)))
    assert got == want
    assert stats.accepted == stats.drafted  # every draft accepted
    # The first token comes from the prompt; each round emits its accepted drafts and one more,
    # but the last round may be cut at max_new_tokens.
    assert len(want) - 1 <= stats.accepted + stats.rounds <= len(want)
    assert stats.accept_length == pytest.approx((stats.accepted + stats.rounds) / stats.rounds)
    assert_same_cache(cache, want_cache)
    assert_same_mtp(drafter, scratch_mtp(m, tokens + got))


@pytest.mark.parametrize('wrong_from', [0, 1, 2, 4])
def test_partly_wrong_drafts_roll_back(wrong_from):
    m = model(seed=5)
    tokens = prompt(4, 7)
    want, want_cache = greedy(m, tokens, 15)

    def choose(logits, position, step):
        i = position - len(tokens)
        right = step < wrong_from  # the round's first wrong_from drafts are right
        if i >= len(want):
            return 0
        return want[i] if right else (want[i] + 1) % 256

    got, cache, drafter, stats = generate(m, tokens, 15, draft_tokens=4, choose_draft=choose)
    assert got == want
    assert all(a == min(wrong_from, 4) for a in stats.accepted_per_round[:-1])
    assert_same_cache(cache, want_cache)
    assert_same_mtp(drafter, scratch_mtp(m, tokens + got))


@pytest.mark.parametrize('at', [1, 2, 3, 5])
def test_a_stop_token_inside_accepted_drafts_ends_like_greedy(at):
    # Round 1 emits free[1] .. free[5] (4 accepted drafts and the model's token): a stop at 1 to 3
    # leaves accepted drafts after it in the cache, which generate rolls back.
    m = model()
    tokens = prompt(6, 5)
    free, _ = greedy(m, tokens, 12)
    assert len(set(free[:6])) == 6, 'pick a prompt whose first tokens differ'
    stop = free[at]
    want, want_cache = greedy(m, tokens, 12, eos=[stop])
    got, cache, drafter, _ = generate(m, tokens, 12, draft_tokens=4, eos=[stop], choose_draft=oracle(free, len(tokens)))
    assert got == want and got[-1] == stop
    assert_same_cache(cache, want_cache)
    assert_same_mtp(drafter, scratch_mtp(m, tokens + got))


def test_one_token_and_a_one_token_prompt():
    m = model()
    assert generate(m, [7], 1)[0] == greedy(m, [7], 1)[0]
    got, cache, drafter, _ = generate(m, [7], 6, draft_tokens=2)
    want, want_cache = greedy(m, [7], 6)
    assert got == want
    assert_same_cache(cache, want_cache)


def test_a_model_without_an_mtp_layer_is_refused():
    c = tiny_config()
    m = GlmMoeDsaModel(c, tiny_weights(c), NumpyBackend())
    with pytest.raises(ValueError, match='MTP'):
        MTPDrafter(m)


def test_the_mtp_step_follows_the_vllm_wiring():
    """e = enorm(embed(t)) (zero at position 0), x = eh_proj([e, hnorm(h)]), layer, shared_head.norm,
    lm_head; written out with NumPy against MTPDrafter.step."""
    m = model()
    w, eps = m.weights, m.eps

    def rms(x, weight):
        return x / np.sqrt(np.mean(x * x) + eps) * weight

    tokens = prompt(9, 3)
    cache, hiddens = m.new_cache(), []
    for t in tokens:
        hiddens.append(m.forward_hidden(t, cache)[1])
    drafter = MTPDrafter(m)
    for p in range(2):
        state_before = len(drafter.state.latents)
        logits, hidden = drafter.step(tokens[p + 1], hiddens[p], p)
        e = np.zeros(32, np.float32) if p == 0 else w['model.embed_tokens.weight'][tokens[p + 1]]
        x = w['model.layers.4.eh_proj.weight'] @ np.concatenate((rms(e, w['model.layers.4.enorm.weight']),
                                                                  rms(hiddens[p], w['model.layers.4.hnorm.weight'])))
        # Run the same layer on a copy of the cache to get the layer output.
        from glm_vx.model import LayerCache
        state = LayerCache()
        for name in ('latents', 'rope_keys', 'index_keys'):
            for row in getattr(drafter.state, name).array[:state_before]:
                getattr(state, name).append(row)
        y, _ = m.forward_layer(4, x.astype(np.float32), p, state)
        g = rms(y, w['model.layers.4.shared_head.norm.weight'])
        np.testing.assert_allclose(hidden, g, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(logits, w['lm_head.weight'] @ g, rtol=1e-5, atol=1e-5)


# Sampling. Chi-square tests with fixed seeds; each threshold is about p = 1e-4 for its degrees of
# freedom, and the always-accept rule, which is wrong, fails the same test.

from glm_vx import speculative  # noqa: E402


def chi_square(counts, probs):
    counts, expected = np.asarray(counts, float), np.asarray(probs, float) * np.sum(counts)
    return float(np.sum((counts - expected) ** 2 / expected))


def round_samples(rule, target, draft, trials, seed):
    """(first emitted, second emitted given the first draft was kept) over many rounds of k = 2."""
    rng = np.random.default_rng(seed)
    first, second = np.zeros(5), np.zeros(5)
    for _ in range(trials):
        drafts = [speculative.sample(q, rng) for q in draft]
        accepted, last = rule(target, draft, drafts, rng)
        emitted = drafts[:accepted] + [last]
        first[emitted[0]] += 1
        if accepted >= 1:
            second[emitted[1]] += 1
    return first, second


def always_accept(target, draft, drafts, rng):
    return len(drafts), speculative.sample(target[len(drafts)], rng)


def test_speculative_sampling_keeps_the_target_distribution():
    rng = np.random.default_rng(11)
    target = [rng.dirichlet(np.ones(5)) for _ in range(3)]
    draft = [rng.dirichlet(np.ones(5)) for _ in range(2)]
    first, second = round_samples(speculative.accept_sampled, target, draft, 20000, seed=1)
    assert chi_square(first, target[0]) < 24 and chi_square(second, target[1]) < 24
    wrong_first, _ = round_samples(always_accept, target, draft, 20000, seed=1)
    assert chi_square(wrong_first, target[0]) > 100  # the test can tell


def test_identical_draft_and_target_accept_every_draft():
    rng = np.random.default_rng(2)
    p = [rng.dirichlet(np.ones(5)) for _ in range(4)]
    for _ in range(200):
        drafts = [speculative.sample(q, rng) for q in p[:3]]
        assert speculative.accept_sampled(p, p[:3], drafts, rng)[0] == 3


def sampling_model():
    c = dict(tiny_config(), num_nextn_predict_layers=1, vocab_size=8)
    w = tiny_weights(c, seed=3, mtp=True)
    w['lm_head.weight'] *= 8  # spread but uneven distributions; the drafts are accepted about half the time
    return GlmMoeDsaModel(c, w, NumpyBackend())


def exact_marginals(m, prompt, temperature):
    """P(token i of the output) for i = 0, 1, 2, by enumerating the model's own distributions."""
    def dist(tokens):
        cache = m.new_cache()
        for t in tokens:
            logits = m.forward(t, cache)
        return speculative.probabilities(logits, temperature)
    p0 = dist(prompt)
    p1 = {a: dist(prompt + [a]) for a in range(8)}
    m1 = sum(p0[a] * p1[a] for a in range(8))
    m2 = sum(p0[a] * p1[a][b] * dist(prompt + [a, b]) for a in range(8) for b in range(8))
    return p0, m1, m2


def test_sampled_generation_has_the_models_distribution():
    m, prompt, temperature = sampling_model(), [1, 2, 3], 1.0
    want = exact_marginals(m, prompt, temperature)
    rng = np.random.default_rng(5)
    counts, drafted, accepted = np.zeros((3, 8)), 0, 0
    for _ in range(1200):
        out, _, _, stats = speculative.generate(m, prompt, 3, draft_tokens=2, temperature=temperature, rng=rng)
        for i, t in enumerate(out):
            counts[i, t] += 1
        drafted, accepted = drafted + stats.drafted, accepted + stats.accepted
    assert 0.2 < accepted / drafted < 0.8  # both paths taken often
    for i in range(3):
        assert chi_square(counts[i], want[i]) < 30, (i, counts[i], want[i] * 1200)


def test_sampling_needs_an_rng_and_its_own_drafts():
    m = sampling_model()
    with pytest.raises(ValueError):
        speculative.generate(m, [1], 3, temperature=1.0)
    with pytest.raises(ValueError):
        speculative.generate(m, [1], 3, temperature=1.0, rng=np.random.default_rng(0),
                             choose_draft=lambda logits, position, step: 0)
