"""ResidentGLM.forward_many (MTP verification, #16) under the CPU SIMT emulator: one to four tokens
as one step give, bit for bit, what forward() gives one token at a time: logits, the final hidden
state, the device caches, and so every later step."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import TOKENS, build, fp8_model
from gpu.testing.context import Context

PREFIX = TOKENS[:6]  # past index_topk = 4: the selections are sparse


def device_rows(resident, upto):
    out = {}
    for layer in range(resident.model.n_layers):
        for name, caches, width in (('latents', resident.latents, resident.rank),
                                    ('rope_keys', resident.rope_keys, resident.rot),
                                    ('index_keys', resident.index_keys, resident.index_dim)):
            if caches[layer] is not None:
                out[name, layer] = np.frombuffer(caches[layer].download(), np.float32).reshape(-1, width)[:upto].copy()
    return out


def run(model, step, **options):
    """(one at a time, forward_many) on two residents of `model`: logits, the next token's logits, caches."""
    results = []
    for many in (False, True):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, **options)
        try:
            for t in PREFIX:
                resident.forward(t, logits=False)
            if many:
                logits, hidden = resident.forward_many(step)
                assert hidden.shape == (len(step), resident.hidden)
            else:
                logits = np.stack([resident.forward(t) for t in step])
            after = resident.forward(TOKENS[-1])
            results.append((logits, after, device_rows(resident, len(PREFIX) + len(step) + 1), dict(resident.launches)))
        finally:
            resident.close()
    return results


def assert_same(results):
    (logits, after, caches, _), (many_logits, many_after, many_caches, _) = results
    np.testing.assert_array_equal(many_logits.view(np.uint32), logits.view(np.uint32))
    np.testing.assert_array_equal(many_after.view(np.uint32), after.view(np.uint32))
    assert caches.keys() == many_caches.keys()
    for key in caches:
        np.testing.assert_array_equal(many_caches[key].view(np.uint32), caches[key].view(np.uint32), err_msg=str(key))


@pytest.mark.parametrize('count', [1, 2, 4])
def test_float32_tokens_as_one_step(count):
    assert_same(run(build(seed=3), TOKENS[6:6 + count]))


@pytest.mark.parametrize('device_experts', [False, True])
@pytest.mark.parametrize('kernels', ['hand', 'x4'])
def test_fp8_tokens_as_one_step(kernels, device_experts):
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    results = run(model, TOKENS[6:10], fp8_kernels=kernels, device_experts=device_experts)
    assert_same(results)
    launches = results[1][3]
    if kernels == 'x4':
        assert launches.get('fp8_matvec4_rows', 0) > 0  # the projections read each weight once
    else:
        assert 'fp8_matvec4_rows' not in launches


def test_hidden_is_the_final_norm_of_each_position():
    model = build(seed=5)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32)
    try:
        for t in PREFIX:
            resident.forward(t, logits=False)
        logits, hidden = resident.forward_many(TOKENS[6:9])
    finally:
        resident.close()
    cache = model.new_cache()
    for t in PREFIX:
        model.forward(t, cache)
    for i, t in enumerate(TOKENS[6:9]):
        want_logits, want_hidden = model.forward_hidden(t, cache)
        np.testing.assert_allclose(hidden[i], want_hidden, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(logits[i], want_logits, rtol=2e-5, atol=2e-5)


def test_limits():
    model = build(seed=3)
    resident = ResidentGLM(model, ctx=Context(), max_tokens=8)
    try:
        with pytest.raises(ValueError, match='1 to 4'):
            resident.forward_many([])
        with pytest.raises(ValueError, match='1 to 4'):
            resident.forward_many([1, 2, 3, 4, 5])
        for t in TOKENS[:6]:
            resident.forward(t, logits=False)
        with pytest.raises(ValueError, match='max_tokens'):
            resident.forward_many([1, 2, 3])
        assert resident.position == 6
    finally:
        resident.close()
    graph = ResidentGLM(model, ctx=Context(), max_tokens=8, graph=True, device_topk=True, device_experts=True)
    try:
        with pytest.raises(ValueError, match='graph'):
            graph.forward_many([1])
    finally:
        graph.close()
