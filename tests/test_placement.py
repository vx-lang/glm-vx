import numpy as np
import pytest

from glm_vx.admission import Verdict
from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.placement import ExpertPlacement
from glm_vx.tiny import tiny_config

TOKENS = [3, 17, 5, 5, 9, 200, 1, 44, 8, 2, 31]


def digest(w):
    return hash(np.ascontiguousarray(w).tobytes())


class Recording(NumpyBackend):
    """NumPy backend that remembers a content hash of every weight matrix it multiplies."""
    def __init__(self):
        self.weights_seen = set()
        self.ops = 0

    def matvec(self, w, x):
        self.weights_seen.add(digest(w))
        self.ops += 1
        return super().matvec(w, x)


@pytest.fixture(scope='module')
def fixture():
    config = tiny_config()
    rng = np.random.default_rng(2)
    weights = tiny_weights(config, seed=9)
    for name, w in weights.items():
        weights[name] = (rng.normal(0, 1 / np.sqrt(w.shape[-1]), w.shape).astype(np.float32)
                         if 'norm' not in name else rng.uniform(.7, 1.3, w.shape).astype(np.float32))
        weights[name].flags.writeable = False
    return config, weights


def expert_pointers(weights, layers):
    return {digest(w) for n, w in weights.items()
            if '.mlp.experts.' in n and int(n.split('.layers.')[1].split('.')[0]) in layers}


def baseline(config, weights, run):
    return run(GlmMoeDsaModel(config, weights, NumpyBackend()))


def per_token(model):
    cache = model.new_cache()
    return np.stack([model.forward(t, cache) for t in TOKENS])


def batched(model):
    cache = model.new_cache()
    model.prefill_chunk(TOKENS[:6], cache, output_logits=False)
    return model.prefill_chunk(TOKENS[6:], cache)


@pytest.mark.parametrize('run', [per_token, batched], ids=['per_token', 'batched_prefill'])
@pytest.mark.parametrize('host_layers', [{1}, {2, 3}, {1, 2, 3}, set()])
def test_split_runs_each_expert_on_its_assigned_backend_with_identical_logits(fixture, run, host_layers):
    config, weights = fixture
    primary, host = Recording(), Recording()
    model = GlmMoeDsaModel(config, weights, primary, expert_placement=ExpertPlacement(host, host_layers))
    got = run(model)
    np.testing.assert_array_equal(got, baseline(config, weights, run))

    moe = {1, 2, 3}
    host_experts = expert_pointers(weights, host_layers)
    gpu_experts = expert_pointers(weights, moe - host_layers)
    assert host.weights_seen <= host_experts                 # the host backend ran only host-layer experts...
    assert not primary.weights_seen & host_experts           # ...and the primary never touched them
    if gpu_experts:
        assert primary.weights_seen & gpu_experts            # primary-layer experts do run on the primary
    if host_layers:
        assert host.weights_seen and host.ops > 0            # (some tokens do route to host experts)
    else:
        assert host.ops == 0


def test_non_expert_work_never_reaches_the_host_backend(fixture):
    config, weights = fixture
    primary, host = Recording(), Recording()
    model = GlmMoeDsaModel(config, weights, primary, expert_placement=ExpertPlacement(host, {1, 2, 3}))
    per_token(model)
    non_expert = {digest(w) for n, w in weights.items() if '.mlp.experts.' not in n and w.ndim == 2}
    assert not host.weights_seen & non_expert
    assert primary.weights_seen & non_expert                 # attention, router, shared experts, lm_head stay primary


def test_from_verdict_keeps_the_first_moe_layers_on_the_primary(fixture):
    config, _ = fixture
    v = Verdict('hybrid', '', 8, 0, 0, 'fp8', 'bf16', 0, 0, 0, moe_layers=3, expert_layers_on_gpu=1,
                expert_layers_on_host=2, gpu_used_gib=0, host_used_gib=0)
    placement = ExpertPlacement.from_verdict(v, config, NumpyBackend())
    assert placement.host_layers == {2, 3}
    wrong = Verdict('hybrid', '', 8, 0, 0, 'fp8', 'bf16', 0, 0, 0, moe_layers=75, expert_layers_on_gpu=4,
                    expert_layers_on_host=71, gpu_used_gib=0, host_used_gib=0)
    with pytest.raises(ValueError):
        ExpertPlacement.from_verdict(wrong, config, NumpyBackend())


def test_placement_with_packed_weights_is_rejected(fixture):
    config, weights = fixture
    with pytest.raises(ValueError):
        GlmMoeDsaModel(config, weights, NumpyBackend(), packed_weights=True,
                       expert_placement=ExpertPlacement(NumpyBackend(), {1}))
