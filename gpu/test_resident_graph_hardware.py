"""ResidentGLM graph=True on a real GPU (#25, #26): the decode step captured once and replayed every
token, against the per-launch path, bit for bit. The long run selects up to 700 tokens, so the
replayed graph runs 1, 2 and 3 MLA splits from the step buffer. Needs a GPU, gpu/build (gpu/build.sh)
and the gpu/spawn expert kernels (gpu/spawn/build.sh)."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import SPAWN, TOKENS, build, expert_models


@pytest.fixture(scope='module')
def ctx():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    if not (SPAWN / 'moe_gate_up.ptx').exists():
        pytest.skip('gpu/spawn expert kernels not built (gpu/spawn/build.sh)')
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    yield context
    context.close()


def both(ctx, model, tokens, max_tokens):
    graph = ResidentGLM(model, ctx=ctx, max_tokens=max_tokens, device_experts=True, graph=True)
    plain = ResidentGLM(model, ctx=ctx, max_tokens=max_tokens, device_experts=True)
    return graph.prefill(tokens), plain.prefill(tokens), graph


@pytest.mark.parametrize('kind', ['float32', 'fused', 'int4', 'fp8', 'bf16'])
def test_graph_replay_equals_the_per_launch_path_on_the_gpu(ctx, kind):
    got, want, graph = both(ctx, expert_models(kind), TOKENS, 32)
    np.testing.assert_array_equal(got, want)
    assert graph.replays == len(TOKENS) - 1


def test_graph_replay_through_several_mla_splits(ctx):
    steps = 700
    model = build(seed=79, index_topk=1024, max_position_embeddings=1024)
    tokens = np.random.default_rng(79).integers(0, model.config['vocab_size'], size=steps).tolist()
    got, want, graph = both(ctx, model, tokens, steps + 8)
    np.testing.assert_array_equal(got, want)
    assert graph.replays == steps - 1
