"""ResidentGLM with fused_attention on a real GPU: each layer's MLA decode in one launch, gpu/spawn's
mla_decode (the block of each head's last split merges that head), against mla_partial then
mla_merge. The sequence grows past 512 selected tokens, so later steps run 2 and 3 splits per head
and reuse one arrival-count buffer across every layer and step. Needs a GPU, gpu/build/kernels.ptx
(gpu/build.sh) and gpu/build/spawn/mla_decode.ptx (gpu/spawn/build.sh)."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import SPAWN, build

STEPS = 700


@pytest.fixture(scope='module')
def ctx():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    if not (SPAWN / 'mla_decode.ptx').exists():
        pytest.skip('gpu/spawn mla_decode not built (gpu/spawn/build.sh)')
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    yield context
    context.close()


def test_fused_attention_matches_partial_then_merge_on_the_gpu(ctx):
    model = build(seed=71, index_topk=1024, max_position_embeddings=1024)
    tokens = np.random.default_rng(71).integers(0, model.config['vocab_size'], size=STEPS).tolist()
    fused = ResidentGLM(model, ctx=ctx, max_tokens=STEPS + 8, fused_attention=True)
    plain = ResidentGLM(model, ctx=ctx, max_tokens=STEPS + 8)
    got, want = fused.prefill(tokens), plain.prefill(tokens)
    assert fused.launches['mla_decode'] == model.n_layers * STEPS
    assert 'mla_partial' not in fused.launches and 'mla_merge' not in fused.launches
    np.testing.assert_array_equal(np.frombuffer(fused.arrivals.download(), np.int32), 0)
    # The same algorithm; only exp differs (libdevice's expf against the hand kernels' ex2.approx).
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    assert np.array_equal(got.argmax(-1), want.argmax(-1))
