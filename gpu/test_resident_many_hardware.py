"""ResidentGLM.forward_many (MTP verification, #16) on a real GPU: one to four tokens as one step
give, bit for bit, the logits, caches and next step that forward() gives one token at a time,
with fp8_matvec4_rows reading each FP8 weight once for all of them. Needs a GPU, gpu/build
(gpu/build.sh) and the gpu/spawn kernels (gpu/spawn/build.sh)."""
import pytest

from gpu.test_resident import SPAWN, build, fp8_model
from gpu.test_resident_many import TOKENS, assert_same, run


@pytest.fixture(scope='module')
def ctx():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    if not (SPAWN / 'fp8_matvec4_rows.ptx').exists():
        pytest.skip('fp8_matvec4_rows not built (gpu/spawn/build.sh)')
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    yield context
    context.close()


@pytest.mark.parametrize('count', [1, 4])
def test_float32_tokens_as_one_step_on_the_gpu(ctx, count):
    assert_same(run(build(seed=3), TOKENS[6:6 + count], ctx=lambda: ctx))


@pytest.mark.parametrize('kernels', ['hand', 'x4'])
def test_fp8_tokens_as_one_step_on_the_gpu(ctx, kernels):
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    results = run(model, TOKENS[6:10], ctx=lambda: ctx, fp8_kernels=kernels, device_experts=True)
    assert_same(results)
    assert (results[1][3].get('fp8_matvec4_rows', 0) > 0) == (kernels == 'x4')
