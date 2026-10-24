"""ResidentGLM with device_experts on a real GPU: the routed experts through gpu/spawn's
*_expert_matvec kernels, chosen from the router's ids on the device, against the host loop that
reads the ids back, for each weight format. Needs a GPU, gpu/build/kernels.ptx (gpu/build.sh)
and the spawn kernels in gpu/build/spawn (gpu/spawn/build.sh)."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import SPAWN, EXPERT_KERNELS, TOKENS, expert_models, refuse_id_reads


@pytest.fixture(scope='module')
def ctx():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    if not all((SPAWN / f'{k}.ptx').exists() for k in EXPERT_KERNELS):
        pytest.skip('gpu/spawn expert kernels not built (gpu/spawn/build.sh)')
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    yield context
    context.close()


@pytest.mark.parametrize('kind', ['float32', 'fused', 'int4', 'fp8', 'bf16'])
def test_device_experts_equal_the_host_loop_on_the_gpu(ctx, kind):
    model = expert_models(kind)
    device = ResidentGLM(model, ctx=ctx, max_tokens=32, device_experts=True)
    host = ResidentGLM(model, ctx=ctx, max_tokens=32)
    refuse_id_reads(device)
    got, want = device.prefill(TOKENS), host.prefill(TOKENS)
    # The spawn matvecs follow the hand kernels' order (gpu/test_spawn_hardware.py checks them bit
    # for bit), so the logits should agree exactly.
    np.testing.assert_array_equal(got, want)
    assert device.launches['expert_scaled_add'] > 0 and 'scaled_add' not in device.launches
