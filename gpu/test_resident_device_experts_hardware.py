"""ResidentGLM with device_experts on a real GPU: the routed experts in two launches, gpu/spawn's
*moe_gate_up and *moe_down, which read the router's ids on the device, against the host loop that
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
    # The fused kernels' dot products follow the hand matvecs' order, but moe_gate_up's sigmoid is
    # libdevice's expf where the hand swiglu uses ex2.approx (gpu/test_spawn_hardware.py, EXP): on
    # the A100 (2026-10-25) the logits differ by at most 1.2e-7. On the CPU, where both are libm's
    # expf, gpu/test_resident.py checks them bit for bit.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert np.array_equal(got.argmax(-1), want.argmax(-1))
    prefix = '' if kind in ('float32', 'fused') else kind + '_'
    assert device.launches[prefix + 'moe_down'] > 0 and 'scaled_add' not in device.launches
