"""MTP speculative decoding on a real GPU (gpu/speculative.py, #16): generate() against plain greedy
decoding on the same ResidentGLM, tokens and cache rows bit for bit, with the MTP layer drafting on
the device and forward_many verifying. Needs a GPU, gpu/build (gpu/build.sh) and the gpu/spawn
kernels (gpu/spawn/build.sh)."""
import numpy as np
import pytest

from gpu import speculative
from gpu.resident import ResidentGLM
from gpu.test_resident import SPAWN
from gpu.test_resident_many import device_rows
from gpu.test_speculative_resident import PROMPT, mtp_model


@pytest.fixture(scope='module')
def ctx():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    if not (SPAWN / 'fp8_matvec4_rows.ptx').exists():
        pytest.skip('gpu/spawn kernels not built (gpu/spawn/build.sh)')
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    yield context
    context.close()


@pytest.mark.parametrize('device_experts', [False, True])
@pytest.mark.parametrize('k', [1, 3])
def test_generation_is_greedy_decoding_on_the_gpu(ctx, k, device_experts):
    model = mtp_model(seed=5)
    runs = []
    for spec in (False, True):
        resident = ResidentGLM(model, ctx=ctx, max_tokens=64, mtp=True, device_experts=device_experts)
        try:
            if spec:
                tokens, stats = speculative.generate(resident, PROMPT, 24, draft_tokens=k)
            else:
                tokens = speculative.greedy(resident, PROMPT, 24)
            rows = {key: v for key, v in device_rows(resident, resident.position).items() if key[1] < model.n_layers}
            runs.append((tokens, rows, resident.position))
        finally:
            resident.close()
    (want, want_rows, want_position), (got, rows, position) = runs
    assert got == want and position == want_position
    for key in want_rows:
        np.testing.assert_array_equal(rows[key].view(np.uint32), want_rows[key].view(np.uint32), err_msg=str(key))
