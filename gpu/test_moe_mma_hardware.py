"""gpu/spawn/moe_mma on a real GPU of sm_89 or later (#63): no CPU copy exists (inline PTX), so these are its only
tests. Unavailable hardware or an unbuilt kernel is a skip, not validation success."""
from pathlib import Path

import numpy as np
import pytest

SPAWN = Path(__file__).parent / 'build' / 'spawn'


@pytest.fixture(scope='module')
def context():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 9))
    except CUDAUnavailable as error:
        pytest.skip('real GPU of sm_89+ unavailable: ' + str(error))
    if not (SPAWN / 'moe_mma' / 'fp8_moe_mma_down_at.ptx').exists():
        pytest.skip('gpu/spawn/moe_mma/build.sh has not run')
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.mark.parametrize('rows, cols, experts, k', [(256, 256, 17, 9), (512, 512, 12, 5), (128, 768, 9, 1)])
def test_fp8_moe_mma_down_matches_the_model_of_its_arithmetic(context, rows, cols, experts, k):
    # validation/moe_mma_check's model: act quantized as the kernel does, float64 dots; only the tensor cores'
    # f32 sums differ. Against down2 (f32 act) the difference is act's E4M3 rounding: about 1e-2 of max |out|.
    from validation.moe_mma_check import run
    r = run(context, context.stream(), SPAWN, np.random.default_rng(rows + cols + k), rows, cols, experts, k, repeats=1)
    assert r['finite']
    assert r['mma_vs_model'] < 2e-5 and r['mma16_vs_model'] < 2e-5
    assert r['mma16_vs_down2'] < 2e-3  # f16 act: about 5e-4 relative a value
    assert r['down2_vs_model'] < 5e-2
