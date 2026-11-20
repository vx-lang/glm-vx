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


VXMM16 = SPAWN / 'moe_mma' / 'fp8_moe_vxmm16_down_at.ptx'


@pytest.mark.parametrize('rows, experts, k', [(6144, 257, 9), (256, 17, 9), (512, 12, 5), (128, 9, 1)])
def test_fp8_moe_vxmm16_down_matches_the_model_and_mma16(context, rows, experts, k):
    # moe_mma='vx16': fp8_moe_mma16_down_at's arithmetic as a plain Vx matmul (the codes' [16, 256] blocks in HBM),
    # an expert width of 256. 6144 x 257 x 9 is one rank of 8 on GLM-5.3.
    if not VXMM16.exists():
        pytest.skip('fp8_moe_vxmm16_down_at not built: it needs Vx 10c6f478f or later')
    from validation.moe_mma_check import run
    r = run(context, context.stream(), SPAWN, np.random.default_rng(rows + experts + k), rows, 256, experts, k, repeats=1)
    assert r['finite']
    assert r['vxmm16_vs_model'] < 2e-5 and r['vxmm16_vs_mma16'] < 2e-5


def test_resident_moe_mma_vx16_gives_mma16s_logits(context):
    # ResidentGLM(moe_mma='vx16') launches fp8_moe_vxmm16_down_at where the expert width is 256, with the codes' tiles
    # as a view of the same weights; the logits agree with moe_mma=True's. The arithmetic is the same but the f32 sums
    # are in another order, and over the layers the logits drift apart: 7.9e-5 at most on the H200 (2026-11-20), of
    # logits up to about 1. A wrong tile or scale would be off by O(0.1-1); the kernel's own test bounds it at 2e-5.
    if not VXMM16.exists():
        pytest.skip('fp8_moe_vxmm16_down_at not built: it needs Vx 10c6f478f or later')
    from gpu.resident import ResidentGLM
    from gpu.test_resident import TOKENS, fp8_model
    model, _ = fp8_model(43, hidden_size=272, moe_intermediate_size=256)
    logits = {}
    for mode in (True, 'vx16'):
        resident = ResidentGLM(model, ctx=context, max_tokens=32, device_experts=True, fp8_kernels='cvt', moe_down2=True,
                               moe_mma=mode)
        try:
            logits[mode] = resident.prefill(TOKENS)
            used = {k for k in resident.launches if 'moe_mma' in k}
        finally:
            resident.close()
        assert used == {'moe_mma/fp8_moe_vxmm16_down_at' if mode == 'vx16' else 'moe_mma/fp8_moe_mma16_down_at'}, used
    drift = float(np.abs(logits['vx16'] - logits[True]).max() / np.abs(logits[True]).max())
    assert drift < 1e-3, drift
    assert np.array_equal(logits['vx16'].argmax(-1), logits[True].argmax(-1))

