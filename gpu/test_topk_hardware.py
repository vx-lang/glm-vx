"""Real GPU gate for the device top-k. Skipped only when no SM80 device is present."""
from pathlib import Path

import numpy as np
import pytest

from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.topk import TopK, scratch_ints


@pytest.fixture
def device():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        yield ctx, TopK.from_ptx(ctx, Path(__file__).parent / 'build/topk.ptx')
    finally:
        ctx.close()


def rule(scores, k):
    return np.argsort(-scores, kind='stable')[:min(k, scores.size)].astype('i4')


@pytest.mark.parametrize('tokens,k', [(1, 4), (33, 33), (300, 299), (4096, 2048), (70000, 2048)])
@pytest.mark.parametrize('ties', [False, True])
def test_topk_device(device, tokens, k, ties):
    ctx, tk = device
    scores = np.random.default_rng(tokens + k).normal(size=tokens).astype('f4')
    if ties:
        scores = np.round(scores * 3) / 3
        scores[::5] = np.float32(-0.0)
        scores[1::5] = np.float32(0.0)
    out = tk.topk_stable(ctx.tensor(scores.shape, data=scores), k)
    ctx.synchronize()
    np.testing.assert_array_equal(np.frombuffer(out.download(), 'i4'), rule(scores, k))


def test_topk_device_nan_last(device):
    ctx, tk = device
    scores = np.random.default_rng(1).normal(size=700).astype('f4')
    scores[::4] = np.nan
    out = tk.topk_stable(ctx.tensor(scores.shape, data=scores), 600)
    ctx.synchronize()
    np.testing.assert_array_equal(np.frombuffer(out.download(), 'i4'), rule(scores, 600))


def test_topk_wrapper_rejects_small_buffers(device):
    ctx, tk = device
    scores = ctx.tensor((100,), data=np.zeros(100, 'f4'))
    with pytest.raises(ValueError):
        tk.topk_stable(scores, 10, scratch=ctx.tensor((scratch_ints(100, 10) - 1,), 'int32'))
    with pytest.raises(ValueError):
        tk.topk_stable(scores, 10, out=ctx.tensor((9,), 'int32'))
