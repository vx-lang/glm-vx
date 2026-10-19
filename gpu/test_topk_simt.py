"""Device top-k (topk.vx) under the CPU SIMT emulator (NOT GPU validation) against the host rule
np.argsort(-scores, kind='stable')[:min(k, tokens)]. Rebuild with gpu/testing/build_simt.sh."""
import ctypes as C

import numpy as np
import pytest

from gpu.test_simt import Simt

SCRATCH_BASE = 66824


@pytest.fixture(scope='module')
def simt():
    return Simt()


def host_rule(scores, k):
    return np.argsort(-scores, kind='stable')[:min(k, scores.size)].astype('i4')


def device_topk(simt, scores, k, block=256):
    scores = np.ascontiguousarray(scores, dtype='f4')
    kk = max(min(k, scores.size), 0)
    out = np.full(kk + 2, -77, 'i4')  # sentinels past the count must survive
    scratch = np.full(SCRATCH_BASE + 2 * kk + 8, 0x5A5A5A5A, 'i4')  # contents must not matter
    simt.run('glm_vx_gpu_topk_stable', out[1:], scores.view('i4'), scratch, scores.size, k, grid=1, block=block)
    return out


def check(simt, scores, k):
    scores = np.asarray(scores, dtype='f4')
    expected = host_rule(scores, k)
    out = device_topk(simt, scores, k)
    # out was passed offset by one element; the written ids start at out[1].
    assert out[0] == -77
    np.testing.assert_array_equal(out[1:1 + expected.size], expected)
    assert (out[1 + expected.size:] == -77).all()


def test_key_orders_like_floats(simt):
    f = simt.lib.glm_vx_topk_key
    f.argtypes, f.restype = [C.c_int32], C.c_int64
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(size=500) * 10.0 ** rng.integers(-40, 38, 500), [0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45, 3.4e38, -3.4e38]]).astype('f4')
    keys = np.array([f(int(b)) for b in values.view('i4')])
    order = np.argsort(values, kind='stable')
    assert (np.diff(keys[order]) >= 0).all()
    assert f(int(np.float32(0.0).view('i4'))) == f(int(np.float32(-0.0).view('i4')))
    assert keys.min() >= 0 and keys.max() < 2**32
    for bits in (0x7FC00000, 0xFFC00001, 0x7F800001, 0xFF800001):
        assert f(np.uint32(bits).view('i4').item()) == 0  # every NaN sorts below -inf (key 0x7FFFFF)


@pytest.mark.parametrize('tokens', [1, 2, 31, 32, 33, 63, 255, 256, 257, 511, 513, 1000, 2047, 2048, 2049, 5000])
@pytest.mark.parametrize('k', [1, 7, 256, 2048])
def test_random_sizes(simt, tokens, k):
    scores = np.random.default_rng(tokens * 7 + k).normal(size=tokens).astype('f4')
    check(simt, scores, k)


@pytest.mark.parametrize('tokens,k', [(1, 1), (1, 2048), (5, 5), (5, 9), (300, 300), (300, 299), (300, 301), (2048, 2048)])
def test_k_versus_tokens(simt, tokens, k):
    check(simt, np.random.default_rng(tokens + k).normal(size=tokens), k)


@pytest.mark.parametrize('tokens,k', [(1, 1), (40, 7), (1000, 2048), (3000, 2048), (4096, 2048)])
def test_all_equal_scores_take_lowest_positions(simt, tokens, k):
    check(simt, np.full(tokens, 3.25, 'f4'), k)


@pytest.mark.parametrize('distinct', [2, 3, 8])
@pytest.mark.parametrize('tokens,k', [(100, 10), (257, 100), (3000, 2048), (4097, 1500)])
def test_few_distinct_values_heavy_ties(simt, distinct, tokens, k):
    rng = np.random.default_rng(distinct * 31 + tokens)
    check(simt, rng.choice(np.linspace(-1, 1, distinct).astype('f4'), size=tokens), k)


def test_ties_straddle_the_threshold_across_chunks(simt):
    # 1000 equal winners-at-threshold spread over every thread chunk; k cuts through them.
    scores = np.zeros(3000, 'f4')
    scores[::7] = 5.0
    scores[3::11] = 9.0
    for k in (1, 100, 428, 429, 430, 700, 1500):
        check(simt, scores, k)


def test_signed_zero_ties_like_numpy(simt):
    scores = np.array([0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -1.0, 0.0], 'f4')
    check(simt, scores, 8)
    check(simt, scores, 4)
    check(simt, np.tile(np.array([-0.0, 0.0], 'f4'), 700), 500)


def test_inf_values(simt):
    scores = np.array([1.0, np.inf, -np.inf, 0.0, np.inf, -np.inf, 2.0], 'f4')
    for k in range(1, 9):
        check(simt, scores, k)


def test_nan_sorts_last_ties_by_position(simt):
    rng = np.random.default_rng(3)
    scores = rng.normal(size=600).astype('f4')
    nan_bits = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0xFFC12345], np.uint32).view('f4')
    scores[rng.choice(600, 90, replace=False)] = np.resize(nan_bits, 90)
    for k in (10, 400, 510, 511, 512, 600):  # 510 finite values: k past them returns NaN positions
        check(simt, scores, k)
    check(simt, np.full(50, np.nan, 'f4'), 20)
    mixed = np.array([np.nan, -np.inf, np.nan, -np.inf, 1.0], 'f4')
    for k in range(1, 7):
        check(simt, mixed, k)


@pytest.mark.parametrize('seed', range(6))
def test_random_sizes_around_boundaries(simt, seed):
    rng = np.random.default_rng(100 + seed)
    tokens = int(rng.choice([1, 31, 32, 33, 255, 256, 257, 8191, 8192, 8193]) + rng.integers(-1, 2))
    tokens = max(tokens, 1)
    k = int(rng.integers(1, 3000))
    # quantized scores force ties at every size
    check(simt, np.round(rng.normal(size=tokens) * 4) / 4, k)


def test_large_tokens_k_2048(simt):
    rng = np.random.default_rng(7)
    check(simt, rng.normal(size=70000).astype('f4'), 2048)
    check(simt, np.round(rng.normal(size=70000) * 20).astype('f4'), 2048)  # ~100s of ties per value


def test_invalid_arguments_write_nothing(simt):
    scores = np.arange(10, dtype='f4')
    for tokens, k, block in [(0, 4, 256), (10, 0, 256), (10, -3, 256), (10, 4, 128), (10, 4, 64)]:
        out = np.full(8, -77, 'i4')
        scratch = np.zeros(SCRATCH_BASE + 64, 'i4')
        simt.run('glm_vx_gpu_topk_stable', out, scores.view('i4'), scratch, tokens, k, grid=1, block=block)
        assert (out == -77).all()
