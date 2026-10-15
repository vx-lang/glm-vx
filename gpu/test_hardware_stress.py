"""Real-device correctness at GLM-5.3 shapes with discriminating inputs.

The smaller hardware tests use tiny shapes, and the benchmark's MLA inputs make
attention nearly uniform (scores ~0.01), which cannot expose softmax or
split-merge errors. These use peaked softmax, edge selection counts and the
real head/rank/dimension sizes. Skipped only when no SM80+ GPU exists.
"""
import ctypes
from pathlib import Path
import numpy as np
import pytest
from gpu.gemm import GEMMKernels
from gpu.runtime import CUDAContext, CUDAUnavailable, GLMKernels
from gpu.test_gemm import tf32_rna

BUILD = Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def ctx():
    try:
        c = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        yield c
    finally:
        c.close()


def mla_oracle(q, qr, kv, kr, ids, scale):
    k, r = kv[ids].astype('f8'), kr[ids].astype('f8')
    s = (q.astype('f8') @ k.T + qr.astype('f8') @ r.T) * scale
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ k, p


@pytest.mark.parametrize('count,splits', [(2048, 8), (2047, 8), (1, 8), (7, 8), (9, 8), (2048, 1), (513, 5)])
@pytest.mark.parametrize('sharpness', [1.0, 4.0])
def test_mla_real_shape_peaked(ctx, count, splits, sharpness):
    heads, rank, rdim, tokens, scale = 64, 512, 64, 4096, .0625
    rng = np.random.default_rng(1000 + count + splits)
    q = rng.normal(0, sharpness, (heads, rank)).astype('f4')
    qr = rng.normal(0, sharpness, (heads, rdim)).astype('f4')
    kv = rng.normal(0, 1, (tokens, rank)).astype('f4')
    kr = rng.normal(0, 1, (tokens, rdim)).astype('f4')
    ids = rng.choice(tokens, count, replace=False).astype('i4')
    expected, probs = mla_oracle(q, qr, kv, kr, ids, scale)
    assert probs.max() > 5 / count or count < 8  # genuinely non-uniform softmax
    with ctx.load_ptx(BUILD / 'kernels.ptx') as mod, ctx.stream() as stream:
        partial, merge = mod.kernel('glm_vx_gpu_mla_partial'), mod.kernel('glm_vx_gpu_mla_merge')
        up = lambda a: ctx.tensor(a.shape, data=a)
        qd, qrd, kvd, krd, idsd = map(up, (q, qr, kv, kr, ids))
        num = ctx.tensor((heads, splits, rank))
        stats = ctx.tensor((heads, splits, 2))
        out = ctx.tensor((heads, rank))
        I = ctypes.c_int32
        partial.launch((heads * splits,), (rank,), [num, stats, qd, qrd, kvd, krd, idsd, I(heads), I(rank), I(rdim),
                       I(tokens), I(count), I(0), I(splits), ctypes.c_float(scale)], stream=stream)
        merge.launch((heads,), (256,), [out, num, stats, I(heads), I(rank), I(splits)], stream=stream)
        stream.synchronize()
        actual = np.frombuffer(out.download(), 'f4').reshape(heads, rank)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-4 * max(1., float(np.abs(expected).max())))


@pytest.mark.parametrize('heads,dim,tokens', [(32, 128, 2048), (32, 128, 4097), (64, 128, 8192)])
def test_index_scores_real_shape(ctx, heads, dim, tokens):
    rng = np.random.default_rng(55 + tokens)
    q = rng.normal(0, 1, (heads, dim)).astype('f4')
    keys = rng.normal(0, 1, (tokens, dim)).astype('f4')
    w = rng.normal(0, 1, heads).astype('f4')
    scale = dim ** -.5
    expected = (np.maximum(q.astype('f8') @ keys.astype('f8').T * scale, 0) * w.astype('f8')[:, None]).sum(0)
    kernels = GLMKernels(ctx.load_ptx(BUILD / 'kernels.ptx'))
    ts = [ctx.tensor(x.shape, data=x.tobytes()) for x in (q, keys, w)]
    out = ctx.tensor((tokens,))
    stream = ctx.stream()
    kernels.index_scores(*ts, scale, out=out, stream=stream)
    stream.synchronize()
    actual = np.frombuffer(out.download(), 'f4')
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4 * max(1., float(np.abs(expected).max())))


@pytest.mark.parametrize('m,n,k', [(257, 2048, 6144), (64, 6144, 2048), (1, 6144, 6144), (130, 12288, 6144)])
@pytest.mark.parametrize('mode', ['f32', 'tf32_rna'])
def test_gemm_real_shape(ctx, m, n, k, mode):
    rng = np.random.default_rng(m * 31 + n)
    a = rng.normal(size=(m, k)).astype('f4')
    b = (rng.normal(size=(k, n)) / np.sqrt(k)).astype('f4')
    gemm = GEMMKernels(ctx.load_ptx(BUILD / 'gemm.ptx'))
    ad, bd = ctx.tensor(a.shape, data=a), ctx.tensor(b.shape, data=b)
    out = ctx.tensor((m, n))
    stream = ctx.stream()
    gemm.gemm(ad, bd, precision=mode, out=out, stream=stream)
    stream.synchronize()
    actual = np.frombuffer(out.download(), 'f4').reshape(m, n)
    aa, bb = (tf32_rna(a), tf32_rna(b)) if mode == 'tf32_rna' else (a, b)
    expected = aa.astype('f8') @ bb.astype('f8')
    bound = 2e-5 * np.maximum(1., np.abs(aa.astype('f8')) @ np.abs(bb.astype('f8')))
    assert np.all(np.abs(actual - expected) <= bound), float(np.abs(actual - expected).max())
