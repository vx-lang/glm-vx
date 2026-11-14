"""#29's small-kernel rewrites on a real GPU: matvec_rows at the block sizes the emulator cannot run
(up to 1024 threads) against the hand matvec bit for bit, the register router against its selection
rule, and the eight-heads-a-group index scores against their order model. Unavailable hardware is a
skip, not validation success."""
from pathlib import Path
import numpy as np
import pytest
from gpu.runtime import CUDAContext, CUDAUnavailable

BUILD = Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def context():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        yield ctx
    finally:
        ctx.close()


def _run(context, path, name, grid, block, args):
    tensors = [context.tensor(a.shape, str(a.dtype), data=a.tobytes()) if isinstance(a, np.ndarray) else a
               for a in args]
    context.load_ptx(path).kernel(name).launch((grid,), (block,), [_arg(t) for t in tensors])
    context.synchronize()
    return [np.frombuffer(t.download(), dtype=a.dtype).reshape(a.shape) if isinstance(a, np.ndarray) else None
            for t, a in zip(tensors, args)]


def _arg(value):
    import ctypes
    if isinstance(value, float):
        return ctypes.c_float(value)
    if isinstance(value, int):
        return ctypes.c_int32(value)
    return value


@pytest.mark.parametrize('shape', [(256, 6144), (32, 6144), (7, 131), (3, 8192)])
@pytest.mark.parametrize('block', [1024, 256, 32])
def test_matvec_rows_is_the_hand_matvec_bitwise(context, shape, block):
    from gpu.test_spawn_vx_host import strided_sums, warp_tree
    rng = np.random.default_rng(shape[0] + block)
    w = rng.normal(size=shape).astype('f4'); x = rng.normal(size=shape[1]).astype('f4')
    out = np.zeros(shape[0], 'f4')
    rows = _run(context, BUILD / 'matvec_rows.ptx', 'glm_vx_gpu_matvec_rows', shape[0], block, [out, w, x, *shape])[0]
    hand = _run(context, BUILD / 'kernels.ptx', 'glm_vx_gpu_matvec', (shape[0] + 3) // 4, 128, [out, w, x, *shape])[0]
    np.testing.assert_array_equal(rows.view(np.uint32), hand.view(np.uint32))
    np.testing.assert_array_equal(rows.view(np.uint32), warp_tree(strided_sums(w * x, 32)).view(np.uint32))


@pytest.mark.parametrize('experts,k', [(256, 8), (160, 8), (33, 5), (7, 7)])
def test_router_is_its_selection_rule(context, experts, k):
    from gpu.test_simt import router_model
    rng = np.random.default_rng(experts + k)
    # Distinct scores: the GPU's expf may differ from NumPy's in the last place, which only a tie sees.
    logits = rng.permutation(np.linspace(-3, 3, experts)).astype('f4'); bias = np.zeros(experts, 'f4')
    indices, weights = _run(context, BUILD / 'kernels.ptx', 'glm_vx_gpu_router', 1, 32,
                            [np.full(8, -7, 'i4'), np.full(8, -7, 'f4'), logits, bias, experts, k, 2.5])[:2]
    ids, expected = router_model(logits, bias, k, 2.5)
    np.testing.assert_array_equal(indices[:k], ids)
    np.testing.assert_allclose(weights[:k], expected, rtol=1e-5)
    assert (indices[k:] == -7).all()


@pytest.mark.parametrize('heads,tokens,dim', [(32, 64, 128), (17, 9, 40), (9, 5, 128)])
def test_index_scores_heads_in_groups(context, heads, tokens, dim):
    from gpu.testing.index_reference import inputs, ordered_f32
    q, keys, weights = inputs(heads, tokens, dim)
    scale = float(np.float32(dim ** -.5))
    out = _run(context, BUILD / 'kernels.ptx', 'glm_vx_gpu_index_scores', (tokens + 3) // 4, 128,
               [np.zeros(tokens, 'f4'), q, keys, weights, tokens, heads, dim, scale])[0]
    np.testing.assert_allclose(out, ordered_f32(q, keys, weights, scale), rtol=3e-5, atol=3e-6)
