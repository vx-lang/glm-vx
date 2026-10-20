"""Kernels built through Vx's own GPU path (gpu/spawn) against the hand-pipeline kernels on a
real device. Skipped when no supported CUDA GPU exists; build with gpu/spawn/build.sh first."""
from pathlib import Path

import ctypes as C
import numpy as np
import pytest

from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.spawn.launch import SpawnKernel

BUILD = Path(__file__).parent / 'build'


@pytest.fixture(scope='module')
def context():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as e:
        pytest.skip('real GPU unavailable: ' + str(e))
    try:
        yield ctx
    finally:
        ctx.close()


def upload(context, array):
    return context.tensor(array.shape, str(array.dtype), array.tobytes())


def download(tensor):
    return np.frombuffer(tensor.download(), dtype=tensor.dtype).reshape(tensor.shape)


# Sizes cover a tail, one element, and more elements than threads (grid-stride loop).
@pytest.mark.parametrize('n,grid,block', [(1, 1, 32), (1000, 4, 128), (6144, 3, 256), (100003, 7, 96)])
def test_spawn_add_matches_hand_kernel(context, n, grid, block):
    rng = np.random.default_rng(n)
    a, b = rng.normal(size=n).astype('f4'), rng.normal(size=n).astype('f4')
    da, db = upload(context, a), upload(context, b)
    spawn_out, hand_out = context.tensor((n,)), context.tensor((n,))
    SpawnKernel(context, BUILD / 'spawn/add.ptx').launch(grid, block, n=n, a=da, b=db, out=spawn_out)
    hand = context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_add')
    hand.launch(grid, block, [hand_out, da, db, C.c_int32(n)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), a + b)
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


def test_spawn_add_in_place(context):
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=4097).astype('f4'), rng.normal(size=4097).astype('f4')
    da, db = upload(context, a), upload(context, b)
    SpawnKernel(context, BUILD / 'spawn/add.ptx').launch(8, 128, n=a.size, a=da, b=db, out=da)
    context.synchronize()
    np.testing.assert_array_equal(download(da), a + b)


def test_spawn_add_rejects_wrong_arguments(context):
    kernel = SpawnKernel(context, BUILD / 'spawn/add.ptx')
    x = context.tensor((4,))
    with pytest.raises(TypeError):
        kernel.launch(1, 32, n=4, a=x, out=x)
    with pytest.raises(TypeError):
        kernel.launch(1, 32, n=4, a=x, b=context.tensor((4,), 'int32'), out=x)


@pytest.mark.parametrize('rows,cols,block', [(7, 131, 64), (64, 6144, 256), (1, 1, 32)])
def test_spawn_matvec_matches_hand_kernel(context, rows, cols, block):
    rng = np.random.default_rng(rows * cols)
    w, x = rng.normal(size=(rows, cols)).astype('f4'), rng.normal(size=cols).astype('f4')
    dw, dx = upload(context, w.reshape(-1)), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // (block // 32))
    SpawnKernel(context, BUILD / 'spawn/matvec.ptx').launch(grid, block, rows=rows, cols=cols, weight=dw, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'kernels.ptx').kernel('glm_vx_gpu_matvec')
    hand.launch(grid, block, [hand_out, dw, dx, C.c_int32(rows), C.c_int32(cols)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))
    np.testing.assert_allclose(download(spawn_out), w @ x, rtol=1e-4, atol=1e-4)


def test_spawn_bf16_decode_is_exact_for_every_encoding(context):
    # One column per row: out[r] = decode(r) * 1.0, so every bf16 code passes through the decode.
    codes = np.arange(65536, dtype=np.uint16)
    dw, dx, out = upload(context, codes), upload(context, np.ones(1, 'f4')), context.tensor((65536,))
    SpawnKernel(context, BUILD / 'spawn/bf16_matvec.ptx').launch(65536 // 8, 256, rows=65536, cols=1, weight=dw, x=dx, out=out)
    context.synchronize()
    expected = (codes.astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_array_equal(download(out).view(np.uint32)[~np.isnan(expected)],
                                  expected.view(np.uint32)[~np.isnan(expected)])
    assert np.isnan(download(out)[np.isnan(expected)]).all()


@pytest.mark.parametrize('rows,cols', [(5, 131), (96, 6144)])
def test_spawn_bf16_matvec_matches_hand_kernel(context, rows, cols):
    rng = np.random.default_rng(rows + cols)
    w = (rng.normal(size=(rows, cols)).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=cols).astype('f4')
    dw, dx = upload(context, w.reshape(-1)), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // 8)
    SpawnKernel(context, BUILD / 'spawn/bf16_matvec.ptx').launch(grid, 256, rows=rows, cols=cols, weight=dw, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'bf16.ptx').kernel('glm_vx_gpu_bf16_matvec')
    hand.launch(grid, 256, [hand_out, dw, dx, C.c_int32(rows), C.c_int32(cols)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


@pytest.mark.parametrize('rows,cols,scale_kind', [(5, 131, 0), (64, 6144, 1), (3, 257, 1)])
def test_spawn_int4_matvec_matches_hand_kernel(context, rows, cols, scale_kind):
    rng = np.random.default_rng(rows * 7 + cols)
    groups, row_bytes = -(-cols // 128), -(-cols // 2)
    packed = rng.integers(0, 256, size=rows * row_bytes, dtype=np.uint8)
    if cols % 2:  # odd tail: the padding nibble is zero
        packed.reshape(rows, row_bytes)[:, -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=rows * groups).astype(np.float32)
    if scale_kind:
        scales = (magnitudes.view(np.uint32) >> 16).astype(np.uint16)
    else:
        scales = magnitudes.astype(np.float16).view(np.uint16)
    x = rng.normal(size=cols).astype('f4')
    dp, ds, dx = upload(context, packed), upload(context, scales), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // 8)
    SpawnKernel(context, BUILD / 'spawn/int4_matvec.ptx').launch(
        grid, 256, rows=rows, cols=cols, scales=ds, scale_kind=scale_kind, packed=dp, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'int4.ptx').kernel('glm_vx_gpu_int4_matvec')
    hand.launch(grid, 256, [hand_out, dp, ds, dx, C.c_int32(rows), C.c_int32(cols), C.c_int32(scale_kind)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


@pytest.mark.parametrize('rows,cols', [(5, 131), (300, 6144)])
def test_spawn_fp8_matvec_matches_hand_kernel(context, rows, cols):
    rng = np.random.default_rng(rows * 11 + cols)
    weight = rng.integers(0, 256, size=rows * cols, dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = rng.uniform(0.001, 0.1, size=-(-rows // 128) * -(-cols // 128)).astype('f4')
    x = rng.normal(size=cols).astype('f4')
    dw, ds, dx = upload(context, weight), upload(context, scales), upload(context, x)
    spawn_out, hand_out = context.tensor((rows,)), context.tensor((rows,))
    grid = -(-rows // 8)
    SpawnKernel(context, BUILD / 'spawn/fp8_matvec.ptx').launch(
        grid, 256, rows=rows, cols=cols, scales=ds, weight=dw, x=dx, out=spawn_out)
    hand = context.load_ptx(BUILD / 'fp8.ptx').kernel('glm_vx_gpu_fp8_matvec')
    hand.launch(grid, 256, [hand_out, dw, ds, dx, C.c_int32(rows), C.c_int32(cols)])
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_out), download(hand_out))


@pytest.mark.parametrize('tokens,k,ties', [(1, 1, False), (300, 64, True), (5000, 2048, True), (70000, 2048, False)])
def test_spawn_topk_matches_stable_argsort_and_hand_kernel(context, tokens, k, ties):
    from gpu.topk import TopK, scratch_ints
    rng = np.random.default_rng(tokens + k)
    scores = rng.integers(-4, 4, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    if tokens > 10:
        scores[3], scores[7] = np.nan, -0.0  # NaN sorts last; -0.0 ties +0.0
    kk = min(k, tokens)
    d_scores = upload(context, scores)
    spawn_ids = context.tensor((kk,), 'int32')
    SpawnKernel(context, BUILD / 'spawn/topk.ptx').launch(
        1, 256, tokens=tokens, k=k, scratch=context.tensor((scratch_ints(tokens, k),), 'int32'),
        scores=d_scores.view((tokens,), 'int32'), out_ids=spawn_ids)
    hand_ids = TopK.from_ptx(context, BUILD / 'topk.ptx').topk_stable(d_scores, k)
    context.synchronize()
    np.testing.assert_array_equal(download(spawn_ids), np.argsort(-scores, kind='stable')[:kk])
    np.testing.assert_array_equal(download(spawn_ids), download(hand_ids))


def hand_kernel(context, module, name):
    return context.load_ptx(BUILD / module).kernel(name)


def compare(context, spawn_name, hand, grid, block, spawn_args, hand_args, outs):
    """Launch the spawn kernel and the hand kernel on separate outputs, then require bitwise
    equality. outs maps each spawn output name to its (spawn tensor, hand tensor) pair."""
    SpawnKernel(context, BUILD / f'spawn/{spawn_name}.ptx').launch(grid, block, **spawn_args)
    hand.launch(grid, block, hand_args)
    context.synchronize()
    for name, (spawn_out, hand_out) in outs.items():
        np.testing.assert_array_equal(download(spawn_out), download(hand_out), err_msg=name)


@pytest.mark.parametrize('n,grid,block', [(1, 1, 32), (1000, 4, 128), (100003, 7, 96)])
@pytest.mark.parametrize('kernel', ['copy', 'scale'])
def test_spawn_elementwise_matches_hand_kernel(context, kernel, n, grid, block):
    x = np.random.default_rng(n).normal(size=n).astype('f4')
    dx = upload(context, x)
    s, h = context.tensor((n,)), context.tensor((n,))
    alpha = np.float32(0.37)
    if kernel == 'copy':
        compare(context, 'copy', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_copy'), grid, block,
                dict(n=n, x=dx, out=s), [h, dx, C.c_int32(n)], {'out': (s, h)})
        np.testing.assert_array_equal(download(s), x)
    else:
        compare(context, 'scale', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_scale'), grid, block,
                dict(n=n, x=dx, alpha=float(alpha), out=s), [h, dx, C.c_float(alpha), C.c_int32(n)], {'out': (s, h)})
        np.testing.assert_array_equal(download(s), x * alpha)


@pytest.mark.parametrize('n,grid,block,slot', [(1, 1, 32, 0), (6144, 3, 256, 5), (100003, 7, 96, 7)])
def test_spawn_scaled_add_matches_hand_kernel(context, n, grid, block, slot):
    rng = np.random.default_rng(n + slot)
    base, x, weights = (rng.normal(size=s).astype('f4') for s in (n, n, 8))
    dx, dw = upload(context, x), upload(context, weights)
    s, h = upload(context, base), upload(context, base)  # accumulated in place
    compare(context, 'scaled_add', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_scaled_add'), grid, block,
            dict(slot=slot, weights=dw, n=n, x=dx, out=s), [h, dx, dw, C.c_int32(slot), C.c_int32(n)], {'out': (s, h)})
    np.testing.assert_array_equal(download(s), base + x * weights[slot])


@pytest.mark.parametrize('heads,dim,grid,block', [(1, 2, 1, 32), (16, 64, 2, 256), (64, 64, 3, 96)])
def test_spawn_rope_glm_matches_hand_kernel(context, heads, dim, grid, block):
    rng = np.random.default_rng(heads * dim)
    x = rng.normal(size=heads * dim).astype('f4')
    angles = rng.uniform(0, 6.3, size=dim // 2)
    cos, sin = np.cos(angles).astype('f4'), np.sin(angles).astype('f4')
    dx, dc, ds = upload(context, x), upload(context, cos), upload(context, sin)
    s, h = context.tensor((heads * dim,)), context.tensor((heads * dim,))
    compare(context, 'rope_glm', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_rope_glm'), grid, block,
            dict(heads=heads, dim=dim, x=dx, cosine=dc, sine=ds, out=s),
            [h, dx, dc, ds, C.c_int32(heads), C.c_int32(dim)], {'out': (s, h)})


@pytest.mark.parametrize('heads,dim,x_stride,out_stride,grid,block', [(1, 2, 2, 2, 1, 32), (64, 64, 256, 64, 3, 96), (16, 64, 192, 128, 2, 256)])
def test_spawn_rope_strided_matches_hand_kernel(context, heads, dim, x_stride, out_stride, grid, block):
    rng = np.random.default_rng(heads + dim + x_stride)
    x = rng.normal(size=heads * x_stride).astype('f4')
    angles = rng.uniform(0, 6.3, size=dim // 2)
    cos, sin = np.cos(angles).astype('f4'), np.sin(angles).astype('f4')
    dx, dc, ds = upload(context, x), upload(context, cos), upload(context, sin)
    # Untouched gaps between strided heads must stay as they were in both outputs.
    fill = rng.normal(size=heads * out_stride).astype('f4')
    s, h = upload(context, fill), upload(context, fill)
    compare(context, 'rope_strided', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_rope_strided'), grid, block,
            dict(heads=heads, dim=dim, x_stride=x_stride, out_stride=out_stride, x=dx, cosine=dc, sine=ds, out=s),
            [h, dx, dc, ds, C.c_int32(heads), C.c_int32(dim), C.c_int32(x_stride), C.c_int32(out_stride)], {'out': (s, h)})


@pytest.mark.parametrize('tokens,heads,dim', [(1, 1, 1), (37, 16, 128), (1000, 32, 128)])
def test_spawn_index_scores_matches_hand_kernel(context, tokens, heads, dim):
    rng = np.random.default_rng(tokens * heads + dim)
    q, keys = rng.normal(size=heads * dim).astype('f4'), rng.normal(size=tokens * dim).astype('f4')
    weights = (rng.normal(size=heads) * heads ** -0.5).astype('f4')
    scale = np.float32(dim ** -0.5)
    dq, dk, dw = upload(context, q), upload(context, keys), upload(context, weights)
    s, h = context.tensor((tokens,)), context.tensor((tokens,))
    compare(context, 'index_scores', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_index_scores'), (tokens + 3) // 4, 128,
            dict(tokens=tokens, heads=heads, dim=dim, q=dq, keys=dk, scale=float(scale), weights=dw, out=s),
            [h, dq, dk, dw, C.c_int32(tokens), C.c_int32(heads), C.c_int32(dim), C.c_float(scale)], {'out': (s, h)})


@pytest.mark.parametrize('heads,rows,cols,w_stride,x_stride,out_stride', [
    (1, 1, 1, 1, 1, 1), (16, 64, 192, 64 * 192, 256, 64), (8, 37, 131, 37 * 131 + 5, 140, 40)])
def test_spawn_matvec_heads_matches_hand_kernel(context, heads, rows, cols, w_stride, x_stride, out_stride):
    rng = np.random.default_rng(heads * rows + cols)
    w = rng.normal(size=heads * w_stride).astype('f4')
    x = rng.normal(size=heads * x_stride).astype('f4')
    dw, dx = upload(context, w), upload(context, x)
    fill = rng.normal(size=heads * out_stride).astype('f4')
    s, h = upload(context, fill), upload(context, fill)
    grid = -(-(heads * rows) // 4)
    compare(context, 'matvec_heads', hand_kernel(context, 'kernels.ptx', 'glm_vx_gpu_matvec_heads'), grid, 128,
            dict(heads=heads, rows=rows, cols=cols, w_stride=w_stride, x_stride=x_stride, weight=dw, x=dx,
                 out_stride=out_stride, out=s),
            [h, dw, dx, C.c_int32(heads), C.c_int32(rows), C.c_int32(cols), C.c_int32(w_stride),
             C.c_int32(x_stride), C.c_int32(out_stride)], {'out': (s, h)})


@pytest.mark.parametrize('m,n,k', [(1, 1, 0), (1, 1, 1), (7, 9, 131), (33, 65, 256)])
def test_spawn_gemm_f32_matches_hand_kernel(context, m, n, k):
    rng = np.random.default_rng(m * n + k)
    a, b = rng.normal(size=max(m * k, 1)).astype('f4'), rng.normal(size=max(k * n, 1)).astype('f4')
    da, db = upload(context, a), upload(context, b)
    s, h = upload(context, np.full(m * n, 7, 'f4')), upload(context, np.full(m * n, 7, 'f4'))
    compare(context, 'gemm_f32', hand_kernel(context, 'gemm.ptx', 'glm_vx_gpu_gemm_f32'), -(-(m * n) // 128), 128,
            dict(m=m, n=n, k=k, a=da, b=db, out=s), [h, da, db, C.c_int32(m), C.c_int32(n), C.c_int32(k)], {'out': (s, h)})


# Batched and grouped matmuls: experts == 0 (dense, expert_ids unread) and experts > 0 with an
# out-of-range id (-1 and E) in the routing, which must write zeros.
MATMUL_CASES = [(1, 1, 1, 0), (5, 37, 131, 0), (6, 300, 257, 3)]


def routing(context, rng, m, experts):
    if not experts:
        return upload(context, np.zeros(1, 'i4')), C.c_uint64(0)
    ids = rng.integers(0, experts, size=m).astype('i4')
    ids[0], ids[-1] = -1, experts
    d = upload(context, ids)
    return d, d


@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_spawn_bf16_matmul_matches_hand_kernel(context, m, n, k, experts):
    rng = np.random.default_rng(m + n + k + experts)
    e = max(experts, 1)
    w = (rng.normal(size=e * n * k).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=m * k).astype('f4')
    dw, dx = upload(context, w), upload(context, x)
    spawn_ids, hand_ids = routing(context, rng, m, experts)
    s, h = context.tensor((m * n,)), context.tensor((m * n,))
    compare(context, 'bf16_matmul', hand_kernel(context, 'bf16.ptx', 'glm_vx_gpu_bf16_matmul'), -(-(m * n) // 8), 256,
            dict(rows_m=m, rows_n=n, cols=k, experts=experts, expert_ids=spawn_ids, weight=dw, x=dx, out=s),
            [h, dw, dx, hand_ids, C.c_int32(m), C.c_int32(n), C.c_int32(k), C.c_int32(experts)], {'out': (s, h)})


@pytest.mark.parametrize('scale_kind', [0, 1])
@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_spawn_int4_matmul_matches_hand_kernel(context, m, n, k, experts, scale_kind):
    rng = np.random.default_rng(m * 3 + n + k + experts + scale_kind)
    e, groups, row_bytes = max(experts, 1), -(-k // 128), -(-k // 2)
    packed = rng.integers(0, 256, size=(e * n, row_bytes), dtype=np.uint8)
    if k % 2:
        packed[:, -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=e * n * groups).astype(np.float32)
    scales = ((magnitudes.view(np.uint32) >> 16).astype(np.uint16) if scale_kind
              else magnitudes.astype(np.float16).view(np.uint16))
    x = rng.normal(size=m * k).astype('f4')
    dp, ds, dx = upload(context, packed.reshape(-1)), upload(context, scales), upload(context, x)
    spawn_ids, hand_ids = routing(context, rng, m, experts)
    s, h = context.tensor((m * n,)), context.tensor((m * n,))
    compare(context, 'int4_matmul', hand_kernel(context, 'int4.ptx', 'glm_vx_gpu_int4_matmul'), -(-(m * n) // 8), 256,
            dict(rows_m=m, rows_n=n, cols=k, experts=experts, expert_ids=spawn_ids, scales=ds, scale_kind=scale_kind,
                 packed=dp, x=dx, out=s),
            [h, dp, ds, dx, hand_ids, C.c_int32(m), C.c_int32(n), C.c_int32(k), C.c_int32(experts),
             C.c_int32(scale_kind)], {'out': (s, h)})


@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_spawn_fp8_matmul_matches_hand_kernel(context, m, n, k, experts):
    rng = np.random.default_rng(m * 5 + n + k + experts)
    e = max(experts, 1)
    weight = rng.integers(0, 256, size=e * n * k, dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0  # no NaN codes
    scales = rng.uniform(0.001, 0.1, size=e * -(-n // 128) * -(-k // 128)).astype('f4')
    x = rng.normal(size=m * k).astype('f4')
    dw, ds, dx = upload(context, weight), upload(context, scales), upload(context, x)
    spawn_ids, hand_ids = routing(context, rng, m, experts)
    s, h = context.tensor((m * n,)), context.tensor((m * n,))
    compare(context, 'fp8_matmul', hand_kernel(context, 'fp8_batched.ptx', 'glm_vx_gpu_fp8_matmul'), -(-(m * n) // 8), 256,
            dict(rows_m=m, rows_n=n, cols=k, experts=experts, expert_ids=spawn_ids, weight=dw, x=dx, scales=ds, out=s),
            [h, dw, ds, dx, hand_ids, C.c_int32(m), C.c_int32(n), C.c_int32(k), C.c_int32(experts)], {'out': (s, h)})
