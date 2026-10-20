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
