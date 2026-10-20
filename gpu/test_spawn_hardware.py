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
