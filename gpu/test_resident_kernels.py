"""Kernels added for the device-resident decode path, under the CPU SIMT emulator
(test-only lanes running the same compiled Vx; NOT GPU validation)."""
import numpy as np
import pytest

from gpu.test_simt import Simt


@pytest.fixture(scope='module')
def simt():
    return Simt()


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize('n', [1, 31, 256, 1000])
def test_add_copy_scale(simt, n):
    r = rng(n)
    a, b = r.normal(size=n).astype('f4'), r.normal(size=n).astype('f4')
    out = np.zeros(n, 'f4')
    simt.run('glm_vx_gpu_add', out, a, b, n, grid=4, block=64)
    np.testing.assert_array_equal(out, a + b)
    inplace = a.copy()
    simt.run('glm_vx_gpu_add', inplace, inplace, b, n, grid=3, block=32)
    np.testing.assert_array_equal(inplace, a + b)
    out[:] = 0
    simt.run('glm_vx_gpu_copy', out, a, n, grid=2, block=32)
    np.testing.assert_array_equal(out, a)
    simt.run('glm_vx_gpu_scale', out, a, 0.37, n, grid=2, block=64)
    np.testing.assert_array_equal(out, a * np.float32(0.37))


def test_scaled_add_reads_the_slot_and_rounds_twice(simt):
    r = rng(3)
    n = 517
    x, base = r.normal(size=n).astype('f4'), r.normal(size=n).astype('f4')
    weights = np.array([0.1, 0.7, 0.25], 'f4')
    for slot in range(3):
        out = base.copy()
        simt.run('glm_vx_gpu_scaled_add', out, x, weights, slot, n, grid=3, block=64)
        np.testing.assert_array_equal(out, base + x * weights[slot])


@pytest.mark.parametrize('rows,dim,block', [(1, 64, 32), (3, 64, 64), (2, 100, 128), (1, 257, 256)])
def test_layernorm_matches_numpy(simt, rows, dim, block):
    r = rng(rows * dim)
    x = (r.normal(size=(rows, dim)) * 3 + 1).astype('f4')
    w, b = r.normal(size=dim).astype('f4'), r.normal(size=dim).astype('f4')
    out = np.zeros_like(x)
    simt.run('glm_vx_gpu_layernorm', out, x, w, b, rows, dim, 1e-6, grid=rows, block=block)
    x64 = x.astype('f8')
    mean = x64.mean(-1, keepdims=True)
    expected = (x64 - mean) / np.sqrt(((x64 - mean) ** 2).mean(-1, keepdims=True) + 1e-6) * w + b
    np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)


def shifted(array, offset):
    """array.reshape(-1)[offset:], zero-padded back to full length so strided reads stay in bounds."""
    flat = array.reshape(-1)
    return np.concatenate([flat[offset:], np.zeros(offset, flat.dtype)])


def test_rope_strided_equals_rope_glm_on_gathered_rows(simt):
    r = rng(5)
    heads, nope, rot = 6, 10, 8
    q = r.normal(size=(heads, nope + rot)).astype('f4')
    cos, sin = r.normal(size=rot // 2).astype('f4'), r.normal(size=rot // 2).astype('f4')
    strided = np.zeros((heads, rot), 'f4')
    # Head h of the shifted buffer starts at h*(nope+rot), the same floats as q[h, nope:].
    simt.run('glm_vx_gpu_rope_strided', strided, shifted(q, nope), cos, sin, heads, rot, nope + rot, rot, grid=2, block=32)
    expected = np.zeros((heads, rot), 'f4')
    simt.run('glm_vx_gpu_rope_glm', expected, np.ascontiguousarray(q[:, nope:]), cos, sin, heads, rot, grid=2, block=32)
    np.testing.assert_array_equal(strided, expected)


def test_rope_strided_scatters_into_wider_rows(simt):
    r = rng(6)
    heads, dim, wide = 4, 8, 20
    x = r.normal(size=(heads, wide)).astype('f4')
    cos, sin = r.normal(size=dim // 2).astype('f4'), r.normal(size=dim // 2).astype('f4')
    out = np.full((heads, wide), -7.0, 'f4')
    simt.run('glm_vx_gpu_rope_strided', out, x, cos, sin, heads, dim, wide, wide, grid=2, block=32)
    ref = np.zeros((heads, dim), 'f4')
    simt.run('glm_vx_gpu_rope_glm', ref, np.ascontiguousarray(x[:, :dim]), cos, sin, heads, dim, grid=2, block=32)
    np.testing.assert_array_equal(out[:, :dim], ref)
    assert (out[:, dim:] == -7.0).all()  # untouched pass-through columns


@pytest.mark.parametrize('heads,rows,cols', [(1, 5, 33), (4, 7, 70), (3, 16, 128)])
def test_matvec_heads_is_bitwise_per_head_matvec(simt, heads, rows, cols):
    r = rng(heads * rows)
    w = r.normal(size=(heads, rows, cols)).astype('f4')
    x = r.normal(size=(heads, cols)).astype('f4')
    out = np.zeros((heads, rows), 'f4')
    simt.run('glm_vx_gpu_matvec_heads', out, w, x, heads, rows, cols, rows * cols, cols, rows,
             grid=(heads * rows + 1) // 2, block=64)
    for h in range(heads):
        ref = np.zeros(rows, 'f4')
        simt.run('glm_vx_gpu_matvec', ref, w[h].copy(), x[h].copy(), rows, cols, grid=(rows + 1) // 2, block=64)
        np.testing.assert_array_equal(out[h], ref)


def test_matvec_heads_strided_weight_and_output(simt):
    # Value up-projection layout: weight [heads, nope+value, rank] read from row `nope`, rows=value.
    r = rng(9)
    heads, nope, value, rank = 3, 5, 6, 40
    kv = r.normal(size=(heads, nope + value, rank)).astype('f4')
    att = r.normal(size=(heads, rank)).astype('f4')
    out = np.zeros((heads, value), 'f4')
    simt.run('glm_vx_gpu_matvec_heads', out, shifted(kv, nope * rank), att, heads, value, rank,
             (nope + value) * rank, rank, value, grid=(heads * value + 1) // 2, block=64)
    for h in range(heads):
        ref = np.zeros(value, 'f4')
        simt.run('glm_vx_gpu_matvec', ref, kv[h, nope:].copy(), att[h].copy(), value, rank, grid=(value + 1) // 2, block=64)
        np.testing.assert_array_equal(out[h], ref)
