"""The tensor-core probes of gpu/spawn/probes (glm-vx #10) against numpy.

Two kinds of test:
- Model tests (any machine): numpy replays each probe's index math against the fragment layouts of
  the PTX ISA (the tables the probes cite in their comments), so a wrong fragment, staging or
  descriptor formula fails here without a GPU.
- Hardware tests: launch the built PTX (gpu/spawn/probes/build.sh, into gpu/build/spawn/probes/)
  through gpu/spawn/launch.py and compare with numpy. mma_bf16_sm80 needs sm_80 or newer; the
  wgmma probes are sm_90a code and run only on compute capability 9.0 (H100/H200). Skipped
  without such a GPU or without the PTX.
"""
from pathlib import Path

import numpy as np
import pytest

BUILD = Path(__file__).parent / 'build' / 'spawn' / 'probes'


# --- numeric formats ----------------------------------------------------------------------------

def bf16_bits(x):
    """Truncate f32 to bfloat16 bits (the values are then exact in bf16)."""
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def bf16_value(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def e4m3_table():
    """All 256 e4m3fn (OCP FP8 E4M3) codes as f32; S.1111.111 is NaN, no infinity."""
    codes = np.arange(256)
    sign = np.where(codes & 128, -1.0, 1.0)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    value = np.where(exponent == 0, mantissa / 8 * 2.0 ** -6, (1 + mantissa / 8) * 2.0 ** (exponent - 7.0))
    value = sign * value
    value[(exponent == 15) & (mantissa == 7)] = np.nan
    return value.astype(np.float32)


def e4m3_bits(x):
    """Nearest finite e4m3 code of each value."""
    table = e4m3_table()
    finite = np.flatnonzero(~np.isnan(table))
    nearest = np.abs(np.asarray(x, np.float32)[..., None] - table[finite]).argmin(-1)
    return finite[nearest].astype(np.uint8)


def words(bits):
    """Row-major [R, K] of 16- or 8-bit codes as i32 words: lower k in the lower bits."""
    return np.ascontiguousarray(bits).view(np.int32)


# --- the PTX ISA fragment layouts ----------------------------------------------------------------

def mma_m16n8k16_a(lane, i):
    """(row, col) of a_i, PTX ISA 9.7.14.5.8, .bf16 A of m16n8k16."""
    g, t = lane >> 2, lane % 4
    row = g if i in (0, 1, 4, 5) else g + 8
    col = t * 2 + (i & 1) + (8 if i >= 4 else 0)
    return row, col


def mma_m16n8k16_b(lane, i):
    """(row, col) = (k, n) of b_i, PTX ISA 9.7.14.5.8, .bf16 B of m16n8k16."""
    g, t = lane >> 2, lane % 4
    return t * 2 + (i & 1) + (8 if i >= 2 else 0), g


def mma_m16n8_c(lane, i):
    """(row, col) of c_i / d_i, PTX ISA 9.7.14.5.8, .f32 C/D of m16n8k16."""
    g, t = lane >> 2, lane % 4
    return g + (8 if i >= 2 else 0), t * 2 + (i & 1)


def wgmma_m64n64_d(tid, i):
    """(row, col) of d_i, PTX ISA 9.7.15.5.2, .f32 D of m64nNk16 (N = 64: i in 0..31)."""
    warp, lane = tid // 32, tid % 32
    g, t = lane >> 2, lane % 4
    return 16 * warp + g + 8 * ((i >> 1) & 1), 8 * (i >> 2) + t * 2 + (i & 1)


def kmajor_element_byte(desc, row, k_byte):
    """Shared byte address of (row, k byte) of a K-major, no-swizzle wgmma operand: PTX ISA
    9.7.15.5.1 — core matrices of 8 rows x 16 bytes; LBO steps k by 16 bytes, SBO steps 8 rows."""
    start, lbo, sbo = (desc & 0x3FFF) << 4, ((desc >> 16) & 0x3FFF) << 4, ((desc >> 32) & 0x3FFF) << 4
    assert desc >> 62 == 0, 'the probes use no swizzle'
    return start + (row >> 3) * sbo + (row & 7) * 16 + (k_byte >> 4) * lbo + (k_byte & 15)


# --- probe index math, as in the .vx files --------------------------------------------------------

def mma_probe_model(a_words, b_words, m, n, k):
    """mma_bf16_sm80.vx: the probe's loads, the mma semantics per the fragment tables, its stores."""
    a_bits = a_words.view(np.uint16).reshape(-1)
    b_bits = b_words.view(np.uint16).reshape(-1)
    out = np.full(m * n, np.nan, np.float64)
    word_count = k // 2
    for tile in range((m // 16) * (n // 64)):
        m0, n0 = tile // (n // 64) * 16, tile % (n // 64) * 64
        acc = np.zeros((32, 32))  # [lane, register]
        for kw in range(0, word_count, 8):
            # Gather the tile each lane's registers hold, by the fragment tables, from the words the
            # probe loads; a missing or doubled element would leave NaN or overwrite.
            a_tile = np.full((16, 16), np.nan)
            b_tiles = np.full((8, 16, 8), np.nan)
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                a_lo, a_hi = (m0 + g) * word_count + t, (m0 + g + 8) * word_count + t
                regs = [a_lo + kw, a_hi + kw, a_lo + kw + 4, a_hi + kw + 4]
                for i in range(8):
                    word = regs[i // 2]
                    a_tile[mma_m16n8k16_a(lane, i)] = bf16_value(a_bits[2 * word + (i & 1)])
                for j in range(8):
                    row = (n0 + g) * word_count + t + 8 * j * word_count + kw
                    for i in range(4):
                        word = [row, row + 4][i // 2]
                        b_tiles[j][mma_m16n8k16_b(lane, i)] = bf16_value(b_bits[2 * word + (i & 1)])
            assert not np.isnan(a_tile).any() and not np.isnan(b_tiles).any()
            for j in range(8):
                d = a_tile @ b_tiles[j]
                for lane in range(32):
                    for q in range(4):
                        acc[lane, 4 * j + q] += d[mma_m16n8_c(lane, q)]
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            out_lo = (m0 + g) * n + n0 + 2 * t
            out_hi = out_lo + 8 * n
            for j in range(8):
                col = 8 * j
                out[out_lo + col], out[out_lo + col + 1] = acc[lane, 4 * j], acc[lane, 4 * j + 1]
                out[out_hi + col], out[out_hi + col + 1] = acc[lane, 4 * j + 2], acc[lane, 4 * j + 3]
    return out.reshape(m, n)


def stage_block(smem_words, src_words, row0, words_per_row, kw):
    """stage_block() of the wgmma probes: 128 threads x 16 words into the core-matrix layout."""
    flat = src_words.reshape(-1)
    for tid in range(128):
        for p in range(16):
            idx = tid + 128 * p
            r, w = idx >> 5, idx & 31
            smem_words[((r >> 3) * 8 + (w >> 2)) * 32 + (r & 7) * 4 + (w & 3)] = flat[(row0 + r) * words_per_row + kw + w]


def kmajor_desc(addr):
    return ((addr & 262143) >> 4) | (8 << 16) | (64 << 32)


def read_operand(smem_bytes, base, desc, step, k_bytes):
    """The 64 x k_bytes operand one wgmma step reads through its descriptor (+16 encoded a step)."""
    d = desc + 16 * step
    block = np.zeros((64, k_bytes), np.uint8)
    for row in range(64):
        for kb in range(k_bytes):
            block[row, kb] = smem_bytes[kmajor_element_byte(d, row, kb) - base]
    return block


def wgmma_probe_model(a_words, b_words, m, n, k, *, fp8=False, a_scale=None, b_scale=None):
    """wgmma_bf16_sm90a.vx / wgmma_fp8_sm90a.vx: staging, descriptors, steps, D layout, stores."""
    elem_bytes = 1 if fp8 else 2
    word_count = k * elem_bytes // 4
    k_block = 128 if fp8 else 64                  # elements per staged block (128 bytes a row)
    blocks = k // k_block
    table = e4m3_table()

    def decode(raw):  # [64, bytes] uint8 -> [64, k] float64
        if fp8:
            return table[raw].astype(np.float64)
        return bf16_value(np.ascontiguousarray(raw).view(np.uint16)).astype(np.float64)

    out = np.full((m, n), np.nan)
    base = 0x400  # any 16-byte aligned shared address; the descriptor encodes it
    for tile in range((m // 64) * (n // 64)):
        m0, n0 = tile // (n // 64) * 64, tile % (n // 64) * 64
        acc = np.zeros((128, 32))
        for kb in range(blocks):
            kw = kb * 32
            a_smem, b_smem = np.zeros(2048, np.int32), np.zeros(2048, np.int32)
            stage_block(a_smem, a_words, m0, word_count, kw)
            stage_block(b_smem, b_words, n0, word_count, kw)
            a_bytes, b_bytes = a_smem.view(np.uint8), b_smem.view(np.uint8)
            d = np.zeros((64, 64))
            for step in range(4):  # k16 (bf16) or k32 (fp8): 32 bytes of k each
                a_op = decode(read_operand(a_bytes, base, kmajor_desc(base), step, 32))
                b_op = decode(read_operand(b_bytes, base, kmajor_desc(base), step, 32))
                d += a_op @ b_op.T
            # Each thread's registers, by the D table; the fp8 probe scales the block's partial.
            for tid in range(128):
                for i in range(32):
                    row, col = wgmma_m64n64_d(tid, i)
                    scale = 1.0
                    if fp8:
                        scale = a_scale[m0 + row, kb] * b_scale[(n0 // 128), kb]
                    acc[tid, i] += d[row, col] * scale
        for tid in range(128):
            lane, warp = tid & 31, tid >> 5
            g, t = lane >> 2, lane & 3
            r_lo = m0 + 16 * warp + g
            for j in range(8):
                out[r_lo, n0 + 8 * j + 2 * t], out[r_lo, n0 + 8 * j + 2 * t + 1] = acc[tid, 4 * j], acc[tid, 4 * j + 1]
                out[r_lo + 8, n0 + 8 * j + 2 * t], out[r_lo + 8, n0 + 8 * j + 2 * t + 1] = acc[tid, 4 * j + 2], acc[tid, 4 * j + 3]
    return out


def bf16_case(m, n, k, seed):
    rng = np.random.default_rng(seed)
    a, b = bf16_bits(rng.normal(size=(m, k))), bf16_bits(rng.normal(size=(n, k)))
    reference = bf16_value(a).astype(np.float64) @ bf16_value(b).astype(np.float64).T
    return words(a), words(b), reference


def fp8_case(m, n, k, seed):
    rng = np.random.default_rng(seed)
    a, b = e4m3_bits(rng.normal(size=(m, k)) * 4), e4m3_bits(rng.normal(size=(n, k)) * 4)
    a_scale = rng.uniform(0.25, 2.0, size=(m, k // 128)).astype(np.float32)
    b_scale = rng.uniform(0.25, 2.0, size=(-(-n // 128), k // 128)).astype(np.float32)
    table = e4m3_table().astype(np.float64)
    av, bv = table[a], table[b]
    reference = np.zeros((m, n))
    for kb in range(k // 128):
        block = av[:, kb * 128:(kb + 1) * 128] @ bv[:, kb * 128:(kb + 1) * 128].T
        reference += block * a_scale[:, kb:kb + 1] * np.repeat(b_scale[:, kb], 128)[:n][None, :]
    return words(a), words(b), a_scale, b_scale, reference


# --- model tests ------------------------------------------------------------------------------------

def test_e4m3_table_matches_known_codes():
    table = e4m3_table()
    assert table[0x00] == 0 and table[0x38] == 1.0 and table[0x7E] == 448.0 and table[0x01] == 2.0 ** -9
    assert table[0xB8] == -1.0 and np.isnan(table[0x7F]) and np.isnan(table[0xFF])


@pytest.mark.parametrize('m,n,k', [(16, 64, 16), (32, 128, 48)])
def test_mma_probe_index_math_matches_numpy(m, n, k):
    a, b, reference = bf16_case(m, n, k, m + n + k)
    np.testing.assert_allclose(mma_probe_model(a, b, m, n, k), reference, rtol=1e-12, atol=1e-12)


def test_wgmma_descriptor_reads_the_staged_block():
    # Each (row, k byte) the descriptor addresses is the byte stage_block put there, and the
    # layout puts the 2048 words of a block in 2048 distinct places.
    src_bytes = (np.arange(64 * 128) % 251).astype(np.uint8).reshape(64, 128)
    smem = np.zeros(2048, np.int32)
    stage_block(smem, src_bytes.view(np.int32), 0, 32, 0)
    for step in range(4):
        got = read_operand(smem.view(np.uint8), 0x400, kmajor_desc(0x400), step, 32)
        np.testing.assert_array_equal(got, src_bytes[:, 32 * step:32 * step + 32])
    assert len({((r >> 3) * 8 + (w >> 2)) * 32 + (r & 7) * 4 + (w & 3) for r in range(64) for w in range(32)}) == 2048


@pytest.mark.parametrize('m,n,k', [(64, 64, 64), (64, 128, 128)])
def test_wgmma_bf16_probe_index_math_matches_numpy(m, n, k):
    a, b, reference = bf16_case(m, n, k, m * 3 + n + k)
    np.testing.assert_allclose(wgmma_probe_model(a, b, m, n, k), reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('m,n,k', [(64, 64, 128), (64, 192, 256)])
def test_wgmma_fp8_probe_index_math_matches_numpy(m, n, k):
    a, b, a_scale, b_scale, reference = fp8_case(m, n, k, m + n * 5 + k)
    got = wgmma_probe_model(a, b, m, n, k, fp8=True, a_scale=a_scale, b_scale=b_scale)
    # The model multiplies the two scales in f32, as the probe does; the reference in f64.
    np.testing.assert_allclose(got, reference, rtol=1e-6, atol=1e-6 * np.abs(reference).max())


# --- hardware tests ---------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def context():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as e:
        pytest.skip('real GPU unavailable: ' + str(e))
    try:
        yield ctx
    finally:
        ctx.close()


def probe(context, name, capability=None):
    from gpu.spawn.launch import SpawnKernel
    if capability and context.compute_capability != capability:
        pytest.skip(f'{name} is sm_{capability[0]}{capability[1]}a code; this GPU is {context.compute_capability}')
    path = BUILD / f'{name}.ptx'
    if not path.exists():
        pytest.skip(f'{path} not built: run gpu/spawn/probes/build.sh')
    return SpawnKernel(context, path)


def upload(context, array):
    return context.tensor(array.shape, str(array.dtype), array.tobytes())


def download(tensor):
    return np.frombuffer(tensor.download(), dtype=tensor.dtype).reshape(tensor.shape)


def run_bf16(context, kernel, m, n, k, seed):
    a, b, reference = bf16_case(m, n, k, seed)
    out = context.tensor((m * n,))
    kernel.launch(out=out, a=upload(context, a.reshape(-1)), b=upload(context, b.reshape(-1)),
                  rows_m=m, rows_n=n, cols=k)
    context.synchronize()
    # bf16 products are exact in f32; only the order of the f32 sums differs from numpy's.
    np.testing.assert_allclose(download(out).reshape(m, n), reference, rtol=1e-4, atol=1e-3 * np.sqrt(k))


@pytest.mark.parametrize('m,n,k', [(16, 64, 16), (64, 128, 4096), (48, 192, 1040)])
def test_mma_bf16_sm80_matches_numpy(context, m, n, k):
    run_bf16(context, probe(context, 'mma_bf16_sm80'), m, n, k, m + n + k)


@pytest.mark.parametrize('m,n,k', [(64, 64, 64), (128, 192, 4096)])
def test_wgmma_bf16_sm90a_matches_numpy(context, m, n, k):
    run_bf16(context, probe(context, 'wgmma_bf16_sm90a', (9, 0)), m, n, k, m * 3 + n + k)


@pytest.mark.parametrize('name', ['wgmma_fp8_sm90a', 'wgmma_fp8_fn_sm90a'])
@pytest.mark.parametrize('m,n,k', [(64, 64, 128), (128, 192, 4096)])
def test_wgmma_fp8_sm90a_matches_numpy(context, name, m, n, k):
    kernel = probe(context, name, (9, 0))
    a, b, a_scale, b_scale, reference = fp8_case(m, n, k, m + n * 5 + k)
    out = context.tensor((m * n,))
    kernel.launch(out=out, a=upload(context, a.reshape(-1)), b=upload(context, b.reshape(-1)),
                  a_scale=upload(context, a_scale.reshape(-1)), b_scale=upload(context, b_scale.reshape(-1)),
                  rows_m=m, rows_n=n, cols=k)
    context.synchronize()
    np.testing.assert_allclose(download(out).reshape(m, n), reference, rtol=1e-4, atol=1e-4 * np.abs(reference).max())
