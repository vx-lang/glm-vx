"""The gpu/spawn kernels, run on the CPU by Vx's own runtime (gpu/spawn/host.py), against NumPy.

With no CUDA device, Vx runs each `across` region in order on the host, so these tests need no
GPU: they check what the kernels compute (indexing, guards, tails, padding, routing, ties) on
any machine. Where a kernel's order is that of the hand kernel (lane-strided partials, then the
warp tree 16, 8, 4, 2, 1 and the block's second stage, all in float32 without FMA), the
reference follows it and must match bit for bit; that pins the order the reduction library of
Vx #260 must keep. Kernels with exp (libm on the host, libdevice on the GPU) compare closely
instead. Parity with the hand kernels on a GPU is gpu/test_spawn_hardware.py's.

Skipped unless gpu/spawn/build.sh made the libraries (VX_DISPATCH_LIB set).
"""
from pathlib import Path

import numpy as np
import pytest

BUILD = Path(__file__).parent / 'build' / 'spawn'
F = np.float32
CLOSE = dict(rtol=2e-6, atol=1e-37)  # exp: libm expf on the host, within an ulp or two of NumPy's


@pytest.fixture(scope='module')
def run():
    if not (BUILD / 'libadd.so').exists():
        pytest.skip('no kernel libraries: run gpu/spawn/build.sh with VX_DISPATCH_LIB set')
    from gpu.spawn.host import Host
    host = Host()
    try:
        yield lambda name, **values: host(BUILD / f'{name}.ptx', **values)
    finally:
        host.close()


def bits_equal(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, F).view(np.uint32), np.asarray(expected, F).view(np.uint32))


# --- The hand kernels' reduction order ----------------------------------------------------

def warp_tree(v):
    """vxk_warp_sum over the last axis (32): offsets 16, 8, 4, 2, 1; the total is element 0."""
    v = np.array(v, F)
    for o in (16, 8, 4, 2, 1):
        v[..., :o] = v[..., :o] + v[..., o:2 * o]
    return v[..., 0]


def warp_max_tree(v):
    v = np.array(v, F)
    for o in (16, 8, 4, 2, 1):
        a, b = v[..., :o], v[..., o:2 * o]
        v[..., :o] = np.where(b > a, b, a)
    return v[..., 0]


def block_sum(partials, threads):
    """vxk_block_sum: partials [..., 1024], warp totals past threads / 32 pad with 0."""
    warps = warp_tree(partials.reshape(*partials.shape[:-1], 32, 32))
    warps[..., threads // 32:] = 0
    return warp_tree(warps)


def block_max(partials, threads):
    warps = warp_max_tree(partials.reshape(*partials.shape[:-1], 32, 32))
    warps[..., threads // 32:] = warps[..., :1]
    return warp_max_tree(warps)


def strided_sums(terms, width):
    """Lane l's float32 running sum of terms[..., l], terms[..., l + width], .. (zero-padded lanes
    past the end add nothing)."""
    n = terms.shape[-1]
    out = np.zeros((*terms.shape[:-1], width), F)
    for start in range(0, n, width):
        chunk = terms[..., start:start + width]
        out[..., :chunk.shape[-1]] = out[..., :chunk.shape[-1]] + chunk
    return out


def block_partials(terms, threads):
    """Thread t's partial over d = t, t + threads, ..; threads past `threads` hold 0."""
    out = np.zeros((*terms.shape[:-1], 1024), F)
    out[..., :threads] = strided_sums(terms, threads)
    return out


# --- Decoders -----------------------------------------------------------------------------

def bf16(codes):
    return (codes.astype(np.uint32) << 16).view(F)


def fp8(codes):
    codes = codes.astype(np.int32)
    e, m = (codes >> 3) & 15, codes & 7
    magnitude = np.where(e > 0, (8 + m).astype(F) * np.exp2(np.maximum(e - 1, 0)).astype(F) / F(512), m.astype(F) / F(512))
    magnitude = np.where((e == 15) & (m == 7), F(np.nan), magnitude).astype(F)
    return np.where(codes & 128, -magnitude, magnitude).astype(F)


def nibbles(packed, high):
    v = ((packed.astype(np.int32) >> (4 if high else 0)) & 15)
    return np.where(v >= 8, v - 16, v).astype(F)


def scales16(raw, kind):
    return bf16(raw) if kind else raw.view(np.float16).astype(F)


# --- Elementwise --------------------------------------------------------------------------

@pytest.mark.parametrize('n', [1, 1000, 100003])
def test_elementwise_kernels(run, n):
    rng = np.random.default_rng(n)
    a, b = rng.normal(size=n).astype(F), rng.normal(size=n).astype(F)
    bits_equal(run('add', out=np.zeros(n, F), a=a, b=b, n=n)['out'], a + b)
    bits_equal(run('copy', out=np.zeros(n, F), x=a, n=n)['out'], a)
    bits_equal(run('scale', out=np.zeros(n, F), x=a, alpha=0.37, n=n)['out'], a * F(0.37))
    weights = rng.normal(size=8).astype(F)
    bits_equal(run('scaled_add', out=b.copy(), x=a, weights=weights, slot=5, n=n)['out'], b + a * weights[5])


@pytest.mark.parametrize('heads,dim,x_stride,out_stride', [(1, 2, 2, 2), (16, 64, 64, 64), (8, 64, 192, 128)])
def test_rope(run, heads, dim, x_stride, out_stride):
    rng = np.random.default_rng(heads + dim + x_stride)
    x = rng.normal(size=heads * x_stride).astype(F)
    angles = rng.uniform(0, 6.3, size=dim // 2)
    cos, sin = np.cos(angles).astype(F), np.sin(angles).astype(F)
    fill = rng.normal(size=(heads, out_stride)).astype(F)
    xs = x.reshape(heads, x_stride)
    a, b = xs[:, 0:dim:2], xs[:, 1:dim:2]
    expected = fill.copy()
    expected[:, :dim // 2] = a * cos - b * sin
    expected[:, dim // 2:dim] = b * cos + a * sin
    got = run('rope_strided', out=fill.copy(), x=x, cosine=cos, sine=sin, heads=heads, dim=dim,
              x_stride=x_stride, out_stride=out_stride)['out']
    bits_equal(got, expected)  # the gaps between heads stay as they were
    if x_stride == dim:
        got = run('rope_glm', out=np.zeros((heads, dim), F), x=x, cosine=cos, sine=sin, heads=heads, dim=dim)['out']
        bits_equal(got, expected[:, :dim])


@pytest.mark.parametrize('n', [1, 1000, 12288])
def test_swiglu(run, n):
    rng = np.random.default_rng(n + 9)
    gate, up = (rng.normal(size=n) * 6).astype(F), rng.normal(size=n).astype(F)
    gate[:min(n, 4)] = [0.0, -0.0, 20.0, -20.0][:min(n, 4)]
    z = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, F(1) / (F(1) + z), z / (F(1) + z)).astype(F)
    np.testing.assert_allclose(run('swiglu', out=np.zeros(n, F), gate=gate, up=up, n=n)['out'], gate * sigmoid * up, **CLOSE)


# --- Warp per row -------------------------------------------------------------------------

@pytest.mark.parametrize('rows,cols', [(1, 1), (7, 131), (64, 6144)])
def test_matvec(run, rows, cols):
    rng = np.random.default_rng(rows * cols)
    w, x = rng.normal(size=(rows, cols)).astype(F), rng.normal(size=cols).astype(F)
    got = run('matvec', out=np.zeros(rows, F), weight=w, x=x, rows=rows, cols=cols)['out']
    bits_equal(got, warp_tree(strided_sums(w * x, 32)))
    np.testing.assert_allclose(got, w.astype('f8') @ x, rtol=1e-4, atol=1e-4)


def test_bf16_decode_is_exact_for_every_encoding(run):
    codes = np.arange(65536, dtype=np.uint16)
    got = run('bf16_matvec', out=np.zeros(65536, F), weight=codes, x=np.ones(1, F), rows=65536, cols=1)['out']
    with np.errstate(invalid='ignore'):
        expected = bf16(codes) + F(0)  # the row sum starts at +0.0, so -0.0 comes out as +0.0
    finite = ~np.isnan(expected)
    bits_equal(got[finite], expected[finite])
    assert np.isnan(got[~finite]).all()


@pytest.mark.parametrize('rows,cols', [(5, 131), (96, 6144)])
def test_bf16_matvec(run, rows, cols):
    rng = np.random.default_rng(rows + cols)
    w = (rng.normal(size=(rows, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=cols).astype(F)
    got = run('bf16_matvec', out=np.zeros(rows, F), weight=w, x=x, rows=rows, cols=cols)['out']
    bits_equal(got, warp_tree(strided_sums(bf16(w) * x, 32)))


def int4_lane_sums(packed, scales, x, k, kind):
    """The int4 kernels' lane partials: per 128-column group, lane l takes bytes g*64 + l and
    g*64 + 32 + l (two columns each) into `part`, then sum += part * scale. packed [r, bytes]
    and x [r, k] (one x row per output row)."""
    rows, groups = packed.shape[0], -(-k // 128)
    lane = np.arange(32)
    total = np.zeros((rows, 32), F)
    for g in range(groups):
        part = np.zeros((rows, 32), F)
        for step in range(2):
            byte = g * 64 + step * 32 + lane
            col = byte * 2
            ok = col < k
            b = packed[:, np.minimum(byte, packed.shape[1] - 1)]
            lo = nibbles(b, 0) * x[:, np.minimum(col, k - 1)]
            part = np.where(ok, part + lo, part).astype(F)
            hi = nibbles(b, 1) * x[:, np.minimum(col + 1, k - 1)]
            part = np.where(ok & (col + 1 < k), part + hi, part).astype(F)
        total = total + part * scales16(scales[:, g], kind)[:, None]
    return total


@pytest.mark.parametrize('rows,cols,scale_kind', [(5, 131, 0), (64, 6144, 1), (3, 257, 1)])
def test_int4_matvec(run, rows, cols, scale_kind):
    rng = np.random.default_rng(rows * 7 + cols)
    groups, row_bytes = -(-cols // 128), -(-cols // 2)
    packed = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    if cols % 2:
        packed[:, -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=(rows, groups)).astype(F)
    scales = (magnitudes.view(np.uint32) >> 16).astype(np.uint16) if scale_kind else magnitudes.astype(np.float16).view(np.uint16)
    x = rng.normal(size=cols).astype(F)
    got = run('int4_matvec', out=np.zeros(rows, F), packed=packed, scales=scales, x=x, rows=rows, cols=cols,
              scale_kind=scale_kind)['out']
    bits_equal(got, warp_tree(int4_lane_sums(packed, scales, np.broadcast_to(x, (rows, cols)), cols, scale_kind)))


def lane_run_order(terms, run):
    """Reorder columns so that strided_sums(.., 32) adds lane l's run of `run` consecutive columns
    of each 32 * run-column tile, tile by tile (zeros pad the last tile)."""
    rows, cols = terms.shape
    tiles = -(-cols // (32 * run))
    padded = np.zeros((rows, tiles * 32 * run), F)
    padded[:, :cols] = terms
    return padded.reshape(rows, tiles, 32, run).transpose(0, 1, 3, 2).reshape(rows, -1)


def bf16_matvec4_case(rows, cols, seed):
    """(weight words, x, expected) for bf16_matvec4: its order, bit for bit."""
    rng = np.random.default_rng(seed)
    w = (rng.normal(size=(rows, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=cols).astype(F)
    return w.view(np.uint32).reshape(-1), x, warp_tree(strided_sums(lane_run_order(bf16(w) * x, 8), 32))


# Eight weights a lane step (a u32 view of the bfloat16 bits, #30): (5, 136) has a partial tile,
# 6144 and 16384 are GLM-5.3's widths, 520 a tile and a few lanes more.
@pytest.mark.parametrize('rows,cols', [(5, 136), (96, 6144), (3, 16384), (7, 520)])
def test_bf16_matvec4(run, rows, cols):
    words, x, expected = bf16_matvec4_case(rows, cols, rows + cols + 4)
    bits_equal(run('bf16_matvec4', out=np.zeros(rows, F), weight=words, x=x, rows=rows, cols=cols)['out'], expected)


def int4_matvec4_lane_sums(packed, scales, x, kind):
    """int4_matvec4's lane partials: lane l of tile t takes columns t * 1024 + 32l .. + 31 (one
    group) into `part` in order, then sum += part * scale."""
    rows, cols = packed.shape[0], packed.shape[1] * 2
    codes = np.empty((rows, cols), F)
    codes[:, 0::2], codes[:, 1::2] = nibbles(packed, 0), nibbles(packed, 1)
    lane = np.arange(32)
    total = np.zeros((rows, 32), F)
    for t in range(-(-cols // 1024)):
        start = t * 1024 + 32 * lane
        valid = start < cols
        part = np.zeros((rows, 32), F)
        for j in range(32):
            col = np.minimum(start + j, cols - 1)
            part = part + codes[:, col] * x[col]
        scale = scales16(scales[:, np.minimum(start // 128, scales.shape[1] - 1)], kind)
        total = np.where(valid, total + part * scale, total).astype(F)
    return total


def test_half_decode_is_exact_for_every_encoding(run):
    # int4_matvec4 with scale_kind 0: row r has code 1 in column 0, x = 1 and scale bits r, so its
    # sum is +0 + 1 * half(r), device.vx's half_decode! of every binary16 encoding. Lanes 1 to 3
    # share the group and add 0 * half(r), NaN for the two infinities, as NaN encodings give NaN.
    rows, cols = 65536, 128
    packed = np.zeros((rows, cols // 2), np.uint8)
    packed[:, 0] = 1
    scales = np.arange(rows, dtype=np.uint16).reshape(rows, 1)
    x = np.zeros(cols, F)
    x[0] = 1
    got = run('int4_matvec4', out=np.zeros(rows, F), packed=packed.view(np.uint32).reshape(-1), scales=scales.reshape(-1),
              x=x, rows=rows, cols=cols, scale_kind=0)['out']
    with np.errstate(invalid='ignore'):
        expected = scales.reshape(-1).view(np.float16).astype(F) + F(0)
    finite = np.isfinite(expected)
    bits_equal(got[finite], expected[finite])
    assert np.isnan(got[~finite]).all()


def int4_matvec4_case(rows, cols, kind, seed):
    """(packed bytes [rows, cols / 2], scale bits [rows, groups], x, expected) for int4_matvec4."""
    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
    magnitudes = rng.uniform(0.001, 0.1, size=(rows, cols // 128)).astype(F)
    scales = (magnitudes.view(np.uint32) >> 16).astype(np.uint16) if kind else magnitudes.astype(np.float16).view(np.uint16)
    x = rng.normal(size=cols).astype(F)
    return packed, scales, x, warp_tree(int4_matvec4_lane_sums(packed, scales, x, kind))


# 32 codes a lane step (a u32 view of the packed bytes, #30): 128 and 512 are partial tiles, 1152
# a tile and a group, 6144 and 16384 GLM-5.3's widths; both scale kinds.
@pytest.mark.parametrize('rows,cols,scale_kind', [(5, 128, 0), (3, 512, 1), (9, 1152, 1), (64, 6144, 1), (4, 16384, 0)])
def test_int4_matvec4(run, rows, cols, scale_kind):
    packed, scales, x, expected = int4_matvec4_case(rows, cols, scale_kind, rows * 5 + cols)
    got = run('int4_matvec4', out=np.zeros(rows, F), packed=packed.view(np.uint32).reshape(-1), scales=scales.reshape(-1),
              x=x, rows=rows, cols=cols, scale_kind=scale_kind)['out']
    bits_equal(got, expected)
    per_col = scales16(scales, scale_kind).repeat(64, axis=1).astype('f8')
    reference = (nibbles(packed, 0) * per_col) @ x[0::2].astype('f8') + (nibbles(packed, 1) * per_col) @ x[1::2].astype('f8')
    np.testing.assert_allclose(got, reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('rows,cols', [(5, 131), (300, 1000)])
def test_fp8_matvec(run, rows, cols):
    rng = np.random.default_rng(rows * 11 + cols)
    weight = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0
    scale_cols = -(-cols // 128)
    scales = rng.uniform(0.001, 0.1, size=(-(-rows // 128), scale_cols)).astype(F)
    x = rng.normal(size=cols).astype(F)
    got = run('fp8_matvec', out=np.zeros(rows, F), weight=weight, scales=scales, x=x, rows=rows, cols=cols)['out']
    per_col = scales[np.arange(rows)[:, None] // 128, np.arange(cols)[None, :] // 128]
    # Columns tile * 128 + step * 32 + lane run tile by tile, step by step: lane l's order is l, l + 32, ..
    bits_equal(got, warp_tree(strided_sums((fp8(weight) * per_col) * x, 32)))


def fp8_matvec4_case(rows, cols, seed, scales=None, nan_rows=()):
    """scales: a list cycled over the scale blocks in place of uniform(0.001, 0.1); nan_rows: rows
    that get one NaN code each (the others have none)."""
    rng = np.random.default_rng(seed)
    weight = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0
    for r in nan_rows:
        weight[r, rng.integers(cols)] = rng.choice([0x7F, 0xFF])
    shape = (-(-rows // 128), -(-cols // 128))
    if scales is None:
        scales = rng.uniform(0.001, 0.1, size=shape).astype(F)
    else:
        scales = np.resize(np.asarray(scales, F), shape).astype(F)
    x = rng.normal(size=cols).astype(F)
    per_col = scales[np.arange(rows)[:, None] // 128, np.arange(cols)[None, :] // 128]
    # Lane l takes columns tile * 512 + 16l .. + 15 of each tile in turn.
    expected = warp_tree(strided_sums(lane_run_order((fp8(weight) * per_col) * x, 16), 32))
    return weight, scales, x, expected


# Sixteen codes a lane step (a u32 view of the bytes, cols % 16 == 0): (5, 144) has a partial
# tile, (300, 1008) one that ends inside a scale block, 576 is kv_a's width, and 6144 and 16384
# are GLM-5.3's q_a/MLP and o_proj widths.
@pytest.mark.parametrize('rows,cols', [(5, 144), (300, 1008), (130, 6144), (4, 16384), (3, 640), (2, 576)])
def test_fp8_matvec4(run, rows, cols):
    weight, scales, x, expected = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    got = run('fp8_matvec4', out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1), scales=scales.reshape(-1),
              x=x, rows=rows, cols=cols)['out']
    bits_equal(got, expected)


# The fast path (no NaN select, scale * 2^120 folded, #30) and the exact one must give the same
# bits: scales at and around 2^8 (|scale| >= 256 takes the exact path), subnormal and zero scales,
# negative ones, and NaN codes (the exact path for that lane step; the row's sum is NaN).
EDGE_SCALES = [300.0, 256.0, 255.99, -300.0, 2.0 ** -130, 1e-3, 0.0, -0.002, 2.0 ** 7]
NAN_ROWS = (1, 4, 129)


@pytest.mark.parametrize('rows,cols', [(5, 1008), (130, 1152)])
def test_fp8_matvec4_fast_and_exact_paths_agree(run, rows, cols):
    nan_rows = [r for r in NAN_ROWS if r < rows]
    weight, scales, x, expected = fp8_matvec4_case(rows, cols, rows + cols, EDGE_SCALES, nan_rows)
    words = weight.view(np.uint32).reshape(-1)
    got = run('fp8_matvec4', out=np.zeros(rows, F), weight=words, scales=scales.reshape(-1), x=x, rows=rows,
              cols=cols)['out']
    nan = np.isin(np.arange(rows), nan_rows)
    assert np.isnan(got[nan]).all() and np.isnan(expected[nan]).all()
    bits_equal(got[~nan], expected[~nan])
    xs = np.random.default_rng(cols).normal(size=(4, cols)).astype(F)
    xs[0] = x
    many = run('fp8_matvec4_rows', out=np.zeros((4, rows), F), weight=words, scales=scales.reshape(-1),
               x=xs.reshape(-1), rows=rows, cols=cols, m=4)['out']
    assert np.isnan(many[:, nan]).all()
    bits_equal(many[0][~nan], expected[~nan])
    for v in range(1, 4):
        single = run('fp8_matvec4', out=np.zeros(rows, F), weight=words, scales=scales.reshape(-1), x=xs[v],
                     rows=rows, cols=cols)['out']
        bits_equal(many[v][~nan], single[~nan])


# fp8_matvec4_rows (MTP verification, #16): m vectors against one read of the weight. Each row of
# out must be fp8_matvec4's result for its vector, bit for bit, so that verifying k + 1 tokens at
# once gives the logits that decoding them one by one gives; rows past m stay as they were.
@pytest.mark.parametrize('m', [1, 2, 3, 4])
@pytest.mark.parametrize('rows,cols', [(5, 144), (300, 1008), (130, 6144), (2, 576)])
def test_fp8_matvec4_rows_is_fp8_matvec4_per_vector(run, rows, cols, m):
    weight, scales, _, _ = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    rng = np.random.default_rng(rows + cols + m)
    xs = rng.normal(size=(4, cols)).astype(F)
    out = np.full((4, rows), 7.0, F)
    got = run('fp8_matvec4_rows', out=out, weight=weight.view(np.uint32).reshape(-1), scales=scales.reshape(-1),
              x=xs.reshape(-1), rows=rows, cols=cols, m=m)['out']
    per_col = scales[np.arange(rows)[:, None] // 128, np.arange(cols)[None, :] // 128]
    for v in range(4):
        if v < m:
            bits_equal(got[v], warp_tree(strided_sums(lane_run_order((fp8(weight) * per_col) * xs[v], 16), 32)))
            single = run('fp8_matvec4', out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1),
                         scales=scales.reshape(-1), x=xs[v], rows=rows, cols=cols)['out']
            bits_equal(got[v], single)
        else:
            bits_equal(got[v], np.full(rows, 7.0, F))


@pytest.mark.parametrize('m,cols', [(0, 144), (5, 144), (2, 100)])
def test_fp8_matvec4_rows_refuses_other_counts_and_widths(run, m, cols):
    weight = np.zeros((3, cols), np.uint8)
    got = run('fp8_matvec4_rows', out=np.full((4, 3), 7.0, F), weight=np.zeros(3 * cols // 4 + 1, np.uint32),
              scales=np.ones(2, F), x=np.ones(4 * cols, F), rows=3, cols=cols, m=m)['out']
    want = np.full((4, 3), 7.0, F)
    want[:min(max(m, 0), 4)] = 0.0  # m > 4, or a width that is not a multiple of 16: zeros; m < 1: nothing
    bits_equal(got, want)


def test_fp8_matvec4_decodes_every_code(run):
    # Row r holds the r-th code at column r (each code but NaN, then two zeros), scale 1, x 1:
    # every row's sum is its code's decoded value.
    codes = np.array([c for c in range(256) if c & 0x7F != 0x7F] + [0, 0], np.uint8)  # 256 columns
    rows, cols = 256, 256
    weight = np.zeros((rows, cols), np.uint8)
    weight[np.arange(rows), np.arange(cols)] = codes
    x = np.ones(cols, F)
    got = run('fp8_matvec4', out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1),
              scales=np.ones(4, F), x=x, rows=rows, cols=cols)['out']
    bits_equal(got, F(0) + fp8(codes))  # a sum starts at +0, so code 0x80 (-0) comes out +0


# fp8_matvec_fma and fp8_matvec_fma_tiles (#30): one fma a code, two rows a warp, so not fp8_matvec4's
# bits (the two differ from each other in order too). Rows must be within
# 2e-6 of the largest |output| of the float64 sum. Both kernels measure 1e-7 to 5e-7 on the host at
# these shapes, at any activation size from 1 down to 1e-9 (the kernel reads x at 2^40 so that no step
# falls to denormals); an error of 1e-4 fails.
FMA_TOLERANCE = 2e-6


def fp8_reference(weight, scales, x):
    """(float64 row sums, rows whose terms are all finite)."""
    rows, cols = weight.shape
    per_col = scales[np.arange(rows)[:, None] // 128, np.arange(cols)[None, :] // 128].astype(np.float64)
    terms = fp8(weight).astype(np.float64) * per_col * x.astype(np.float64)
    return terms.sum(axis=1), np.isfinite(terms).all(axis=1)


def assert_fma_close(got, weight, scales, x, rows=None):
    want, finite = fp8_reference(weight, scales, x)
    keep = finite if rows is None else finite & rows
    err = np.abs(np.asarray(got, np.float64) - want)[keep]
    worst = float(err.max() / np.abs(want[keep]).max())
    assert worst <= FMA_TOLERANCE, f'{worst:.1e} of the largest output'


# fp8_matvec_cvt_tiles_ref (glm-vx #38): the codes at their true values, the integer-decode build of the
# kernel whose sm_90 build uses Hopper's conversion (the same bits; gpu/test_spawn_hardware.py).
FMA_KERNELS = ['fp8_matvec_fma', 'fp8_matvec_fma_tiles', 'fp8_matvec_cvt_tiles_ref']


# Odd row counts leave a warp one row (1, 17) or a block a partial group (6); widths below 512 or not
# a multiple of it end in a partial tile (144, 576, 640, 1008).
@pytest.mark.parametrize('kernel', FMA_KERNELS)
@pytest.mark.parametrize('rows,cols', [(5, 144), (300, 1008), (130, 6144), (4, 16384), (3, 640), (2, 576), (8, 2048),
                                       (1, 576), (6, 576), (17, 1008)])
def test_fp8_matvec_fma(run, kernel, rows, cols):
    weight, scales, x, _ = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    words = weight.view(np.uint32).reshape(-1)
    got = run(kernel, out=np.zeros(rows, F), weight=words, scales=scales.reshape(-1), x=x, rows=rows,
              cols=cols)['out']
    assert_fma_close(got, weight, scales, x)
    four = run('fp8_matvec4', out=np.zeros(rows, F), weight=words, scales=scales.reshape(-1), x=x, rows=rows,
               cols=cols)['out']
    assert_fma_close(four, weight, scales, x)  # fp8_matvec4 is within it too


# Steps off the fma path: scales of 256 and more, subnormal and zero scales, and NaN codes (that
# step goes code by code, and the row is NaN). (130, 2304) puts them in both halves of a turn.
@pytest.mark.parametrize('kernel', FMA_KERNELS)
@pytest.mark.parametrize('rows,cols', [(5, 1008), (130, 1152), (130, 2304)])
def test_fp8_matvec_fma_exact_steps(run, kernel, rows, cols):
    nan_rows = [r for r in NAN_ROWS if r < rows]
    weight, scales, x, _ = fp8_matvec4_case(rows, cols, rows + cols, EDGE_SCALES, nan_rows)
    got = run(kernel, out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1),
              scales=scales.reshape(-1), x=x, rows=rows, cols=cols)['out']
    nan = np.isin(np.arange(rows), nan_rows)
    assert np.isnan(got[nan]).all()
    assert_fma_close(got, weight, scales, x, rows=~nan)


@pytest.mark.parametrize('kernel', FMA_KERNELS)
@pytest.mark.parametrize('size', [1e-5, 1e-9])
def test_fp8_matvec_fma_keeps_its_precision_for_small_activations(run, kernel, size):
    # Read at 2^-120 scale, as from_bits gives the codes, these steps would be denormals (7e-5
    # relative error at 1e-7); x at 2^40 keeps them normal.
    rows, cols = 130, 6144
    weight, scales, x, _ = fp8_matvec4_case(rows, cols, 7)
    x = (x * F(size)).astype(F)
    got = run(kernel, out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1),
              scales=scales.reshape(-1), x=x, rows=rows, cols=cols)['out']
    assert np.abs(got).max() > 0
    assert_fma_close(got, weight, scales, x)


@pytest.mark.parametrize('kernel', FMA_KERNELS)
def test_fp8_matvec_fma_decodes_every_code(run, kernel):
    # As test_fp8_matvec4_decodes_every_code: one code a row, scale 1, x 1. Every product and
    # rounding is exact here (code * 2^-80 is a normal float32), so the bits are the code's.
    codes = np.array([c for c in range(256) if c & 0x7F != 0x7F] + [0, 0], np.uint8)
    rows, cols = 256, 256
    weight = np.zeros((rows, cols), np.uint8)
    weight[np.arange(rows), np.arange(cols)] = codes
    got = run(kernel, out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1),
              scales=np.ones(4, F), x=np.ones(cols, F), rows=rows, cols=cols)['out']
    bits_equal(got, F(0) + fp8(codes))


# fp8_matvec_heads_cvt_ref (#42): with period = group and offset 0, fp8_matvec_cvt_tiles for a
# block-diagonal matvec, each group of rows with its own x. Every row is the dense kernel's row on its
# group's x, bit for bit. Shapes: GLM-5.3's two
# MLA absorbs (64 heads cut to 3), and a group of 16 that splits scale blocks across groups.
@pytest.mark.parametrize('heads,group,cols,x_stride', [(3, 512, 128, 192), (3, 128, 512, 512), (2, 256, 256, 300)])
def test_fp8_matvec_heads_cvt_is_cvt_tiles_on_each_group(run, heads, group, cols, x_stride):
    rows = heads * group
    weight, scales, _, _ = fp8_matvec4_case(rows, cols, rows + cols + x_stride)
    x = np.random.default_rng(rows + cols).normal(size=heads * x_stride).astype(F)
    got = run('fp8_matvec_heads_cvt_ref', out=np.full(rows, 7, F), weight=weight.view(np.uint32).reshape(-1),
              scales=scales.reshape(-1), x=x, rows=rows, cols=cols, group=group, x_stride=x_stride,
              period=group, offset=0)['out']
    for g in range(heads):
        part = weight[g * group:(g + 1) * group]
        want = run('fp8_matvec_cvt_tiles_ref', out=np.zeros(group, F), weight=np.ascontiguousarray(part).view(np.uint32).reshape(-1),
                   scales=np.ascontiguousarray(scales[g * group // 128:(g + 1) * group // 128]).reshape(-1),
                   x=np.ascontiguousarray(x[g * x_stride:g * x_stride + cols]), rows=group, cols=cols)['out']
        bits_equal(got[g * group:(g + 1) * group], want)


def test_fp8_matvec_heads_cvt_small_groups_and_bad_shapes(run):
    heads, group, cols, x_stride = 5, 16, 144, 150
    rows = heads * group
    weight, scales, _, _ = fp8_matvec4_case(rows, cols, 11)
    x = np.random.default_rng(3).normal(size=heads * x_stride).astype(F)
    got = run('fp8_matvec_heads_cvt_ref', out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1),
              scales=scales.reshape(-1), x=x, rows=rows, cols=cols, group=group, x_stride=x_stride,
              period=group, offset=0)['out']
    dense = fp8(weight).astype(np.float64) * np.repeat(np.repeat(scales.astype(np.float64), 128, 0), 128, 1)[:rows, :cols]
    want = np.concatenate([dense[g * group:(g + 1) * group] @ x[g * x_stride:g * x_stride + cols].astype(np.float64)
                           for g in range(heads)])
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()
    # group % 4, zero, rows % group, stride, a period shorter than its group, an offset below zero
    for bad in (dict(group=10), dict(group=0), dict(group=32), dict(x_stride=-1), dict(period=14), dict(offset=-1),
                dict(period=16, offset=2)):
        values = {**dict(rows=rows, cols=cols, group=group, x_stride=x_stride, period=group, offset=0), **bad}
        out = run('fp8_matvec_heads_cvt_ref', out=np.full(rows, 7, F), weight=weight.view(np.uint32).reshape(-1),
                  scales=scales.reshape(-1), x=x, **values)['out']
        bits_equal(out, np.full(rows, 7, F))


def fp8_dense(weight, scales):
    rows, cols = weight.shape
    return fp8(weight).astype(np.float64) * np.repeat(np.repeat(scales.astype(np.float64), 128, 0), 128, 1)[:rows, :cols]


# GLM's kv_b_proj holds nope + value rows a head (GLM-5.3: 192 + 256), so its 128-row blocks of scales
# straddle heads. The value absorb reads the stored matrix in place: output row (head g, i) is stored row
# g * period + offset + i, with that row's own scales. Here heads of 64 + 96 rows start 0, 32 and 64 rows
# into a block.
@pytest.mark.parametrize('cols', [128, 384])
def test_fp8_matvec_heads_cvt_reads_value_rows_in_place(run, cols):
    heads, nope, value = 3, 64, 96
    period = nope + value
    weight, scales, _, _ = fp8_matvec4_case(heads * period, cols, cols + 5)
    x = np.random.default_rng(cols).normal(size=heads * cols).astype(F)
    got = run('fp8_matvec_heads_cvt_ref', out=np.full(heads * value, 7, F), weight=weight.view(np.uint32).reshape(-1),
              scales=scales.reshape(-1), x=x, rows=heads * value, cols=cols, group=value, x_stride=cols,
              period=period, offset=nope)['out']
    dense = fp8_dense(weight, scales)
    want = np.concatenate([dense[g * period + nope:(g + 1) * period] @ x[g * cols:(g + 1) * cols].astype(np.float64)
                           for g in range(heads)])
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()


# fp8_matvec_cols_cvt_ref (#42), the query absorb: out[g, c] = sum over d < depth of w[g * period + offset + d, c]
# * x[g * x_stride + d], on the stored matrix. GLM-5.3's shape cut to 3 heads (nope 192 of 448 rows, so the
# heads start 0, 64 and 0 rows into a block), depths shorter than one twelve-row step of the 16 subgroups,
# and 400 rows: two steps and a tail.
@pytest.mark.parametrize('groups,cols,depth,period,offset,x_stride', [(3, 512, 192, 448, 0, 256), (2, 128, 37, 50, 5, 40),
                                                                     (3, 256, 8, 8, 0, 8), (1, 384, 100, 228, 128, 100),
                                                                     (2, 128, 400, 410, 3, 400)])
def test_fp8_matvec_cols_cvt_sums_rows_down_the_columns(run, groups, cols, depth, period, offset, x_stride):
    weight, scales, _, _ = fp8_matvec4_case(groups * period, cols, groups + cols + depth)
    x = np.random.default_rng(depth).normal(size=groups * x_stride).astype(F)
    got = run('fp8_matvec_cols_cvt_ref', out=np.full(groups * cols, 7, F), weight=weight.view(np.uint32).reshape(-1),
              scales=scales.reshape(-1), x=x, groups=groups, cols=cols, depth=depth, period=period, offset=offset,
              x_stride=x_stride)['out']
    dense = fp8_dense(weight, scales)
    want = np.concatenate([x[g * x_stride:g * x_stride + depth].astype(np.float64)
                           @ dense[g * period + offset:g * period + offset + depth] for g in range(groups)])
    assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max()


def test_fp8_matvec_cols_cvt_zero_depth_and_bad_shapes(run):
    groups, cols, depth, period, x_stride = 2, 128, 16, 16, 16
    weight, scales, _, _ = fp8_matvec4_case(groups * period, cols, 9)
    x = np.ones(groups * x_stride, F)
    args = dict(weight=weight.view(np.uint32).reshape(-1), scales=scales.reshape(-1), x=x, groups=groups)
    good = dict(cols=cols, depth=depth, period=period, offset=0, x_stride=x_stride)
    got = run('fp8_matvec_cols_cvt_ref', out=np.full(groups * cols, 7, F), **args, **{**good, 'depth': 0})['out']
    bits_equal(got, np.zeros(groups * cols, F))
    for bad in (dict(period=15), dict(offset=-1), dict(x_stride=-1), dict(depth=-1), dict(cols=112)):
        out = run('fp8_matvec_cols_cvt_ref', out=np.full(groups * cols, 7, F), **args, **{**good, **bad})['out']
        bits_equal(out, np.full(groups * cols, 7, F))


@pytest.mark.parametrize('kernel', FMA_KERNELS)
def test_fp8_matvec_fma_writes_zeros_for_other_widths(run, kernel):
    got = run(kernel, out=np.full(3, 7.0, F), weight=np.zeros(3 * 100 // 4 + 1, np.uint32),
              scales=np.ones(2, F), x=np.ones(100, F), rows=3, cols=100)['out']
    bits_equal(got, np.zeros(3, F))


# --- Routed experts chosen on the device ---------------------------------------------------
# Each *_expert_matvec reads the router's ids and runs the slot-th expert in ascending id order,
# so it must equal the plain kernel on that expert's slice, bit for bit.

EXPERT_CASES = [(1, 1, 0), (12, 8, 3), (256, 8, 7)]  # (experts, k, slot)


def expert_ids(rng, experts, k):
    return rng.choice(experts, size=k, replace=False).astype(np.int32)


@pytest.mark.parametrize('experts,k,slot', EXPERT_CASES)
def test_expert_matvec(run, experts, k, slot):
    rng = np.random.default_rng(experts * 31 + slot)
    rows, cols = 7, 131
    w, x, ids = rng.normal(size=(experts, rows, cols)).astype(F), rng.normal(size=cols).astype(F), expert_ids(rng, experts, k)
    e = int(np.sort(ids)[slot])
    got = run('expert_matvec', out=np.zeros(rows, F), weight=w, x=x, ids=ids, k=k, slot=slot, experts=experts,
              rows=rows, cols=cols)['out']
    bits_equal(got, run('matvec', out=np.zeros(rows, F), weight=w[e], x=x, rows=rows, cols=cols)['out'])


@pytest.mark.parametrize('experts,k,slot', EXPERT_CASES)
def test_bf16_expert_matvec(run, experts, k, slot):
    rng = np.random.default_rng(experts * 37 + slot)
    rows, cols = 5, 131
    w = (rng.normal(size=(experts, rows, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x, ids = rng.normal(size=cols).astype(F), expert_ids(rng, experts, k)
    e = int(np.sort(ids)[slot])
    got = run('bf16_expert_matvec', out=np.zeros(rows, F), weight=w, x=x, ids=ids, k=k, slot=slot, experts=experts,
              rows=rows, cols=cols)['out']
    bits_equal(got, run('bf16_matvec', out=np.zeros(rows, F), weight=w[e], x=x, rows=rows, cols=cols)['out'])


@pytest.mark.parametrize('scale_kind', [0, 1])
@pytest.mark.parametrize('experts,k,slot', EXPERT_CASES)
def test_int4_expert_matvec(run, experts, k, slot, scale_kind):
    rng = np.random.default_rng(experts * 41 + slot)
    rows, cols = 5, 257
    groups, row_bytes = -(-cols // 128), -(-cols // 2)
    packed = rng.integers(0, 256, size=(experts, rows, row_bytes), dtype=np.uint8)
    packed[..., -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=(experts, rows, groups)).astype(F)
    scales = (magnitudes.view(np.uint32) >> 16).astype(np.uint16) if scale_kind else magnitudes.astype(np.float16).view(np.uint16)
    x, ids = rng.normal(size=cols).astype(F), expert_ids(rng, experts, k)
    e = int(np.sort(ids)[slot])
    got = run('int4_expert_matvec', out=np.zeros(rows, F), packed=packed, scales=scales, x=x, ids=ids, k=k, slot=slot,
              experts=experts, rows=rows, cols=cols, scale_kind=scale_kind)['out']
    bits_equal(got, run('int4_matvec', out=np.zeros(rows, F), packed=packed[e], scales=scales[e], x=x, rows=rows,
                        cols=cols, scale_kind=scale_kind)['out'])


@pytest.mark.parametrize('experts,k,slot', EXPERT_CASES)
def test_fp8_expert_matvec(run, experts, k, slot):
    rng = np.random.default_rng(experts * 43 + slot)
    rows, cols = 130, 300  # two scale rows and three scale columns per expert
    weight = rng.integers(0, 256, size=(experts, rows, cols), dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0
    scales = rng.uniform(0.001, 0.1, size=(experts, -(-rows // 128), -(-cols // 128))).astype(F)
    x, ids = rng.normal(size=cols).astype(F), expert_ids(rng, experts, k)
    e = int(np.sort(ids)[slot])
    got = run('fp8_expert_matvec', out=np.zeros(rows, F), weight=weight, scales=scales, x=x, ids=ids, k=k, slot=slot,
              experts=experts, rows=rows, cols=cols)['out']
    bits_equal(got, run('fp8_matvec', out=np.zeros(rows, F), weight=weight[e], scales=scales[e], x=x, rows=rows,
                        cols=cols)['out'])


def test_expert_out_of_range_writes_zeros(run):
    w = np.ones((2, 3, 4), F)
    got = run('expert_matvec', out=np.full(3, 7, F), weight=w, x=np.ones(4, F), ids=np.array([5], np.int32), k=1,
              slot=0, experts=2, rows=3, cols=4)['out']
    bits_equal(got, np.zeros(3, F))


@pytest.mark.parametrize('experts,k', [(1, 1), (256, 8)])
def test_moe_sum_in_ascending_id_order(run, experts, k):
    """Every slot through expert_matvec and expert_scaled_add, the host never reading the ids,
    equals resident.py's loop (argsort the ids on the host, matvec, scaled_add) bit for bit."""
    rng = np.random.default_rng(experts + k)
    rows, cols = 9, 64
    w, x = rng.normal(size=(experts, rows, cols)).astype(F), rng.normal(size=cols).astype(F)
    ids, weights = expert_ids(rng, experts, k), rng.uniform(0.01, 1, size=k).astype(F)
    total = np.zeros(rows, F)
    for slot in range(k):
        y = run('expert_matvec', out=np.zeros(rows, F), weight=w, x=x, ids=ids, k=k, slot=slot, experts=experts,
                rows=rows, cols=cols)['out']
        total = run('expert_scaled_add', out=total, x=y, weights=weights, ids=ids, k=k, slot=slot, n=rows)['out']
    expected = np.zeros(rows, F)
    for slot in np.argsort(ids, kind='stable'):
        y = run('matvec', out=np.zeros(rows, F), weight=w[ids[slot]], x=x, rows=rows, cols=cols)['out']
        expected = run('scaled_add', out=expected, x=y, weights=weights, slot=int(slot), n=rows)['out']
    bits_equal(total, expected)


# The fused MoE kernels: moe_gate_up for all k experts, then moe_down, must leave what the
# per-expert kernels leave (gate, up, swiglu, down, expert_scaled_add for each slot from zeros).

def moe_weights(rng, fmt, experts, rows, cols):
    """One projection's weights for all experts in `fmt`: (tensors by kernel name, extra args)."""
    if fmt == 'bf16':
        return {'weight': (rng.normal(size=(experts, rows, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)}, {}
    if fmt == 'fp8':
        codes = rng.integers(0, 256, size=(experts, rows, cols), dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0
        return {'weight': codes, 'scales': rng.uniform(0.001, 0.1, size=(experts, -(-rows // 128), -(-cols // 128))).astype(F)}, {}
    if fmt == 'int4':
        packed = rng.integers(0, 256, size=(experts, rows, -(-cols // 2)), dtype=np.uint8)
        if cols % 2:
            packed[..., -1] &= 0x0F
        magnitudes = rng.uniform(0.001, 0.1, size=(experts, rows, -(-cols // 128))).astype(F)
        return {'packed': packed, 'scales': (magnitudes.view(np.uint32) >> 16).astype(np.uint16)}, {'scale_kind': 1}
    return {'weight': rng.normal(size=(experts, rows, cols)).astype(F) * F(0.2)}, {}


def moe_prefix(fmt):
    return '' if fmt == 'f32' else fmt + '_'


def per_expert_moe(run, fmt, gate, up, down, x, ids, weights, experts, inter, hidden):
    k, total = len(ids), np.zeros(hidden, F)
    for slot in range(k):
        common = dict(ids=ids, k=k, slot=slot, experts=experts)
        (gw, ge), (uw, ue), (dw, de) = gate, up, down
        g = run(moe_prefix(fmt) + 'expert_matvec', out=np.zeros(inter, F), x=x, rows=inter, cols=hidden, **gw, **ge, **common)['out']
        u = run(moe_prefix(fmt) + 'expert_matvec', out=np.zeros(inter, F), x=x, rows=inter, cols=hidden, **uw, **ue, **common)['out']
        a = run('swiglu', out=np.zeros(inter, F), gate=g, up=u, n=inter)['out']
        y = run(moe_prefix(fmt) + 'expert_matvec', out=np.zeros(hidden, F), x=a, rows=hidden, cols=inter, **dw, **de, **common)['out']
        total = run('expert_scaled_add', out=total, x=y, weights=weights, ids=ids, k=k, slot=slot, n=hidden)['out']
    return total


def fused_moe(run, fmt, gate, up, down, x, ids, weights, experts, inter, hidden):
    k = len(ids)
    (gw, extra), (uw, _), (dw, _) = gate, up, down
    gate_args = {'gate' if n in ('weight', 'packed') else 'gate_scales': v for n, v in gw.items()}
    gate_args.update({'up' if n in ('weight', 'packed') else 'up_scales': v for n, v in uw.items()})
    act = run(moe_prefix(fmt) + 'moe_gate_up', act=np.full(k * inter, 7, F), x=x, ids=ids, k=k, experts=experts,
              inter=inter, up_row=0, stride=inter, cols=hidden, **gate_args, **extra)['act']
    down_args = {'down' if n in ('weight', 'packed') else 'scales': v for n, v in dw.items()}
    return run(moe_prefix(fmt) + 'moe_down', out=np.full(hidden, 7, F), act=act, weights=weights, ids=ids, k=k,
               experts=experts, rows=hidden, cols=inter, **down_args, **extra)['out']


@pytest.mark.parametrize('fmt', ['f32', 'bf16', 'fp8', 'int4'])
@pytest.mark.parametrize('experts,k,inter,hidden', [(1, 1, 3, 5), (12, 8, 9, 131), (256, 8, 130, 300), (40, 13, 9, 131), (64, 32, 5, 70)])
def test_fused_moe_equals_the_per_expert_kernels(run, fmt, experts, k, inter, hidden):
    rng = np.random.default_rng(experts * 7 + inter)
    gate, up = moe_weights(rng, fmt, experts, inter, hidden), moe_weights(rng, fmt, experts, inter, hidden)
    down = moe_weights(rng, fmt, experts, hidden, inter)
    x, ids = rng.normal(size=hidden).astype(F), expert_ids(rng, experts, k)
    weights = rng.uniform(0.01, 1, size=k).astype(F)
    args = (gate, up, down, x, ids, weights, experts, inter, hidden)
    bits_equal(fused_moe(run, fmt, *args), per_expert_moe(run, fmt, *args))


# fp8_moe4_gate_up and fp8_moe4_down (#31): the u32 view of the codes, each term fp8_matvec4's.
# They must leave what fp8_matvec4 on each expert's slice, swiglu, fp8_matvec4 of the down rows and
# scaled_add in ascending id order leave, bit for bit. Widths are multiples of 16.
@pytest.mark.parametrize('experts,k,inter,hidden', [(1, 1, 16, 16), (12, 8, 16, 144), (256, 8, 144, 304), (40, 13, 16, 144), (64, 32, 16, 80)])
def test_fp8_moe4_equals_fp8_matvec4_per_expert(run, experts, k, inter, hidden):
    rng = np.random.default_rng(experts * 5 + inter)
    (gate, _), (up, _), (down, _) = (moe_weights(rng, 'fp8', experts, r, c) for r, c in
                                     ((inter, hidden), (inter, hidden), (hidden, inter)))
    x, ids = rng.normal(size=hidden).astype(F), expert_ids(rng, experts, k)
    weights = rng.uniform(0.01, 1, size=k).astype(F)
    words = lambda w: w['weight'].view(np.uint32).reshape(-1)
    act = run('fp8_moe4_gate_up', act=np.full(k * inter, 7, F), gate=words(gate), gate_scales=gate['scales'].reshape(-1),
              up=words(up), up_scales=up['scales'].reshape(-1), x=x, ids=ids, k=k, experts=experts, inter=inter,
              up_row=0, stride=inter, cols=hidden)['act']
    got = run('fp8_moe4_down', out=np.full(hidden, 7, F), down=words(down), scales=down['scales'].reshape(-1), act=act,
              weights=weights, ids=ids, k=k, experts=experts, rows=hidden, cols=inter)['out']
    matvec4 = lambda w, e, vec, rows, cols: run('fp8_matvec4', out=np.zeros(rows, F), weight=w['weight'][e].view(np.uint32).reshape(-1),
                                                scales=w['scales'][e].reshape(-1), x=vec, rows=rows, cols=cols)['out']
    expected, want_act = np.zeros(hidden, F), np.zeros(k * inter, F)
    for s, slot in enumerate(np.argsort(ids, kind='stable')):
        e = int(ids[slot])
        a = run('swiglu', out=np.zeros(inter, F), gate=matvec4(gate, e, x, inter, hidden), up=matvec4(up, e, x, inter, hidden),
                n=inter)['out']
        want_act[s * inter:(s + 1) * inter] = a
        expected = run('scaled_add', out=expected, x=matvec4(down, e, a, hidden, inter), weights=weights, slot=int(slot),
                       n=hidden)['out']
    bits_equal(act, want_act)
    bits_equal(got, expected)


# fp8_matvec_fma_tiles3 (#30): q_a, kv_a and the indexer's wk in one launch. Each matrix's rows are
# fp8_matvec_fma_tiles' for it, bit for bit, in one out with each matrix from a multiple of 8;
# rows_c = 0 leaves C's place unwritten.
@pytest.mark.parametrize('three,one', [('fp8_matvec_fma_tiles3', 'fp8_matvec_fma_tiles'),
                                       ('fp8_matvec_cvt_tiles3_ref', 'fp8_matvec_cvt_tiles_ref')])
@pytest.mark.parametrize('rows_a,rows_b,rows_c,cols', [(20, 9, 5, 640), (16, 13, 0, 144), (256, 72, 16, 1024)])
def test_fp8_matvec_fma_tiles3_is_tiles_per_matrix(run, three, one, rows_a, rows_b, rows_c, cols):
    pad = lambda n: -(-n // 8) * 8
    mats = [fp8_matvec4_case(r, cols, r * 3 + cols)[:2] for r in (rows_a, rows_b, max(rows_c, 1))]
    x = np.random.default_rng(cols).normal(size=cols).astype(F)
    words = lambda w: w.view(np.uint32).reshape(-1)
    out = np.full(pad(rows_a) + pad(rows_b) + pad(rows_c), 7.0, F)
    got = run(three, out=out, x=x, cols=cols,
              weight_a=words(mats[0][0]), scales_a=mats[0][1].reshape(-1), rows_a=rows_a,
              weight_b=words(mats[1][0]), scales_b=mats[1][1].reshape(-1), rows_b=rows_b,
              weight_c=words(mats[2][0]), scales_c=mats[2][1].reshape(-1), rows_c=rows_c)['out']
    starts = (0, pad(rows_a), pad(rows_a) + pad(rows_b))
    for (w, sc), rows, at in zip(mats, (rows_a, rows_b, rows_c), starts):
        want = run(one, out=np.zeros(rows, F), weight=words(w), scales=sc.reshape(-1), x=x,
                   rows=rows, cols=cols)['out'] if rows else np.zeros(0, F)
        bits_equal(got[at:at + rows], want)
        bits_equal(got[at + rows:at + pad(rows)], np.full(pad(rows) - rows, 7.0, F))  # padding untouched


def dequantized(w, e):
    """Expert e's [rows, cols] fp8 matrix in float64, its scales applied."""
    codes, scales = w['weight'][e], w['scales'][e]
    rows, cols = codes.shape
    return fp8(codes).astype(np.float64) * scales[np.arange(rows)[:, None] // 128, np.arange(cols)[None, :] // 128]


MOE_FMA = ['fp8_moe_fma_', 'fp8_moe_cvt_']  # + gate_up / down; the cvt family through its _ref builds


def moe_kernel(family, half):
    return family + half + ('_ref' if family == 'fp8_moe_cvt_' else '')


# fp8_moe_fma_gate_up and fp8_moe_fma_down (#31): fp8_matvec_fma_tiles' dot products, so not
# fp8_moe4's bits. act must be within FMA_TOLERANCE of silu(g) * u from float64 g and u (of its
# largest |act|), and the down sum within it of the float64 sum over the experts of weight * (W x)
# from the kernel's own act. (256, 8, 1024, 2048) takes the whole-tile loads; the others end in
# partial tiles.
@pytest.mark.parametrize('experts,k,inter,hidden', [(1, 1, 16, 16), (12, 8, 16, 144), (256, 8, 144, 304),
                                                    (40, 13, 16, 144), (64, 32, 16, 80), (256, 8, 1024, 2048)])
@pytest.mark.parametrize('family', MOE_FMA)
def test_fp8_moe_fma_within_tolerance(run, family, experts, k, inter, hidden):
    rng = np.random.default_rng(experts * 7 + inter)
    (gate, _), (up, _), (down, _) = (moe_weights(rng, 'fp8', experts, r, c) for r, c in
                                     ((inter, hidden), (inter, hidden), (hidden, inter)))
    x, ids = rng.normal(size=hidden).astype(F), expert_ids(rng, experts, k)
    weights = rng.uniform(0.01, 1, size=k).astype(F)
    words = lambda w: w['weight'].view(np.uint32).reshape(-1)
    act = run(moe_kernel(family, 'gate_up'), act=np.full(k * inter, 7, F), gate=words(gate), gate_scales=gate['scales'].reshape(-1),
              up=words(up), up_scales=up['scales'].reshape(-1), x=x, ids=ids, k=k, experts=experts, inter=inter,
              up_row=0, stride=inter, cols=hidden)['act']
    got = run(moe_kernel(family, 'down'), out=np.full(hidden, 7, F), down=words(down), scales=down['scales'].reshape(-1), act=act,
              weights=weights, ids=ids, k=k, experts=experts, rows=hidden, cols=inter)['out']
    want_act, want = np.zeros(k * inter), np.zeros(hidden)
    for s, slot in enumerate(np.argsort(ids, kind='stable')):
        e = int(ids[slot])
        g, u = dequantized(gate, e) @ x.astype(np.float64), dequantized(up, e) @ x.astype(np.float64)
        want_act[s * inter:(s + 1) * inter] = g * (0.5 + 0.5 * np.tanh(g / 2)) * u
        want += float(weights[slot]) * (dequantized(down, e) @ act[s * inter:(s + 1) * inter].astype(np.float64))
    assert np.abs(act - want_act).max() <= FMA_TOLERANCE * np.abs(want_act).max()
    assert np.abs(got - want).max() <= FMA_TOLERANCE * np.abs(want).max()


@pytest.mark.parametrize('family', MOE_FMA)
def test_fp8_moe_fma_down_odd_rows_and_out_of_range_ids(run, family):
    # A last block with one row (rows odd), and an id past the experts adding nothing.
    rng = np.random.default_rng(3)
    experts, rows, cols = 6, 145, 144
    (down, _) = moe_weights(rng, 'fp8', experts, rows, cols)
    ids, weights = np.array([4, 9, 1], np.int32), np.array([0.5, 0.25, 2.0], F)
    act = rng.normal(size=3 * cols).astype(F)
    got = run(moe_kernel(family, 'down'), out=np.full(rows, 7, F), down=down['weight'].view(np.uint32).reshape(-1),
              scales=down['scales'].reshape(-1), act=act, weights=weights, ids=ids, k=3, experts=experts, rows=rows,
              cols=cols)['out']
    # Ascending ids: 1 (slot 2) has rank 0, 4 (slot 0) rank 1, 9 (slot 1) rank 2 and is out of range.
    want = 2.0 * (dequantized(down, 1) @ act[:cols].astype(np.float64)) \
        + 0.5 * (dequantized(down, 4) @ act[cols:2 * cols].astype(np.float64))
    assert np.abs(got - want).max() <= FMA_TOLERANCE * np.abs(want).max()


@pytest.mark.parametrize('family', MOE_FMA)
def test_fp8_moe_fma_gate_up_reads_the_fused_layout(run, family):
    # gate_up as one [E, 2 * inter, cols] matrix (stride 2 * inter, up rows from inter), as GLM's
    # checkpoints may store it: the same act as the two halves passed separately.
    rng = np.random.default_rng(11)
    experts, k, inter, hidden = 12, 8, 144, 640
    (both, _) = moe_weights(rng, 'fp8', experts, 2 * inter, hidden)
    x, ids = rng.normal(size=hidden).astype(F), expert_ids(rng, experts, k)
    words = lambda a: np.ascontiguousarray(a).view(np.uint32).reshape(-1)
    common = dict(act=np.zeros(k * inter, F), x=x, ids=ids, k=k, experts=experts, inter=inter, cols=hidden)
    fused = run(moe_kernel(family, 'gate_up'), gate=words(both['weight']), gate_scales=both['scales'].reshape(-1),
                up=words(both['weight']), up_scales=both['scales'].reshape(-1), up_row=inter, stride=2 * inter,
                **common)['act']
    # Rows inter.. start a new 128-row scale block only when inter % 128 == 0: 144 is not, so cut
    # the scales row-wise from the dequantized values instead of slicing the blocks.
    want = np.zeros(k * inter)
    for s, slot in enumerate(np.argsort(ids, kind='stable')):
        m = dequantized(both, int(ids[slot])) @ x.astype(np.float64)
        g, u = m[:inter], m[inter:]
        want[s * inter:(s + 1) * inter] = g * (0.5 + 0.5 * np.tanh(g / 2)) * u
    assert np.abs(fused - want).max() <= FMA_TOLERANCE * np.abs(want).max()


def test_fused_moe_gate_up_reads_the_fused_layout(run):
    """gate_up as one [E, 2 * inter, cols] tensor (stride 2 * inter, up rows from inter) equals
    separate gate and up projections cut from it."""
    rng = np.random.default_rng(5)
    experts, k, inter, hidden = 12, 8, 9, 131
    gate_up = rng.normal(size=(experts, 2 * inter, hidden)).astype(F)
    x, ids = rng.normal(size=hidden).astype(F), expert_ids(rng, experts, k)
    common = dict(act=np.zeros(k * inter, F), x=x, ids=ids, k=k, experts=experts, inter=inter, cols=hidden)
    got = run('moe_gate_up', gate=gate_up, up=gate_up, up_row=inter, stride=2 * inter, **common)['act']
    want = run('moe_gate_up', gate=np.ascontiguousarray(gate_up[:, :inter]), up=np.ascontiguousarray(gate_up[:, inter:]),
               up_row=0, stride=inter, **common)['act']
    bits_equal(got, want)


def test_fused_moe_out_of_range_id_adds_zero(run):
    rng = np.random.default_rng(9)
    experts, inter, hidden = 4, 3, 6
    gate, up = moe_weights(rng, 'f32', experts, inter, hidden), moe_weights(rng, 'f32', experts, inter, hidden)
    down = moe_weights(rng, 'f32', experts, hidden, inter)
    x, weights = rng.normal(size=hidden).astype(F), np.array([0.5, 0.25], F)
    args = (gate, up, down, x)
    got = fused_moe(run, 'f32', *args, np.array([2, 9], np.int32), weights, experts, inter, hidden)
    only = fused_moe(run, 'f32', *args, np.array([2], np.int32), weights[:1], experts, inter, hidden)
    bits_equal(got, only)


@pytest.mark.parametrize('heads,rows,cols,w_stride,x_stride,out_stride', [
    (1, 1, 1, 1, 1, 1), (8, 37, 131, 37 * 131 + 5, 140, 40)])
def test_matvec_heads(run, heads, rows, cols, w_stride, x_stride, out_stride):
    rng = np.random.default_rng(heads * rows + cols)
    w = rng.normal(size=heads * w_stride).astype(F)
    x = rng.normal(size=heads * x_stride).astype(F)
    fill = rng.normal(size=(heads, out_stride)).astype(F)
    got = run('matvec_heads', out=fill.copy(), weight=w, x=x, heads=heads, rows=rows, cols=cols,
              w_stride=w_stride, x_stride=x_stride, out_stride=out_stride)['out']
    expected = fill.copy()
    for h in range(heads):
        wh = w[h * w_stride:h * w_stride + rows * cols].reshape(rows, cols)
        expected[h, :rows] = warp_tree(strided_sums(wh * x[h * x_stride:h * x_stride + cols], 32))
    bits_equal(got, expected)


@pytest.mark.parametrize('tokens,heads,dim', [(1, 1, 1), (37, 16, 128)])
def test_index_scores(run, tokens, heads, dim):
    rng = np.random.default_rng(tokens * heads + dim)
    q, keys = rng.normal(size=(heads, dim)).astype(F), rng.normal(size=(tokens, dim)).astype(F)
    weights = (rng.normal(size=heads) * heads ** -0.5).astype(F)
    scale = F(dim ** -0.5)
    got = run('index_scores', out=np.zeros(tokens, F), q=q, keys=keys, weights=weights, tokens=tokens,
              heads=heads, dim=dim, scale=float(scale))['out']
    score = np.zeros(tokens, F)
    for h in range(heads):
        s = warp_tree(strided_sums(keys * q[h], 32)) * scale
        score = score + weights[h] * np.where(s < 0, F(0), s)
    bits_equal(got, score)


def test_gemm_f32(run):
    m, n, k = 33, 65, 256
    rng = np.random.default_rng(m * n + k)
    a, b = rng.normal(size=(m, k)).astype(F), rng.normal(size=(k, n)).astype(F)
    expected = np.zeros((m, n), F)
    for q in range(k):
        expected = expected + a[:, q:q + 1] * b[q]
    bits_equal(run('gemm_f32', out=np.full((m, n), 7, F), a=a, b=b, m=m, n=n, k=k)['out'], expected)


# Dense (experts 0) and routed with out-of-range ids (-1 and E), which write zeros.
MATMUL_CASES = [(1, 1, 1, 0), (5, 37, 131, 0), (6, 30, 257, 3)]


def routing(rng, m, experts):
    if not experts:
        return np.zeros(1, np.int32), np.zeros(m, np.int64), np.ones(m, bool)
    ids = rng.integers(0, experts, size=m).astype(np.int32)
    ids[0], ids[-1] = -1, experts
    valid = (ids >= 0) & (ids < experts)
    return ids, np.where(valid, ids, 0), valid


@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_bf16_matmul(run, m, n, k, experts):
    rng = np.random.default_rng(m + n + k + experts)
    e = max(experts, 1)
    w = (rng.normal(size=(e, n, k)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=(m, k)).astype(F)
    ids, expert, valid = routing(rng, m, experts)
    got = run('bf16_matmul', out=np.zeros((m, n), F), weight=w, x=x, expert_ids=ids, rows_m=m, rows_n=n,
              cols=k, experts=experts)['out']
    expected = warp_tree(strided_sums(bf16(w[expert]) * x[:, None, :], 32))
    bits_equal(got, np.where(valid[:, None], expected, F(0)))


@pytest.mark.parametrize('scale_kind', [0, 1])
@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_int4_matmul(run, m, n, k, experts, scale_kind):
    rng = np.random.default_rng(m * 3 + n + k + experts + scale_kind)
    e, groups, row_bytes = max(experts, 1), -(-k // 128), -(-k // 2)
    packed = rng.integers(0, 256, size=(e, n, row_bytes), dtype=np.uint8)
    if k % 2:
        packed[..., -1] &= 0x0F
    magnitudes = rng.uniform(0.001, 0.1, size=(e, n, groups)).astype(F)
    scales = (magnitudes.view(np.uint32) >> 16).astype(np.uint16) if scale_kind else magnitudes.astype(np.float16).view(np.uint16)
    x = rng.normal(size=(m, k)).astype(F)
    ids, expert, valid = routing(rng, m, experts)
    got = run('int4_matmul', out=np.zeros((m, n), F), packed=packed, scales=scales, x=x, expert_ids=ids,
              rows_m=m, rows_n=n, cols=k, experts=experts, scale_kind=scale_kind)['out']
    expected = np.zeros((m, n), F)
    for i in range(m):
        if valid[i]:
            expected[i] = warp_tree(int4_lane_sums(packed[expert[i]], scales[expert[i]],
                                                   np.broadcast_to(x[i], (n, k)), k, scale_kind))
    bits_equal(got, expected)


@pytest.mark.parametrize('m,n,k,experts', MATMUL_CASES)
def test_fp8_matmul(run, m, n, k, experts):
    rng = np.random.default_rng(m * 5 + n + k + experts)
    e, scale_cols = max(experts, 1), -(-k // 128)
    weight = rng.integers(0, 256, size=(e, n, k), dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0
    scales = rng.uniform(0.001, 0.1, size=(e, -(-n // 128), scale_cols)).astype(F)
    x = rng.normal(size=(m, k)).astype(F)
    ids, expert, valid = routing(rng, m, experts)
    got = run('fp8_matmul', out=np.zeros((m, n), F), weight=weight, scales=scales, x=x, expert_ids=ids,
              rows_m=m, rows_n=n, cols=k, experts=experts)['out']
    lane = np.arange(32)
    expected = np.zeros((m, n), F)
    for i in range(m):
        if not valid[i]:
            continue
        terms = fp8(weight[expert[i]]) * x[i]  # [n, k]
        total = np.zeros((n, 32), F)
        for tile in range(scale_cols):
            part = np.zeros((n, 32), F)
            for step in range(4):
                col = tile * 128 + step * 32 + lane
                part = np.where(col < k, part + terms[:, np.minimum(col, k - 1)], part).astype(F)
            total = total + part * scales[expert[i], np.arange(n) // 128, tile][:, None]
        expected[i] = warp_tree(total)
    bits_equal(got, expected)


# --- Block per row ------------------------------------------------------------------------

def rms_reference(v, w, eps, threads):
    total = block_sum(block_partials(v * v, threads), threads)
    inv = F(1) / np.sqrt(total / F(v.shape[-1]) + F(eps))
    return v * inv[:, None] * w


@pytest.mark.parametrize('rows,dim,threads', [(1, 6144, 256), (5, 131, 64), (3, 1, 32), (4, 2048, 1024)])
def test_rmsnorm_rows(run, rows, dim, threads):
    rng = np.random.default_rng(rows * dim)
    x, w = rng.normal(size=(rows, dim)).astype(F), rng.normal(size=dim).astype(F)
    got = run('rmsnorm_rows', out=np.zeros((rows, dim), F), x=x, weight=w, rows=rows, dim=dim, eps=1e-5,
              threads=threads)['out']
    bits_equal(got, rms_reference(x, w, 1e-5, threads))


@pytest.mark.parametrize('rows,dim,threads', [(1, 6144, 256), (7, 333, 96)])
def test_residual_rmsnorm(run, rows, dim, threads):
    rng = np.random.default_rng(rows + dim)
    x, r, w = rng.normal(size=(rows, dim)).astype(F), rng.normal(size=(rows, dim)).astype(F), rng.normal(size=dim).astype(F)
    got = run('residual_rmsnorm', out=np.zeros((rows, dim), F), residual_out=np.zeros((rows, dim), F), x=x,
              residual=r, weight=w, rows=rows, dim=dim, eps=1e-5, threads=threads)
    bits_equal(got['residual_out'], x + r)
    bits_equal(got['out'], rms_reference(x + r, w, 1e-5, threads))


@pytest.mark.parametrize('rows,dim,threads', [(1, 128, 128), (9, 333, 64)])
def test_layernorm(run, rows, dim, threads):
    rng = np.random.default_rng(rows * 3 + dim)
    x = (rng.normal(size=(rows, dim)) + 3).astype(F)
    w, b = rng.normal(size=dim).astype(F), rng.normal(size=dim).astype(F)
    got = run('layernorm', out=np.zeros((rows, dim), F), x=x, weight=w, bias=b, rows=rows, dim=dim, eps=1e-6,
              threads=threads)['out']
    mean = (block_sum(block_partials(x, threads), threads) / F(dim))[:, None]
    c = x - mean
    inv = (F(1) / np.sqrt(block_sum(block_partials(c * c, threads), threads) / F(dim) + F(1e-6)))[:, None]
    bits_equal(got, c * inv * w + b)


@pytest.mark.parametrize('rows,dim,threads', [(1, 154880, 1024), (4, 154880, 256), (5, 131, 64), (3, 1, 32)])
def test_argmax_rows_is_numpys_argmax(run, rows, dim, threads):
    # Few distinct values, so most rows tie at their maximum: the first index wins, as numpy.argmax.
    rng = np.random.default_rng(rows * dim + 11)
    x = rng.integers(-3, 4, size=(rows, dim)).astype(F)
    x[0, :] = -np.inf
    x[0, dim // 2] = -0.0
    if dim > 1:
        x[0, dim - 1] = 0.0  # equal to -0.0: still the first
    got = run('argmax_rows', out=np.full(rows, 7, np.int32), x=x, rows=rows, dim=dim, threads=threads)['out']
    np.testing.assert_array_equal(got, np.argmax(x, axis=1))


def test_argmax_rows_refuses_a_bad_launch(run):
    x = np.ones((2, 64), F)
    for dim, threads in [(0, 64), (64, 48), (64, 2048)]:
        got = run('argmax_rows', out=np.full(2, 7, np.int32), x=x, rows=2, dim=dim, threads=threads)['out']
        np.testing.assert_array_equal(got, [-1, -1])


@pytest.mark.parametrize('rows,dim,threads', [(1, 154880, 256), (5, 131, 64), (3, 1, 32), (4, 2048, 1024)])
def test_softmax_rows(run, rows, dim, threads):
    rng = np.random.default_rng(rows * dim + 5)
    x = (rng.normal(size=(rows, dim)) * 3).astype(F)
    got = run('softmax_rows', out=np.zeros((rows, dim), F), x=x, rows=rows, dim=dim, threads=threads)['out']
    # Each thread's running max starts at x[0]; then vxk_block_max (padding with warp 0's).
    partials = np.repeat(x[:, :1], 1024, axis=1)
    lanes = np.full((rows, threads), -np.inf, F)
    for start in range(0, dim, threads):
        chunk = x[:, start:start + threads]
        lanes[:, :chunk.shape[1]] = np.where(chunk > lanes[:, :chunk.shape[1]], chunk, lanes[:, :chunk.shape[1]])
    partials[:, :threads] = np.where(lanes > x[:, :1], lanes, x[:, :1])
    maximum = block_max(partials, threads)
    assert (maximum == x.max(axis=1)).all()
    e = np.exp(x - maximum[:, None]).astype(F)
    np.testing.assert_allclose(got, e / block_sum(block_partials(e, threads), threads)[:, None], **CLOSE)


# --- Selection ----------------------------------------------------------------------------

@pytest.mark.parametrize('tokens,k,ties', [(1, 1, False), (300, 64, True), (5000, 2048, True), (70000, 2048, False)])
def test_topk_matches_stable_argsort(run, tokens, k, ties):
    from gpu.topk import spawn_buffers
    rng = np.random.default_rng(tokens + k)
    scores = rng.integers(-4, 4, size=tokens).astype(F) if ties else rng.normal(size=tokens).astype(F)
    if tokens > 10:
        scores[3], scores[7] = np.nan, -0.0  # NaN sorts last; -0.0 ties +0.0
    kk = min(k, tokens)
    scratch, order_a, order_b = (np.zeros(s, t) for s, t in spawn_buffers(tokens, k))
    got = run('topk', out_ids=np.full(kk, -7, np.int32), scores=scores.view(np.int32), scratch=scratch,
              order_a=order_a, order_b=order_b, tokens=tokens, k=k)['out_ids']
    np.testing.assert_array_equal(got, np.argsort(-scores, kind='stable')[:kk])


# topk_sort (#32): the same selection for tokens <= 2048 by one shared-memory bitonic sort.
@pytest.mark.parametrize('tokens,k,ties', [(1, 1, False), (2, 2048, True), (64, 2048, False), (300, 64, True),
                                           (1000, 2048, True), (2048, 2048, False), (2048, 16, True), (1500, 1500, False)])
def test_topk_sort_matches_stable_argsort(run, tokens, k, ties):
    rng = np.random.default_rng(tokens * 3 + k)
    scores = rng.integers(-4, 4, size=tokens).astype(F) if ties else rng.normal(size=tokens).astype(F)
    if tokens > 10:
        scores[3], scores[7], scores[9] = np.nan, -0.0, np.inf  # NaN sorts last; -0.0 ties +0.0
        scores[5] = -np.inf
    kk = min(k, tokens)
    got = run('topk_sort', out_ids=np.full(kk, -7, np.int32), scores=scores.view(np.int32), tokens=tokens, k=k)['out_ids']
    np.testing.assert_array_equal(got, np.argsort(-scores, kind='stable')[:kk])


def test_topk_sort_leaves_longer_contexts_to_topk(run):
    got = run('topk_sort', out_ids=np.full(4, -7, np.int32), scores=np.zeros(2049, F).view(np.int32), tokens=2049, k=4)
    np.testing.assert_array_equal(got['out_ids'], -7)


# topk_chunks then topk_merge rounds (#32): the same selection past 2048 tokens, each 2048-position
# chunk sorted by its own block, then sorted rows merged pairwise, the first 2048 kept.
def topk_long(run, scores, tokens, k):
    chunks = -(-tokens // 2048)
    src = run('topk_chunks', lists=np.zeros((chunks, 2048), np.int64), scores=scores.view(np.int32), tokens=tokens)['lists']
    out_ids, lists = np.full((1, k), -7, np.int32), chunks
    while True:
        out = run('topk_merge', dst=np.zeros((chunks, 2048), np.int64), src=src, out_ids=out_ids, lists=lists,
                  tokens=tokens, k=k)
        src, out_ids, lists = out['dst'], out['out_ids'], -(-lists // 2)
        if lists == 1:
            return out_ids[0, :min(k, tokens)]


@pytest.mark.parametrize('tokens,k,ties', [(2049, 2048, True), (4096, 2048, False), (5000, 2048, True), (6145, 64, False),
                                           (20000, 2048, True), (70000, 2048, False), (9000, 1, True)])
def test_topk_chunks_and_merge_match_stable_argsort(run, tokens, k, ties):
    rng = np.random.default_rng(tokens * 5 + k)
    scores = rng.integers(-4, 4, size=tokens).astype(F) if ties else rng.normal(size=tokens).astype(F)
    scores[3], scores[7], scores[9], scores[5] = np.nan, -0.0, np.inf, -np.inf  # NaN sorts last; -0.0 ties +0.0
    scores[-1] = np.inf  # the last position, in the last chunk, ranks second
    np.testing.assert_array_equal(topk_long(run, scores, tokens, k), np.argsort(-scores, kind='stable')[:min(k, tokens)])


# The same under graph replay: tokens from the step buffer, launches sized for max_tokens (one
# chunks_at, then ceil(log2(chunks)) merge_at rounds), whatever the step's context.
@pytest.mark.parametrize('tokens', [100, 2048, 2049, 6000, 8192, 9000, 20000])
def test_topk_chunks_and_merge_at_read_tokens_from_the_step(run, tokens):
    max_tokens, k = 20000, 2048
    chunks = -(-max_tokens // 2048)
    rng = np.random.default_rng(tokens)
    scores = rng.integers(-50, 50, size=max_tokens).astype(F)
    step = np.array([tokens - 1, tokens, min(k, tokens), 1], np.int32)
    out_ids = np.full((1, k), -7, np.int32)
    lists = run('topk_chunks_at', lists=np.zeros((chunks, 2048), np.int64), scores=scores.view(np.int32), step=step,
                max_tokens=max_tokens)['lists']
    for r in range((chunks - 1).bit_length()):
        out = run('topk_merge_at', dst=np.zeros((chunks, 2048), np.int64), src=lists, out_ids=out_ids, step=step,
                  max_tokens=max_tokens, round=r, k=k)
        lists, out_ids = out['dst'], out['out_ids']
    if tokens <= 2048:
        np.testing.assert_array_equal(out_ids, -7)  # topk_sort_at's
    else:
        np.testing.assert_array_equal(out_ids[0], np.argsort(-scores[:tokens], kind='stable')[:k])


# topk_radix rounds 0 .. 8 (#32): the same selection by radix select, eight one-byte rounds over
# the key and the position reversed, then a gather and one sort. Each round's last block reads and
# clears hist and keeps the threshold so far in state, which needs no start value; round 8's sorts
# the gathered values and clears count.
def topk_radix(run, scores, tokens, k, at=None):
    hist, state = np.zeros(256, np.uint32), np.full(5, 99, np.int64)
    vals, count = np.zeros(2048, np.int64), np.zeros(1, np.uint32)
    size = dict(tokens=tokens) if at is None else dict(step=at[0], max_tokens=at[1])
    for r in range(9):
        out = run('topk_radix' + ('' if at is None else '_at'), out_ids=np.full(k, -7, np.int32), hist=hist,
                  state=state, vals=vals, count=count, scores=scores.view(np.int32), k=k, round=r,
                  arrivals=np.zeros(1, np.int32), **size)
        hist, state, vals, count = out['hist'], out['state'], out['vals'], out['count']
        assert not hist.any()  # ready for the next round
    assert not count.any()  # and for the next call
    return out['out_ids'][:min(k, tokens)]


@pytest.mark.parametrize('tokens,k,ties', [(2049, 2048, True), (5000, 2048, True), (6145, 64, False), (20000, 2048, True),
                                           (70000, 2048, False), (9000, 1, True), (3000, 2048, 'equal'), (100, 7, False)])
def test_topk_radix_matches_stable_argsort(run, tokens, k, ties):
    rng = np.random.default_rng(tokens * 5 + k)
    if ties == 'equal':
        scores = np.zeros(tokens, F)  # every round but the last ties; the position bytes decide
    else:
        scores = rng.integers(-4, 4, size=tokens).astype(F) if ties else rng.normal(size=tokens).astype(F)
    scores[3], scores[7], scores[9], scores[5] = np.nan, -0.0, np.inf, -np.inf  # NaN sorts last; -0.0 ties +0.0
    scores[-1] = np.inf
    got = topk_radix(run, scores, tokens, k)
    np.testing.assert_array_equal(got, np.argsort(-scores, kind='stable')[:min(k, tokens)])


# The same under graph replay: tokens from the step buffer, nine launches of ceil(max_tokens / 4096)
# blocks whatever the step's context; up to 2048 tokens they write no ids.
@pytest.mark.parametrize('tokens', [100, 2048, 2049, 6000, 9000, 20000])
def test_topk_radix_at_reads_tokens_from_the_step(run, tokens):
    max_tokens, k = 20000, 2048
    rng = np.random.default_rng(tokens)
    scores = rng.integers(-50, 50, size=max_tokens).astype(F)
    step = np.array([tokens - 1, tokens, min(k, tokens), 1], np.int32)
    got = topk_radix(run, scores, tokens, k, at=(step, max_tokens))
    if tokens <= 2048:
        np.testing.assert_array_equal(got, -7)  # topk_sort_at's
    else:
        np.testing.assert_array_equal(got, np.argsort(-scores[:tokens], kind='stable')[:k])


# topk_select (#32): the same set as the stable argsort's first k, in ascending position order, by
# a radix select of the threshold in shared memory and a scan; no sort.
def stable_set(scores, k):
    return np.sort(np.argsort(-scores, kind='stable')[:min(k, len(scores))])


@pytest.mark.parametrize('tokens,k,ties', [(2049, 2048, True), (5000, 2048, True), (6145, 64, False), (20000, 2048, True),
                                           (70000, 2048, False), (9000, 1, True), (3000, 2048, 'equal'), (100, 7, False),
                                           (100, 200, True), (1, 2048, False), (1025, 1024, 'equal'), (4097, 2048, 'few')])
def test_topk_select_is_the_stable_set_in_position_order(run, tokens, k, ties):
    rng = np.random.default_rng(tokens * 7 + k)
    if ties == 'equal':
        scores = np.zeros(tokens, F)  # one key: the lowest positions are taken
    elif ties == 'few':
        scores = rng.integers(0, 2, size=tokens).astype(F)  # the threshold bucket is most of the context
    else:
        scores = rng.integers(-4, 4, size=tokens).astype(F) if ties else rng.normal(size=tokens).astype(F)
    if tokens > 10:
        scores[3], scores[7], scores[9], scores[5] = np.nan, -0.0, np.inf, -np.inf  # NaN sorts last; -0.0 ties +0.0
        scores[-1] = np.inf
    kk = min(k, tokens)
    got = run('topk_select', out_ids=np.full(kk + 3, -7, np.int32), scores=scores.view(np.int32), tokens=tokens, k=k)['out_ids']
    np.testing.assert_array_equal(got[:kk], stable_set(scores, k))
    np.testing.assert_array_equal(got[kk:], -7)


# Under graph replay: tokens from the step buffer; up to k it writes nothing (#43's positions).
@pytest.mark.parametrize('tokens', [100, 2048, 2049, 6000, 20000])
def test_topk_select_at_reads_tokens_from_the_step(run, tokens):
    max_tokens, k = 20000, 2048
    scores = np.random.default_rng(tokens).integers(-50, 50, size=max_tokens).astype(F)
    step = np.array([tokens - 1, tokens, min(k, tokens), 1], np.int32)
    got = run('topk_select_at', out_ids=np.full(k, -7, np.int32), scores=scores.view(np.int32), step=step, k=k)['out_ids']
    if tokens <= k:
        np.testing.assert_array_equal(got, -7)
    else:
        np.testing.assert_array_equal(got, stable_set(scores[:tokens], k))


# topk_select_bitmap_at (#63): topk_select_at's set and order, from the bins of the keys' top 12 bits that
# the scores kernel counted. It reads the scores once and leaves the bins zero.
def score_bins(scores):
    """index_scores_*_hist_at's bins: each key's top 12 bits (topk_key32: NaN is 0, -0.0 is +0.0's key)."""
    b = scores.view(np.uint32).astype(np.uint64)
    key = np.where(b >= 0x80000000, 0xFFFFFFFF - b, b ^ 0x80000000)
    key = np.where(b == 0x80000000, 0x80000000, key)
    key = np.where((b & 0x7FFFFFFF) > 0x7F800000, 0, key)
    return np.bincount((key >> 20).astype(np.int64), minlength=4096).astype(np.uint32)


def bitmap_scores(tokens, kind, rng):
    if kind == 'equal':
        s = np.zeros(tokens, F)  # one key: the lowest positions are taken
    elif kind == 'few':
        s = rng.integers(0, 2, size=tokens).astype(F)  # the threshold bin is most of the context
    elif kind == 'narrow':
        s = (5 + 0.01 * rng.normal(size=tokens)).astype(F)  # one bin holds more than 2048 keys
    elif kind == 'ints':
        s = rng.integers(-50, 50, size=tokens).astype(F)
    elif kind == 'recent':
        s = rng.normal(size=tokens).astype(F)
        s[-3000:] += 3  # the last positions score higher, as recent tokens often do
    elif kind == 'negative':
        s = -np.abs(rng.normal(size=tokens)).astype(F)
    elif kind == 'nan':
        s = rng.normal(size=tokens).astype(F)
        s[rng.random(tokens) < 0.8] = np.nan  # the threshold is NaN's key: NaN ties, the lowest positions first
    elif kind in ('bin2048', 'bin2049'):  # 1001 keys above a bin of exactly 2048 (listed) or 2049 equal keys
        s = np.full(tokens, -1, F)
        at = 10 + rng.permutation(tokens - 10)
        s[at[:1000]] = 100
        s[at[1000:1000 + int(kind[3:])]] = 1
    else:
        s = rng.normal(size=tokens).astype(F)
    s[[3, 5, 7, 9]] = np.nan, -np.inf, -0.0, np.inf
    return s


@pytest.mark.parametrize('tokens,k,kind', [(2049, 2048, 'ints'), (3001, 2048, 'normal'), (32768, 2048, 'normal'),
                                           (32768, 2048, 'ints'), (32768, 2048, 'recent'), (32768, 2048, 'narrow'),
                                           (32768, 2048, 'negative'), (65536, 2048, 'normal'), (65536, 2048, 'equal'),
                                           (5000, 2048, 'few'), (4096, 2048, 'nan'), (6000, 1, 'normal'),
                                           (6000, 4000, 'ints'), (65536, 65535, 'ints'), (6000, 3049, 'bin2048'),
                                           (6000, 3050, 'bin2049')])
def test_topk_select_bitmap_at_is_the_stable_set_in_position_order(run, tokens, k, kind):
    scores = bitmap_scores(tokens, kind, np.random.default_rng(tokens * 7 + k))
    step = np.array([tokens - 1, tokens, k, 1], np.int32)
    got = run('topk_select_bitmap_at', out_ids=np.full(k + 3, -7, np.int32), scores=scores.view(np.int32),
              hist=score_bins(scores), step=step, k=k)
    np.testing.assert_array_equal(got['out_ids'][:k], stable_set(scores, k))
    np.testing.assert_array_equal(got['out_ids'][k:], -7)
    np.testing.assert_array_equal(got['hist'], 0)


def test_topk_select_bitmap_at_cases_reach_both_of_its_paths():
    """The threshold bin of 'bin2048' holds 2048 keys, which the kernel lists; 'bin2049''s holds one more,
    so the kernel reads the scores again. k takes the whole bin, so a key the list lost would show."""
    listed = set()
    for tokens, k, kind in [(6000, 3049, 'bin2048'), (6000, 3050, 'bin2049')]:
        bins = score_bins(bitmap_scores(tokens, kind, np.random.default_rng(tokens * 7 + k)))
        above = np.cumsum(bins[::-1])[::-1]
        listed.add(bool(bins[np.flatnonzero(above >= k).max()] <= 2048))
    assert listed == {True, False}


# Under graph replay: tokens from the step buffer, the scores past it ignored; up to k it writes nothing
# and still zeroes the bins.
@pytest.mark.parametrize('tokens', [100, 2048, 2049, 6000, 20000])
def test_topk_select_bitmap_at_reads_tokens_from_the_step(run, tokens):
    max_tokens, k = 20000, 2048
    scores = np.random.default_rng(tokens).integers(-50, 50, size=max_tokens).astype(F)
    scores[tokens:] = np.inf  # never selected: past tokens
    step = np.array([tokens - 1, tokens, min(k, tokens), 1], np.int32)
    bins = score_bins(scores[:tokens]) if tokens > k else np.arange(4096, dtype=np.uint32)
    got = run('topk_select_bitmap_at', out_ids=np.full(k, -7, np.int32), scores=scores.view(np.int32), hist=bins,
              step=step, k=k)
    if tokens <= k:
        np.testing.assert_array_equal(got['out_ids'], -7)
    else:
        np.testing.assert_array_equal(got['out_ids'], stable_set(scores[:tokens], k))
    np.testing.assert_array_equal(got['hist'], 0)


# kv_rows_bf16_at (#39, ResidentGLM(fuse_glue=True)): store_row_bf16_at of the latent row, and of
# rope_strided's packed [rotated even, rotated odd] (one rounding an operation), in one launch.
@pytest.mark.parametrize('rank,rot', [(512, 64), (333, 10), (16, 0)])
def test_kv_rows_bf16_at_is_store_rope_store(run, rank, rot):
    rng = np.random.default_rng(rank + rot)
    latent_row, rope_x = rng.normal(size=rank).astype(F), rng.normal(size=max(rot, 2)).astype(F)
    angle = rng.normal(size=max(rot // 2, 1)).astype(F)
    cosine, sine = np.cos(angle).astype(F), np.sin(angle).astype(F)
    step = np.array([5, 6, 6, 1], np.int32)
    latents = rng.integers(0, 1 << 16, size=(9, rank)).astype(np.uint16)
    rope_keys = rng.integers(0, 1 << 16, size=(9, max(rot, 1))).astype(np.uint16)
    got = run('kv_rows_bf16_at', latents=latents.copy(), rope_keys=rope_keys.copy(), latent_row=latent_row,
              rope_x=rope_x, cosine=cosine, sine=sine, step=step, rank=rank, rot=rot)
    want_latents = run('store_row_bf16_at', cache=latents.copy(), row=latent_row, step=step, width=rank)['cache']
    np.testing.assert_array_equal(got['latents'], want_latents)
    if rot:
        a, b, p = rope_x[0:rot:2], rope_x[1:rot:2], rot // 2
        rotated = np.concatenate((a * cosine[:p] - b * sine[:p], b * cosine[:p] + a * sine[:p])).astype(F)
        want_rope = run('store_row_bf16_at', cache=rope_keys.copy(), row=rotated, step=step, width=rot)['cache']
        np.testing.assert_array_equal(got['rope_keys'], want_rope)
    else:
        np.testing.assert_array_equal(got['rope_keys'], rope_keys)


@pytest.mark.parametrize('experts,k,ties', [(256, 8, False), (256, 8, True), (7, 3, True), (1, 1, False), (33, 8, True)])
def test_router(run, experts, k, ties):
    rng = np.random.default_rng(experts * 10 + k)
    if ties:
        logits = rng.integers(-3, 3, size=experts).astype(F)
        bias = np.tile(np.array([0.0, 0.5], F), experts)[:experts].copy()
    else:
        logits, bias = rng.normal(size=experts).astype(F), (rng.normal(size=experts) * 0.1).astype(F)
    got = run('router', indices=np.full(k, -7, np.int32), weights=np.zeros(k, F), logits=logits, bias=bias,
              experts=experts, k=k, scale=2.5)
    z = np.exp(-np.abs(logits))
    sigmoid = np.where(logits >= 0, F(1) / (F(1) + z), z / (F(1) + z)).astype(F)
    score = sigmoid + bias
    ids = np.lexsort((np.arange(experts), -score))[:k]  # highest score, then lowest id
    np.testing.assert_array_equal(got['indices'], ids)
    chosen = np.zeros(32, F)
    chosen[:k] = sigmoid[ids]
    np.testing.assert_allclose(got['weights'], chosen[:k] / (warp_tree(chosen) + F(1e-20)) * F(2.5), **CLOSE)


# bf16_matvec_rows: a block a row, the hand bf16_matvec's order (lane-strided sums, then the warp tree), bit for
# bit; 6144 is the router's width, 136 a partial run of lanes, 8192 the most columns it takes.
@pytest.mark.parametrize('rows,cols', [(256, 6144), (5, 136), (33, 8192), (1, 8)])
def test_bf16_matvec_rows(run, rows, cols):
    rng = np.random.default_rng(rows + cols)
    w = (rng.normal(size=(rows, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=cols).astype(F)
    got = run('bf16_matvec_rows', out=np.zeros(rows, F), weight=w.view(np.uint32).reshape(-1), x=x, rows=rows, cols=cols)
    bits_equal(got['out'], warp_tree(strided_sums(bf16(w) * x, 32)))


# router_bf16 (fuse_router): the gate's rows in the hand bf16_matvec's order (lane-strided sums, then the
# warp tree), then router's choice and weights, bit for bit; slot k (fold_shared's) untouched. Ties: a
# zero gate and a bias that repeats, so the lower id must win.
@pytest.mark.parametrize('experts,cols,k,ties', [(256, 6144, 8, False), (7, 136, 3, True), (33, 520, 8, False),
                                                 (256, 64, 8, True), (1, 8, 1, False)])
def test_router_bf16(run, experts, cols, k, ties):
    rng = np.random.default_rng(experts + cols + k)
    w = (rng.normal(size=(experts, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=cols).astype(F)
    if ties:
        w[...] = 0
        bias = np.tile(np.array([0.0, 0.5], F), experts)[:experts].copy()
    else:
        bias = (rng.normal(size=experts) * 0.1).astype(F)
    expected = warp_tree(strided_sums(bf16(w) * x, 32))
    ids, weights = np.full(k + 1, -7, np.int32), np.full(k + 1, 3.0, F)
    got = run('router_bf16', indices=ids, weights=weights, logits=np.zeros(experts, F), gate=w.view(np.uint32).reshape(-1),
              x=x, bias=bias, experts=experts, cols=cols, k=k, scale=2.5, arrivals=np.zeros(1, np.uint32))
    bits_equal(got['logits'], expected)
    want = run('router', indices=ids[:k].copy(), weights=weights[:k].copy(), logits=np.asarray(expected, F), bias=bias,
               experts=experts, k=k, scale=2.5)
    np.testing.assert_array_equal(got['indices'][:k], want['indices'])
    bits_equal(got['weights'][:k], want['weights'])
    assert got['indices'][k] == -7 and got['weights'][k] == 3.0


# --- MLA decode ---------------------------------------------------------------------------

def mla_reference(q, qr, kv, kr, selected, heads, rank, rope, tokens, count, stride, splits, scale, threads):
    tasks = heads * splits
    num, stats = np.zeros((tasks, rank), F), np.zeros((tasks, 2), F)
    for task in range(tasks):
        h, split = divmod(task, splits)
        acc, m, l = np.zeros(rank, F), F(0), F(0)
        for j in range(split, count, splits):
            token = selected[h * stride + j]
            if not 0 <= token < tokens:
                continue
            terms = np.zeros(1024, F)
            terms[:rank] = q[h] * kv[token]
            for d in range(rope):  # rope_dim <= threads here: one rope term per thread
                terms[d % threads] = terms[d % threads] + qr[h, d] * kr[token, d]
            score = block_sum(terms, threads) * F(scale)
            if l > 0:
                next_m = max(m, score)
                alpha, beta = F(np.exp(F(m - next_m))), F(np.exp(F(score - next_m)))
            else:
                next_m, alpha, beta = score, F(0), F(1)
            acc = acc * alpha + beta * kv[token]
            m, l = F(next_m), F(l * alpha + beta)
        num[task], stats[task] = acc, (m, l)
    out = np.zeros((heads, rank), F)
    for h in range(heads):
        st = stats[h * splits:(h + 1) * splits]
        live = st[:, 1] > 0
        if live.any():
            factor = np.exp(st[:, 0] - st[live, 0].max()).astype(F)
            s, den = np.zeros(rank, F), F(0)
            for t in np.flatnonzero(live):
                s = s + num[h * splits + t] * factor[t]
                den = F(den + st[t, 1] * factor[t])
            out[h] = s / den
    return num, stats, out


@pytest.mark.parametrize('heads,rank,rope,tokens,count,splits,stride,threads', [
    (2, 48, 4, 7, 5, 3, 0, 64), (2, 48, 4, 7, 5, 8, 0, 64), (2, 48, 4, 7, 0, 3, 0, 64),
    (8, 512, 64, 1024, 96, 4, 0, 512), (4, 64, 64, 300, 37, 4, 37, 64)])
def test_mla(run, heads, rank, rope, tokens, count, splits, stride, threads):
    rng = np.random.default_rng(heads + rank + count + splits)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)).astype(F), rng.normal(size=(tokens, rope)).astype(F)
    lists = heads if stride else 1
    selected = rng.integers(0, tokens, size=max(1, lists * max(count, 1))).astype(np.int32)
    if count:
        selected[1 % selected.size] = -1
        selected[-1] = tokens + 5
    scale = rank ** -0.5
    part = run('mla_partial', numerator=np.zeros((heads * splits, rank), F), stats=np.zeros((heads * splits, 2), F),
               q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, selected=selected, heads=heads, rank=rank,
               rope_dim=rope, tokens=tokens, count=count, selection_stride=stride, splits=splits, scale=scale,
               threads=threads)
    out = run('mla_merge', out=np.zeros((heads, rank), F), numerator=part['numerator'], stats=part['stats'],
              heads=heads, rank=rank, splits=splits)['out']
    num, stats, expected = mla_reference(q, qr, kv, kr, selected, heads, rank, rope, tokens, count, stride, splits,
                                         scale, threads)
    np.testing.assert_allclose(part['stats'], stats, rtol=1e-5, atol=1e-37)
    np.testing.assert_allclose(part['numerator'], num, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('heads,rank,rope,tokens,count,splits,stride,threads', [
    (2, 48, 4, 7, 5, 3, 0, 64), (2, 48, 4, 7, 5, 8, 0, 64), (2, 48, 4, 7, 0, 3, 0, 64), (3, 48, 4, 7, 5, 1, 0, 64),
    (8, 512, 64, 1024, 96, 4, 0, 512), (4, 64, 64, 300, 37, 4, 37, 64)])
def test_mla_decode_equals_partial_then_merge(run, heads, rank, rope, tokens, count, splits, stride, threads):
    """mla_decode (one launch; head h's last split merges h) leaves mla_partial's partials and
    mla_merge's result, bit for bit."""
    rng = np.random.default_rng(heads * 3 + rank + count + splits)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)).astype(F), rng.normal(size=(tokens, rope)).astype(F)
    selected = rng.integers(0, tokens, size=max(1, (heads if stride else 1) * max(count, 1))).astype(np.int32)
    if count:
        selected[1 % selected.size] = -1
        selected[-1] = tokens + 5
    common = dict(q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, selected=selected, heads=heads, rank=rank,
                  rope_dim=rope, tokens=tokens, count=count, selection_stride=stride, splits=splits,
                  scale=rank ** -0.5, threads=threads)
    part = run('mla_partial', numerator=np.zeros((heads * splits, rank), F), stats=np.zeros((heads * splits, 2), F), **common)
    want = run('mla_merge', out=np.zeros((heads, rank), F), numerator=part['numerator'], stats=part['stats'],
               heads=heads, rank=rank, splits=splits)['out']
    got = run('mla_decode', out=np.full((heads, rank), 7, F), numerator=np.zeros((heads, splits, rank), F),
              stats=np.zeros((heads, splits, 2), F), arrivals=np.zeros(heads, np.int32), **common)
    bits_equal(got['numerator'].reshape(heads * splits, rank), part['numerator'])
    bits_equal(got['stats'].reshape(heads * splits, 2), part['stats'])
    bits_equal(got['out'], want)


def mla_float64(q, qr, kv, kr, selected, stride, count, tokens, scale):
    """Attention of each head over its selected tokens in float64 (invalid ids skipped)."""
    out = np.zeros(q.shape, np.float64)
    for h in range(q.shape[0]):
        ids = [int(selected[h * stride + j]) for j in range(count)]
        ids = [i for i in ids if 0 <= i < tokens]
        if not ids:
            continue
        s = (kv[ids].astype(np.float64) @ q[h] + kr[ids].astype(np.float64) @ qr[h]) * scale
        p = np.exp(s - s.max())
        out[h] = p @ kv[ids].astype(np.float64) / p.sum()
    return out


# mla_chunks_at (#33): mla_decode_at's arguments and merge, 32 tokens a chunk; equal to mla_decode_at
# to rounding (another order of the sums) and to float64 attention.
@pytest.mark.parametrize('heads,rank,rope,tokens,stride,splits_max', [(2, 48, 4, 700, 0, 3), (3, 64, 64, 900, 1, 16)])
def test_mla_chunks_at_matches_mla_decode_at(run, heads, rank, rope, tokens, stride, splits_max):
    rng = np.random.default_rng(heads * 11 + rank)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)).astype(F), rng.normal(size=(tokens, rope)).astype(F)
    scale = rank ** -0.5
    common = dict(q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, heads=heads, rank=rank, rope_dim=rope,
                  selection_stride=stride, scale=scale, threads=64, splits_max=splits_max)
    for count, splits in ((0, 1), (5, 1), (33, 1), (300, 2), (640, 3)):
        selected = rng.integers(0, tokens, size=max(1, (heads if stride else 1) * max(count, 1))).astype(np.int32)
        if count > 3:
            selected[1], selected[-1] = -1, tokens + 5  # skipped
        values = dict(selected=selected, step=np.array([tokens - 1, tokens, count, splits], np.int32), **common)
        shapes = dict(out=np.full((heads, rank), 7, F), numerator=np.zeros((heads, splits_max, rank), F),
                      stats=np.zeros((heads, splits_max, 2), F), arrivals=np.zeros(heads, np.int32))
        want = run('mla_decode_at', **values, **{k: v.copy() for k, v in shapes.items()})['out']
        got = run('mla_chunks_at', **values, **{k: v.copy() for k, v in shapes.items()})['out']
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got, mla_float64(q, qr, kv, kr, selected, stride, count, tokens, scale), rtol=1e-4, atol=1e-5)


# mla_heads_at (#33): four heads a block, each latent row read once for them; each head's arithmetic is
# mla_chunks_at's, so out, and the numerators and stats laid out per head, are equal bit for bit.
@pytest.mark.parametrize('heads,rank,rope,tokens,splits_max', [(4, 48, 4, 700, 3), (8, 64, 64, 900, 16), (4, 1024, 8, 300, 2)])
def test_mla_heads_at_equals_mla_chunks_at(run, heads, rank, rope, tokens, splits_max):
    rng = np.random.default_rng(heads * 13 + rank)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)).astype(F), rng.normal(size=(tokens, rope)).astype(F)
    scale = rank ** -0.5
    common = dict(q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, heads=heads, rank=rank, rope_dim=rope,
                  selection_stride=0, scale=scale, threads=64, splits_max=splits_max)
    groups = heads // 4
    for count, splits in ((0, 1), (5, 1), (33, 1), (300, 2), (min(640, tokens), min(3, splits_max))):
        selected = rng.integers(0, tokens, size=max(1, count)).astype(np.int32)
        if count > 3:
            selected[1], selected[-1] = -1, tokens + 5  # skipped
        values = dict(selected=selected, step=np.array([tokens - 1, tokens, count, splits], np.int32), **common)
        want = run('mla_chunks_at', out=np.full((heads, rank), 7, F), numerator=np.zeros((heads, splits_max, rank), F),
                   stats=np.zeros((heads, splits_max, 2), F), arrivals=np.zeros(heads, np.int32), **values)
        got = run('mla_heads_at', out=np.full((groups, 4 * rank), 7, F), numerator=np.zeros((groups, splits_max, 4 * rank), F),
                  stats=np.zeros((groups, splits_max, 8), F), arrivals=np.zeros(groups, np.int32), **values)
        bits_equal(got['out'].reshape(heads, rank), want['out'])
        per_head = lambda a, n: a.reshape(groups, splits_max, 4, n).transpose(0, 2, 1, 3).reshape(heads, splits_max, n)[:, :splits]
        bits_equal(per_head(got['numerator'], rank), want['numerator'][:, :splits])
        bits_equal(per_head(got['stats'], 2), want['stats'][:, :splits])
        np.testing.assert_allclose(got['out'].reshape(heads, rank), mla_float64(q, qr, kv, kr, selected, 0, count, tokens, scale),
                                   rtol=1e-4, atol=1e-5)


# mla_flash_at + mla_flash_merge_at (#33): each lane's query and sums in registers, one token at a time;
# out agrees with mla_chunks_at to rounding (the softmax folds one token at a time, not 32) and with
# float64 attention. Ids out of range are skipped; splits past step's write nothing.
def flash(run, heads, splits_max, values):
    groups, rank = heads // 16, values['rank']
    got = run('mla_flash_at', numerator=np.zeros((groups, splits_max, 16 * rank), F), stats=np.zeros((groups, splits_max, 32), F),
              **{k: v for k, v in values.items() if k not in ('threads',)})
    out = run('mla_flash_merge_at', out=np.full((heads, rank), 7, F), numerator=got['numerator'], stats=got['stats'],
              step=values['step'], heads=heads, rank=rank, splits_max=splits_max)['out']
    return got, out


@pytest.mark.parametrize('heads,tokens,splits_max', [(16, 700, 3), (32, 300, 16)])
def test_mla_flash_at_matches_mla_chunks_at_and_float64(run, heads, tokens, splits_max):
    rank, rope = 512, 64
    rng = np.random.default_rng(heads + tokens)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)).astype(F), rng.normal(size=(tokens, rope)).astype(F)
    scale = rank ** -0.5
    common = dict(q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, heads=heads, rank=rank, rope_dim=rope,
                  selection_stride=0, scale=scale, splits_max=splits_max)
    for count, splits in ((0, 1), (5, 1), (33, 2), (min(200, tokens), min(3, splits_max))):
        selected = rng.integers(0, tokens, size=max(1, count)).astype(np.int32)
        if count > 3:
            selected[1], selected[-1] = -1, tokens + 5  # skipped
        values = dict(selected=selected, step=np.array([tokens - 1, tokens, count, splits], np.int32), **common)
        got, out = flash(run, heads, splits_max, values)
        assert (got['numerator'][:, splits:] == 0).all() and (got['stats'][:, splits:] == 0).all()
        want = run('mla_chunks_at', out=np.full((heads, rank), 7, F), numerator=np.zeros((heads, splits_max, rank), F),
                   stats=np.zeros((heads, splits_max, 2), F), arrivals=np.zeros(heads, np.int32), threads=64, **values)['out']
        np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out, mla_float64(q, qr, kv, kr, selected, 0, count, tokens, scale), rtol=1e-4, atol=1e-5)


def to_bf16(x):
    """float32 -> bf16 bits, round to nearest even (finite values)."""
    u = np.asarray(x, F).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


# mla_q_bf16_at + mla_tc_at + mla_flash_merge_at (#33): the bf16 cache on the tensor cores, one
# 32-token chunk a split.
# The products are of bf16 values (the queries and P rounded to bf16) with f32 sums, so out agrees
# with float64 attention over the bf16 cache and queries to bf16 rounding.
@pytest.mark.parametrize('heads,tokens,splits_max', [(16, 700, 8), (32, 300, 4)])
def test_mla_tc_at_matches_float64_on_bf16_values(run, heads, tokens, splits_max):
    rank, rope = 512, 64
    rng = np.random.default_rng(heads + tokens + 1)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = to_bf16(rng.normal(size=(tokens, rank))), to_bf16(rng.normal(size=(tokens, rope)))
    scale = rank ** -0.5
    common = dict(q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, heads=heads, rank=rank,
                  rope_dim=rope, selection_stride=0, scale=scale, splits_max=splits_max)
    as_f32 = lambda b: bf16(b)
    for count in (0, 5, 33, min(32 * splits_max, tokens, 200)):
        splits = max(1, (count + 31) // 32)
        selected = rng.integers(0, tokens, size=max(1, 32 * splits)).astype(np.int32)
        if count > 3:
            selected[1], selected[count - 1] = -1, tokens + 5  # skipped
        values = dict(selected=selected, step=np.array([tokens - 1, tokens, count, splits], np.int32), **common)
        groups = heads // 16
        qs = run('mla_q_bf16_at', q_t=np.zeros((groups, 1024, 8), np.uint16), qr_t=np.zeros((groups, 128, 8), np.uint16),
                 q_latent=q, q_rope=qr, heads=heads, rank=rank, rope_dim=rope)
        tc = {k: v for k, v in values.items() if k not in ('q_latent', 'q_rope')}
        got = run('mla_tc_at', numerator=np.zeros((groups * splits_max * 16, rank), F), stats=np.zeros((groups * splits_max, 32), F),
                  q_t=qs['q_t'].reshape(-1, 8), qr_t=qs['qr_t'].reshape(-1, 8), **tc)
        got = dict(numerator=got['numerator'].reshape(groups, splits_max, 16 * rank), stats=got['stats'].reshape(groups, splits_max, 32))
        assert (got['numerator'][:, splits:] == 0).all() and (got['stats'][:, splits:] == 0).all()
        out = run('mla_flash_merge_at', out=np.full((heads, rank), 7, F), numerator=got['numerator'], stats=got['stats'],
                  step=values['step'], heads=heads, rank=rank, splits_max=splits_max)['out']
        want = mla_float64(as_f32(to_bf16(q)), as_f32(to_bf16(qr)), as_f32(kv), as_f32(kr), selected, 0, count, tokens, scale)
        np.testing.assert_allclose(out, want, rtol=0, atol=1e-2 * max(1.0, np.abs(want).max()))


@pytest.mark.parametrize('heads,rank,rope,stride', [(16, 256, 64, 0), (16, 512, 32, 0), (16, 512, 64, 10), (8, 512, 64, 0)])
def test_mla_flash_at_writes_nothing_off_its_shape(run, heads, rank, rope, stride):
    rng = np.random.default_rng(rank + rope + stride)
    tokens, groups = 50, max(1, heads // 16)
    values = dict(q_latent=rng.normal(size=heads * rank).astype(F), q_rope=rng.normal(size=heads * rope).astype(F),
                  cache_latent=rng.normal(size=tokens * rank).astype(F), cache_rope=rng.normal(size=tokens * rope).astype(F),
                  selected=np.arange(heads * 10, dtype=np.int32), step=np.array([tokens - 1, tokens, 10, 1], np.int32),
                  heads=heads, rank=rank, rope_dim=rope, selection_stride=stride, splits_max=2, scale=0.1)
    got = run('mla_flash_at', numerator=np.zeros((groups, 2, 16 * rank), F), stats=np.zeros((groups, 2, 32), F), **values)
    assert (got['numerator'] == 0).all() and (got['stats'] == 0).all()


def test_mla_heads_at_writes_nothing_for_per_head_selections(run):
    heads, rank, rope, tokens = 4, 32, 4, 50
    rng = np.random.default_rng(3)
    got = run('mla_heads_at', out=np.full((1, 4 * rank), 7, F), numerator=np.zeros((1, 2, 4 * rank), F),
              stats=np.zeros((1, 2, 8), F), arrivals=np.zeros(1, np.int32), q_latent=rng.normal(size=heads * rank).astype(F),
              q_rope=rng.normal(size=heads * rope).astype(F), cache_latent=rng.normal(size=tokens * rank).astype(F),
              cache_rope=rng.normal(size=tokens * rope).astype(F), selected=np.arange(heads * 10, dtype=np.int32),
              step=np.array([tokens - 1, tokens, 10, 1], np.int32), heads=heads, rank=rank, rope_dim=rope,
              selection_stride=10, splits_max=2, scale=0.1, threads=64)
    assert (got['numerator'] == 0).all() and (got['stats'] == 0).all() and (got['out'] == 7).all()


@pytest.mark.parametrize('heads,rank,rope,tokens,stride,threads,splits_max', [
    (2, 48, 4, 700, 0, 64, 3), (4, 64, 64, 900, 0, 64, 16)])
def test_mla_decode_at_equals_mla_decode_for_each_splits(run, heads, rank, rope, tokens, stride, threads, splits_max):
    """mla_decode_at (glm-vx #27) reads tokens, count and splits from the step buffer and runs
    splits_max splits per head, the ones past splits idle; for splits 1 to
    3 as a replayed graph would, it equals mla_decode with those values bit for bit. (The host
    path keeps its own counts; gpu/test_spawn_hardware.py checks the arrivals buffer.)"""
    rng = np.random.default_rng(heads + rank + splits_max)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)).astype(F), rng.normal(size=(tokens, rope)).astype(F)
    common = dict(q_latent=q, q_rope=qr, cache_latent=kv, cache_rope=kr, heads=heads, rank=rank, rope_dim=rope,
                  selection_stride=stride, scale=rank ** -0.5, threads=threads)
    for count, splits in ((5, 1), (300, 2), (640, 3)):
        selected = rng.permutation(tokens)[:count].astype(np.int32)
        want = run('mla_decode', out=np.full((heads, rank), 7, F), numerator=np.zeros((heads, splits, rank), F),
                   stats=np.zeros((heads, splits, 2), F), arrivals=np.zeros(heads, np.int32), selected=selected,
                   tokens=tokens, count=count, splits=splits, **common)
        got = run('mla_decode_at', out=np.full((heads, rank), 7, F), numerator=np.zeros((heads, splits_max, rank), F),
                  stats=np.zeros((heads, splits_max, 2), F), arrivals=np.zeros(heads, np.int32), selected=selected,
                  step=np.array([tokens - 1, tokens, count, splits], np.int32), splits_max=splits_max, **common)
        bits_equal(got['numerator'][:, :splits], want['numerator'])
        bits_equal(got['stats'][:, :splits], want['stats'])
        bits_equal(got['out'], want['out'])


# index_scores_tc_at (#32, ResidentGLM(index_kernel='tc')): q and the key rows rounded to bf16, the
# products on the tensor cores with f32 sums, against float64 sums of the same bf16 values; out past
# step[1]'s tokens is not written.
@pytest.mark.parametrize('heads,dim,tokens,max_tokens', [(32, 128, 300, 384), (8, 64, 129, 256), (5, 128, 40, 40),
                                                         (32, 64, 1, 128), (17, 128, 255, 256)])
def test_index_scores_tc_at_matches_float64_on_bf16_values(run, heads, dim, tokens, max_tokens):
    rng = np.random.default_rng(heads + dim + tokens)
    q, keys = rng.normal(size=(heads, dim)).astype(F), rng.normal(size=(max_tokens, dim)).astype(F)
    weights, scale = rng.normal(size=heads).astype(F), dim ** -0.5
    out = run('index_scores_tc_at', out=np.full(max_tokens, 7, F), q=q, keys=keys, weights=weights,
              step=np.array([tokens - 1, tokens, 0, 0], np.int32), max_tokens=max_tokens, heads=heads, dim=dim,
              scale=scale)['out']
    qb, kb = bf16(to_bf16(q)).astype(np.float64), bf16(to_bf16(keys[:tokens])).astype(np.float64)
    terms = np.maximum(kb @ qb.T * scale, 0) * weights.astype(np.float64)
    np.testing.assert_allclose(out[:tokens], terms.sum(-1), rtol=0, atol=1e-5 * np.abs(terms).sum(-1).max())
    assert (out[tokens:] == 7).all()


@pytest.mark.parametrize('heads,dim', [(33, 128), (32, 96), (0, 64)])
def test_index_scores_tc_at_writes_nothing_off_its_shape(run, heads, dim):
    rng = np.random.default_rng(heads + dim)
    out = run('index_scores_tc_at', out=np.full(128, 7, F), q=rng.normal(size=(max(heads, 1), dim)).astype(F),
              keys=rng.normal(size=(128, dim)).astype(F), weights=np.ones(max(heads, 1), F),
              step=np.array([99, 100, 0, 0], np.int32), max_tokens=128, heads=heads, dim=dim, scale=0.1)['out']
    assert (out == 7).all()


@pytest.mark.parametrize('n', [256, 300, 6144])
def test_tp_all_reduce_one_rank(run, n):
    # One rank (the host path cannot run several: a rank waits on the others' flags): x stays x, and
    # each call copies x into the half of its epoch's parity, raises every block's flag to the epoch.
    blocks = -(-n // 256)
    rng = np.random.default_rng(n)
    mine, flags, epochs = np.full(2 * n, -1.0, F), np.zeros(blocks, np.uint32), np.zeros(blocks, np.uint32)
    other = np.zeros(2 * n, F), np.zeros(blocks, np.uint32)
    for call in range(3):
        x = rng.normal(size=n).astype(F)
        values = dict(x=x.copy(), mine=mine, my_flags=flags, epochs=epochs, ranks=1, n=n)
        values.update({f'd{r}': mine if r == 0 else other[0] for r in range(8)})
        values.update({f'f{r}': flags if r == 0 else other[1] for r in range(8)})
        got = run('tp_all_reduce', **values)
        bits_equal(got['x'], x)
        mine, flags, epochs = got['mine'], got['my_flags'], got['epochs']
        assert (epochs == call + 1).all() and (flags == call + 1).all()
        half = (call + 1) % 2
        bits_equal(mine[half * n:(half + 1) * n], x)


@pytest.mark.parametrize('n,blocks', [(6144, 8), (6144, 16), (300, 1), (1029, 2)])
def test_tp_sum_rmsnorm_one_rank(run, n, blocks):
    # One rank (the host path cannot run several: a rank spins until the others' words arrive): the sum is x;
    # residual_out = x + residual and out its RMSNorm in residual_rmsnorm8's order at 1024 threads, bit for bit;
    # each call pushes x's bits with the call's epoch into its half of the inbox and counts the call.
    rng = np.random.default_rng(n + blocks)
    inbox, epochs, ss = np.zeros(2 * 8 * n, np.uint64), np.zeros(blocks, np.uint32), np.zeros(1024, F)
    for call in range(3):
        x, r, w = (rng.normal(size=n).astype(F) for _ in range(3))
        values = dict(out=np.zeros(n, F), residual_out=np.zeros(n, F), x=x.view(np.uint32), residual=r, weight=w,
                      ss=ss, epochs=epochs, inbox=inbox, rank=0, ranks=1, n=n, eps=1e-5, blocks=blocks,
                      arrivals=np.zeros(1, np.int32))
        values.update({f'p{i}': inbox for i in range(8)})  # the ranks past `ranks` get this rank's own
        got = run('tp_sum_rmsnorm', **values)
        bits_equal(got['residual_out'], x + r)
        bits_equal(got['out'], rms_reference((x + r)[None], w, 1e-5, 1024)[0])
        epochs, inbox, ss = got['epochs'], got['p0'], got['ss']
        assert (epochs == call + 1).all()
        words = inbox[(call + 1) % 2 * 8 * n:][:n]
        assert ((words >> np.uint64(32)) == call + 1).all()
        assert ((words & np.uint64(0xFFFFFFFF)).astype(np.uint32) == x.view(np.uint32)).all()


@pytest.mark.parametrize('kernel', ['tc', 'flash'])
def test_mla_padded_heads_do_not_change_the_real_ones(run, kernel):
    # Tensor parallelism over 8 GPUs leaves GLM-5.3 8 heads a rank; ResidentGLM pads them to the 16 of
    # mla_tc_at's tile (and mla_flash_at's group) with query rows of zeros (#57). Heads 0..7's results must
    # be the same bits whatever rows 8..15 hold.
    heads, real, rank, rope, tokens, splits_max = 16, 8, 512, 64, 300, 4
    rng = np.random.default_rng(57)
    q, qr = rng.normal(size=(heads, rank)).astype(F), rng.normal(size=(heads, rope)).astype(F)
    kv, kr = rng.normal(size=(tokens, rank)), rng.normal(size=(tokens, rope))
    if kernel == 'tc':
        kv, kr = to_bf16(kv), to_bf16(kr)
    else:
        kv, kr = kv.astype(F), kr.astype(F)
    count, splits = 100, 4
    selected = rng.integers(0, tokens, size=128).astype(np.int32)
    step = np.array([tokens - 1, tokens, count, splits], np.int32)
    outs = []
    for pad in ('zeros', 'noise'):
        q2, qr2 = q.copy(), qr.copy()
        if pad == 'zeros':
            q2[real:], qr2[real:] = 0, 0
        common = dict(cache_latent=kv, cache_rope=kr, selected=selected, step=step, heads=heads, rank=rank, rope_dim=rope,
                      selection_stride=0, splits_max=splits_max, scale=rank ** -0.5)
        numerator, stats = np.zeros((1, splits_max, 16 * rank), F), np.zeros((1, splits_max, 32), F)
        if kernel == 'tc':
            qs = run('mla_q_bf16_at', q_t=np.zeros((1, 1024, 8), np.uint16), qr_t=np.zeros((1, 128, 8), np.uint16),
                     q_latent=q2, q_rope=qr2, heads=heads, rank=rank, rope_dim=rope)
            got = run('mla_tc_at', numerator=numerator.reshape(splits_max * 16, rank), stats=stats.reshape(splits_max, 32),
                      q_t=qs['q_t'].reshape(-1, 8), qr_t=qs['qr_t'].reshape(-1, 8), **common)
        else:
            got = run('mla_flash_at', numerator=numerator, stats=stats, q_latent=q2, q_rope=qr2, **common)
        out = run('mla_flash_merge_at', out=np.zeros((heads, rank), F), numerator=got['numerator'].reshape(1, splits_max, -1),
                  stats=got['stats'].reshape(1, splits_max, 32), step=step, heads=heads, rank=rank, splits_max=splits_max)['out']
        assert np.isfinite(out).all()
        outs.append(out[:real])
    assert np.abs(outs[0]).max() > 0
    bits_equal(outs[0], outs[1])


@pytest.mark.parametrize('ranks,n', [(2, 6144), (3, 300), (8, 257)])
def test_sum_ranks_adds_in_rank_order(run, ranks, n):
    g = np.random.default_rng(ranks * n).normal(size=(ranks, n)).astype(F) * np.logspace(-3, 3, ranks, dtype=F)[:, None]
    want = g[0] + g[1]
    for r in range(2, ranks):
        want = want + g[r]
    got = run('sum_ranks', out=np.zeros(n, F), gathered=g, ranks=ranks, n=n)['out']
    bits_equal(got, want)
