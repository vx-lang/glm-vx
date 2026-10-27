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


def fp8_word_order(terms):
    """fp8_matvec4's per-lane order: lane l takes columns tile * 128 + 4l .. + 3 of each tile in
    turn. Reorder the columns so that strided_sums(.., 32) adds them in that order (zeros pad the
    last tile, in lanes that add nothing there)."""
    rows, cols = terms.shape
    tiles = -(-cols // 128)
    padded = np.zeros((rows, tiles * 128), F)
    padded[:, :cols] = terms
    return padded.reshape(rows, tiles, 32, 4).transpose(0, 1, 3, 2).reshape(rows, tiles * 128)


def fp8_matvec4_case(rows, cols, seed):
    rng = np.random.default_rng(seed)
    weight = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    weight[(weight & 0x7F) == 0x7F] = 0
    scales = rng.uniform(0.001, 0.1, size=(-(-rows // 128), -(-cols // 128))).astype(F)
    x = rng.normal(size=cols).astype(F)
    per_col = scales[np.arange(rows)[:, None] // 128, np.arange(cols)[None, :] // 128]
    expected = warp_tree(strided_sums(fp8_word_order((fp8(weight) * per_col) * x), 32))
    return weight, scales, x, expected


# Four codes a load (a u32 view of the bytes), decoded through a shared-memory table: (5, 132) has
# a partial last tile, (300, 1000) one with a word past the end of some lanes, and 6144 and 16384
# are GLM-5.3's q_a/MLP and o_proj widths (tile counts that are and are not multiples of 4).
@pytest.mark.parametrize('rows,cols', [(5, 132), (300, 1000), (130, 6144), (4, 16384), (3, 640)])
def test_fp8_matvec4(run, rows, cols):
    weight, scales, x, expected = fp8_matvec4_case(rows, cols, rows * 13 + cols)
    got = run('fp8_matvec4', out=np.zeros(rows, F), weight=weight.view(np.uint32).reshape(-1), scales=scales.reshape(-1),
              x=x, rows=rows, cols=cols)['out']
    bits_equal(got, expected)


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
@pytest.mark.parametrize('experts,k,inter,hidden', [(1, 1, 3, 5), (12, 8, 9, 131), (256, 8, 130, 300)])
def test_fused_moe_equals_the_per_expert_kernels(run, fmt, experts, k, inter, hidden):
    rng = np.random.default_rng(experts * 7 + inter)
    gate, up = moe_weights(rng, fmt, experts, inter, hidden), moe_weights(rng, fmt, experts, inter, hidden)
    down = moe_weights(rng, fmt, experts, hidden, inter)
    x, ids = rng.normal(size=hidden).astype(F), expert_ids(rng, experts, k)
    weights = rng.uniform(0.01, 1, size=k).astype(F)
    args = (gate, up, down, x, ids, weights, experts, inter, hidden)
    bits_equal(fused_moe(run, fmt, *args), per_expert_moe(run, fmt, *args))


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
