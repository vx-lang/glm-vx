#!/usr/bin/env python3
"""Write index_scores_mma_at.vx (#32): the indexer's scores on the tensor cores, the warp's whole
work one inline-PTX block in an mlir! macro. Run from anywhere; rerun after editing, and commit the
output (build.sh checks it with --check).

out[t] = sum over the 32 heads of weights[h] * relu(scale * (q[h] . keys[t])), dim 128, for every
t < step[1]; out past step[1] is not written. A warp takes 16 tokens:

- Each lane loads its two key rows (tokens g = lane / 4 and g + 8) as float4s straight from the
  f32 cache and rounds each pair to bf16 with one cvt.rn.bf16x2.f32. The dims are permuted so a
  lane's mma fragment is one contiguous float4: in k-step s, the fragment columns {2c, 2c + 1}
  and {2c + 8, 2c + 9} of lane column c = lane % 4 are dims 16 s + 4 c + {0, 1} and + {2, 3}. The
  dot product does not depend on the order of the dims, and q uses the same permutation.
- For each 8-head tile n: the B fragment (head 8 n + g) the same way, eight
  mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32 over the 128 dims, then w * relu(scale * s) of the
  lane's two heads added into its two tokens' sums.
- Two butterfly shuffles add the four lanes of a token; lane c == 0 stores it.

Precision: q and k in bf16, products exact, f32 sums (SGLang rounds both to FP8 here). The heads
are added in another order than index_scores_at's, so out agrees with it to bf16 rounding.

`python3 gen_index_mma.py --check` exits 1 if the committed output differs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIM, HEADS = 128, 32
STEPS, TILES = DIM // 16, HEADS // 8
TOKENS_PER_BLOCK, WARPS = 128, 4


def warp_ptx(bf16_keys=False):
    """The PTX of one lane's work, returning ($0) the score of token first + lane. Operands: $1 to $4
    the key rows first + g, + 8, + 16 and + 24 (byte address of dim 4 c), $5 q row g (dim 4 c),
    $6 weights[2 c], $7 scale. With bf16_keys the key rows are bf16 (index_key_bf16_at's words, the same
    cvt.rn.bf16x2.f32 of the same values): one 8-byte load is the two A registers that the f32 path
    makes from a float4, so the scores are the same bits from half the bytes."""
    p = ['{', '.reg .b32 a<%d>;' % (8 * STEPS), '.reg .f32 x<4>, y<4>, z<4>, c<8>;', '.reg .b32 b<2>, l, src;',
         '.reg .f32 t, acc<4>, w0, w1, sh<4>;', '.reg .pred q1, q2, q3;']
    for s in range(STEPS):
        for m, (k0, k1) in enumerate((('$1', '$2'), ('$3', '$4'))):
            base = 8 * s + 4 * m
            if bf16_keys:
                p.append(f'ld.global.nc.v2.b32 {{a{base}, a{base + 2}}}, [{k0}+{32 * s}];')
                p.append(f'ld.global.nc.v2.b32 {{a{base + 1}, a{base + 3}}}, [{k1}+{32 * s}];')
                continue
            p.append(f'ld.global.nc.v4.f32 {{x0, x1, x2, x3}}, [{k0}+{64 * s}];')
            p.append(f'ld.global.nc.v4.f32 {{y0, y1, y2, y3}}, [{k1}+{64 * s}];')
            # A of tile m: (row g, cols 2c..), (row g + 8, ..), (row g, cols 2c + 8..), (row g + 8, ..);
            # cvt.rn.bf16x2.f32 d, hi, lo puts the lower k in the low half.
            p += [f'cvt.rn.bf16x2.f32 a{base}, x1, x0;', f'cvt.rn.bf16x2.f32 a{base + 1}, y1, y0;',
                  f'cvt.rn.bf16x2.f32 a{base + 2}, x3, x2;', f'cvt.rn.bf16x2.f32 a{base + 3}, y3, y2;']
    p += [f'mov.f32 acc{i}, 0f00000000;' for i in range(4)]
    for n in range(TILES):
        p += [f'mov.f32 c{i}, 0f00000000;' for i in range(8)]
        for s in range(STEPS):
            p.append(f'ld.global.nc.v4.f32 {{z0, z1, z2, z3}}, [$5+{n * 8 * DIM * 4 + 64 * s}];')
            p += ['cvt.rn.bf16x2.f32 b0, z1, z0;', 'cvt.rn.bf16x2.f32 b1, z3, z2;']
            for m in range(2):
                base = 8 * s + 4 * m
                cs = ', '.join(f'c{4 * m + i}' for i in range(4))
                p.append(f'mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{cs}}}, '
                         f'{{a{base}, a{base + 1}, a{base + 2}, a{base + 3}}}, {{b0, b1}}, {{{cs}}};')
        # Tile m: c[4m], c[4m+1] row g (+ 16 m), heads 8 n + 2 c and + 1; c[4m+2], c[4m+3] row g + 8 (+ 16 m).
        p.append(f'ld.global.nc.v2.f32 {{w0, w1}}, [$6+{32 * n}];')
        for m in range(2):
            for i, acc, w in ((0, 2 * m, 'w0'), (1, 2 * m, 'w1'), (2, 2 * m + 1, 'w0'), (3, 2 * m + 1, 'w1')):
                p += [f'mul.f32 t, c{4 * m + i}, $7;', 'max.f32 t, t, 0f00000000;',
                      f'fma.rn.f32 acc{acc}, {w}, t, acc{acc};']
    # acc0..3: rows g, g + 8, g + 16, g + 24 of this lane's column c; add the four columns.
    for i in range(4):
        for m in (1, 2):
            p += [f'shfl.sync.bfly.b32 t, acc{i}, {m}, 31, -1;', f'add.f32 acc{i}, acc{i}, t;']
    # Lane l takes row l: acc[l / 8] of lane 4 (l % 8).
    p += ['mov.u32 l, %laneid;', 'and.b32 src, l, 7;', 'shl.b32 src, src, 2;']
    p += [f'shfl.sync.idx.b32 sh{i}, acc{i}, src, 31, -1;' for i in range(4)]
    p += ['shr.u32 l, l, 3;', 'setp.eq.u32 q1, l, 1;', 'setp.eq.u32 q2, l, 2;', 'setp.eq.u32 q3, l, 3;',
          'mov.f32 $0, sh0;', '@q1 mov.f32 $0, sh1;', '@q2 mov.f32 $0, sh2;', '@q3 mov.f32 $0, sh3;', '}']
    return ' '.join(p)


F32_1E_4, F32_448, F32_1 = '0f38D1B717', '0f43E00000', '0f3F800000'


def key_fp8_ptx():
    """index_key_fp8_at's lane: $1 the address of key dims 4 l .. 4 l + 3 (f32). Returns ($0) the four
    e4m3 codes in one word (low half) and the row's scale's bits (high half). SGLang's indexer_k.cuh:
    each value rounded to bf16 first, scale = max(1e-4, amax) / 448, codes = e4m3(x * (1 / scale)),
    round to nearest, saturating."""
    p = ['{', '.reg .f32 x<4>, m, t, s, inv;', '.reg .b16 h<3>;', '.reg .b32 w, sb;',
         'ld.global.v4.f32 {x0, x1, x2, x3}, [$1];']
    for i in range(4):
        p += [f'cvt.rn.bf16.f32 h2, x{i};', f'cvt.f32.bf16 x{i}, h2;']
    p += ['abs.f32 m, x0;'] + [f'abs.f32 t, x{i}; max.f32 m, m, t;' for i in (1, 2, 3)]
    p += [f'shfl.sync.bfly.b32 t, m, {k}, 31, -1; max.f32 m, m, t;' for k in (16, 8, 4, 2, 1)]
    p += [f'max.f32 s, m, {F32_1E_4};', f'div.rn.f32 s, s, {F32_448};', f'div.rn.f32 inv, {F32_1}, s;']
    p += [f'mul.f32 x{i}, x{i}, inv;' for i in range(4)]
    p += ['cvt.rn.satfinite.e4m3x2.f32 h0, x1, x0;', 'cvt.rn.satfinite.e4m3x2.f32 h1, x3, x2;',
          'mov.b32 w, {h0, h1};', 'mov.b32 sb, s;', 'mov.b64 $0, {w, sb};', '}']
    return ' '.join(p)


def scores_fp8_ptx():
    """index_scores_fp8_at's lane, returning ($0) the score of token first + lane. Operands: $1 to $4
    the code rows first + g, + 8, + 16, + 24 (byte 16 c), $5 to $8 their scales, $9 q row g (f32 dim
    16 c), $10 weights[2 c], $11 scale (index_dim ** -0.5).

    In k-step s = 2 j + h (k32 each), the fragment bytes {4c..4c+3} and {16+4c..} of lane column c
    are dims 64 j + 16 c + 8 h + {0..3} and + {4..7}: each lane loads 16 contiguous bytes of a row
    per j. q is quantized here, as SGLang's fused_q_indexer_rope_first_quant: one scale a head,
    max(1e-4, amax) / 448, with q_scale * scale folded into the head's weight."""
    p = ['{', '.reg .b32 a<32>, b<8>, l, src;', '.reg .f32 x<32>, c<8>, t, m, s, inv, qs0, qs1, w0, w1, ks<4>, acc<4>, sh<4>;',
         '.reg .b16 h<2>;', '.reg .pred q1, q2, q3;']
    # A: a[16 j + 4 r + i] = word i of row r (0..3: first + g, + 8, + 16, + 24) at 64 j + 16 c.
    for j in range(2):
        for r in range(4):
            base = 16 * j + 4 * r
            p.append(f'ld.global.nc.v4.u32 {{a{base}, a{base + 1}, a{base + 2}, a{base + 3}}}, [${r + 1}+{64 * j}];')
    for r in range(4):
        p.append(f'ld.global.nc.f32 ks{r}, [${r + 5}];')
    p += [f'mov.f32 acc{i}, 0f00000000;' for i in range(4)]
    for n in range(TILES):
        # q of head 8 n + g, dims 64 j + 16 c + {0..15}: x[16 j + k].
        for j in range(2):
            for i in range(4):
                xs = ', '.join(f'x{16 * j + 4 * i + e}' for e in range(4))
                p.append(f'ld.global.nc.v4.f32 {{{xs}}}, [$9+{n * 8 * DIM * 4 + 256 * j + 16 * i}];')
        p += ['abs.f32 m, x0;'] + [f'abs.f32 t, x{i}; max.f32 m, m, t;' for i in range(1, 32)]
        p += [f'shfl.sync.bfly.b32 t, m, {k}, 31, -1; max.f32 m, m, t;' for k in (1, 2)]
        p += [f'max.f32 s, m, {F32_1E_4};', f'div.rn.f32 s, s, {F32_448};', f'div.rn.f32 inv, {F32_1}, s;']
        p += [f'mul.f32 x{i}, x{i}, inv;' for i in range(32)]
        # b[4 j + i]: word i (dims 64 j + 16 c + 4 i ..) of this head's codes.
        for j in range(2):
            for i in range(4):
                k = 16 * j + 4 * i
                p += [f'cvt.rn.satfinite.e4m3x2.f32 h0, x{k + 1}, x{k};', f'cvt.rn.satfinite.e4m3x2.f32 h1, x{k + 3}, x{k + 2};',
                      f'mov.b32 b{4 * j + i}, {{h0, h1}};']
        p += [f'mov.f32 c{i}, 0f00000000;' for i in range(8)]
        for j in range(2):
            for h in range(2):
                bw = f'b{4 * j + 2 * h}, b{4 * j + 2 * h + 1}'
                for m in range(2):
                    r0, r1 = 2 * m, 2 * m + 1   # rows g + 16 m and g + 8 + 16 m
                    aw = (f'a{16 * j + 4 * r0 + 2 * h}, a{16 * j + 4 * r1 + 2 * h}, '
                          f'a{16 * j + 4 * r0 + 2 * h + 1}, a{16 * j + 4 * r1 + 2 * h + 1}')
                    cs = ', '.join(f'c{4 * m + i}' for i in range(4))
                    p.append(f'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{{cs}}}, {{{aw}}}, {{{bw}}}, {{{cs}}};')
        # The weights of heads 8 n + 2 c and + 1: q_scale from lanes 8 c and 8 c + 4 (g = 2 c, 2 c + 1).
        p += ['mov.u32 l, %laneid;', 'and.b32 src, l, 3;', 'shl.b32 src, src, 3;',
              'shfl.sync.idx.b32 qs0, s, src, 31, -1;', 'add.u32 src, src, 4;', 'shfl.sync.idx.b32 qs1, s, src, 31, -1;',
              f'ld.global.nc.v2.f32 {{w0, w1}}, [$10+{32 * n}];', 'mul.f32 w0, w0, qs0;', 'mul.f32 w1, w1, qs1;',
              'mul.f32 w0, w0, $11;', 'mul.f32 w1, w1, $11;']
        for m in range(2):
            for i, acc, w in ((0, 2 * m, 'w0'), (1, 2 * m, 'w1'), (2, 2 * m + 1, 'w0'), (3, 2 * m + 1, 'w1')):
                p += [f'max.f32 t, c{4 * m + i}, 0f00000000;', f'fma.rn.f32 acc{acc}, {w}, t, acc{acc};']
    # acc0..3: rows g, g + 8, g + 16, g + 24; times each row's key scale (relu(q . k * ks) = relu(q . k) * ks).
    p += [f'mul.f32 acc{i}, acc{i}, ks{i};' for i in range(4)]
    for i in range(4):
        for m in (1, 2):
            p += [f'shfl.sync.bfly.b32 t, acc{i}, {m}, 31, -1;', f'add.f32 acc{i}, acc{i}, t;']
    p += ['mov.u32 l, %laneid;', 'and.b32 src, l, 7;', 'shl.b32 src, src, 2;']
    p += [f'shfl.sync.idx.b32 sh{i}, acc{i}, src, 31, -1;' for i in range(4)]
    p += ['shr.u32 l, l, 3;', 'setp.eq.u32 q1, l, 1;', 'setp.eq.u32 q2, l, 2;', 'setp.eq.u32 q3, l, 3;',
          'mov.f32 $0, sh0;', '@q1 mov.f32 $0, sh1;', '@q2 mov.f32 $0, sh2;', '@q3 mov.f32 $0, sh3;', '}']
    return ' '.join(p)


HELPERS = """fn address1(t: &Tensor<f32, [?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return mlir!( inputs : (%t = t : memref<?xf32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?xf32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  };
}

fn address2(t: &Tensor<f32, [?, ?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return mlir!( inputs : (%t = t : memref<?x?xf32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?x?xf32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  };
}

fn address2i(t: &Tensor<i32, [?, ?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return mlir!( inputs : (%t = t : memref<?x?xi32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?x?xi32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  };
}
"""


def key_fp8_source():
    return f'''// Generated by gen_index_mma.py: the indexer key row at step[0] (f32, from index_key_at or store_row) to
// FP8 E4M3 codes and a scale, for index_scores_fp8_at (#32), as SGLang 0.5.22's indexer_k.cuh does:
// each value rounded to bf16, scale = max(1e-4, amax) / 448, codes = e4m3(x * (1 / scale)), round to
// nearest, saturating. One warp; codes [max_tokens, 32] holds a row's 128 codes as 32 words.
// Needs dim == {DIM}; otherwise it writes nothing.
// spawn-include: ../device.vx

{HELPERS}
fn index_key_fp8_lane(k: i64) on Topology::Device -> i64
ensures writes_only()
{{
  return mlir!( inputs : (%k = k : i64), returns : i64, dialects : ["llvm"], ensures : [writes_only()]) {{
    %r = llvm.inline_asm has_side_effects "{key_fp8_ptx()}", "=l,l" %k : (i64) -> i64
    macro.yield %r : i64
  }};
}}

pub fn glm_vx_spawn_index_key_fp8_at(codes: &mut Tensor<i32, [?, ?], Memory::HBM>, scales: &mut Tensor<f32, [?], Memory::HBM>,
                                     keys: &Tensor<f32, [?, ?], Memory::HBM>, step: &Tensor<i32, [?], Memory::HBM>,
                                     max_tokens: i32, dim: i32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..1 across blocks {{
      for lane in 0..32 across lanes {{
        let pos: i32 = step[0];
        if dim == {DIM} && pos >= 0 && pos < max_tokens {{
          let v: i64 = index_key_fp8_lane(address2(keys) + ((pos as i64) * {DIM} + 4 * (lane as i64)) * 4);
          codes[pos][lane] = (v & 4294967295) as i32;
          if lane == 0 {{ scales[pos] = f32::from_bits(((v >> 32) & 4294967295) as u32); }}
        }}
      }}
    }}
  }}
  return;
}}
'''


def scores_fp8_source():
    return f'''// Generated by gen_index_mma.py: the indexer's scores in FP8 E4M3 on the tensor cores (#32), SGLang
// 0.5.22's recipe: key rows quantized by index_key_fp8_at (codes, scales), each head's q here (one scale
// a head), mma.sync.m16n8k32.e4m3 with f32 sums, then sum over heads of weights[h] * q_scale[h] * scale
// * relu(q8[h] . k8[t]), times the row's scale. Each warp takes 32 tokens. Needs heads == {HEADS} and
// dim == {DIM}; otherwise it writes nothing. bf16 (index_scores_mma_at, index_kernel='mma') is the
// higher-precision version to come back to.
// spawn-include: ../device.vx

{HELPERS}
fn index_scores_fp8_warp(k0: i64, k1: i64, k2: i64, k3: i64, s0: i64, s1: i64, s2: i64, s3: i64, q: i64, w: i64,
                         scale: f32) on Topology::Device -> f32
ensures writes_only()
{{
  return mlir!( inputs : (%k0 = k0 : i64, %k1 = k1 : i64, %k2 = k2 : i64, %k3 = k3 : i64, %s0 = s0 : i64, %s1 = s1 : i64,
                          %s2 = s2 : i64, %s3 = s3 : i64, %q = q : i64, %w = w : i64, %sc = scale : f32), returns : f32,
                dialects : ["llvm"], ensures : [writes_only()]) {{
    %r = llvm.inline_asm has_side_effects "{scores_fp8_ptx()}", "=f,l,l,l,l,l,l,l,l,l,l,f" %k0, %k1, %k2, %k3, %s0, %s1, %s2, %s3, %q, %w, %sc : (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32) -> f32
    macro.yield %r : f32
  }};
}}

pub fn glm_vx_spawn_index_scores_fp8_at(out: &mut Tensor<f32, [?], Memory::HBM>, q: &Tensor<f32, [?, ?], Memory::HBM>,
                                        codes: &Tensor<i32, [?, ?], Memory::HBM>, scales: &Tensor<f32, [?], Memory::HBM>,
                                        weights: &Tensor<f32, [?], Memory::HBM>, step: &Tensor<i32, [?], Memory::HBM>,
                                        max_tokens: i32, heads: i32, dim: i32, scale: f32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..(max_tokens + {TOKENS_PER_BLOCK - 1}) / {TOKENS_PER_BLOCK} across blocks {{
      for wp in 0..{WARPS} across subgroups {{
        for lane in 0..32 across lanes {{
          let first: i32 = b * {TOKENS_PER_BLOCK} + wp * 32;
          let tokens: i32 = step[1];
          // Whole warps take this branch together (mma.sync and the shuffles need all 32 lanes).
          if heads == {HEADS} && dim == {DIM} && max_tokens > 0 && first < tokens {{
            let g: i32 = lane / 4;
            let c16: i64 = 16 * ((lane % 4) as i64);
            let last: i32 = max_tokens - 1;
            let kb: i64 = address2i(codes);
            let sb: i64 = address1(scales);
            let r0: i64 = ((first + g).min(last)) as i64;
            let r1: i64 = ((first + g + 8).min(last)) as i64;
            let r2: i64 = ((first + g + 16).min(last)) as i64;
            let r3: i64 = ((first + g + 24).min(last)) as i64;
            let score: f32 = index_scores_fp8_warp(kb + r0 * {DIM} + c16, kb + r1 * {DIM} + c16, kb + r2 * {DIM} + c16,
                                                   kb + r3 * {DIM} + c16, sb + r0 * 4, sb + r1 * 4, sb + r2 * 4, sb + r3 * 4,
                                                   address2(q) + ((g as i64) * {DIM} + c16) * 4,
                                                   address1(weights) + c16 / 2, scale);
            let token: i32 = first + lane;
            if token < tokens && token < max_tokens {{ out[token] = score; }}
          }}
        }}
      }}
    }}
  }}
  return;
}}
'''



def source():
    ptx = warp_ptx()
    return f'''// Generated by gen_index_mma.py: the indexer's scores on the tensor cores (#32), a warp's work one
// inline-PTX block in an mlir! macro; the layout is in the generator's docstring. Needs heads == {HEADS} and dim == {DIM}, keys with max_tokens
// rows; otherwise it writes nothing. A block of {WARPS} warps takes {TOKENS_PER_BLOCK} tokens.
// spawn-include: ../device.vx

fn address1(t: &Tensor<f32, [?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{{
  return mlir!( inputs : (%t = t : memref<?xf32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {{
    %p = memref.extract_aligned_pointer_as_index %t : memref<?xf32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  }};
}}

fn address2(t: &Tensor<f32, [?, ?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{{
  return mlir!( inputs : (%t = t : memref<?x?xf32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {{
    %p = memref.extract_aligned_pointer_as_index %t : memref<?x?xf32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  }};
}}

// One lane's part of a warp's 32 scores (gen_index_mma.py's warp_ptx): the score of token first + lane.
// It reads the keys, q and weights through their addresses and writes no memory.
fn index_scores_mma_warp(k0: i64, k1: i64, k2: i64, k3: i64, q: i64, w: i64, scale: f32) on Topology::Device -> f32
ensures writes_only()
{{
  return mlir!( inputs : (%k0 = k0 : i64, %k1 = k1 : i64, %k2 = k2 : i64, %k3 = k3 : i64, %q = q : i64, %w = w : i64,
                          %s = scale : f32), returns : f32, dialects : ["llvm"], ensures : [writes_only()]) {{
    %r = llvm.inline_asm has_side_effects "{ptx}", "=f,l,l,l,l,l,l,f" %k0, %k1, %k2, %k3, %q, %w, %s : (i64, i64, i64, i64, i64, i64, f32) -> f32
    macro.yield %r : f32
  }};
}}

pub fn glm_vx_spawn_index_scores_mma_at(out: &mut Tensor<f32, [?], Memory::HBM>, q: &Tensor<f32, [?, ?], Memory::HBM>,
                                        keys: &Tensor<f32, [?, ?], Memory::HBM>, weights: &Tensor<f32, [?], Memory::HBM>,
                                        step: &Tensor<i32, [?], Memory::HBM>, max_tokens: i32, heads: i32, dim: i32,
                                        scale: f32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..(max_tokens + {TOKENS_PER_BLOCK - 1}) / {TOKENS_PER_BLOCK} across blocks {{
      for wp in 0..{WARPS} across subgroups {{
        for lane in 0..32 across lanes {{
          let first: i32 = b * {TOKENS_PER_BLOCK} + wp * 32;
          let tokens: i32 = step[1];
          // Whole warps take this branch together (mma.sync and the shuffles need all 32 lanes).
          if heads == {HEADS} && dim == {DIM} && max_tokens > 0 && first < tokens {{
            let g: i32 = lane / 4;
            let c4: i64 = 4 * ((lane % 4) as i64);
            let last: i32 = max_tokens - 1;
            let kb: i64 = address2(keys);
            let r0: i64 = ((first + g).min(last)) as i64;
            let r1: i64 = ((first + g + 8).min(last)) as i64;
            let r2: i64 = ((first + g + 16).min(last)) as i64;
            let r3: i64 = ((first + g + 24).min(last)) as i64;
            let score: f32 = index_scores_mma_warp(kb + (r0 * {DIM} + c4) * 4, kb + (r1 * {DIM} + c4) * 4,
                                                   kb + (r2 * {DIM} + c4) * 4, kb + (r3 * {DIM} + c4) * 4,
                                                   address2(q) + ((g as i64) * {DIM} + c4) * 4,
                                                   address1(weights) + 2 * c4, scale);
            let token: i32 = first + lane;
            if token < tokens && token < max_tokens {{ out[token] = score; }}
          }}
        }}
      }}
    }}
  }}
  return;
}}
'''


# The scores kernels with the top-k's first pass folded in (#32): each also counts its scores' keys by their
# top 12 bits into hist[4096] for topk_select_ilp_hist_at, which then reads the scores twice, not three
# times, and zeroes hist for the next layer. A warp adds once a distinct bin, by its lowest lane
# (match.any.sync, popc), with a global atomic.
HIST_HELPERS = """
// Order-preserving map from float32 bits to [0, 2^32), as topk_select_at's: bigger key == bigger score;
// NaN is 0, and -0.0 is +0.0's key.
macro_rules! topk_key32 {
    ($bits:expr) => {
        unsafe {
          let dv_b: u32 = $bits as u32;
          let mut dv_key: u32 = dv_b ^ 2147483648u32;
          if dv_b >= 2147483648u32 { dv_key = 4294967295u32 - dv_b; }
          if dv_b == 2147483648u32 { dv_key = 2147483648u32; }
          if (dv_b & 2147483647u32) > 2139095040u32 { dv_key = 0u32; }
          dv_key
        }
    };
}

// The lanes of this warp whose x equals this lane's (match.any.sync); all 32 lanes must call it.
fn match_any(x: i32) on Topology::Device -> i32
ensures writes_only()
{
  return mlir!( inputs : (%x = x : i32), returns : i32, dialects : ["llvm"], ensures : [writes_only()]) {
    %r = llvm.inline_asm has_side_effects "match.any.sync.b32 $0, $1, -1;", "=r,r" %x : (i32) -> i32
    macro.yield %r : i32
  };
}

fn popc(x: i32) on Topology::Device -> i32
ensures writes_only()
{
  return mlir!( inputs : (%x = x : i32), returns : i32, dialects : ["llvm"], ensures : [writes_only()]) {
    %r = llvm.inline_asm "popc.b32 $0, $1;", "=r,r" %x : (i32) -> i32
    macro.yield %r : i32
  };
}
"""

HIST_WRITE = """            let token: i32 = first + lane;
            let ok: bool = token < tokens && token < max_tokens;
            if ok {{ out[token] = score; }}
            // The score's bin for topk_select_ilp_hist_at: the warp's lanes with one bin add once, by the
            // lowest of them. Every lane calls match_any (the whole warp is in this branch).
            let mut bin: i32 = 4096;
            if ok {{ bin = (topk_key32!(score.to_bits()) >> 20u32) as i32; }}
            let same: u32 = match_any(bin) as u32;
            let below: u32 = (1u32 << (lane as u32)) - 1u32;
            if bin < 4096 && (same & below) == 0u32 {{
              let _o = atomic_fetch_add_u32(hist, bin, popc(same as i32) as u32, AtomicOrdering::Relaxed);
            }}"""


def hist_variant(text, name):
    """`text` (a scores kernel) as <name>_hist_at: a `hist` parameter after `out`, and the bins counted."""
    entry = f'pub fn glm_vx_spawn_{name}_at(out: &mut Tensor<f32, [?], Memory::HBM>, '
    assert text.count(entry) == 1, name
    pad = ' ' * len(f'pub fn glm_vx_spawn_{name}_hist_at(')
    text = text.replace(entry, f'pub fn glm_vx_spawn_{name}_hist_at(out: &mut Tensor<f32, [?], Memory::HBM>, '
                               f'hist: &mut Tensor<u32, [?], Memory::HBM>,\n{pad}')
    write = """            let token: i32 = first + lane;
            if token < tokens && token < max_tokens { out[token] = score; }"""
    assert text.count(write) == 1, name
    text = text.replace(write, HIST_WRITE.replace('{{', '{').replace('}}', '}'))
    head, rest = text.split('// spawn-include: ../device.vx\n', 1)
    head = head + ('// With hist: the top-k\'s first pass folded in (gen_index_mma.py\'s HIST_HELPERS): hist[4096]\n'
                   '// counts the keys\' top 12 bits for topk_select_ilp_hist_at, which zeroes it after use.\n')
    return (head + '// spawn-include: ../device.vx\n// spawn-vxc: -fweak-relaxed-atomics\nimport core::sync::atomic;\n'
            + HIST_HELPERS + rest)


def key_bf16_source():
    return f'''// Generated by gen_index_mma.py: the indexer key row at step[0] (f32, from index_key_at or store_row) in
// bf16 for index_scores_mma16_at (#32): words [max_tokens, 64], word j dims 2j (low half) and 2j + 1, by
// the cvt.rn.bf16x2.f32 that index_scores_mma_at rounds the f32 row with, so its scores are the same.
// One warp, 4 dims a lane. Needs dim == {DIM}; otherwise it writes nothing.
// spawn-include: ../device.vx

{HELPERS}
fn index_key_bf16_lane(k: i64) on Topology::Device -> i64
ensures writes_only()
{{
  // One lane's 4 dims at the byte address k: ld.global.nc.v4.f32, then two cvt.rn.bf16x2.f32 (the low
  // half of each word is the lower dim). The result is word0 | word1 << 32.
  return mlir!( inputs : (%k = k : i64), returns : i64, dialects : ["llvm"], ensures : [writes_only()]) {{
    %p = llvm.inttoptr %k : i64 to !llvm.ptr<1>
    %x = llvm.load %p {{alignment = 16 : i64, invariant}} : !llvm.ptr<1> -> vector<4xf32>
    %h = llvm.fptrunc %x : vector<4xf32> to vector<4xbf16>
    %r = llvm.bitcast %h : vector<4xbf16> to i64
    macro.yield %r : i64
  }};
}}

pub fn glm_vx_spawn_index_key_bf16_at(words: &mut Tensor<i32, [?, ?], Memory::HBM>, keys: &Tensor<f32, [?, ?], Memory::HBM>,
                                      step: &Tensor<i32, [?], Memory::HBM>, max_tokens: i32, dim: i32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..1 across blocks {{
      for lane in 0..32 across lanes {{
        let pos: i32 = step[0];
        if dim == {DIM} && pos >= 0 && pos < max_tokens {{
          let v: i64 = index_key_bf16_lane(address2(keys) + ((pos as i64) * {DIM} + 4 * (lane as i64)) * 4);
          words[pos][2 * lane] = (v & 4294967295) as i32;
          words[pos][2 * lane + 1] = ((v >> 32) & 4294967295) as i32;
        }}
      }}
    }}
  }}
  return;
}}
'''


def source16():
    """index_scores_mma_at over the bf16 key rows of index_key_bf16_at: index_scores_mma16_at."""
    text = source()
    def swap(old, new):
        nonlocal text
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    swap('// Generated by gen_index_mma.py: the indexer\'s scores on the tensor cores (#32),',
         '// Generated by gen_index_mma.py: index_scores_mma_at over bf16 key rows (index_key_bf16_at\'s words\n'
         '// [max_tokens, 64]: half the bytes, the same scores bit for bit). The indexer\'s scores on the tensor cores (#32),')
    swap(warp_ptx(), warp_ptx(bf16_keys=True))
    swap('fn index_scores_mma_warp(', 'fn index_scores_mma16_warp(')
    swap('let score: f32 = index_scores_mma_warp(', 'let score: f32 = index_scores_mma16_warp(')
    swap('pub fn glm_vx_spawn_index_scores_mma_at(out: &mut Tensor<f32, [?], Memory::HBM>, q: &Tensor<f32, [?, ?], Memory::HBM>,\n'
         '                                        keys: &Tensor<f32, [?, ?], Memory::HBM>,',
         'pub fn glm_vx_spawn_index_scores_mma16_at(out: &mut Tensor<f32, [?], Memory::HBM>, q: &Tensor<f32, [?, ?], Memory::HBM>,\n'
         '                                          keys: &Tensor<i32, [?, ?], Memory::HBM>,')
    swap('let kb: i64 = address2(keys);', 'let kb: i64 = address2i(keys);')
    for r in range(4):
        swap(f'kb + (r{r} * {DIM} + c4) * 4', f'kb + (r{r} * {DIM} + c4) * 2')
    swap('fn address2(t: &Tensor<f32, [?, ?], Memory::HBM>) on Topology::Device -> i64',
         HELPERS[HELPERS.index('fn address2i('):].rstrip() + '\n\nfn address2(t: &Tensor<f32, [?, ?], Memory::HBM>) on Topology::Device -> i64')
    return text


def main():
    outputs = {'index_scores_mma_at.vx': source(), 'index_key_fp8_at.vx': key_fp8_source(),
               'index_scores_fp8_at.vx': scores_fp8_source(),
               'index_scores_mma_hist_at.vx': hist_variant(source(), 'index_scores_mma'),
               'index_scores_fp8_hist_at.vx': hist_variant(scores_fp8_source(), 'index_scores_fp8'),
               'index_key_bf16_at.vx': key_bf16_source(), 'index_scores_mma16_at.vx': source16(),
               'index_scores_mma16_hist_at.vx': hist_variant(source16(), 'index_scores_mma16')}
    stale = [name for name, text in outputs.items()
             if not (HERE / name).exists() or (HERE / name).read_text() != text]
    if '--check' in sys.argv:
        if stale:
            print(f'{", ".join(stale)}: not what gen_index_mma.py writes; rerun it', file=sys.stderr)
            sys.exit(1)
        return
    for name, text in outputs.items():
        (HERE / name).write_text(text)


if __name__ == '__main__':
    main()
