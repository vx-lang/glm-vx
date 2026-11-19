#!/usr/bin/env python3
"""Write fp8_moe_mma_down_at.vx (#63, docs/TP-SERVING-PLAN.md item 4): the routed experts' down projection on the
FP8 tensor cores, each warp's work one inline-PTX block in an mlir! macro (as gpu/spawn/index_mma). Run from
anywhere; rerun after editing, and commit the output (build.sh checks it with --check).

out[r] = sum over the k experts, in ascending id order, of weights[slot] * (down[e][r] . act[s]), for the [rows, cols]
FP8 E4M3 down matrices with their 128x128 float scales, act[s * cols ..] expert s's input (s its place in ascending
id order, as fp8_moe_cvt_gate_up writes it). A block takes 16 rows, one warp an expert (k <= 9: GLM-5.3's 8 routed
experts and, with fold_shared, the shared one):

- Each lane loads 16 contiguous bytes of rows g = lane / 4 and g + 8 at column 16 c (c = lane % 4) of each
  64-column chunk, straight from global memory into its mma.sync.m16n8k32 A fragments: in a chunk's two k32
  steps, the fragment bytes {4c..4c+3} and {16+4c..} of lane column c are columns 16 c + {0..3} and + {4..7},
  then + {8..11} and + {12..15}. The dot does not depend on the order of the columns, and act uses the same order.
- act is quantized here, as SGLang's per-token-group quant before its FP8 GEMMs: one scale a group of 128
  columns, max(1e-10, amax) / 448, codes cvt.rn.satfinite.e4m3x2.f32. Every column of B holds act, so every
  column of C is the dot; c0 is row g's, c2 row g + 8's.
- A 128-column group's two chunks go into one f32 sum, times the weight scale and act's scale; the group sums are
  added in column order, then times the router weight. Lanes 0..15 return rows 0..15's values.

Precision: weights and act in E4M3 (SGLang's), products exact, the tensor cores' f32 sums. Not
fp8_moe_cvt_down's bits: compare within tolerance (validation/moe_mma_check.py).

`python3 gen_moe_mma.py --check` exits 1 if the committed output differs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS_PER_BLOCK, WARPS = 16, 9
SEGMENT = 256  # columns a call of the warp's PTX takes: two 128-column scale groups
# ptxas's budget (build.py: .maxnreg). Unbudgeted it takes 127 registers, one block of 9 warps an SM; at 72 (172 bytes
# spilled) three fit: 12.7 against 14.0 us at TP8's shapes on the H200 (ncu, evidence/h200-20261118-z18).
MAXNREG = 72


def down_ptx():
    """One warp's 16 rows x 256 columns of one expert. $0: lane j < 16 gets row j's sum (scaled), lanes 16.. 0.
    $1, $2: rows g and g + 8 of the segment at byte 16 c; $3: act of the segment at column 16 c (bytes, f32);
    $4: the address of the segment's two weight scales, loaded after the weights (an address the weights do not
    wait for: passed as values, the scales' load came first and the weights' waited a round trip for it)."""
    p = ['{', '.reg .b32 a<32>, b<8>, l, src;', '.reg .f32 x<32>, c<4>, t, m, s, inv, f, ws<2>, acc0, acc2, v0, v2;',
         '.reg .b16 h<2>;', '.reg .pred p8, p16;']
    # a[8 q + 4 r + i]: word i of row r (0: g, 1: g + 8) in chunk q (0..3), 64 q + 16 c.
    for q in range(4):
        for r in range(2):
            base = 8 * q + 4 * r
            p.append(f'ld.global.nc.L1::no_allocate.v4.u32 {{a{base}, a{base + 1}, a{base + 2}, a{base + 3}}}, [${r + 1}+{64 * q}];')
    p += ['ld.global.nc.f32 ws0, [$4+0];', 'ld.global.nc.f32 ws1, [$4+4];']
    p += ['mov.f32 acc0, 0f00000000;', 'mov.f32 acc2, 0f00000000;']
    for grp in range(2):
        # act columns 128 grp + 64 h + 16 c + {0..15} (h = 0, 1): x[16 h + ..], bytes 512 grp + 256 h + 16 i.
        for h in range(2):
            for i in range(4):
                xs = ', '.join(f'x{16 * h + 4 * i + e}' for e in range(4))
                p.append(f'ld.global.nc.v4.f32 {{{xs}}}, [$3+{512 * grp + 256 * h + 16 * i}];')
        p += ['abs.f32 m, x0;'] + [f'abs.f32 t, x{i}; max.f32 m, m, t;' for i in range(1, 32)]
        p += [f'shfl.sync.bfly.b32 t, m, {k}, 31, -1; max.f32 m, m, t;' for k in (1, 2)]
        p += ['max.f32 s, m, 0f2EDBE6FF;', 'div.rn.f32 s, s, 0f43E00000;', 'div.rn.f32 inv, 0f3F800000, s;']
        p += [f'mul.f32 x{i}, x{i}, inv;' for i in range(32)]
        for j in range(8):
            p += [f'cvt.rn.satfinite.e4m3x2.f32 h0, x{4 * j + 1}, x{4 * j};',
                  f'cvt.rn.satfinite.e4m3x2.f32 h1, x{4 * j + 3}, x{4 * j + 2};', f'mov.b32 b{j}, {{h0, h1}};']
        p += [f'mov.f32 c{i}, 0f00000000;' for i in range(4)]
        for h in range(2):
            q = 2 * grp + h
            for step in range(2):
                a0, a8 = 8 * q + 2 * step, 8 * q + 4 + 2 * step
                bj = 4 * h + 2 * step
                p.append(f'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{c0, c1, c2, c3}}, '
                         f'{{a{a0}, a{a8}, a{a0 + 1}, a{a8 + 1}}}, {{b{bj}, b{bj + 1}}}, {{c0, c1, c2, c3}};')
        p += [f'mul.f32 f, s, ws{grp};', 'fma.rn.f32 acc0, f, c0, acc0;', 'fma.rn.f32 acc2, f, c2, acc2;']
    p += ['mov.u32 l, %laneid;', 'and.b32 src, l, 7;', 'shl.b32 src, src, 2;',
          'shfl.sync.idx.b32 v0, acc0, src, 31, -1;', 'shfl.sync.idx.b32 v2, acc2, src, 31, -1;',
          'setp.lt.u32 p8, l, 8;', 'setp.lt.u32 p16, l, 16;',
          'mov.f32 $0, 0f00000000;', '@p16 mov.f32 $0, v2;', '@p8 mov.f32 $0, v0;', '}']
    return ' '.join(p)


def down16_ptx():
    """down_ptx with act in f16 instead of E4M3, for sm_90, whose mma.sync on E4M3 is f16 HMMA after ptxas unpacks
    both operands (evidence/h200-20261118-z19/sass-lowering.txt): the weights unpacked once here (cvt.rn.f16x2.e4m3x2,
    exact), act scaled by a power of two a group (its max into [2^14, 2^15), exact) and rounded to f16, then
    mma.sync.m16n8k16.f16 with f32 sums. In chunk q's k16 step m, lane column c's fragment columns {2c, 2c + 1} and
    {2c + 8, 2c + 9} are the chunk's columns 16 c + 4 m + {0, 1} and + {2, 3}, in A and B alike. Operands as
    down_ptx's."""
    p = ['{', '.reg .b32 a<32>, u<4>, b<8>, l, src, eb, ib;',
         '.reg .f32 x<32>, c<4>, t, m, s, inv, f, ws<2>, acc0, acc2, v0, v2;', '.reg .b16 lo, hi;', '.reg .pred p8, p16;']
    for q in range(4):
        for r in range(2):
            base = 8 * q + 4 * r
            p.append(f'ld.global.nc.L1::no_allocate.v4.u32 {{a{base}, a{base + 1}, a{base + 2}, a{base + 3}}}, [${r + 1}+{64 * q}];')
    p += ['ld.global.nc.f32 ws0, [$4+0];', 'ld.global.nc.f32 ws1, [$4+4];']
    p += ['mov.f32 acc0, 0f00000000;', 'mov.f32 acc2, 0f00000000;']
    for grp in range(2):
        for h in range(2):
            for i in range(4):
                xs = ', '.join(f'x{16 * h + 4 * i + e}' for e in range(4))
                p.append(f'ld.global.nc.v4.f32 {{{xs}}}, [$3+{512 * grp + 256 * h + 16 * i}];')
        p += ['abs.f32 m, x0;'] + [f'abs.f32 t, x{i}; max.f32 m, m, t;' for i in range(1, 32)]
        p += [f'shfl.sync.bfly.b32 t, m, {k}, 31, -1; max.f32 m, m, t;' for k in (1, 2)]
        # s = 2^(E - 14) for m in [2^E, 2^(E+1)); 1/s exactly; m at least 2^-100 (all zeros: zeros).
        p += ['max.f32 m, m, 0f0D800000;', 'mov.b32 eb, m;', 'and.b32 eb, eb, 2139095040;', 'sub.u32 eb, eb, 117440512;',
              'mov.b32 s, eb;', 'sub.u32 ib, 2130706432, eb;', 'mov.b32 inv, ib;']
        p += [f'mul.f32 x{i}, x{i}, inv;' for i in range(32)]
        p += [f'cvt.rn.f16x2.f32 b{j}, x{2 * j + 1}, x{2 * j};' for j in range(8)]  # chunk h = 0: b0..b7 later reused
        p += [f'mov.f32 c{i}, 0f00000000;' for i in range(4)]
        for h in range(2):
            if h == 1:
                p += [f'cvt.rn.f16x2.f32 b{j}, x{16 + 2 * j + 1}, x{16 + 2 * j};' for j in range(8)]
            q = 2 * grp + h
            for mstep in range(4):
                # word mstep of each row: codes 16 c + 4 m + {0..3} -> two f16x2 (u0/u1 row g, u2/u3 row g + 8)
                p += [f'mov.b32 {{lo, hi}}, a{8 * q + mstep};', 'cvt.rn.f16x2.e4m3x2 u0, lo;', 'cvt.rn.f16x2.e4m3x2 u1, hi;',
                      f'mov.b32 {{lo, hi}}, a{8 * q + 4 + mstep};', 'cvt.rn.f16x2.e4m3x2 u2, lo;', 'cvt.rn.f16x2.e4m3x2 u3, hi;']
                p.append(f'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{c0, c1, c2, c3}}, {{u0, u2, u1, u3}}, '
                         f'{{b{2 * mstep}, b{2 * mstep + 1}}}, {{c0, c1, c2, c3}};')
        p += [f'mul.f32 f, s, ws{grp};', 'fma.rn.f32 acc0, f, c0, acc0;', 'fma.rn.f32 acc2, f, c2, acc2;']
    p += ['mov.u32 l, %laneid;', 'and.b32 src, l, 7;', 'shl.b32 src, src, 2;',
          'shfl.sync.idx.b32 v0, acc0, src, 31, -1;', 'shfl.sync.idx.b32 v2, acc2, src, 31, -1;',
          'setp.lt.u32 p8, l, 8;', 'setp.lt.u32 p16, l, 16;',
          'mov.f32 $0, 0f00000000;', '@p16 mov.f32 $0, v2;', '@p8 mov.f32 $0, v0;', '}']
    return ' '.join(p)


def slot_ptx():
    """The slot (0..k-1) of the expert of rank $3 in ascending id order among ids[0..k-1] ($1 their address, $2 k <=
    WARPS), the warp's own: lane j < k loads ids[j] and counts the ids below it through shuffles; a ballot finds the
    lane whose count is $3. No shared memory or barrier, unlike device.vx's id_rank! and slot_of_rank! (about 1.7 us
    a launch before any weight load could start, evidence/h200-20261118-z18). -1 if none (ids not distinct)."""
    p = ['{', '.reg .b32 l, id, other, r, ball;', '.reg .b64 addr;', '.reg .pred inr, lt, hit;',
         'mov.u32 l, %laneid;', 'mov.u32 id, 2147483647;', 'setp.lt.s32 inr, l, $2;',
         'mul.wide.u32 addr, l, 4;', 'add.s64 addr, addr, $1;', '@inr ld.global.nc.u32 id, [addr];', 'mov.u32 r, 0;']
    for o in range(WARPS):
        p += [f'shfl.sync.idx.b32 other, id, {o}, 31, -1;', 'setp.lt.s32 lt, other, id;', '@lt add.u32 r, r, 1;']
    p += ['setp.eq.u32 hit, r, $3;', 'and.pred hit, hit, inr;', 'vote.sync.ballot.b32 ball, hit, -1;',
          'bfind.u32 $0, ball;', '}']
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

fn address1i(t: &Tensor<i32, [?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return mlir!( inputs : (%t = t : memref<?xi32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?xi32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  };
}

fn address1u(t: &Tensor<u32, [?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return mlir!( inputs : (%t = t : memref<?xi32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?xi32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  };
}
"""


E4M3_HEAD = '''// Generated by gen_moe_mma.py: the routed experts' down projection on the FP8 tensor cores (#63), SGLang's
// recipe: act quantized to E4M3 in groups of 128 (amax / 448), mma.sync.m16n8k32.e4m3 straight from the FP8
// weights, f32 sums times the 128x128 weight scales and act's, then the router weights, the experts added in
// ascending id order.'''
F16_HEAD = '''// Generated by gen_moe_mma.py: fp8_moe_mma_down_at with act in f16 (a power-of-two scale a group of 128, exact)
// and the FP8 weights unpacked to f16 here, mma.sync.m16n8k16.f16 with f32 sums: on sm_90 the same tensor-core
// instruction ptxas makes of the E4M3 mma, without rounding act to E4M3.
//'''


def down_source(name='fp8_moe_mma_down_at', head=None, ptx=None):
    head, ptx = head or E4M3_HEAD, ptx or down_ptx
    return f'''{head} A block takes {ROWS_PER_BLOCK} rows, a warp an expert. Arguments as fp8_moe_cvt_down. Needs
// k <= {WARPS}, rows % {ROWS_PER_BLOCK} == 0 and cols % {SEGMENT} == 0; otherwise out is zeros (ResidentGLM(moe_mma=True)
// keeps down2 there).
// spawn-include: ../device.vx
// spawn-maxnreg: {MAXNREG}

{HELPERS}
fn down_mma_warp(r0: i64, r8: i64, x: i64, ws: i64) on Topology::Device -> f32
ensures writes_only()
{{
  return mlir!( inputs : (%r0 = r0 : i64, %r8 = r8 : i64, %x = x : i64, %ws = ws : i64), returns : f32,
                dialects : ["llvm"], ensures : [writes_only()]) {{
    %r = llvm.inline_asm has_side_effects "{ptx()}", "=f,l,l,l,l" %r0, %r8, %x, %ws : (i64, i64, i64, i64) -> f32
    macro.yield %r : f32
  }};
}}

fn slot_of_rank_warp(ids: i64, k: i32, rank: i32) on Topology::Device -> i32
ensures writes_only()
{{
  return mlir!( inputs : (%i = ids : i64, %k = k : i32, %r = rank : i32), returns : i32, dialects : ["llvm"],
                ensures : [writes_only()]) {{
    %s = llvm.inline_asm has_side_effects "{slot_ptx()}", "=r,l,r,r" %i, %k, %r : (i64, i32, i32) -> i32
    macro.yield %s : i32
  }};
}}

pub fn glm_vx_spawn_{name}(out: &mut Tensor<f32, [?], Memory::HBM>, down: &Tensor<u32, [?], Memory::HBM>,
                                        scales: &Tensor<f32, [?], Memory::HBM>, act: &Tensor<f32, [?], Memory::HBM>,
                                        weights: &Tensor<f32, [?], Memory::HBM>, ids: &Tensor<i32, [?], Memory::HBM>,
                                        k: i32, experts: i32, rows: i32, cols: i32) -> void {{
  spawn on(Topology::Device) {{
    for rb in 0..(rows + {ROWS_PER_BLOCK - 1}) / {ROWS_PER_BLOCK} across blocks {{
      let mut part = unsafe {{ Tensor<f32, [{WARPS}, {ROWS_PER_BLOCK}], Memory::SMEM>::uninit() }};
      for w in 0..{WARPS} across subgroups {{
        for lane in 0..32 across lanes {{
          let mut v: f32 = 0.0;
          // Whole warps take these branches together (mma.sync and the shuffles need all 32 lanes).
          if w < k && k <= {WARPS} && rows % {ROWS_PER_BLOCK} == 0 && cols % {SEGMENT} == 0 && cols > 0 {{
            let slot: i32 = slot_of_rank_warp(address1i(ids), k, w);
            let expert: i32 = if slot >= 0 && slot < k {{ ids[slot] }} else {{ -1 }};
            if expert >= 0 && expert < experts {{
              let n: i64 = cols as i64;
              let e: i64 = expert as i64;
              let row0: i64 = (rb * {ROWS_PER_BLOCK}) as i64;
              let c16: i64 = 16 * ((lane % 4) as i64);
              let rg: i64 = address1u(down) + (e * (rows as i64) + row0 + ((lane / 4) as i64)) * n + c16;
              let xb: i64 = address1(act) + ((w as i64) * n + c16) * 4;
              let tiles: i64 = n / 128;
              let sb: i64 = address1(scales) + (e * (((rows as i64) + 127) / 128) + row0 / 128) * tiles * 4;
              let mut seg: i64 = 0;
              while seg < n / {SEGMENT} {{
                v = v + down_mma_warp(rg + seg * {SEGMENT}, rg + 8 * n + seg * {SEGMENT}, xb + seg * {SEGMENT * 4}, sb + seg * 8);
                seg = seg + 1;
              }}
              v = v * weights[slot];
            }}
          }}
          if lane < {ROWS_PER_BLOCK} {{ part[w][lane] = v; }}
        }}
      }}
      for j in 0..{ROWS_PER_BLOCK} across lanes {{
        if rb * {ROWS_PER_BLOCK} + j < rows {{
          let mut sum: f32 = 0.0;
          for q in 0..{WARPS} {{ sum = sum + part[q][j]; }}
          out[rb * {ROWS_PER_BLOCK} + j] = sum;
        }}
      }}
    }}
  }}
  return;
}}
'''


GU_WARPS = 8  # gate_up: warps a block, each an eighth of the columns
GU_MAXNREG = 128  # two blocks of 8 warps an SM: TP8's 144 blocks in one wave


def gate_up16_ptx():
    """One warp's share of a 16-row tile of gate and the same 16 rows of up against x: one 256-column segment (two
    scale groups). All 16 of the lane's weight loads (rows g and g + 8 of gate and up, 4 chunks of 16 bytes) are
    issued first, then the scales and x; x is put in f16 a group at a time (a power of two, as down16_ptx) and
    feeds both tiles' mma.sync.m16n8k16. $0: lanes 0..15 gate rows 0..15's segment sums (weight scale x x's scale
    applied), lanes 16..31 up's. $1, $2: gate rows g, g + 8 at the segment's byte 16 c; $3, $4: up's; $5: x of
    the segment at column 16 c (f32 bytes); $6, $7: the two groups' gate and up scales (addresses)."""
    p = ['{', '.reg .b32 a<32>, d<32>, u<4>, b<8>, l, src, eb, ib;',
         '.reg .f32 x<32>, cg<4>, cu<4>, t, m, s, inv, f, gs<2>, us<2>, ag0, ag2, au0, au2, v0, v2, v4, v6;',
         '.reg .b16 lo, hi;', '.reg .pred p8, p16, p24;']
    # a: gate, d: up; [8 q + 4 r + i]: word i of row r (0: g, 1: g + 8) in chunk q
    for name, regs in (('a', (1, 2)), ('d', (3, 4))):
        for q in range(4):
            for r in range(2):
                base = 8 * q + 4 * r
                p.append(f'ld.global.nc.L1::no_allocate.v4.u32 {{{name}{base}, {name}{base + 1}, {name}{base + 2}, '
                         f'{name}{base + 3}}}, [${regs[r]}+{64 * q}];')
    p += ['ld.global.nc.f32 gs0, [$6+0];', 'ld.global.nc.f32 gs1, [$6+4];',
          'ld.global.nc.f32 us0, [$7+0];', 'ld.global.nc.f32 us1, [$7+4];']
    p += [f'mov.f32 {r}, 0f00000000;' for r in ('ag0', 'ag2', 'au0', 'au2')]
    for grp in range(2):
        for h in range(2):
            for i in range(4):
                xs = ', '.join(f'x{16 * h + 4 * i + e}' for e in range(4))
                p.append(f'ld.global.nc.v4.f32 {{{xs}}}, [$5+{512 * grp + 256 * h + 16 * i}];')
        p += ['abs.f32 m, x0;'] + [f'abs.f32 t, x{i}; max.f32 m, m, t;' for i in range(1, 32)]
        p += [f'shfl.sync.bfly.b32 t, m, {k}, 31, -1; max.f32 m, m, t;' for k in (1, 2)]
        p += ['max.f32 m, m, 0f0D800000;', 'mov.b32 eb, m;', 'and.b32 eb, eb, 2139095040;', 'sub.u32 eb, eb, 117440512;',
              'mov.b32 s, eb;', 'sub.u32 ib, 2130706432, eb;', 'mov.b32 inv, ib;']
        p += [f'mul.f32 x{i}, x{i}, inv;' for i in range(32)]
        p += [f'mov.f32 cg{i}, 0f00000000;' for i in range(4)] + [f'mov.f32 cu{i}, 0f00000000;' for i in range(4)]
        for h in range(2):
            p += [f'cvt.rn.f16x2.f32 b{j}, x{16 * h + 2 * j + 1}, x{16 * h + 2 * j};' for j in range(8)]
            q = 2 * grp + h
            for mstep in range(4):
                for name, c in (('a', 'cg'), ('d', 'cu')):
                    p += [f'mov.b32 {{lo, hi}}, {name}{8 * q + mstep};', 'cvt.rn.f16x2.e4m3x2 u0, lo;',
                          'cvt.rn.f16x2.e4m3x2 u1, hi;', f'mov.b32 {{lo, hi}}, {name}{8 * q + 4 + mstep};',
                          'cvt.rn.f16x2.e4m3x2 u2, lo;', 'cvt.rn.f16x2.e4m3x2 u3, hi;',
                          f'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{c}0, {c}1, {c}2, {c}3}}, {{u0, u2, u1, u3}}, '
                          f'{{b{2 * mstep}, b{2 * mstep + 1}}}, {{{c}0, {c}1, {c}2, {c}3}};']
        p += [f'mul.f32 f, s, gs{grp};', 'fma.rn.f32 ag0, f, cg0, ag0;', 'fma.rn.f32 ag2, f, cg2, ag2;',
              f'mul.f32 f, s, us{grp};', 'fma.rn.f32 au0, f, cu0, au0;', 'fma.rn.f32 au2, f, cu2, au2;']
    p += ['mov.u32 l, %laneid;', 'and.b32 src, l, 7;', 'shl.b32 src, src, 2;',
          'shfl.sync.idx.b32 v0, ag0, src, 31, -1;', 'shfl.sync.idx.b32 v2, ag2, src, 31, -1;',
          'shfl.sync.idx.b32 v4, au0, src, 31, -1;', 'shfl.sync.idx.b32 v6, au2, src, 31, -1;',
          'setp.lt.u32 p8, l, 8;', 'setp.lt.u32 p16, l, 16;', 'setp.lt.u32 p24, l, 24;',
          'mov.f32 $0, v6;', '@p24 mov.f32 $0, v4;', '@p16 mov.f32 $0, v2;', '@p8 mov.f32 $0, v0;', '}']
    return ' '.join(p)


def gate_up_source():
    return f'''// Generated by gen_moe_mma.py: the routed experts' gate and up projections and SwiGLU on the tensor cores (#63,
// docs/TP-SERVING-PLAN.md item 4): act[s * inter + i] = silu(gate . x) * (up . x) for expert s (ascending id order)
// and row i, as fp8_moe_cvt_gate_up. The FP8 weights are unpacked to f16 in registers (exact), x is put in f16 with
// a power-of-two scale a group of 128, mma.sync.m16n8k16.f16 with f32 sums, times the 128x128 weight scales and x's.
// A block takes 16 rows of an expert's gate and the same 16 of up; its {GU_WARPS} warps an eighth of the columns
// each, a 256-column segment a call (16 loads of 16 bytes a lane in flight); their sums added in warp order, then
// silu(g) * u (device.vx's sigmoid!). Arguments as fp8_moe_cvt_gate_up. Needs k <= {WARPS}, inter % 16 == 0 and
// cols % {GU_WARPS * SEGMENT} == 0; otherwise act is zeros (ResidentGLM(moe_mma=True) keeps fp8_moe_cvt_gate_up there).
// spawn-include: ../device.vx
// spawn-maxnreg: {GU_MAXNREG}

{HELPERS}
fn slot_of_rank_warp(ids: i64, k: i32, rank: i32) on Topology::Device -> i32
ensures writes_only()
{{
  return mlir!( inputs : (%i = ids : i64, %k = k : i32, %r = rank : i32), returns : i32, dialects : ["llvm"],
                ensures : [writes_only()]) {{
    %s = llvm.inline_asm has_side_effects "{slot_ptx()}", "=r,l,r,r" %i, %k, %r : (i64, i32, i32) -> i32
    macro.yield %s : i32
  }};
}}

fn gate_up_mma_warp(g0: i64, g8: i64, u0: i64, u8: i64, x: i64, gs: i64, us: i64) on Topology::Device -> f32
ensures writes_only()
{{
  return mlir!( inputs : (%g0 = g0 : i64, %g8 = g8 : i64, %u0 = u0 : i64, %u8 = u8 : i64, %x = x : i64, %gs = gs : i64,
                          %us = us : i64), returns : f32, dialects : ["llvm"], ensures : [writes_only()]) {{
    %r = llvm.inline_asm has_side_effects "{gate_up16_ptx()}", "=f,l,l,l,l,l,l,l" %g0, %g8, %u0, %u8, %x, %gs, %us : (i64, i64, i64, i64, i64, i64, i64) -> f32
    macro.yield %r : f32
  }};
}}

pub fn glm_vx_spawn_fp8_moe_mma16_gate_up_at(act: &mut Tensor<f32, [?], Memory::HBM>, gate: &Tensor<u32, [?], Memory::HBM>,
                                             gate_scales: &Tensor<f32, [?], Memory::HBM>, up: &Tensor<u32, [?], Memory::HBM>,
                                             up_scales: &Tensor<f32, [?], Memory::HBM>, x: &Tensor<f32, [?], Memory::HBM>,
                                             ids: &Tensor<i32, [?], Memory::HBM>, k: i32, experts: i32, inter: i32,
                                             up_row: i32, stride: i32, cols: i32) -> void {{
  spawn on(Topology::Device) {{
    for tb in 0..(k * inter + 15) / 16 across blocks {{
      let mut part = unsafe {{ Tensor<f32, [{GU_WARPS}, 32], Memory::SMEM>::uninit() }};
      for w in 0..{GU_WARPS} across subgroups {{
        for lane in 0..32 across lanes {{
          let mut v: f32 = 0.0;
          // Whole warps take these branches together (mma.sync, the shuffles and the ballot need all 32 lanes).
          if k <= {WARPS} && inter % 16 == 0 && inter > 0 && cols % {GU_WARPS * SEGMENT} == 0 && cols > 0 {{
            let s: i32 = (tb * 16) / inter;
            let slot: i32 = slot_of_rank_warp(address1i(ids), k, s);
            let expert: i32 = if slot >= 0 && slot < k {{ ids[slot] }} else {{ -1 }};
            if expert >= 0 && expert < experts {{
              let n: i64 = cols as i64;
              let e: i64 = expert as i64;
              let row0: i64 = (tb * 16 - s * inter) as i64;
              let g: i64 = (lane / 4) as i64;
              let c16: i64 = 16 * ((lane % 4) as i64);
              let first: i64 = (w as i64) * (n / {GU_WARPS});
              let gr: i64 = address1u(gate) + (e * (stride as i64) + row0 + g) * n + first + c16;
              let ur: i64 = address1u(up) + (e * (stride as i64) + (up_row as i64) + row0 + g) * n + first + c16;
              let xb: i64 = address1(x) + (first + c16) * 4;
              let tiles: i64 = n / 128;
              let srows: i64 = ((stride as i64) + 127) / 128;
              let gsb: i64 = address1(gate_scales) + ((e * srows + row0 / 128) * tiles + first / 128) * 4;
              let usb: i64 = address1(up_scales) + ((e * srows + ((up_row as i64) + row0) / 128) * tiles + first / 128) * 4;
              let mut seg: i64 = 0;
              while seg < n / {GU_WARPS * SEGMENT} {{
                v = v + gate_up_mma_warp(gr + seg * {SEGMENT}, gr + 8 * n + seg * {SEGMENT}, ur + seg * {SEGMENT},
                                         ur + 8 * n + seg * {SEGMENT}, xb + seg * {SEGMENT * 4}, gsb + seg * 8, usb + seg * 8);
                seg = seg + 1;
              }}
            }}
          }}
          part[w][lane] = v;
        }}
      }}
      for j in 0..16 across lanes {{
        if tb * 16 + j < k * inter && cols >= 0 {{
          let mut gt: f32 = 0.0;
          let mut ut: f32 = 0.0;
          for q in 0..{GU_WARPS} {{
            gt = gt + part[q][j];
            ut = ut + part[q][16 + j];
          }}
          act[tb * 16 + j] = gt * sigmoid!(gt) * ut;
        }}
      }}
    }}
  }}
  return;
}}
'''


def main():
    outputs = {'fp8_moe_mma_down_at.vx': down_source(),
               'fp8_moe_mma16_down_at.vx': down_source('fp8_moe_mma16_down_at', F16_HEAD, down16_ptx),
               'fp8_moe_mma16_gate_up_at.vx': gate_up_source()}
    stale = [name for name, text in outputs.items()
             if not (HERE / name).exists() or (HERE / name).read_text() != text]
    if '--check' in sys.argv:
        if stale:
            print(f'{", ".join(stale)}: not what gen_moe_mma.py writes; rerun it', file=sys.stderr)
            sys.exit(1)
        return
    for name, text in outputs.items():
        (HERE / name).write_text(text)


if __name__ == '__main__':
    main()
