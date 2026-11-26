#!/usr/bin/env python3
"""Write the indexer's scores for a whole prefill chunk in one launch (#63, item 5b): index_q_bf16_at.vx,
index_scores_chunk_at.vx and its _hist_at twin. Run from anywhere; rerun after editing, and commit the
outputs (build.sh checks them with --check).

scores[i][j] = sum over the 32 heads of weights[i][h] * relu(scale * (q[i][h] . keys[j])), dim 128, for
query i < rows (position start + i) and key j <= start + i; the other elements are not written.

- index_q_bf16_at rounds q [rows, 32 * 128] (f32) to bf16 words [rows, 32 * 64], word w = dims 2w (low
  half) and 2w + 1, by the cvt.rn.bf16x2.f32 that index_key_bf16_at rounds the key rows with.
- index_scores_chunk_at: a block of 8 warps takes 256 keys (warp w the 32 keys k0 + 32 w ..) and 64
  queries. Each warp loads its keys' A fragments once (64 words in a REG tensor, index_scores_mma16_at's layout:
  in k-step s, lane column c = lane % 4 holds dims 16 s + 4 c + {0, 1} and {2, 3}, so a fragment is
  two adjacent words). For each query the block copies its q (8 KiB of bf16) into shared memory with
  cp.async, rows padded to 72 words so the B loads of a warp do not share banks, double-buffered: query
  qi + 1 is in flight while the warps compute query qi. A warp's query is index_scores_mma16_at's warp
  work with B from shared memory: 4 head tiles x 8 k-steps x 2 token tiles of
  mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32, then w * relu(scale * s) added over the heads, and two
  butterfly shuffles. A block whose keys are all after its last query's position does nothing; a warp
  whose keys are all after the query skips the query's mma.
- _hist_at also counts each row's scores by their top 12 bits into hist[i * 4096 + bin] (as
  index_scores_mma_hist_at does for one query), for the top-k's first pass.

The q copy and the barrier are nvvm ops in the lane loop (cp.async, bar.sync 1, 256): the shared
buffer is the block's, and the A fragments must stay in registers from one query to the next, so the
loop over queries cannot be a series of lane loops. The fragments are the lane's own REG tensor, written
element by element (32 two-word loads) and read through a shared borrow, which Vx main accepts.

`python3 gen_index_chunk.py --check` exits 1 if a committed output differs.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIM, HEADS = 128, 32
STEPS, TILES = DIM // 16, HEADS // 8
WORDS = DIM // 2            # bf16 words a row of q or keys
WARPS = 8
THREADS = 32 * WARPS
KEYS_PER_BLOCK = 32 * WARPS
QUERIES_PER_BLOCK = 64
PITCH = WORDS + 8           # words a head row in shared memory: 72, so g * 72 + 2 c spreads over the banks
BUFFER = HEADS * PITCH      # words a query's q in shared memory
COPY_WORDS = HEADS * WORDS // THREADS   # words of q each thread copies: 8, two 16-byte cp.async


def a_index(s, m, i):
    """The A register i (0..3) of token tile m (rows g + 16 m, g + 8 + 16 m) in k-step s."""
    return 8 * s + 4 * m + i


def key_loads():
    """index_scores_mma16_at's A loads, one 8-byte load each: (key row 0..3 = g, g + 8, g + 16, g + 24,
    byte offset from word 2 c of the row, the A register of the low word, of the high word)."""
    loads = []
    for s in range(STEPS):
        for m in range(2):
            loads.append((2 * m, 32 * s, a_index(s, m, 0), a_index(s, m, 2)))
            loads.append((2 * m + 1, 32 * s, a_index(s, m, 1), a_index(s, m, 3)))
    return loads


def query_mlir():
    """One lane's score for one query, as the body of an mlir! block: %r the score of key first + lane.
    %a0..%a63 the A fragments, %qs the shared address of word 2 c of head row g of the query's buffer,
    %w the address of weights[i][2 c], %sc the scale."""
    p = ['%qp = llvm.inttoptr %qs : i64 to !llvm.ptr<3>', '%wp = llvm.inttoptr %w : i64 to !llvm.ptr<1>',
         '%zero = llvm.mlir.constant(0.0 : f32) : f32', '%all = llvm.mlir.constant(-1 : i32) : i32',
         '%clamp = llvm.mlir.constant(31 : i32) : i32', '%e0 = llvm.mlir.constant(0 : i32) : i32',
         '%e1 = llvm.mlir.constant(1 : i32) : i32']
    mma = ('{layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = '
           '#nvvm.mma_type<bf16>, multiplicandBPtxType = #nvvm.mma_type<bf16>, shape = #nvvm.shape<m = 16, n = 8, '
           'k = 16>} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>')
    acc = ['%zero'] * 4
    for n in range(TILES):
        c = ['%zero'] * 8
        for s in range(STEPS):
            # B of head 8 n + g, dims 16 s + 4 c + {0..3}: words 8 s + 2 c and + 1 of its row.
            b = f'%b{n}_{s}'
            p += [f'{b}p = llvm.getelementptr %qp[{4 * (8 * n * PITCH + 8 * s)}] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8',
                  f'{b} = llvm.load {b}p {{alignment = 8 : i64}} : !llvm.ptr<3> -> vector<2xi32>',
                  f'{b}x = llvm.extractelement {b}[%e0 : i32] : vector<2xi32>',
                  f'{b}y = llvm.extractelement {b}[%e1 : i32] : vector<2xi32>']
            for m in range(2):
                d = f'%d{n}_{s}_{m}'
                aw = ', '.join(f'%a{a_index(s, m, i)}' for i in range(4))
                cs = ', '.join(c[4 * m:4 * m + 4])
                p.append(f'{d} = nvvm.mma.sync A[{aw}] B[{b}x, {b}y] C[{cs}] {mma}')
                for i in range(4):
                    p.append(f'{d}c{i} = llvm.extractvalue {d}[{i}] : !llvm.struct<(f32, f32, f32, f32)>')
                    c[4 * m + i] = f'{d}c{i}'
        # Tile m: c[4m], c[4m+1] row g (+ 16 m), heads 8 n + 2 c and + 1; c[4m+2], c[4m+3] row g + 8 (+ 16 m).
        # The freeze keeps LLVM from splitting the load in two, which drops its invariant mark (.nc).
        p += [f'%w{n}p = llvm.getelementptr %wp[{32 * n}] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8',
              f'%w{n}l = llvm.load %w{n}p {{alignment = 8 : i64, invariant}} : !llvm.ptr<1> -> vector<2xf32>',
              f'%w{n} = llvm.freeze %w{n}l : vector<2xf32>']
        w = [f'%w{n}x', f'%w{n}y']
        p += [f'{w[0]} = llvm.extractelement %w{n}[%e0 : i32] : vector<2xf32>',
              f'{w[1]} = llvm.extractelement %w{n}[%e1 : i32] : vector<2xf32>']
        for m in range(2):
            for i, a, wi in ((0, 2 * m, 0), (1, 2 * m, 1), (2, 2 * m + 1, 0), (3, 2 * m + 1, 1)):
                t = f'%t{n}_{m}_{i}'
                p += [f'{t}m = llvm.fmul {c[4 * m + i]}, %sc : f32',
                      f'{t} = llvm.intr.maxnum({t}m, %zero) : (f32, f32) -> f32',
                      f'{t}a = llvm.intr.fma({w[wi]}, {t}, {acc[a]}) : (f32, f32, f32) -> f32']
                acc[a] = f'{t}a'
    # acc0..3: rows g, g + 8, g + 16, g + 24 of this lane's column c; add the four columns.
    for i in range(4):
        for m in (1, 2):
            v = f'%x{i}_{m}'
            p += [f'{v}o = llvm.mlir.constant({m} : i32) : i32',
                  f'{v}s = nvvm.shfl.sync bfly %all, {acc[i]}, {v}o, %clamp : f32 -> f32',
                  f'{v} = llvm.fadd {acc[i]}, {v}s : f32']
            acc[i] = v
    # Lane l takes row l: acc[l / 8] of lane 4 (l % 8). nvvm.read.ptx.sreg.laneid would move out of the
    # kernel's loop over blocks, and its values there cost registers, so %laneid stays inline asm.
    p += ['%lane = llvm.inline_asm has_side_effects "mov.u32 $0, %laneid;", "=r" : () -> i32',
          '%seven = llvm.mlir.constant(7 : i32) : i32',
          '%two = llvm.mlir.constant(2 : i32) : i32', '%low = llvm.and %lane, %seven : i32',
          '%src = llvm.shl %low, %two : i32']
    p += [f'%sh{i} = nvvm.shfl.sync idx %all, {acc[i]}, %src, %clamp : f32 -> f32' for i in range(4)]
    p += ['%three = llvm.mlir.constant(3 : i32) : i32', '%row = llvm.lshr %lane, %three : i32',
          '%q1 = llvm.icmp "eq" %row, %e1 : i32',
          '%q2 = llvm.icmp "eq" %row, %two : i32', '%q3 = llvm.icmp "eq" %row, %three : i32',
          '%r1 = llvm.select %q1, %sh1, %sh0 : i1, f32', '%r2 = llvm.select %q2, %sh2, %r1 : i1, f32',
          '%r = llvm.select %q3, %sh3, %r2 : i1, f32']
    return '\n'.join('    ' + line for line in p)


def copy_mlir():
    """The body of an mlir! block that copies 32 bytes from global address %g to shared address %s."""
    return '\n'.join('    ' + line for line in [
        '%sp = llvm.inttoptr %s : i64 to !llvm.ptr<3>', '%gp = llvm.inttoptr %g : i64 to !llvm.ptr<1>',
        '%sp1 = llvm.getelementptr %sp[16] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8',
        '%gp1 = llvm.getelementptr %gp[16] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8',
        'nvvm.cp.async.shared.global %sp, %gp, 16, cache = cg : !llvm.ptr<3>, !llvm.ptr<1>',
        'nvvm.cp.async.shared.global %sp1, %gp1, 16, cache = cg : !llvm.ptr<3>, !llvm.ptr<1>',
        'nvvm.cp.async.commit.group'])


ADDRESS_HELPERS = """fn address2(t: &Tensor<f32, [?, ?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return unsafe { mlir!( inputs : (%t = t : memref<?x?xf32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?x?xf32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  } };
}

fn address2i(t: &Tensor<i32, [?, ?], Memory::HBM>) on Topology::Device -> i64
ensures reads_only(t)
{
  return unsafe { mlir!( inputs : (%t = t : memref<?x?xi32>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {
    %p = memref.extract_aligned_pointer_as_index %t : memref<?x?xi32> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  } };
}
"""


def q_bf16_source():
    return f'''// Generated by gen_index_chunk.py: q [rows, {HEADS * DIM}] (f32) as bf16 words [rows, {HEADS * WORDS}] for
// index_scores_chunk_at: word w is dims 2w (low half) and 2w + 1, by the cvt.rn.bf16x2.f32 of
// index_key_bf16_at. One lane converts 4 dims.
// spawn-include: ../device.vx

{ADDRESS_HELPERS}
fn index_q_bf16_lane(x: i64) on Topology::Device -> i64
ensures writes_only()
{{
  // 4 dims of q at the byte address x: ld.global.nc.v4.f32, then two cvt.rn.bf16x2.f32 (the low half of
  // each word is the lower dim). The result is word0 | word1 << 32.
  return unsafe {{ mlir!( inputs : (%x = x : i64), returns : i64, dialects : ["llvm"], ensures : [writes_only()]) {{
    %p = llvm.inttoptr %x : i64 to !llvm.ptr<1>
    %v = llvm.load %p {{alignment = 16 : i64, invariant}} : !llvm.ptr<1> -> vector<4xf32>
    %h = llvm.fptrunc %v : vector<4xf32> to vector<4xbf16>
    %r = llvm.bitcast %h : vector<4xbf16> to i64
    macro.yield %r : i64
  }} }};
}}

pub fn glm_vx_spawn_index_q_bf16_at(words: &mut Tensor<i32, [?], Memory::HBM>, q: &Tensor<f32, [?, ?], Memory::HBM>,
                                    rows: i32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..rows * {HEADS * DIM // 4 // 256} across blocks {{
      for l in 0..256 across lanes(256) {{
        let v: i64 = index_q_bf16_lane(address2(q) + 16 * ((b * 256 + l) as i64));
        words[2 * (b * 256 + l)] = (v & 4294967295) as i32;
        words[2 * (b * 256 + l) + 1] = ((v >> 32) & 4294967295) as i32;
      }}
    }}
  }}
  return;
}}
'''


def scores_source(hist=False):
    key_lines = '\n'.join(
        f'          let v{n}: i64 = index_chunk_load2(k_{row} + {offset});\n'
        f'          a[{lo}] = (v{n} & 4294967295) as i32;\n          a[{hi}] = ((v{n} >> 32) & 4294967295) as i32;'
        for n, (row, offset, lo, hi) in enumerate(key_loads()))
    read_lines = '\n'.join(f'    %i{k} = arith.constant {k} : index\n    %a{k} = memref.load %a[%i{k}] : memref<64xi32>'
                           for k in range(64))
    name = 'index_scores_chunk_hist_at' if hist else 'index_scores_chunk_at'
    hist_param = '\n' + ' ' * len(f'pub fn glm_vx_spawn_{name}(') + 'hist: &mut Tensor<u32, [?], Memory::HBM>,' if hist else ''
    head = (f"// With hist: the top-k's first pass folded in. hist[i * 4096 + bin] counts row i's scores by their top\n"
            "// 12 bits (index_scores_mma_hist_at's bins); the warp's lanes with one bin add once, by the lowest.\n"
            if hist else '')
    includes = ('// spawn-include: ../device.vx\n// spawn-vxc: -fweak-relaxed-atomics\nimport core::sync::atomic;\n'
                if hist else '// spawn-include: ../device.vx\n')
    helpers = HIST_HELPERS if hist else ''
    store = ('''              if ok { out[i][j] = score; }
              let mut bin: i32 = 4096;
              if ok { bin = (topk_key32!(score.to_bits()) >> 20u32) as i32; }
              let same: u32 = match_any(bin) as u32;
              let below: u32 = (1u32 << (lane as u32)) - 1u32;
              if bin < 4096 && (same & below) == 0u32 {
                let _o = atomic_fetch_add_u32(hist, i * 4096 + bin, popc(same as i32) as u32, AtomicOrdering::Relaxed);
              }''' if hist else '''              if ok { out[i][j] = score; }''')
    return f'''// Generated by gen_index_chunk.py: the indexer's scores for a chunk of queries in one launch (#63), on the
// tensor cores; the layout is in the generator's docstring. scores[i][j] for query i < rows (position
// start + i) and key j <= start + i; the other elements of out are not written. q is index_q_bf16_at's
// words [rows, {HEADS * WORDS}], keys index_key_bf16_at's [max_tokens, {WORDS}], weights [rows, {HEADS}]. A block takes
// {KEYS_PER_BLOCK} keys and {QUERIES_PER_BLOCK} queries: launch with qtiles = ceil(rows / {QUERIES_PER_BLOCK}) and
// ktiles = ceil((start + rows) / {KEYS_PER_BLOCK}). Needs heads == {HEADS}, dim == {DIM} and start + rows <= max_tokens;
// otherwise it writes nothing.
{head}{includes}
{ADDRESS_HELPERS}
fn address_shared(t: &Tensor<i32, [{2 * BUFFER}], Memory::SMEM>) on Topology::Device -> i64
ensures reads_only(t)
{{
  return unsafe {{ mlir!( inputs : (%t = t : memref<{2 * BUFFER}xi32, 3>), returns : i64, dialects : ["memref", "arith"], ensures : [reads_only(t)]) {{
    %p = memref.extract_aligned_pointer_as_index %t : memref<{2 * BUFFER}xi32, 3> -> index
    %r = arith.index_cast %p : index to i64
    macro.yield %r : i64
  }} }};
}}
{helpers}
// Two words of a key row (an 8-byte load); the first in the low half.
fn index_chunk_load2(at: i64) on Topology::Device -> i64
ensures writes_only()
{{
  // An invariant 8-byte load from a global pointer: ld.global.nc.v2.b32. The freeze keeps LLVM from
  // making it one 64-bit load.
  return unsafe {{ mlir!( inputs : (%x = at : i64), returns : i64, dialects : ["llvm"], ensures : [writes_only()]) {{
    %p = llvm.inttoptr %x : i64 to !llvm.ptr<1>
    %v = llvm.load %p {{alignment = 8 : i64, invariant}} : !llvm.ptr<1> -> vector<2xi32>
    %f = llvm.freeze %v : vector<2xi32>
    %r = llvm.bitcast %f : vector<2xi32> to i64
    macro.yield %r : i64
  }} }};
}}

// One lane's score for one query (gen_index_chunk.py's query_mlir), from the A fragments in `$a` and the
// query's q in shared memory: 64 mma.sync with B from ld.shared.v2, then the heads' sum and 12 shfl.sync.
// It reads its addresses and writes no memory.
macro_rules! index_chunk_query {{
    ($a:ident, $qs:expr, $w:expr, $scale:expr) => {{
        unsafe {{
          let dv_qs: i64 = $qs;
          let dv_w: i64 = $w;
          let dv_scale: f32 = $scale;
          let dv_a = &$a;
          let dv_s: f32 = mlir!( inputs : (%a = dv_a : memref<64xi32>, %qs = dv_qs : i64, %w = dv_w : i64, %sc = dv_scale : f32),
                                returns : f32, dialects : ["llvm", "nvvm", "memref", "arith"], ensures : [reads_only(dv_a)]) {{
{read_lines}
{query_mlir()}
    macro.yield %r : f32
          }};
          dv_s
        }}
    }};
}}

// Starts the copy of this thread's 8 words of a query's q into shared memory (cp.async).
fn index_chunk_copy(shared: i64, global: i64) on Topology::Device -> i32
ensures writes_only()
{{
  return unsafe {{ mlir!( inputs : (%s = shared : i64, %g = global : i64), returns : i32, dialects : ["llvm", "nvvm", "arith"], ensures : [writes_only()]) {{
{copy_mlir()}
    %z = arith.constant 0 : i32
    macro.yield %z : i32
  }} }};
}}

// Waits for this thread's copies, then for the block (named barrier 1, all {THREADS} threads).
fn index_chunk_wait(x: i32) on Topology::Device -> i32
ensures writes_only()
{{
  return unsafe {{ mlir!( inputs : (%x = x : i32), returns : i32, dialects : ["llvm", "nvvm"], ensures : [writes_only()]) {{
    %id = llvm.mlir.constant(1 : i32) : i32
    %n = llvm.mlir.constant({THREADS} : i32) : i32
    nvvm.cp.async.wait.group 0
    nvvm.barrier id = %id number_of_threads = %n
    macro.yield %x : i32
  }} }};
}}

pub fn glm_vx_spawn_{name}(out: &mut Tensor<f32, [?, ?], Memory::HBM>,{hist_param}
{' ' * len(f'pub fn glm_vx_spawn_{name}(')}q: &Tensor<i32, [?, ?], Memory::HBM>, keys: &Tensor<i32, [?, ?], Memory::HBM>,
{' ' * len(f'pub fn glm_vx_spawn_{name}(')}weights: &Tensor<f32, [?, ?], Memory::HBM>, rows: i32, start: i32, max_tokens: i32,
{' ' * len(f'pub fn glm_vx_spawn_{name}(')}qtiles: i32, ktiles: i32, heads: i32, dim: i32, scale: f32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..qtiles * ktiles across blocks {{
      let qbuf = unsafe {{ Tensor<i32, [{2 * BUFFER}], Memory::SMEM>::uninit() }};
      for tid in 0..{THREADS} across lanes({THREADS}) {{
        let i0: i32 = (b / ktiles) * {QUERIES_PER_BLOCK};
        let k0: i32 = (b % ktiles) * {KEYS_PER_BLOCK};
        let last_query: i32 = if i0 + {QUERIES_PER_BLOCK} < rows {{ i0 + {QUERIES_PER_BLOCK} - 1 }} else {{ rows - 1 }};
        // The whole block takes this branch or not (its barriers need all {THREADS} threads).
        if heads == {HEADS} && dim == {DIM} && rows > 0 && start >= 0 && start + rows <= max_tokens && i0 < rows
           && k0 <= start + last_query {{
          let lane: i32 = tid % 32;
          let first: i32 = k0 + 32 * (tid / 32);
          let g: i32 = lane / 4;
          let c: i64 = (lane % 4) as i64;
          let top: i32 = max_tokens - 1;
          let kb: i64 = address2i(keys);
          let r0: i64 = ((first + g).min(top)) as i64;
          let r1: i64 = ((first + g + 8).min(top)) as i64;
          let r2: i64 = ((first + g + 16).min(top)) as i64;
          let r3: i64 = ((first + g + 24).min(top)) as i64;
          let k_0: i64 = kb + (r0 * {WORDS} + 2 * c) * 4;
          let k_1: i64 = kb + (r1 * {WORDS} + 2 * c) * 4;
          let k_2: i64 = kb + (r2 * {WORDS} + 2 * c) * 4;
          let k_3: i64 = kb + (r3 * {WORDS} + 2 * c) * 4;
          // The A fragments of the warp's 32 keys (gen_index_chunk.py's key_loads), in registers for every query.
          let mut a = unsafe {{ Tensor<i32, [64], Memory::REG>::uninit() }};
{key_lines}
          let qb: i64 = address2i(q);
          let wb: i64 = address2(weights);
          let sb: i64 = address_shared(&qbuf);
          // This thread copies words {COPY_WORDS} * (tid % 8) .. of head row tid / 8.
          let copy_to: i64 = sb + (((tid / 8) * {PITCH} + {COPY_WORDS} * (tid % 8)) as i64) * 4;
          let copy_from: i64 = ((tid / 8) * {WORDS} + {COPY_WORDS} * (tid % 8)) as i64 * 4;
          let read_at: i64 = sb + ((g * {PITCH}) as i64 + 2 * c) * 4;
          let _c0: i32 = index_chunk_copy(copy_to, qb + (i0 as i64) * {HEADS * WORDS * 4} + copy_from);
          let _w0: i32 = index_chunk_wait(0);
          for qi in 0..{QUERIES_PER_BLOCK} {{
            let i: i32 = i0 + qi;
            let next: i32 = if i + 1 < rows {{ i + 1 }} else {{ rows - 1 }};
            let _c: i32 = index_chunk_copy(copy_to + (((qi + 1) % 2) as i64) * {BUFFER * 4},
                                           qb + (next as i64) * {HEADS * WORDS * 4} + copy_from);
            // Whole warps take this branch together (mma.sync and the shuffles need all 32 lanes).
            if i < rows && first <= start + i {{
              let score: f32 = index_chunk_query!(a, read_at + ((qi % 2) as i64) * {BUFFER * 4},
                                                 wb + ((i as i64) * {HEADS} + 2 * c) * 4, scale);
              let j: i32 = k0 + tid;
              let ok: bool = j <= start + i;
{store}
            }}
            let _w: i32 = index_chunk_wait(qi);
          }}
        }}
      }}
    }}
  }}
  return;
}}
'''


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
  return unsafe { mlir!( inputs : (%x = x : i32), returns : i32, dialects : ["llvm", "nvvm"], ensures : [writes_only()]) {
    %all = llvm.mlir.constant(-1 : i32) : i32
    %r = nvvm.match.sync any %all, %x : i32 -> i32
    macro.yield %r : i32
  } };
}

fn popc(x: i32) on Topology::Device -> i32
ensures writes_only()
{
  return unsafe { mlir!( inputs : (%x = x : i32), returns : i32, dialects : ["llvm"], ensures : [writes_only()]) {
    %r = llvm.intr.ctpop(%x) : (i32) -> i32
    macro.yield %r : i32
  } };
}
"""


def outputs():
    return {'index_q_bf16_at.vx': q_bf16_source(), 'index_scores_chunk_at.vx': scores_source(),
            'index_scores_chunk_hist_at.vx': scores_source(hist=True)}


def main():
    stale = [name for name, text in outputs().items()
             if not (HERE / name).exists() or (HERE / name).read_text() != text]
    if '--check' in sys.argv:
        if stale:
            print(f'{", ".join(stale)}: not what gen_index_chunk.py writes; rerun it', file=sys.stderr)
            sys.exit(1)
        return
    for name, text in outputs().items():
        (HERE / name).write_text(text)


if __name__ == '__main__':
    main()
