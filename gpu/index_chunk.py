"""The DSA indexer's scores for a whole prefill chunk in one launch (#63, item 5b): the host side of
gpu/spawn/index_mma/index_scores_chunk_at (gen_index_chunk.py).

    scores[i][j] = sum_h weights[i][h] * relu(scale * (q[i][h] . keys[j])),  j <= start + i

for the queries i < rows of a chunk at positions start .. start + rows - 1, with 32 heads of dim 128.
q and the key rows are rounded to bf16 (round to nearest even), the products are exact and the sums
are f32. Elements with j > start + i are not written.

- reference(): the scores in float64 over the bf16 values.
- emulate(): the kernel's lanes run in NumPy, from the MLIR text that gen_index_chunk.py writes and the
  addresses and key loads that the Vx code of index_scores_chunk_at computes. It checks the layout, the operand
  numbers, the shared-memory copies and the shuffles without a GPU. It is slow: small cases only.
- IndexChunkScores: launches index_q_bf16_at and index_scores_chunk_at (or its _hist_at twin).
"""
import re
from pathlib import Path

import numpy as np

from gpu.spawn.index_mma import gen_index_chunk as G

BUILD = Path(__file__).parent / 'build' / 'spawn' / 'index_mma'
HEADS, DIM = G.HEADS, G.DIM


def bf16_words(x):
    """f32 [..., 2 n] as bf16 words [..., n] (word w: element 2 w in the low half), round to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).view(np.int32)


def from_bf16_words(words):
    """bf16 words [..., n] as float64 [..., 2 n]."""
    halves = np.ascontiguousarray(words).view(np.uint16).astype(np.uint32) << 16
    return halves.view(np.float32).astype(np.float64)


def tiles(rows, start):
    """(qtiles, ktiles) for a chunk of `rows` queries after `start` keys."""
    return -(-rows // G.QUERIES_PER_BLOCK), -(-(start + rows) // G.KEYS_PER_BLOCK)


def reference(q_words, key_words, weights, start, scale):
    """float64 scores [rows, start + rows]; NaN where the kernel writes nothing (j > start + i).
    q_words [rows, 32 * 64], key_words [>= start + rows, 64], weights [rows, 32]."""
    rows = q_words.shape[0]
    n = start + rows
    q = from_bf16_words(q_words).reshape(rows, HEADS, DIM)
    k = from_bf16_words(key_words[:n])
    s = np.maximum(np.einsum('ihd,jd->ihj', q, k) * float(np.float32(scale)), 0)
    out = np.einsum('ih,ihj->ij', weights.astype(np.float64), s)
    out[np.arange(n)[None, :] > start + np.arange(rows)[:, None]] = np.nan
    return out


def reference_row(q_words, key_words, weights, position, scale):
    """float64 scores of one query at `position` over keys 0 .. position, and their error bound.
    q_words [32 * 64], weights [32]."""
    q = from_bf16_words(q_words).reshape(HEADS, DIM)
    k = from_bf16_words(key_words[:position + 1])
    dots = q @ k.T * float(np.float32(scale))
    w = weights.astype(np.float64)
    scores = w @ np.maximum(dots, 0)
    absolute = np.abs(w) @ (np.abs(q) @ np.abs(k).T * abs(float(np.float32(scale))))
    return scores, 4e-6 * (DIM + HEADS) * absolute + 1e-30


def bound(q_words, key_words, weights, start, scale):
    """An error bound for f32 sums of these terms: f32 rounding of each mma's sum over 128 dims and of
    the sum over the heads, as a multiple of the terms' absolute sum."""
    rows = q_words.shape[0]
    q = np.abs(from_bf16_words(q_words).reshape(rows, HEADS, DIM))
    k = np.abs(from_bf16_words(key_words[:start + rows]))
    absolute = np.einsum('ih,ihj->ij', np.abs(weights.astype(np.float64)),
                         np.einsum('ihd,jd->ihj', q, k) * abs(float(np.float32(scale))))
    return 4e-6 * (DIM + HEADS) * absolute + 1e-30


# ---------------------------------------------------------------------------------------------------
# The MLIR interpreter: the ops of gen_index_chunk.py's query_mlir and copy_mlir, for the 32 lanes of a
# warp at once. An i32 or f32 value holds 32 uint32 words.

def _f(x):
    return x.view(np.float32)


def _u(x):
    return np.asarray(x, np.float32).view(np.uint32)


def _bf16_pair(word):
    """The two bf16 values of each word, low half first, as float64 [32, 2]."""
    lo = (word & 0xFFFF).astype(np.uint32) << 16
    hi = word & 0xFFFF0000
    return np.stack([lo.view(np.float32), hi.astype(np.uint32).view(np.float32)], 1).astype(np.float64)


class Memory:
    """Flat byte spaces: global buffers by base address, and one shared space."""

    def __init__(self):
        self.global_buffers = []    # (base, uint8 array)
        self.shared = None

    def add_global(self, base, array):
        self.global_buffers.append((base, np.ascontiguousarray(array).view(np.uint8).reshape(-1)))
        return base

    def _global(self, address, size):
        for base, data in self.global_buffers:
            if base <= address and address + size <= base + data.size:
                return data, address - base
        raise IndexError(f'global load of {size} bytes at {address:#x} is outside every buffer')

    def load(self, space, address, words):
        out = np.empty((32, words), np.uint32)
        for lane in range(32):
            a = int(address[lane])
            if space == 'shared':
                if a < 0 or a + 4 * words > self.shared.size:
                    raise IndexError(f'shared load at {a:#x}')
                data, at = self.shared, a
            else:
                data, at = self._global(a, 4 * words)
            if at % (4 * words):
                raise ValueError(f'{space} load of {4 * words} bytes at {a:#x} is not aligned')
            out[lane] = data[at:at + 4 * words].view(np.uint32)
        return out

    def copy_to_shared(self, dst, src, size):
        if dst % size or src % size:
            raise ValueError(f'cp.async of {size} bytes from {src:#x} to {dst:#x} is not aligned')
        data, at = self._global(src, size)
        if dst < 0 or dst + size > self.shared.size:
            raise IndexError(f'cp.async to shared {dst:#x}')
        self.shared[dst:dst + size] = data[at:at + size]


_VALUE = re.compile(r'%\w+')


def run_mlir(text, operands, memory):
    """Run the body of an mlir! block (gen_index_chunk.py's query_mlir or copy_mlir) for 32 lanes.
    operands: {'%name': array [32]} for the block's inputs; returns all values by name. A pointer is
    (space, int64 array [32]), a vector [32, n] and a struct [32, 4]. Only the ops that the generator writes."""
    values = {k: np.array(v, copy=True) for k, v in operands.items()}
    lane_ids = np.arange(32, dtype=np.uint32)

    for line in filter(None, (s.strip() for s in text.split('\n'))):
        name, _, inst = line.partition(' = ') if re.match(r'%\w+ = ', line) else ('', '', line)
        op = inst.split()[0].split('(')[0]
        operand_text = re.sub(r'"[^"]*"', '', inst[len(op):])
        args = [values[v] for v in _VALUE.findall(operand_text.split(' : ')[0].split('{')[0])]
        if op == 'llvm.mlir.constant':
            number = re.search(r'\(([-\d.e]+) : (\w+)\)', inst)
            bits = np.float32(number[1]).view(np.uint32) if number[2] == 'f32' else int(number[1]) & 0xFFFFFFFF
            r = np.full(32, bits, np.uint32)
        elif op == 'llvm.inttoptr':
            r = ('shared' if '!llvm.ptr<3>' in inst else 'global', args[0].astype(np.int64))
        elif op == 'llvm.getelementptr':
            space, address = args[0]
            r = (space, address + int(re.search(r'\[(\d+)\]', inst)[1]))
        elif op == 'llvm.load':
            space, address = args[0]
            r = memory.load(space, address, int(re.search(r'vector<(\d+)x', inst)[1]))
        elif op == 'llvm.freeze':
            r = args[0]
        elif op == 'llvm.extractelement':
            r = args[0][:, int(args[1][0])]
        elif op == 'llvm.extractvalue':
            r = args[0][:, int(re.search(r'\[(\d+)\]', inst)[1])]
        elif op == 'nvvm.mma.sync':
            r = _mma(args[0:4], args[4:6], args[6:10], lane_ids)
        elif op == 'nvvm.shfl.sync':
            mask, x, b, clamp = args
            assert (mask == 0xFFFFFFFF).all() and (clamp == 31).all(), line
            lanes = (lane_ids ^ b) if ' bfly ' in inst else b
            r = x[lanes % 32]
        elif op == 'nvvm.read.ptx.sreg.laneid' or (op == 'llvm.inline_asm' and '"mov.u32 $0, %laneid;"' in inst):
            r = lane_ids.copy()
        elif op == 'nvvm.cp.async.shared.global':
            assert ', 16, cache = cg ' in inst, line
            (_, dst), (_, src) = args
            for lane in range(32):
                memory.copy_to_shared(int(dst[lane]), int(src[lane]), 16)
            continue
        elif op == 'nvvm.cp.async.commit.group':
            continue
        elif op == 'llvm.fmul':
            r = _u(_f(args[0]) * _f(args[1]))
        elif op == 'llvm.fadd':
            r = _u(_f(args[0]) + _f(args[1]))
        elif op == 'llvm.intr.maxnum':
            r = _u(np.fmax(_f(args[0]), _f(args[1])))
        elif op == 'llvm.intr.fma':
            r = _u((_f(args[0]).astype(np.float64) * _f(args[1]) + _f(args[2])).astype(np.float32))
        elif op == 'llvm.and':
            r = args[0] & args[1]
        elif op == 'llvm.shl':
            r = (args[0] << args[1]) & 0xFFFFFFFF
        elif op == 'llvm.lshr':
            r = args[0] >> args[1]
        elif op == 'llvm.icmp':
            r = (args[0] == args[1]) if '"eq"' in inst else (args[0] != args[1])
        elif op == 'llvm.select':
            r = np.where(args[0], args[1], args[2])
        else:
            raise NotImplementedError(f'MLIR op {op!r}')
        values[name] = r
    return values


def _mma(a, b, c, lane_ids):
    """mma.sync m16n8k16 .row.col .f32.bf16.bf16.f32 for a warp: the 4 D registers as [32, 4] uint32."""
    # A 16x16 (row), B 16x8 (col), C 16x8: PTX ISA 9.7.14.5.8 (m16n8k16, .bf16).
    am = np.zeros((16, 16)); bm = np.zeros((16, 8)); cm = np.zeros((16, 8))
    g, cc = lane_ids // 4, lane_ids % 4
    pairs = [_bf16_pair(x) for x in a]
    for lane in range(32):
        for reg, (row, col) in enumerate(((g[lane], 2 * cc[lane]), (g[lane] + 8, 2 * cc[lane]),
                                          (g[lane], 2 * cc[lane] + 8), (g[lane] + 8, 2 * cc[lane] + 8))):
            am[row, col:col + 2] = pairs[reg][lane]
    bpairs = [_bf16_pair(x) for x in b]
    for lane in range(32):
        bm[2 * cc[lane]:2 * cc[lane] + 2, g[lane]] = bpairs[0][lane]
        bm[2 * cc[lane] + 8:2 * cc[lane] + 10, g[lane]] = bpairs[1][lane]
    cvals = [_f(x).astype(np.float64) for x in c]
    for lane in range(32):
        cm[g[lane], 2 * cc[lane]] = cvals[0][lane]
        cm[g[lane], 2 * cc[lane] + 1] = cvals[1][lane]
        cm[g[lane] + 8, 2 * cc[lane]] = cvals[2][lane]
        cm[g[lane] + 8, 2 * cc[lane] + 1] = cvals[3][lane]
    dm = (am @ bm + cm).astype(np.float32)
    return np.stack([_u(np.array([dm[g[l] + 8 * (k // 2), 2 * cc[l] + k % 2] for l in range(32)], np.float32))
                     for k in range(4)], 1)


def emulate(q_words, key_words, weights, start, scale, max_tokens=None, out=None, hist=None):
    """The kernel's work, lane by lane: out [rows, >= start + rows] f32, written where the kernel writes.
    Mirrors index_scores_chunk_at's Vx code for the addresses and the branches."""
    rows = q_words.shape[0]
    max_tokens = key_words.shape[0] if max_tokens is None else max_tokens
    qtiles, ktiles = tiles(rows, start)
    if out is None:
        out = np.full((rows, start + rows), np.nan, np.float32)
    mem = Memory()
    QB, KB, WB = 0x10000000, 0x20000000, 0x30000000
    mem.add_global(QB, q_words)
    mem.add_global(KB, key_words)
    mem.add_global(WB, weights.astype(np.float32))
    query_text = G.query_mlir()
    scale_bits = np.full(32, np.float32(scale).view(np.uint32), np.uint32)
    for b in range(qtiles * ktiles):
        mem.shared = np.zeros(2 * G.BUFFER * 4, np.uint8)
        i0, k0 = (b // ktiles) * G.QUERIES_PER_BLOCK, (b % ktiles) * G.KEYS_PER_BLOCK
        last_query = min(i0 + G.QUERIES_PER_BLOCK - 1, rows - 1)
        if not (start + rows <= max_tokens and i0 < rows and k0 <= start + last_query):
            continue
        tid = np.arange(G.THREADS)
        lane, first = tid % 32, k0 + 32 * (tid // 32)
        g, c = lane // 4, lane % 4
        top = max_tokens - 1
        r = [np.minimum(first + g + 8 * m, top) for m in range(4)]
        copy_to = ((tid // 8) * G.PITCH + G.COPY_WORDS * (tid % 8)) * 4
        copy_from = ((tid // 8) * G.WORDS + G.COPY_WORDS * (tid % 8)) * 4
        read_at = (g * G.PITCH + 2 * c) * 4
        a = []
        for w in range(G.WARPS):
            sl = slice(32 * w, 32 * w + 32)
            # The Vx code's 32 index_chunk_load2 calls: a[lo] and a[hi] from one 8-byte load each.
            regs = [None] * 64
            for row, offset, lo, hi in G.key_loads():
                words = mem.load('global', KB + (r[row][sl] * G.WORDS + 2 * c[sl]) * 4 + offset, 2)
                regs[lo], regs[hi] = words[:, 0], words[:, 1]
            a.append(regs)

        def copy(buffer, row):
            for w in range(G.WARPS):
                sl = slice(32 * w, 32 * w + 32)
                run_mlir(G.copy_mlir(), {'%s': (copy_to[sl] + buffer * G.BUFFER * 4).astype(np.uint64),
                                         '%g': (QB + row * G.HEADS * G.WORDS * 4 + copy_from[sl]).astype(np.uint64)}, mem)

        copy(0, i0)
        for qi in range(G.QUERIES_PER_BLOCK):
            i = i0 + qi
            copy((qi + 1) % 2, min(i + 1, rows - 1))
            for w in range(G.WARPS):
                sl = slice(32 * w, 32 * w + 32)
                if not (i < rows and first[32 * w] <= start + i):
                    continue
                ops = {f'%a{k}': a[w][k] for k in range(64)}
                ops.update({'%qs': (read_at[sl] + (qi % 2) * G.BUFFER * 4).astype(np.uint64),
                            '%w': (WB + (i * G.HEADS + 2 * c[sl]) * 4).astype(np.uint64), '%sc': scale_bits})
                score = _f(run_mlir(query_text, ops, mem)['%r'])
                j = k0 + tid[sl]
                ok = j <= start + i
                out[i, j[ok]] = score[ok]
                if hist is not None:
                    bins = topk_bins(score)[ok]
                    np.add.at(hist, i * 4096 + bins, 1)
    return out


def topk_bins(scores):
    """The top 12 bits of topk_key32 (index_scores_mma_hist_at's bins): bigger score, bigger bin."""
    b = np.asarray(scores, np.float32).view(np.uint32).astype(np.uint64)
    key = np.where(b >= 0x80000000, 0xFFFFFFFF - b, b ^ 0x80000000)
    key = np.where(b == 0x80000000, 0x80000000, key)
    key = np.where((b & 0x7FFFFFFF) > 0x7F800000, 0, key)
    return (key >> 20).astype(np.int64)


class IndexChunkScores:
    """The chunk scores on the GPU: q to bf16 words (index_q_bf16_at), then index_scores_chunk_at or,
    with hist, index_scores_chunk_hist_at."""

    def __init__(self, context, build=BUILD):
        from gpu.spawn.launch import SpawnKernel
        self.q_bf16 = SpawnKernel(context, build / 'index_q_bf16_at.ptx')
        self.scores = SpawnKernel(context, build / 'index_scores_chunk_at.ptx')
        self.scores_hist = SpawnKernel(context, build / 'index_scores_chunk_hist_at.ptx')

    def __call__(self, out, q_words, keys, weights, *, rows, start, max_tokens, scale, q=None, hist=None, stream=None):
        """out [rows, >= start + rows] f32; q_words [rows, 2048] int32 (written from q [rows, 4096] f32
        first when q is given); keys [max_tokens, 64] int32 (index_key_bf16_at's words); weights [rows, 32]
        f32; hist [rows * 4096] uint32 or None."""
        if q is not None:
            self.q_bf16.launch(words=q_words, q=q, rows=rows, stream=stream)
        qtiles, ktiles = tiles(rows, start)
        args = dict(out=out, q=q_words, keys=keys, weights=weights, rows=rows, start=start, max_tokens=max_tokens,
                    qtiles=qtiles, ktiles=ktiles, heads=HEADS, dim=DIM, scale=scale)
        if hist is None:
            self.scores.launch(stream=stream, **args)
        else:
            self.scores_hist.launch(stream=stream, hist=hist, **args)
