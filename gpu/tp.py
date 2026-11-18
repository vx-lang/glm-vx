"""Tensor parallelism for ResidentGLM (#34; design #55).

Megatron's split, one GPU a rank, each rank one ResidentGLM on its slice of the weights:

  * attention by head: q_b_proj's and kv_b_proj's rows of the rank's heads (contiguous: heads
    [rank * H/n, (rank + 1) * H/n)), MLA over those heads, o_proj's columns of their values. q_a,
    kv_a, their norms and the indexer are replicated, so every rank keeps the whole latent, rope and
    index-key caches and makes the same top-k selection;
  * the dense MLP, the shared expert and every routed expert by intermediate column: gate_proj's and
    up_proj's rows, down_proj's columns. The router is replicated, so every rank routes alike and
    runs its slice of the same k experts (no expert placement, no all-to-all);
  * lm_head by vocabulary row.

o_proj and down_proj then hold a partial sum on each rank: per layer two cross-rank sums (after the
attention and after the MLP / MoE, whose shared expert is in the same partial), and one gather of
the logits. The sum is an all-gather of the partials and their sum in rank order,
((p0 + p1) + p2) + ..., on every rank, so the ranks keep identical residual streams bit for bit,
and the result does not depend on NCCL's reduction order: the same model split the same way gives
the same bits on n GPUs as n ranks emulated on one (gpu/testing/collective.py). NCCL's all-reduce
(Collective.all_reduce_sum) is faster per call but sums in its own order, so it is the A/B, not the
default.

FP8 Linears split on their 128 x 128 scale blocks: each rank's rows (or columns) must start on a
multiple of 128, as GLM-5.3's do up to 8 ranks (q_b 8 heads x 256 = 2048 rows, kv_b 8 x 448 = 3584,
o_proj 2048 columns, the MLP 1536, an expert 256). INT4 Linears are not split yet.
"""
import re

import numpy as np

from glm_vx.checkpoint import SafetensorsCheckpoint

ROWS, COLS = 'rows', 'cols'
# A Linear's prefix (its name without '.weight') -> how it splits. Anything else is replicated.
_RULES = [
    (re.compile(r'\.self_attn\.(q_b_proj|kv_b_proj)$'), ROWS),
    (re.compile(r'\.self_attn\.o_proj$'), COLS),
    (re.compile(r'\.mlp(\.shared_experts|\.experts\.\d+)?\.(gate_proj|up_proj)$'), ROWS),
    (re.compile(r'\.mlp(\.shared_experts|\.experts\.\d+)?\.down_proj$'), COLS),
    (re.compile(r'^lm_head$'), ROWS),
]
# The fused checkpoint layout's routed experts, whole tensors (no '.weight'): gate_up_proj [E, 2I, H]
# (gate rows, then up rows), down_proj [E, H, I].
_FUSED_GATE_UP = re.compile(r'\.mlp\.experts\.gate_up_proj$')
_FUSED_DOWN = re.compile(r'\.mlp\.experts\.down_proj$')


class TensorParallel:
    """This process's place in the job: rank of size, and the collective that joins the ranks
    (gpu/collective.Collective on GPUs, gpu/testing/collective's LocalCollective in tests). Only
    all_gather(tensor, out, *, stream) is used. one_shot: the per-layer sums through
    gpu/spawn/tp_all_reduce over CUDA IPC peer buffers (PeerSum) in one launch each, not an NCCL
    all-gather and size - 1 adds; the same bits."""

    def __init__(self, rank, size, collective, *, one_shot=False):
        if not 0 <= rank < size:
            raise ValueError(f'rank {rank} is not in 0..{size}')
        self.rank, self.size, self.collective = int(rank), int(size), collective
        self.one_shot = bool(one_shot)


def split(prefix, tie_word_embeddings=False):
    """ROWS, COLS or None (replicated) for the Linear `prefix`."""
    if tie_word_embeddings and prefix == 'model.embed_tokens':
        return ROWS
    for pattern, kind in _RULES:
        if pattern.search(prefix):
            return kind
    return None


class ShardedModel:
    """A GlmMoeDsaModel as rank `rank` of `size` sees it (tp_rank, tp_size: `rank` is the model's MLA
    kv_lora_rank): its slice of each split Linear (through
    _weight and weights.block_fp8 / bf16), and a config with that slice's head count and
    intermediate sizes. The embedding table and every replicated weight are the model's own."""

    def __init__(self, model, rank, size):
        c = model.config
        if bool(c.get('attention_bias', False)):
            raise NotImplementedError('tensor parallelism with attention biases (o_proj would add its bias on every rank)')
        checks = {'num_attention_heads': model.heads, 'vocab_size': int(c['vocab_size'])}
        for key in ('intermediate_size', 'moe_intermediate_size'):
            if key in c:
                checks[key] = int(c[key])
        for key, value in checks.items():
            if value % size:
                raise ValueError(f'{key} {value} does not split over {size} ranks')
        self._model, self.tp_rank, self.tp_size = model, rank, size
        self.tie = bool(c.get('tie_word_embeddings', False))
        self.config = dict(c)
        for key in ('num_attention_heads', 'num_key_value_heads', 'intermediate_size', 'moe_intermediate_size'):
            if key in c:
                self.config[key] = int(c[key]) // size
        self.heads = model.heads // size
        self.weights = _ShardedWeights(model.weights, self)

    def __getattr__(self, name):
        if name == '_model':
            raise AttributeError(name)
        return getattr(self._model, name)

    def part(self, n):
        """The rank's slice of n rows or columns."""
        width = n // self.tp_size
        if width * self.tp_size != n:
            raise ValueError(f'{n} does not split over {self.tp_size} ranks')
        return slice(self.tp_rank * width, (self.tp_rank + 1) * width)

    def slice_dense(self, name, array):
        """The rank's part of a float32 tensor `name` (a Linear's '.weight' or a fused expert tensor)."""
        array = np.asarray(array)
        if _FUSED_GATE_UP.search(name):
            inter = array.shape[1] // 2
            s = self.part(inter)
            return np.concatenate([array[:, s], array[:, inter + s.start:inter + s.stop]], axis=1)
        if _FUSED_DOWN.search(name):
            return array[:, :, self.part(array.shape[2])]
        if not name.endswith('.weight'):
            return array
        kind = split(name[:-len('.weight')], self.tie)
        if kind == ROWS:
            return array[self.part(array.shape[0])]
        if kind == COLS:
            return array[:, self.part(array.shape[1])]
        return array

    def _weight(self, name):
        return self.slice_dense(name, self._model._weight(name))


class _ShardedWeights:
    """The model's weight source with block_fp8 and bf16 giving the rank's slice."""

    def __init__(self, base, model):
        self._base, self._model = base, model

    def __getattr__(self, name):
        if name == '_base':
            raise AttributeError(name)
        return getattr(self._base, name)

    def __call__(self, name):
        return self._model._weight(name)

    __getitem__ = __call__

    def _kind(self, prefix):
        return split(prefix, self._model.tie)

    def _uncopied(self, method, prefix):
        """The base's arrays without its copy where it can give views (SafetensorsCheckpoint): a
        rank copies its slice only, not every full tensor and then the slice."""
        if self._kind(prefix) is not None and isinstance(self._base, SafetensorsCheckpoint):
            return getattr(self._base, method)(prefix, copy=False)
        return getattr(self._base, method)(prefix)

    def block_fp8(self, prefix):
        codes, scales, block, shape = self._uncopied('block_fp8', prefix)
        kind = self._kind(prefix)
        if kind is None:
            return codes, scales, block, shape
        axis = 0 if kind == ROWS else 1
        s = self._model.part(int(shape[axis]))
        edge = int(block[axis])
        if s.start % edge or (s.stop % edge and s.stop != shape[axis]):
            raise ValueError(f'{prefix}: rank {self._model.tp_rank} of {self._model.tp_size} takes {"rows" if axis == 0 else "columns"} '
                             f'{s.start}..{s.stop}, not on its {edge}-wide FP8 scale blocks')
        blocks = slice(s.start // edge, -(-s.stop // edge))
        if axis == 0:
            codes, scales = codes[s], scales[blocks]
        else:
            codes, scales = codes[:, s], scales[:, blocks]
        return np.ascontiguousarray(codes), np.ascontiguousarray(scales), block, codes.shape

    def bf16(self, prefix):
        bits, shape = self._uncopied('bf16', prefix)
        kind = self._kind(prefix)
        if kind == ROWS:
            bits = bits[self._model.part(int(shape[0]))]
        elif kind == COLS:
            bits = bits[:, self._model.part(int(shape[1]))]
        bits = np.ascontiguousarray(bits)
        return bits, bits.shape

    def compressed_int4(self, prefix):
        if self._kind(prefix) is not None:
            raise NotImplementedError(f'{prefix}: tensor parallelism does not split INT4 Linears yet')
        return self._base.compressed_int4(prefix)


TP_ALL_REDUCE_RANKS = 8  # gpu/spawn/tp_all_reduce's buffer parameters


class PeerSum:
    """The buffers of the one-shot cross-rank sum, gpu/spawn/tp_all_reduce, for vectors of n floats:
    this rank's two halves of n floats and a flag a block (one cuMemAlloc allocation, zero at first),
    every other rank's mapped through CUDA IPC (gpu.runtime.PeerBuffer), and this rank's call count a
    block. exchange(handle) takes this rank's 64-byte IPC handle and returns every rank's in rank
    order (handles_by_collective, or handles_by_files for ranks sharing one GPU, where NCCL will not
    run). Every rank must make its PeerSum at the same point, and call it the same number of times."""

    def __init__(self, context, rank, ranks, n, exchange):
        from .runtime import DeviceTensor, PeerBuffer
        if not 1 <= ranks <= TP_ALL_REDUCE_RANKS or not 0 <= rank < ranks:
            raise ValueError(f'tp_all_reduce takes 1 to {TP_ALL_REDUCE_RANKS} ranks, not rank {rank} of {ranks}')
        self.rank, self.ranks, self.n = rank, ranks, n
        self.blocks = -(-n // 256)
        data = -(-2 * n * 4 // 256) * 256
        total = data + self.blocks * 4
        own = context.allocate(total)
        own.upload(bytes(total))
        self.epochs = context.tensor((self.blocks,), 'uint32', bytes(self.blocks * 4))
        handles = exchange(own.ipc_handle())
        if len(handles) != ranks:
            raise ValueError(f'the exchange gave {len(handles)} handles for {ranks} ranks')
        self.buffers = [own if r == rank else PeerBuffer(context, handles[r], total) for r in range(ranks)]
        self.data = [DeviceTensor(b, (2 * n,), 'float32') for b in self.buffers]
        self.flags = [DeviceTensor(b, (self.blocks,), 'uint32', offset=data) for b in self.buffers]

    def arguments(self, x):
        """tp_all_reduce's arguments to sum x (n floats) over the ranks, in place."""
        if x.size != self.n:
            raise ValueError(f'this PeerSum sums {self.n} floats, not {x.size}')
        pad = TP_ALL_REDUCE_RANKS - self.ranks
        data, flags = self.data + [self.data[self.rank]] * pad, self.flags + [self.flags[self.rank]] * pad
        values = dict(x=x, mine=self.data[self.rank], my_flags=self.flags[self.rank], epochs=self.epochs,
                      ranks=self.ranks, n=self.n)
        values.update({f'd{r}': data[r] for r in range(TP_ALL_REDUCE_RANKS)})
        values.update({f'f{r}': flags[r] for r in range(TP_ALL_REDUCE_RANKS)})
        return values

    def close(self):
        for buffer in self.buffers[::-1]:
            buffer.close()


def handles_by_collective(collective, context, stream):
    """PeerSum's exchange through the job's all_gather (the 64 bytes as 16 float32, copied as bits)."""
    def exchange(handle):
        mine = context.tensor((16,), 'float32', handle)
        every = context.tensor((16 * collective.ranks,), 'float32')
        collective.all_gather(mine, every, stream=stream)
        stream.synchronize()
        raw = every.download()
        return [raw[64 * r:64 * (r + 1)] for r in range(collective.ranks)]
    return exchange


def handles_by_files(directory, rank, ranks, timeout=120.0):
    """PeerSum's exchange through files in an empty directory every rank shares (ranks on one GPU)."""
    import os
    import time
    from pathlib import Path

    def exchange(handle):
        root = Path(directory)
        part = root / f'{rank}.tmp'
        part.write_bytes(handle)
        os.replace(part, root / f'{rank}.handle')
        deadline = time.monotonic() + timeout
        while not all((root / f'{r}.handle').exists() for r in range(ranks)):
            if time.monotonic() > deadline:
                raise TimeoutError(f'rank {rank}: not every rank wrote its IPC handle to {root}')
            time.sleep(0.01)
        return [(root / f'{r}.handle').read_bytes() for r in range(ranks)]
    return exchange
