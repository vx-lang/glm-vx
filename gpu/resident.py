"""Device-resident GLM decode: weights, activations and the KV cache stay on the GPU.

`GPUBackend` (gpu/backend.py) runs one primitive per call and reads every result back to the
host, so a decode step is dominated by transfers. `ResidentGLM` runs a whole decode step through
the same Vx kernels with activations and the per-layer MLA / indexer caches in device memory. The
only host traffic per token is:

  * the embedding row up, and the RoPE cos/sin for the position up;
  * nothing per indexer layer: the top-k runs on the device (gpu/topk.vx, the model's exact rule,
    np.argsort(-scores, kind='stable')). ``device_topk=False`` restores the host argsort (scores
    down, ids up) for comparison;
  * per MoE layer, the k routed expert ids down (to pick which expert weights to launch), unless
    ``device_experts=True``: then the experts run in two launches, gpu/spawn's moe_gate_up and
    moe_down, which read the ids on the device (each MoE projection's experts are uploaded back to
    back, one tensor per projection and layer), and a decode step makes no host read until the
    logits;
  * the logits down.

Numerics follow `GlmMoeDsaModel` operation for operation (same kernels, same reduction orders,
same expert aggregation order), so results should agree with the hybrid path to float32 rounding;
`validation/resident_compare.py` checks that on a GPU. Weights are uploaded once and kept.

INT4: a Linear the weight source reports as compressed-tensors INT4 (``is_compressed_int4(prefix)``,
as SafetensorsCheckpoint and glm_vx.quant.CompressedInt4Weights do) stays packed on the device and
runs through ``glm_vx_gpu_int4_matvec`` (gpu/int4.ptx) with its bf16/fp16 group scales; every other
weight is float32. The exception is ``kv_b_proj``, which absorbed MLA reads transposed per head: it
is dequantized exactly (code * scale) to float32 once at upload. The NumPy reference sees the same
exact dequantized weights, so the only differences from it are float32 accumulation order (the INT4
kernel sums code * x per 128-column group, then scales).

FP8: a Linear the source reports as block-scaled FP8 (``is_block_fp8(prefix)``, as
SafetensorsCheckpoint and glm_vx.quant.BlockFP8Weights do) stays as E4M3FN bytes on the device with
its float32 128x128 ``weight_scale_inv`` grid and runs through ``glm_vx_gpu_fp8_matvec``
(gpu/fp8.ptx). Activations stay float32: this is weight-only FP8, the same arithmetic as the NumPy
reference on the exact dequantized weights, not the official per-token FP8 activation quantization.
``kv_b_proj`` is dequantized to float32 as for INT4.

BF16: any other Linear (including the router and lm_head) the source reports as a bfloat16 matrix
(``is_bf16(prefix)``, as SafetensorsCheckpoint and glm_vx.quant.BF16Weights do) stays as its 16-bit
patterns on the device and runs through ``glm_vx_gpu_bf16_matvec`` (gpu/bf16.ptx). That kernel
decodes exactly and sums in the float32 matvec's order, so the logits equal the float32 path on the
same bf16 values bit for bit. Norms, biases, embeddings and ``kv_b_proj`` stay float32.
GGUF packed weights, expert placement and tracing are not supported.

``fused_attention=True`` runs each layer's MLA decode in one launch, gpu/spawn's mla_decode (the
block of each head's last split merges that head), in place of mla_partial then mla_merge. Its exp
is libdevice's expf, not the hand kernels' ex2.approx, so logits differ from the default in the last
bits. It needs Vx's code of a group of blocks (#2185 to #2188, on Vx main from c3716538).

``graph=True`` captures the decode step's layers into a CUDA graph (#26) and replays it every token:
one upload of the step's inputs (the embedding row, cos, sin and the step buffer), one graph launch,
then the final norm, lm_head and the logits down. What changes per token comes from the step buffer
``[position, tokens, count, splits]`` (#25): the ``*_at`` kernels read it on the device, launch with
the grid for ``max_tokens`` (or ``splits_max``), and new cache rows go through fixed rows and
``store_row``. The first step runs the same launches eagerly, so every weight is uploaded and every
kernel loaded before the capture at the second; the arithmetic is the per-launch path's, so the
logits are equal bit for bit. It needs ``device_topk`` and ``device_experts`` (no host read inside a
step). With ``fused_attention`` the graph holds gpu/spawn's mla_decode_at (#27): splits_max splits
per head, those past the step's splits idle but arriving, so the arrival counts still return to zero.

``mla_kernel='chunks'`` runs the fused attention under graph replay through gpu/spawn's
mla_chunks_at (#33): 32 tokens a chunk, equal to mla_decode_at to rounding. ``mla_kernel='heads'``
runs mla_heads_at instead: four heads a block, so each selected latent row is read once for four
heads; each head's arithmetic is mla_chunks_at's, so the logits equal 'chunks' bit for bit. It needs
a head count that is a multiple of 4.

``fp8_kernels='x4'`` runs FP8 Linears through gpu/spawn's fp8_matvec4 and, with ``device_experts``,
the routed experts through fp8_moe4_gate_up/down (#30, #31): sixteen codes a lane step, one 16-byte
load. Their summation order differs from the hand kernels', so the logits are close to the default
path's, not equal; each kernel's own order is pinned by gpu/test_spawn_vx_host.py. Widths that are
not a multiple of 16 keep the default kernels.

``matvec_kernels='spawn'`` runs INT4 and BF16 Linears through gpu/spawn's int4_matvec and
bf16_matvec instead of the hand kernels. gpu/test_spawn_hardware.py checks each against its hand
kernel bit for bit, so the logits do not change; the spawn kernels share device.vx's decode macros.
``matvec_kernels='x4'`` runs them through int4_matvec4 and bf16_matvec4 instead (#30): each lane reads
four consecutive u32 words a step (32 INT4 codes or 8 bfloat16 weights), the shape Vx #2362 makes one
16-byte load. Their summation order differs from the hand kernels', so the logits are close to the
default path's, not equal; gpu/test_spawn_vx_host.py pins each order. INT4 widths that are not a
multiple of 128 and BF16 widths that are not a multiple of 8 keep the hand kernels.

Not a performance claim: no timing is made here.
``device_experts`` needs the gpu/spawn kernels built (gpu/spawn/build.sh into ptx_dir/spawn) with a
vxc that has Vx's lanes:: reductions (see gpu/spawn/README.md); it has not run on a GPU yet.
"""
import ctypes as C
from pathlib import Path

import numpy as np

from kernels.backend import _cached_rope_coefficients
from .runtime import CUDAContext

I = C.c_int32
F = C.c_float
_ELEMENTWISE_BLOCK = 256
TOPK_SORT_TOKENS = 2048  # gpu/spawn/topk_sort.vx sorts contexts up to this many tokens (#32)
TOPK_CHUNK = 2048  # past it, topk_chunks sorts chunks of this many, topk_merge merges them (#32)
_MAX_GRID = 65535


class _Planned:
    """Size-only stand-in that preload records once device memory would run out; never launched."""

    def __init__(self, shape):
        self.shape = tuple(shape)


def _planned(entry):
    return isinstance(entry, _Planned) or (isinstance(entry, tuple) and any(isinstance(e, _Planned) for e in entry))


def _flat(tensor):
    """The tensor as one dimension, for gpu/spawn kernels that take [?] weights."""
    return tensor.view((int(np.prod(tensor.shape)),))


def _f32(value, name, positive=False):
    value = C.c_float(float(value)).value
    if not np.isfinite(value) or (value <= 0 if positive else False):
        raise ValueError(name + ' must be finite' + (' and positive' if positive else '') + ' in float32')
    return value


class ResidentGLM:
    name = 'vx-gpu-resident'

    def __init__(self, model, *, max_tokens=4096, ptx_dir=None, device=0, ctx=None, max_weight_bytes=None,
                 device_topk=True, device_experts=False, fused_attention=False, graph=False, fp8_kernels='hand',
                 topk='auto', mla_kernel='decode', matvec_kernels='hand'):
        if getattr(model, 'packed_weights', False):
            raise NotImplementedError('ResidentGLM supports expanded float32 weights only')
        if getattr(model, 'expert_placement', None) is not None:
            raise NotImplementedError('ResidentGLM does not support expert placement')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        self.model, self.config = model, model.config
        self.max_tokens = max_tokens
        self.device_topk = bool(device_topk)
        self.device_experts = bool(device_experts)
        self.fused_attention = bool(fused_attention)
        self.graph = bool(graph)
        if fp8_kernels not in ('hand', 'x4'):
            raise ValueError("fp8_kernels must be 'hand' or 'x4'")
        self.fp8_kernels = fp8_kernels
        if matvec_kernels not in ('hand', 'spawn', 'x4'):
            raise ValueError("matvec_kernels must be 'hand', 'spawn' or 'x4'")
        self.matvec_kernels = matvec_kernels
        if topk not in ('auto', 'sort', 'radix'):
            raise ValueError("topk must be 'auto', 'sort' or 'radix'")
        self._topk_choice = topk
        if mla_kernel not in ('decode', 'chunks', 'heads'):
            raise ValueError("mla_kernel must be 'decode', 'chunks' or 'heads'")
        self.mla_kernel = mla_kernel
        if mla_kernel != 'decode' and not (self.graph and self.fused_attention):
            raise ValueError(f"mla_kernel={mla_kernel!r} needs graph=True and fused_attention=True")
        if self.graph and not (self.device_topk and self.device_experts):
            raise ValueError('graph=True needs device_topk and device_experts: a captured step cannot read device memory')
        self._step_graph, self._warm = None, False
        self.replays = 0
        self.max_weight_bytes = max_weight_bytes
        self.weight_bytes = 0
        self._room = self._plan = None  # preload's fit check, see preload()
        self._owns_context = ctx is None
        self.ctx = ctx if ctx is not None else CUDAContext(device, minimum_compute_capability=(8, 0))
        self._closed = False
        try:
            ptx = Path(ptx_dir or Path(__file__).parent / 'build') / 'kernels.ptx'
            self.module = self.ctx.load_ptx(ptx)
            self._ptx_dir = ptx.parent
            self._modules = {}
            source = model.weights
            self._is_int4 = getattr(source, 'is_compressed_int4', None) or (lambda prefix: False)
            self._is_fp8 = getattr(source, 'is_block_fp8', None) or (lambda prefix: False)
            self._is_bf16 = getattr(source, 'is_bf16', None) or (lambda prefix: False)
            self._formats = {}
            self._check_quantization(model.config.get('quantization_config'))
            self.stream = self.ctx.stream()
            self._kernels = {}
            self._spawn_kernels = {}
            self.launches = {}
            self._weights = {}
            self._allocate()
            self.position = 0
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check_quantization(q):
        if q is None:
            return
        from glm_vx.quant import block_fp8_scheme, compressed_int4_scheme
        method = q.get('quant_method') if isinstance(q, dict) else None
        if method == 'fp8':
            block_fp8_scheme(q)
        elif method == 'compressed-tensors':
            compressed_int4_scheme(q)
        else:
            raise ValueError(f'quant_method must be compressed-tensors (INT4) or fp8, got {method!r}')

    # ---------------------------------------------------------------- resources
    def close(self):
        if not self._closed:
            self._closed = True
            if self._owns_context:
                self.ctx.close()

    def _tensor(self, shape, dtype='float32', data=None):
        return self.ctx.tensor(tuple(shape), dtype, data)

    def _kernel(self, name):
        kernel = self._kernels.get(name)
        if kernel is None:
            kernel = self._kernels[name] = self.module.kernel('glm_vx_gpu_' + name)
        return kernel

    def _launch(self, name, grid, block, args):
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        self._kernel(name).launch((grid,), (block,), args, stream=self.stream)
        self.launches[name] = self.launches.get(name, 0) + 1

    def _spawn(self, name, **values):
        """Launch gpu/spawn kernel `name` (ptx_dir/spawn/<name>.ptx) with its manifest's shape;
        arguments by parameter name. A test context may supply spawn_kernel(path)."""
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        kernel = self._spawn_kernels.get(name)
        if kernel is None:
            path = self._ptx_dir / 'spawn' / (name + '.ptx')
            factory = getattr(self.ctx, 'spawn_kernel', None)
            if factory is None:
                from .spawn.launch import SpawnKernel
                kernel = SpawnKernel(self.ctx, path)
            else:
                kernel = factory(path)
            self._spawn_kernels[name] = kernel
        kernel.launch(stream=self.stream, **values)
        self.launches[name] = self.launches.get(name, 0) + 1

    def _grid(self, n):
        return max(1, min((n + _ELEMENTWISE_BLOCK - 1) // _ELEMENTWISE_BLOCK, _MAX_GRID))

    def _weight(self, name):
        """Upload a model tensor once; the device copy is kept for the life of this object."""
        tensor = self._weights.get(name)
        if tensor is None:
            tensor = self._upload_weight(name, np.ascontiguousarray(self.model._weight(name), dtype=np.float32))
        return tensor

    def _upload_weight(self, key, array, dtype='float32'):
        if self.max_weight_bytes is not None and self.weight_bytes + array.nbytes > self.max_weight_bytes:
            raise MemoryError(f'device weights would exceed max_weight_bytes={self.max_weight_bytes} at {key!r}')
        if self._plan is None and self._room is not None and array.nbytes > self._room:
            self._plan = 0  # out of device memory: size the rest without uploading it
        if self._plan is not None:
            self._plan += array.nbytes
            self._weights[key] = tensor = _Planned(array.shape)
            return tensor
        if self._room is not None:
            self._room -= array.nbytes
        tensor = self._tensor(array.shape, dtype, array.tobytes())
        self.weight_bytes += array.nbytes
        self._weights[key] = tensor
        return tensor

    def _upload_stacked(self, key, parts, dtype='float32'):
        """Upload equal-shape arrays back to back as one [len(parts), ..] tensor, one part at a
        time (parts is a list of callables returning each array), with _upload_weight's limits."""
        first = np.ascontiguousarray(parts[0]())
        shape = (len(parts),) + first.shape
        nbytes = first.nbytes * len(parts)
        if self.max_weight_bytes is not None and self.weight_bytes + nbytes > self.max_weight_bytes:
            raise MemoryError(f'device weights would exceed max_weight_bytes={self.max_weight_bytes} at {key!r}')
        if self._plan is None and self._room is not None and nbytes > self._room:
            self._plan = 0
        if self._plan is not None:
            self._plan += nbytes
            self._weights[key] = tensor = _Planned(shape)
            return tensor
        if self._room is not None:
            self._room -= nbytes
        tensor = self._tensor(shape, dtype)
        for i, part in enumerate(parts):
            array = first if i == 0 else np.ascontiguousarray(part())
            if array.shape != first.shape or array.dtype != first.dtype:
                tensor.close()
                raise ValueError(f'{key}: part {i} is {array.dtype}{array.shape}, part 0 {first.dtype}{first.shape}')
            tensor.view(first.shape, offset=i * first.nbytes).upload(array.tobytes())
        self.weight_bytes += nbytes
        self._weights[key] = tensor
        return tensor

    def _int4_weight(self, prefix):
        """(packed uint8, scale bits uint16, scale_kind, (N, K)) on the device, uploaded once."""
        key = prefix + ':int4'
        entry = self._weights.get(key)
        if entry is None:
            packed, bits, scale_format, shape = self.model.weights.compressed_int4(prefix)
            kind = {'float16': 0, 'bfloat16': 1}[scale_format]
            entry = (self._upload_weight(key + ':packed', np.ascontiguousarray(packed, np.uint8), 'uint8'),
                     self._upload_weight(key + ':scales', np.ascontiguousarray(bits, np.uint16), 'uint16'),
                     kind, tuple(shape))
            self._weights[key] = entry
        return entry

    def _fp8_weight(self, prefix):
        """(E4M3FN codes uint8, float32 block scales, (N, K)) on the device, uploaded once."""
        key = prefix + ':fp8'
        entry = self._weights.get(key)
        if entry is None:
            codes, scales, block, shape = self.model.weights.block_fp8(prefix)
            rows, cols = shape
            if tuple(block) != (128, 128):
                raise ValueError(f'{prefix}: fp8 kernels implement 128x128 blocks, got {tuple(block)}')
            if codes.shape != (rows, cols) or scales.shape != ((rows + 127) // 128, (cols + 127) // 128):
                raise ValueError(f'{prefix}: FP8 codes {codes.shape} / scales {scales.shape} do not match {shape}')
            entry = (self._upload_weight(key + ':codes', np.ascontiguousarray(codes, np.uint8), 'uint8'),
                     self._upload_weight(key + ':scales', np.ascontiguousarray(scales, np.float32)),
                     (rows, cols))
            self._weights[key] = entry
        return entry

    def _bf16_weight(self, prefix):
        """(bfloat16 bits uint16, (N, K)) on the device, uploaded once."""
        key = prefix + ':bf16'
        entry = self._weights.get(key)
        if entry is None:
            bits, shape = self.model.weights.bf16(prefix)
            if bits.shape != tuple(shape):
                raise ValueError(f'{prefix}: BF16 bits {bits.shape} do not match {shape}')
            entry = (self._upload_weight(key, np.ascontiguousarray(bits, np.uint16), 'uint16'), tuple(shape))
            self._weights[key] = entry
        return entry

    def _format(self, prefix):
        """'int4', 'fp8', 'bf16' or 'float32' for the Linear `prefix`, decided once."""
        fmt = self._formats.get(prefix)
        if fmt is None:
            fmt = ('int4' if self._is_int4(prefix) else 'fp8' if self._is_fp8(prefix)
                   else 'bf16' if self._is_bf16(prefix) else 'float32')
            self._formats[prefix] = fmt
        return fmt

    def _out_features(self, prefix):
        fmt = self._format(prefix)
        if fmt == 'int4':
            return self._int4_weight(prefix)[3][0]
        if fmt == 'fp8':
            return self._fp8_weight(prefix)[2][0]
        if fmt == 'bf16':
            return self._bf16_weight(prefix)[1][0]
        return self._weight(prefix + '.weight').shape[0]

    def _allocate(self):
        m, c = self.model, self.config
        cap = self.max_tokens
        self.hidden, self.heads = m.hidden, m.heads
        if self.mla_kernel == 'heads' and self.heads % 4:
            raise ValueError(f"mla_kernel='heads' needs a multiple of 4 heads, the model has {self.heads}")
        self.rank, self.nope, self.rot, self.value_dim = m.rank, m.nope, m.rot, m.value_dim
        self.index_heads, self.index_dim, self.index_topk = m.index_heads, m.index_dim, m.index_topk
        qk = self.nope + self.rot
        q_rank = int(c['q_lora_rank'])
        widths = [int(c.get('moe_intermediate_size', 0)) * max(1, int(c.get('n_shared_experts', 1))),
                  int(c.get('moe_intermediate_size', 0))]
        if 'dense' in m.mlp_types:
            widths.append(int(c['intermediate_size']))
        width = max(widths)
        t = self._tensor
        # Residual stream ping-pong and per-block scratch.
        half = self.rot // 2
        if self.graph:
            # The step's inputs back to back, one upload per token: x (the embedding row; the
            # ping-pong returns to it after every layer's two swaps), cos, sin, then the int32 step
            # buffer [position, tokens, count, splits] the *_at kernels read.
            self.step_inputs = t((self.hidden + 2 * half + 4,))
            self.x = self.step_inputs.view((self.hidden,))
            self.cos = self.step_inputs.view((half,), offset=self.hidden * 4)
            self.sin = self.step_inputs.view((half,), offset=(self.hidden + half) * 4)
            self.step = self.step_inputs.view((4,), 'int32', offset=(self.hidden + 2 * half) * 4)
        else:
            self.x = t((self.hidden,))
            self.cos, self.sin = t((half,)), t((half,))
        self.x_next, self.norm = t((self.hidden,)), t((self.hidden,))
        self.attn_out, self.ffn_out = t((self.hidden,)), t((self.hidden,))
        self.q_a, self.q_res = t((q_rank,)), t((q_rank,))
        self.query = t((self.heads * qk,))
        self.compressed = t((self.rank + self.rot,))
        self.latent_q, self.rot_q = t((self.heads, self.rank)), t((self.heads, self.rot))
        self.splits_max = 16
        self.numerator = t((self.heads, self.splits_max, self.rank))
        self.stats = t((self.heads, self.splits_max, 2))
        self.attended = t((self.heads, self.rank))
        if self.fused_attention:  # mla_decode's per-head arrival counts: zero once, its last splits re-zero them
            self.arrivals = t((self.heads,), 'int32', np.zeros(self.heads, np.int32))
        self.heads_out = t((self.heads * self.value_dim,))
        self.iq, self.iq_rot = t((self.index_heads * self.index_dim,)), t((self.index_heads * self.index_dim,))
        self.ik, self.ik_norm = t((self.index_dim,)), t((self.index_dim,))
        self.head_weights_raw, self.head_weights = t((self.index_heads,)), t((self.index_heads,))
        self.scores = t((cap,))
        self.selected = t((self.index_topk,), 'int32')
        if self.device_topk:
            from .topk import scratch_ints
            self.topk_scratch = t((scratch_ints(cap, self.index_topk),), 'int32')
            if cap > TOPK_SORT_TOKENS and self.index_topk <= TOPK_CHUNK:
                # topk_chunks' sorted rows and topk_merge's, swapped each round (#32).
                chunks = -(-cap // TOPK_CHUNK)
                self.topk_lists = [t((chunks, TOPK_CHUNK), 'int64') for _ in range(2)]
        self.selected_count = 0
        self.gate, self.up, self.act = t((width,)), t((width,)), t((width,))
        self.fused = t((2 * int(c.get('moe_intermediate_size', 1)),))
        self.mlp_out, self.expert_out = t((self.hidden,)), t((self.hidden,))
        self.moe_sum = t((self.hidden,))
        self.zeros = t((self.hidden,))
        # With graph=True, this step's cache rows before store_row puts them at the position.
        self.latent_new = self.rope_new = self.index_key_new = None
        if self.graph:
            self.latent_new, self.rope_new, self.index_key_new = t((self.rank,)), t((self.rot,)), t((self.index_dim,))
        if 'sparse' in m.mlp_types:
            experts = int(c['n_routed_experts'])
            k = int(c['num_experts_per_tok'])
            self.router_logits = t((experts,))
            self.route_ids, self.route_weights = t((k,), 'int32'), t((k,))
            if self.device_experts:  # each routed expert's SwiGLU output, in ascending id order
                self.moe_act = t((k * int(c['moe_intermediate_size']),))
        vocab = int(c['vocab_size'])
        self.logits = t((vocab,))
        # Per-layer caches, all rows the caller has produced so far stay on the device.
        self.latents = [t((cap, self.rank)) for _ in range(m.n_layers)]
        self.rope_keys = [t((cap, self.rot)) for _ in range(m.n_layers)]
        self.index_keys = [t((cap, self.index_dim)) if m.indexer_types[i] == 'full' else None
                           for i in range(m.n_layers)]

    def reset(self):
        """Start a new sequence. Device caches are overwritten as positions are reused."""
        self.position = 0
        self.selected_count = 0

    # ---------------------------------------------------------------- op helpers
    def _matvec(self, weight, x, out):
        rows, cols = weight.shape
        self._launch('matvec', (rows + 3) // 4, 128, [out, weight, x, I(rows), I(cols)])
        return out

    def _project(self, prefix, x, out):
        """out = W x for the Linear `prefix`: the INT4, FP8 or BF16 kernel on stored weights, else float32 matvec."""
        fmt = self._format(prefix)
        if fmt == 'float32':
            return self._matvec(self._weight(prefix + '.weight'), x, out)
        if fmt == 'int4':
            packed, scales, kind, (rows, cols) = self._int4_weight(prefix)
            if self.matvec_kernels == 'x4' and cols % 128 == 0:
                self._check_shape(prefix, fmt, rows, cols, x, out)
                self._spawn('int4_matvec4', out=out, packed=packed.view((rows * cols // 8,), 'uint32'),
                            scales=_flat(scales), x=x, rows=rows, cols=cols, scale_kind=kind)
                return out
            if self.matvec_kernels == 'spawn':
                self._check_shape(prefix, fmt, rows, cols, x, out)
                self._spawn('int4_matvec', out=out, packed=_flat(packed), scales=_flat(scales), x=x, rows=rows,
                            cols=cols, scale_kind=kind)
                return out
            args = [packed, scales, x, I(rows), I(cols), I(kind)]
        elif fmt == 'fp8':
            codes, scales, (rows, cols) = self._fp8_weight(prefix)
            if self.fp8_kernels == 'x4' and cols % 16 == 0:
                if out.shape != (rows,) or x.shape != (cols,):
                    raise ValueError(f'{prefix}: FP8 weight is {rows}x{cols}, got x {x.shape} -> out {out.shape}')
                self._spawn('fp8_matvec4', out=out, weight=codes.view((rows * cols // 4,), 'uint32'),
                            scales=scales.view((int(np.prod(scales.shape)),)), x=x, rows=rows, cols=cols)
                return out
            args = [codes, scales, x, I(rows), I(cols)]
        else:
            bits, (rows, cols) = self._bf16_weight(prefix)
            if self.matvec_kernels == 'x4' and cols % 8 == 0:
                self._check_shape(prefix, fmt, rows, cols, x, out)
                self._spawn('bf16_matvec4', out=out, weight=bits.view((rows * cols // 2,), 'uint32'), x=x, rows=rows,
                            cols=cols)
                return out
            if self.matvec_kernels == 'spawn':
                self._check_shape(prefix, fmt, rows, cols, x, out)
                self._spawn('bf16_matvec', out=out, weight=_flat(bits), x=x, rows=rows, cols=cols)
                return out
            args = [bits, x, I(rows), I(cols)]
        self._check_shape(prefix, fmt, rows, cols, x, out)
        self._launch(self._format_kernel(fmt), (rows + 3) // 4, 128, [out] + args)
        return out

    @staticmethod
    def _check_shape(prefix, fmt, rows, cols, x, out):
        if out.shape != (rows,) or x.shape != (cols,):
            raise ValueError(f'{prefix}: {fmt.upper()} weight is {rows}x{cols}, got x {x.shape} -> out {out.shape}')

    def _format_kernel(self, fmt):
        name = fmt + '_matvec'
        if name not in self._kernels:
            module = self._modules.get(fmt)
            if module is None:
                module = self._modules[fmt] = self.ctx.load_ptx(self._ptx_dir / (fmt + '.ptx'))
            self._kernels[name] = module.kernel('glm_vx_gpu_' + name)
        return name

    def _topk_sort(self):
        """Whether contexts up to TOPK_SORT_TOKENS select through gpu/spawn's topk_sort (one
        shared-memory sort, the same ids as topk_stable): 'sort' always, 'radix' never, 'auto' when
        it is built."""
        if self._topk_choice == 'auto':
            self._topk_choice = 'sort' if (self._ptx_dir / 'spawn' / 'topk_sort.ptx').exists() else 'radix'
        return self._topk_choice == 'sort'

    def _topk_merge(self):
        """Whether contexts past TOPK_SORT_TOKENS select through gpu/spawn's topk_chunks and topk_merge
        rounds (the same ids as topk_stable): with 'sort', when they are built and index_topk <= 2048.
        Their chunk is topk_sort's limit, so past it there are always two chunks or more."""
        names = ('topk_chunks_at', 'topk_merge_at') if self.graph else ('topk_chunks', 'topk_merge')
        return (self._topk_sort() and hasattr(self, 'topk_lists') and TOPK_SORT_TOKENS == TOPK_CHUNK
                and all((self._ptx_dir / 'spawn' / (n + '.ptx')).exists() for n in names))

    def _topk_chunks_and_merge(self, tokens):
        """topk_chunks, then topk_merge rounds until one row is left; the last writes selected."""
        lists, out_ids = self.topk_lists, self.selected.view((1, self.index_topk))
        cap = self.max_tokens
        if self.graph:
            # Captured for max_tokens: each launch reads tokens from the step buffer.
            self._spawn('topk_chunks_at', lists=lists[0], scores=self.scores.view(self.scores.shape, 'int32'),
                        step=self.step, max_tokens=cap)
            for r in range((-(-cap // TOPK_CHUNK) - 1).bit_length()):
                self._spawn('topk_merge_at', dst=lists[(r + 1) % 2], src=lists[r % 2], out_ids=out_ids, step=self.step,
                            max_tokens=cap, round=r, k=self.index_topk)
            return
        self._spawn('topk_chunks', lists=lists[0], scores=self.scores.view((tokens,), 'int32'), tokens=tokens)
        count, r = -(-tokens // TOPK_CHUNK), 0
        while count > 1:
            self._spawn('topk_merge', dst=lists[(r + 1) % 2], src=lists[r % 2], out_ids=out_ids, lists=count,
                        tokens=tokens, k=self.index_topk)
            count, r = -(-count // 2), r + 1

    def _topk_kernel(self, name=None):
        name = name or ('topk_stable_at' if self.graph else 'topk_stable')
        if name not in self._kernels:
            module = self._modules.get('topk')
            if module is None:
                module = self._modules['topk'] = self.ctx.load_ptx(self._ptx_dir / 'topk.ptx')
            self._kernels[name] = module.kernel('glm_vx_gpu_' + name)
        return name

    def _linear(self, name, x, out, bias=False):
        self._project(name, x, out)
        if bias:
            self._add(out, out, self._weight(name + '.bias'), out.shape[0])
        return out

    def _add(self, out, a, b, n):
        self._launch('add', self._grid(n), _ELEMENTWISE_BLOCK, [out, a, b, I(n)])

    def _copy(self, out, x, n):
        self._launch('copy', self._grid(n), _ELEMENTWISE_BLOCK, [out, x, I(n)])

    def _rmsnorm(self, x, weight, out, eps):
        n = x.shape[0]
        self._launch('rmsnorm', 1, 256, [out, x, weight, I(n), F(_f32(eps, 'epsilon', True))])

    def _swiglu(self, gate, up, out, n):
        self._launch('swiglu', self._grid(n), _ELEMENTWISE_BLOCK, [out, gate, up, I(n)])

    def _rope(self, out, x, heads, dim, x_stride, out_stride):
        n = heads * (dim // 2)
        self._launch('rope_strided', self._grid(n), _ELEMENTWISE_BLOCK,
                     [out, x, self.cos, self.sin, I(heads), I(dim), I(x_stride), I(out_stride)])

    def _row(self, tensor, position, width):
        return tensor.view((width,), offset=position * width * 4)

    def _new_row(self, cache, position, width, scratch):
        """Where this step's row of `cache` is written: the row itself, or with graph=True a
        fixed scratch row that _store_row then puts at the position the step buffer holds."""
        return scratch.view((width,)) if self.graph else self._row(cache, position, width)

    def _store_row(self, cache, scratch, width):
        if self.graph:
            self._launch('store_row', self._grid(width), _ELEMENTWISE_BLOCK, [cache, scratch, self.step, I(width)])

    def _mlp(self, prefix, x, out):
        """down(SwiGLU(gate(x), up(x))) with both projections and the activation resident."""
        width = self._out_features(prefix + '.gate_proj')
        gate, up, act = (t.view((width,)) for t in (self.gate, self.up, self.act))
        self._project(prefix + '.gate_proj', x, gate)
        self._project(prefix + '.up_proj', x, up)
        self._swiglu(gate, up, act, width)
        self._project(prefix + '.down_proj', act, out)

    def _fused_experts(self, base):
        try:
            if self._format(base + '.gate_proj') == 'float32':
                self.model._weight(base + '.gate_proj.weight')
            return False
        except KeyError:
            return True

    def _expert(self, prefix, expert, x, out):
        base = f'{prefix}.experts.{expert}'
        if self._fused_experts(base):
            # Fused checkpoint layout: gate_up_proj[E, 2I, H] and down_proj[E, H, I]. The whole
            # tensors are uploaded once; each expert is a view into them.
            gate_up = self._weight(prefix + '.experts.gate_up_proj')
            down = self._weight(prefix + '.experts.down_proj')
            two_i, hidden = gate_up.shape[1], gate_up.shape[2]
            inter = two_i // 2
            fused = self.fused.view((two_i,))
            self._matvec(gate_up.view((two_i, hidden), offset=expert * two_i * hidden * 4), x, fused)
            gate, up = fused.view((inter,)), fused.view((inter,), offset=inter * 4)
            act = self.act.view((inter,))
            self._swiglu(gate, up, act, inter)
            self._matvec(down.view(down.shape[1:], offset=expert * down.shape[1] * down.shape[2] * 4), act, out)
            return
        self._mlp(base, x, out)

    def _stacked_experts(self, prefix, name):
        """All routed experts' Linear `name` ('gate_proj', 'up_proj', 'down_proj', or 'gate_up' /
        'down' of the fused layout) back to back on the device, uploaded once:
        (format, tensors, (rows, cols), extra arguments)."""
        key = f'{prefix}.experts:{name}:stacked'
        entry = self._weights.get(key)
        if entry is not None:
            return entry
        experts = int(self.config['n_routed_experts'])
        if name in ('gate_up', 'down'):
            tensor = self._weight(f'{prefix}.experts.{name}_proj')
            if _planned(tensor):
                return tensor
            entry = ('float32', (tensor,), tuple(tensor.shape[1:]), {})
            self._weights[key] = entry
            return entry
        linears = [f'{prefix}.experts.{e}.{name}' for e in range(experts)]
        fmt = self._format(linears[0])
        mixed = [p for p in linears if self._format(p) != fmt]
        if mixed:
            raise ValueError(f'device_experts needs one format per expert projection: {linears[0]} is {fmt}, '
                             f'{mixed[0]} is {self._format(mixed[0])}')
        w = self.model.weights
        if fmt == 'int4':
            first = w.compressed_int4(linears[0])
            kind = {'float16': 0, 'bfloat16': 1}[first[2]]
            for p in linears[1:]:
                if w.compressed_int4(p)[2] != first[2]:
                    raise ValueError(f'device_experts needs one INT4 scale format per projection: {p}')
            tensors = (self._upload_stacked(key + ':packed', [lambda p=p: np.asarray(w.compressed_int4(p)[0], np.uint8) for p in linears], 'uint8'),
                       self._upload_stacked(key + ':scales', [lambda p=p: np.asarray(w.compressed_int4(p)[1], np.uint16) for p in linears], 'uint16'))
            shape, extra = tuple(first[3]), {'scale_kind': kind}
        elif fmt == 'fp8':
            first = w.block_fp8(linears[0])
            if tuple(first[2]) != (128, 128):
                raise ValueError(f'{linears[0]}: fp8 kernels implement 128x128 blocks, got {tuple(first[2])}')
            tensors = (self._upload_stacked(key + ':codes', [lambda p=p: np.asarray(w.block_fp8(p)[0], np.uint8) for p in linears], 'uint8'),
                       self._upload_stacked(key + ':scales', [lambda p=p: np.asarray(w.block_fp8(p)[1], np.float32) for p in linears]))
            shape, extra = tuple(first[3]), {}
        elif fmt == 'bf16':
            first = w.bf16(linears[0])
            tensors = (self._upload_stacked(key, [lambda p=p: np.asarray(w.bf16(p)[0], np.uint16) for p in linears], 'uint16'),)
            shape, extra = tuple(first[1]), {}
        else:
            tensors = (self._upload_stacked(key, [lambda p=p: np.asarray(self.model._weight(p + '.weight'), np.float32) for p in linears]),)
            shape, extra = tuple(tensors[0].shape[1:]), {}
        if any(_planned(t) for t in tensors):
            return tensors[0]
        entry = (fmt, tensors, shape, extra)
        self._weights[key] = entry
        return entry

    def _device_moe(self, prefix, x, k):
        """moe_sum = the routed experts' weighted sum in two launches, the ids read on the device:
        gpu/spawn's moe_gate_up (every expert's gate, up and SwiGLU) then moe_down (each output
        row sums its k terms in ascending id order). The same arithmetic as the per-expert
        kernels (*_expert_matvec, swiglu, expert_scaled_add from zeros), to the last bit."""
        experts = int(self.config['n_routed_experts'])
        if self._fused_experts(f'{prefix}.experts.0'):
            gate = up = self._stacked_experts(prefix, 'gate_up')
            down = self._stacked_experts(prefix, 'down')
            stride = gate[2][0]
            inter, up_row = stride // 2, stride // 2
        else:
            gate, up, down = (self._stacked_experts(prefix, n) for n in ('gate_proj', 'up_proj', 'down_proj'))
            inter, up_row = gate[2][0], 0
            stride = inter
        fmt, hidden = gate[0], gate[2][1]
        if up[0] != fmt or down[0] != fmt or up[3] != gate[3] or down[3] != gate[3] or tuple(down[2]) != (hidden, inter):
            raise ValueError(f'{prefix}.experts: gate, up and down need one format and scale kind and matching shapes')
        kernel = ('' if fmt == 'float32' else fmt + '_') + 'moe_'
        common = dict(ids=self.route_ids, k=k, experts=experts, **gate[3])
        gate_args = dict(gate=gate[1][0], up=up[1][0])
        down_args = dict(down=down[1][0])
        if fmt == 'fp8' and self.fp8_kernels == 'x4' and hidden % 16 == 0 and inter % 16 == 0:
            kernel = 'fp8_moe4_'  # the codes as u32, sixteen a lane step (#31)
            words = lambda t: t.view((int(np.prod(t.shape)) // 4,), 'uint32')
            gate_args, down_args = dict(gate=words(gate[1][0]), up=words(up[1][0])), dict(down=words(down[1][0]))
        if fmt in ('int4', 'fp8'):
            gate_args.update(gate_scales=gate[1][1], up_scales=up[1][1])
            down_args.update(scales=down[1][1])
        act = self.moe_act.view((k * inter,))
        self._spawn(kernel + 'gate_up', act=act, x=x, inter=inter, up_row=up_row, stride=stride, cols=hidden,
                    **gate_args, **common)
        self._spawn(kernel + 'down', out=self.moe_sum, act=act, weights=self.route_weights, rows=hidden,
                    cols=inter, **down_args, **common)

    # ---------------------------------------------------------------- blocks
    def _index(self, layer, prefix, x, position):
        """DSA indexer for one layer: refresh its keys, score all causal keys, choose the top-k."""
        m = self.model
        tokens = position + 1
        self._linear(prefix + '.wq_b', self.q_res, self.iq)
        self._linear(prefix + '.wk', x, self.ik)
        self._launch('layernorm', 1, 256, [self.ik_norm, self.ik, self._weight(prefix + '.k_norm.weight'),
                                           self._weight(prefix + '.k_norm.bias'), I(1), I(self.index_dim), F(1e-6)])
        row = self._new_row(self.index_keys[layer], position, self.index_dim, self.index_key_new)
        self._copy(row, self.ik_norm, self.index_dim)
        self._rope(row, self.ik_norm, 1, self.rot, self.rot, self.rot)
        self._store_row(self.index_keys[layer], row, self.index_dim)
        self._copy(self.iq_rot, self.iq, self.index_heads * self.index_dim)
        self._rope(self.iq_rot, self.iq, self.index_heads, self.rot, self.index_dim, self.index_dim)
        self._linear(prefix + '.weights_proj', x, self.head_weights_raw)
        self._launch('scale', self._grid(self.index_heads), _ELEMENTWISE_BLOCK,
                     [self.head_weights, self.head_weights_raw, F(_f32(self.index_heads ** -0.5, 'scale')), I(self.index_heads)])
        scale = F(_f32(self.index_dim ** -0.5, 'scale'))
        if self.graph:
            self._launch('index_scores_at', (self.max_tokens + 3) // 4, 128,
                         [self.scores, self.iq_rot, self.index_keys[layer], self.head_weights, self.step,
                          I(self.index_heads), I(self.index_dim), scale])
        else:
            self._launch('index_scores', (tokens + 3) // 4, 128,
                         [self.scores, self.iq_rot, self.index_keys[layer], self.head_weights, I(tokens), I(self.index_heads),
                          I(self.index_dim), scale])
        self._select(tokens)

    def _select(self, tokens):
        """selected[:min(index_topk, tokens)] = the indexer's top-k of scores[:tokens], stable by position."""
        count = min(self.index_topk, tokens)
        sort = self.device_topk and self._topk_sort()
        if self.graph and sort:
            # Both captured: topk_sort_at selects up to TOPK_SORT_TOKENS, the chunk sorts and merges (or
            # the radix select) past it (#32).
            self._spawn('topk_sort_at', out_ids=self.selected, scores=self.scores.view(self.scores.shape, 'int32'),
                        step=self.step, k=self.index_topk)
            if self._topk_merge():
                self._topk_chunks_and_merge(tokens)
            else:
                self._launch(self._topk_kernel('topk_stable_long_at'), 1, 256,
                             [self.selected, self.scores, self.topk_scratch, self.step, I(self.index_topk),
                              I(TOPK_SORT_TOKENS)])
        elif self.graph:
            self._launch(self._topk_kernel(), 1, 256, [self.selected, self.scores, self.topk_scratch, self.step,
                                                 I(self.index_topk)])
        elif sort and tokens <= TOPK_SORT_TOKENS:
            self._spawn('topk_sort', out_ids=self.selected, scores=self.scores.view((tokens,), 'int32'), tokens=tokens,
                        k=self.index_topk)
        elif sort and self._topk_merge():
            self._topk_chunks_and_merge(tokens)
        elif self.device_topk:
            # Stable by position, exactly as GlmMoeDsaModel._index (TOPK.md); writes ids[:count].
            self._launch(self._topk_kernel(), 1, 256, [self.selected, self.scores.view((tokens,)), self.topk_scratch,
                                                 I(tokens), I(self.index_topk)])
        else:
            scores = np.frombuffer(self.scores.view((tokens,)).download(), dtype=np.float32)
            chosen = np.argsort(-scores, kind='stable')[:count].astype(np.int32)
            self.selected.view((count,)).upload(chosen)
        self.selected_count = count

    def _kv_nope_t(self, prefix):
        kv_name = prefix + '.kv_b_proj.weight'
        kv_t = self._weights.get(kv_name + ':nope_t')
        if kv_t is None:
            kv = np.asarray(self.model._weight(kv_name), dtype=np.float32).reshape(self.heads, self.nope + self.value_dim, self.rank)
            kv_t = self._upload_weight(kv_name + ':nope_t', np.ascontiguousarray(kv[:, :self.nope, :].transpose(0, 2, 1)))
        return kv_t

    def _attention(self, layer, x, position, shared_ok):
        m, c = self.model, self.config
        prefix = f'model.layers.{layer}.self_attn'
        bias = bool(c.get('attention_bias', False))
        qk = self.nope + self.rot
        self._linear(prefix + '.q_a_proj', x, self.q_a, bias)
        self._rmsnorm(self.q_a, self._weight(prefix + '.q_a_layernorm.weight'), self.q_res, 1e-6)
        self._linear(prefix + '.q_b_proj', self.q_res, self.query)
        self._linear(prefix + '.kv_a_proj_with_mqa', x, self.compressed, bias)
        latent_row = self._new_row(self.latents[layer], position, self.rank, self.latent_new)
        self._rmsnorm(self.compressed.view((self.rank,)), self._weight(prefix + '.kv_a_layernorm.weight'), latent_row, 1e-6)
        self._store_row(self.latents[layer], latent_row, self.rank)
        rope_row = self._new_row(self.rope_keys[layer], position, self.rot, self.rope_new)
        self._rope(rope_row, self.compressed.view((self.rot,), offset=self.rank * 4), 1, self.rot, self.rot, self.rot)
        self._store_row(self.rope_keys[layer], rope_row, self.rot)
        if m.indexer_types[layer] == 'full':
            self._index(layer, prefix + '.indexer', x, position)
        elif not shared_ok:
            raise ValueError('Shared DSA layer has no preceding full indexer selection')

        # Absorbed MLA. q_latent[h] = W_k[h]^T q_nope[h]; the transposed nope blocks are built once.
        kv_t = self._kv_nope_t(prefix)
        kv_b = self._weight(prefix + '.kv_b_proj.weight')
        warps = 4
        self._launch('matvec_heads', (self.heads * self.rank + warps - 1) // warps, 128,
                     [self.latent_q, kv_t, self.query, I(self.heads), I(self.rank), I(self.nope),
                      I(self.rank * self.nope), I(qk), I(self.rank)])
        self._rope(self.rot_q, self.query.view((self.heads * qk - self.nope,), offset=self.nope * 4),
                   self.heads, self.rot, qk, self.rot)
        count = self.selected_count
        splits = min(self.splits_max, max(1, (count + 255) // 256))
        block = ((self.rank + 31) // 32) * 32
        scale = F(_f32(qk ** -0.5, 'MLA scale', True))
        tokens = position + 1
        if self.fused_attention and self.graph and self.mla_kernel == 'heads':
            # The per-head buffers viewed per group of four heads (mla_heads_at.vx).
            groups = self.heads // 4
            self._spawn('mla_heads_at', out=self.attended.view((groups, 4 * self.rank)),
                        numerator=self.numerator.view((groups, self.splits_max, 4 * self.rank)),
                        stats=self.stats.view((groups, self.splits_max, 8)), q_latent=self.latent_q, q_rope=self.rot_q,
                        cache_latent=self.latents[layer], cache_rope=self.rope_keys[layer], selected=self.selected,
                        step=self.step, heads=self.heads, rank=self.rank, rope_dim=self.rot, selection_stride=0,
                        splits_max=self.splits_max, scale=scale.value, threads=block,
                        arrivals=self.arrivals.view((groups,)))
        elif self.fused_attention and self.graph:
            self._spawn('mla_chunks_at' if self.mla_kernel == 'chunks' else 'mla_decode_at', out=self.attended, numerator=self.numerator, stats=self.stats,
                        q_latent=self.latent_q, q_rope=self.rot_q, cache_latent=self.latents[layer],
                        cache_rope=self.rope_keys[layer], selected=self.selected, step=self.step, heads=self.heads,
                        rank=self.rank, rope_dim=self.rot, selection_stride=0, splits_max=self.splits_max,
                        scale=scale.value, threads=block, arrivals=self.arrivals)
        elif self.fused_attention:
            self._spawn('mla_decode', out=self.attended, numerator=self.numerator, stats=self.stats,
                        q_latent=self.latent_q, q_rope=self.rot_q, cache_latent=self.latents[layer],
                        cache_rope=self.rope_keys[layer], selected=self.selected, heads=self.heads, rank=self.rank,
                        rope_dim=self.rot, tokens=tokens, count=count, selection_stride=0, splits=splits,
                        scale=scale.value, threads=block, arrivals=self.arrivals)
        elif self.graph:
            self._launch('mla_partial_at', self.heads * self.splits_max, block,
                         [self.numerator, self.stats, self.latent_q, self.rot_q, self.latents[layer], self.rope_keys[layer],
                          self.selected, self.step, I(self.heads), I(self.rank), I(self.rot), I(0), scale])
            self._launch('mla_merge_at', self.heads, 256,
                         [self.attended, self.numerator, self.stats, self.step, I(self.heads), I(self.rank)])
        else:
            self._launch('mla_partial', self.heads * splits, block,
                         [self.numerator, self.stats, self.latent_q, self.rot_q, self.latents[layer], self.rope_keys[layer],
                          self.selected, I(self.heads), I(self.rank), I(self.rot), I(tokens), I(count), I(0), I(splits), scale])
            self._launch('mla_merge', self.heads, 256,
                         [self.attended, self.numerator, self.stats, I(self.heads), I(self.rank), I(splits)])
        value_w = kv_b.view((self.heads * (self.nope + self.value_dim) - self.nope, self.rank), offset=self.nope * self.rank * 4)
        self._launch('matvec_heads', (self.heads * self.value_dim + warps - 1) // warps, 128,
                     [self.heads_out, value_w, self.attended, I(self.heads), I(self.value_dim), I(self.rank),
                      I((self.nope + self.value_dim) * self.rank), I(self.rank), I(self.value_dim)])
        self._linear(prefix + '.o_proj', self.heads_out, self.attn_out, bias)

    def _feed_forward(self, layer, x):
        m, c = self.model, self.config
        prefix = f'model.layers.{layer}.mlp'
        if m.mlp_types[layer] == 'dense':
            self._mlp(prefix, x, self.ffn_out)
            return
        experts = int(c['n_routed_experts'])
        k = int(c['num_experts_per_tok'])
        self._project(prefix + '.gate', x, self.router_logits)
        self._launch('router', 1, 32,
                     [self.route_ids, self.route_weights, self.router_logits,
                      self._weight(prefix + '.gate.e_score_correction_bias'), I(experts), I(k),
                      F(_f32(float(c.get('routed_scaling_factor', 1.0)), 'router scale'))])
        # Reduction order of the pinned official Experts: ascending expert id, whatever the top-k order.
        if self.device_experts:
            self._device_moe(prefix, x, k)
        else:
            self._copy(self.moe_sum, self.zeros, self.hidden)
            self._host_experts(prefix, x, k)
        if int(c.get('n_shared_experts', 1)) > 0:
            self._mlp(prefix + '.shared_experts', x, self.mlp_out)
            self._add(self.moe_sum, self.moe_sum, self.mlp_out, self.hidden)
        self._copy(self.ffn_out, self.moe_sum, self.hidden)

    def _host_experts(self, prefix, x, k):
        """The routed experts with the ids read back to the host to pick each expert's weights."""
        ids = np.frombuffer(self.route_ids.download(), dtype=np.int32)
        for slot in np.argsort(ids, kind='stable'):
            self._expert(prefix, int(ids[slot]), x, self.expert_out)
            self._launch('scaled_add', self._grid(self.hidden), _ELEMENTWISE_BLOCK,
                         [self.moe_sum, self.expert_out, self.route_weights, I(int(slot)), I(self.hidden)])

    # ---------------------------------------------------------------- public
    def forward(self, token_id, *, logits=True):
        """Append one token and return next-token logits (None with logits=False)."""
        m, c = self.model, self.config
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        if not isinstance(token_id, (int, np.integer)) or not 0 <= int(token_id) < int(c['vocab_size']):
            raise ValueError('token_id must be an integer in the vocabulary')
        position = self.position
        if position >= self.max_tokens:
            raise ValueError(f'sequence exceeds max_tokens={self.max_tokens}')
        if position >= int(c.get('max_position_embeddings', 2 ** 31 - 1)):
            raise ValueError('Request exceeds max_position_embeddings')
        embedding = np.ascontiguousarray(m._embedding(int(token_id)), dtype=np.float32)
        cosine, sine = _cached_rope_coefficients(self.rot, float(position), float(m.theta))
        cosine, sine = (np.ascontiguousarray(v, dtype=np.float32) for v in (cosine, sine))
        if self.graph:
            tokens = position + 1
            count = min(self.index_topk, tokens)
            splits = min(self.splits_max, max(1, (count + 255) // 256))
            step = np.array([position, tokens, count, splits], np.int32).view(np.float32)
            self.step_inputs.upload(np.concatenate([embedding, cosine, sine, step]))
            if self._step_graph is not None:
                self._step_graph.launch(self.stream)
                self.replays += 1
                self.selected_count = count
            elif not self._warm:
                self._layers(position)  # uploads every weight and loads every kernel the capture needs
                self._warm = True
            else:
                with self.stream.capture() as graph:
                    self._layers(position)
                self._step_graph = graph
                graph.launch(self.stream)
                self.replays += 1
        else:
            self.x.upload(embedding)
            self.cos.upload(cosine)
            self.sin.upload(sine)
            self._layers(position)
        self.position += 1
        if not logits:
            return None
        self._rmsnorm(self.x, self._weight('model.norm.weight'), self.norm, m.eps)
        head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
        self._project(head, self.norm, self.logits)
        return np.frombuffer(self.logits.download(), dtype=np.float32).copy()

    def _layers(self, position):
        """Every layer of one decode step, from self.x (the embedding row) to the final residual
        in self.x; with graph=True, what the step graph holds."""
        m = self.model
        have_selection = False
        for layer in range(m.n_layers):
            prefix = f'model.layers.{layer}'
            self._rmsnorm(self.x, self._weight(prefix + '.input_layernorm.weight'), self.norm, m.eps)
            self._attention(layer, self.norm, position, have_selection)
            have_selection = True
            self._add(self.x_next, self.x, self.attn_out, self.hidden)
            self.x, self.x_next = self.x_next, self.x
            self._rmsnorm(self.x, self._weight(prefix + '.post_attention_layernorm.weight'), self.norm, m.eps)
            self._feed_forward(layer, self.norm)
            self._add(self.x_next, self.x, self.ffn_out, self.hidden)
            self.x, self.x_next = self.x_next, self.x

    def preload(self, reserve_bytes=256 * 1024 ** 2):
        """Upload every weight decode can use (all layers, every routed expert, the final norm and
        lm_head) and load the kernel modules, so no request pays a first-use upload. Returns the
        bytes this call uploaded.

        With a context that reports free memory (CUDAContext.memory_info), the upload stops before
        free memory minus ``reserve_bytes`` runs out: the remaining weights are only sized, and a
        MemoryError reports the bytes needed against the bytes free. The object then stays usable
        with the weights that did fit (the rest upload lazily as before)."""
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        info = getattr(self.ctx, 'memory_info', None)
        free = info()[0] if callable(info) else None
        self._room = None if free is None else max(0, free - int(reserve_bytes))
        self._plan, before = None, self.weight_bytes
        try:
            self._preload()
            planned = self._plan
        finally:
            self._room = self._plan = None
        if planned is not None:
            self._weights = {k: v for k, v in self._weights.items() if not _planned(v)}
            needed = self.weight_bytes - before + planned
            raise MemoryError(f'ResidentGLM preload needs {needed} bytes ({needed / 1024 ** 3:.2f} GiB) of weights; '
                              f'the device has {free} bytes ({free / 1024 ** 3:.2f} GiB) free, keeping {reserve_bytes} in reserve')
        return self.weight_bytes - before

    def _preload(self):
        m, c = self.model, self.config
        bias = bool(c.get('attention_bias', False))

        def linear(name, with_bias=False):
            self._out_features(name)  # uploads in the Linear's own format
            if self._format(name) != 'float32':
                self._format_kernel(self._format(name))
            if with_bias:
                self._weight(name + '.bias')

        def mlp(prefix):
            for n in ('gate_proj', 'up_proj', 'down_proj'):
                linear(f'{prefix}.{n}')

        for layer in range(m.n_layers):
            p = f'model.layers.{layer}'
            self._weight(p + '.input_layernorm.weight')
            self._weight(p + '.post_attention_layernorm.weight')
            a = p + '.self_attn'
            for n in ('q_a_proj', 'kv_a_proj_with_mqa', 'o_proj'):
                linear(f'{a}.{n}', bias)
            linear(a + '.q_b_proj')
            self._weight(a + '.q_a_layernorm.weight')
            self._weight(a + '.kv_a_layernorm.weight')
            self._kv_nope_t(a)
            self._weight(a + '.kv_b_proj.weight')
            if m.indexer_types[layer] == 'full':
                for n in ('wq_b', 'wk', 'weights_proj'):
                    linear(f'{a}.indexer.{n}')
                self._weight(a + '.indexer.k_norm.weight')
                self._weight(a + '.indexer.k_norm.bias')
            f = p + '.mlp'
            if m.mlp_types[layer] == 'dense':
                mlp(f)
                continue
            linear(f + '.gate')
            self._weight(f + '.gate.e_score_correction_bias')
            if self.device_experts:
                fused = self._fused_experts(f + '.experts.0')
                for n in (('gate_up', 'down') if fused else ('gate_proj', 'up_proj', 'down_proj')):
                    self._stacked_experts(f, n)
            for e in range(0 if self.device_experts else int(c['n_routed_experts'])):
                if self._fused_experts(f'{f}.experts.{e}'):
                    self._weight(f + '.experts.gate_up_proj')
                    self._weight(f + '.experts.down_proj')
                else:
                    mlp(f'{f}.experts.{e}')
            if int(c.get('n_shared_experts', 1)) > 0:
                mlp(f + '.shared_experts')
        self._weight('model.norm.weight')
        linear('model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head')
        if self.device_topk:
            self._topk_kernel()

    def prefill(self, token_ids):
        """Token-by-token prefill; returns the stacked logits of every position."""
        return np.stack([self.forward(t) for t in token_ids]) if len(token_ids) else np.empty((0, int(self.config['vocab_size'])), np.float32)


class ResidentSequenceModel:
    """The scheduler's model interface (new_cache / prefill_token / forward) over one ResidentGLM.

    ResidentGLM holds a single sequence's device caches, so this serves one request at a time
    (Scheduler max_sequences=1) until batched decode (#8). ``new_cache`` resets the position for
    the next request; device buffers are reused, never reallocated. The returned handle only
    guards against a stale request touching the caches after another one took them over.
    ``config['max_position_embeddings']`` is clamped to ``max_tokens`` so the scheduler rejects
    requests that do not fit the caches. Prompt tokens before the last skip lm_head.
    """

    def __init__(self, model, resident):
        self.model, self.resident, self.backend = model, resident, resident
        self.config = dict(model.config)
        limit = int(self.config.get('max_position_embeddings', resident.max_tokens))
        self.config['max_position_embeddings'] = min(limit, int(resident.max_tokens))
        self._current = None

    def new_cache(self):
        self.resident.reset()
        self._current = object()
        return self._current

    def _check(self, cache):
        if cache is None or cache is not self._current:
            raise RuntimeError('ResidentGLM holds one sequence and this request no longer owns it')

    def prefill_token(self, token_id, cache):
        self._check(cache)
        self.resident.forward(token_id, logits=False)

    def forward(self, token_id, cache):
        self._check(cache)
        # Logits come back so the scheduler samples on the host; device-side sampling is #17.
        return self.resident.forward(token_id)

    def close(self):
        self.resident.close()
