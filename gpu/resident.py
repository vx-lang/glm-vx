"""Device-resident GLM decode: weights, activations and the KV cache stay on the GPU.

`GPUBackend` (gpu/backend.py) runs one primitive per call and reads every result back to the
host, so a decode step is dominated by transfers. `ResidentGLM` runs a whole decode step through
the same Vx kernels with activations and the per-layer MLA / indexer caches in device memory. The
only host traffic per token is:

  * the embedding row up, and the RoPE cos/sin for the position up;
  * per full-indexer layer, the indexer scores down and the chosen token ids up (top-k is still
    a stable host argsort, the same tie rule as the model; a device top-k is a known gap);
  * per MoE layer, the k routed expert ids down (to pick which expert weights to launch);
  * the logits down.

Numerics follow `GlmMoeDsaModel` operation for operation (same kernels, same reduction orders,
same expert aggregation order), so results should agree with the hybrid path to float32 rounding;
`validation/resident_compare.py` checks that on a GPU. Weights are float32, uploaded once and kept.
Expanded weights only: packed (GGUF) weights, expert placement and tracing are not supported.

Not a performance claim: no timing is made here, and launches are still one per op.
"""
import ctypes as C
from pathlib import Path

import numpy as np

from kernels.backend import _cached_rope_coefficients
from .runtime import CUDAContext

I = C.c_int32
F = C.c_float
_ELEMENTWISE_BLOCK = 256
_MAX_GRID = 65535


def _f32(value, name, positive=False):
    value = C.c_float(float(value)).value
    if not np.isfinite(value) or (value <= 0 if positive else False):
        raise ValueError(name + ' must be finite' + (' and positive' if positive else '') + ' in float32')
    return value


class ResidentGLM:
    name = 'vx-gpu-resident-f32'

    def __init__(self, model, *, max_tokens=4096, ptx_dir=None, device=0, ctx=None, max_weight_bytes=None):
        if getattr(model, 'packed_weights', False):
            raise NotImplementedError('ResidentGLM supports expanded float32 weights only')
        if getattr(model, 'expert_placement', None) is not None:
            raise NotImplementedError('ResidentGLM does not support expert placement')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        self.model, self.config = model, model.config
        self.max_tokens = max_tokens
        self.max_weight_bytes = max_weight_bytes
        self.weight_bytes = 0
        self._owns_context = ctx is None
        self.ctx = ctx if ctx is not None else CUDAContext(device, minimum_compute_capability=(8, 0))
        self._closed = False
        try:
            ptx = Path(ptx_dir or Path(__file__).parent / 'build') / 'kernels.ptx'
            self.module = self.ctx.load_ptx(ptx)
            self.stream = self.ctx.stream()
            self._kernels = {}
            self._weights = {}
            self._allocate()
            self.position = 0
        except BaseException:
            self.close()
            raise

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

    def _grid(self, n):
        return max(1, min((n + _ELEMENTWISE_BLOCK - 1) // _ELEMENTWISE_BLOCK, _MAX_GRID))

    def _weight(self, name):
        """Upload a model tensor once; the device copy is kept for the life of this object."""
        tensor = self._weights.get(name)
        if tensor is None:
            tensor = self._upload_weight(name, np.ascontiguousarray(self.model._weight(name), dtype=np.float32))
        return tensor

    def _upload_weight(self, key, array):
        if self.max_weight_bytes is not None and self.weight_bytes + array.nbytes > self.max_weight_bytes:
            raise MemoryError(f'device weights would exceed max_weight_bytes={self.max_weight_bytes} at {key!r}')
        tensor = self._tensor(array.shape, 'float32', array.tobytes())
        self.weight_bytes += array.nbytes
        self._weights[key] = tensor
        return tensor

    def _allocate(self):
        m, c = self.model, self.config
        cap = self.max_tokens
        self.hidden, self.heads = m.hidden, m.heads
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
        self.x, self.x_next, self.norm = t((self.hidden,)), t((self.hidden,)), t((self.hidden,))
        self.attn_out, self.ffn_out = t((self.hidden,)), t((self.hidden,))
        self.q_a, self.q_res = t((q_rank,)), t((q_rank,))
        self.query = t((self.heads * qk,))
        self.compressed = t((self.rank + self.rot,))
        self.latent_q, self.rot_q = t((self.heads, self.rank)), t((self.heads, self.rot))
        self.splits_max = 16
        self.numerator = t((self.heads, self.splits_max, self.rank))
        self.stats = t((self.heads, self.splits_max, 2))
        self.attended = t((self.heads, self.rank))
        self.heads_out = t((self.heads * self.value_dim,))
        self.iq, self.iq_rot = t((self.index_heads * self.index_dim,)), t((self.index_heads * self.index_dim,))
        self.ik, self.ik_norm = t((self.index_dim,)), t((self.index_dim,))
        self.head_weights_raw, self.head_weights = t((self.index_heads,)), t((self.index_heads,))
        self.scores = t((cap,))
        self.selected = t((self.index_topk,), 'int32')
        self.selected_count = 0
        self.gate, self.up, self.act = t((width,)), t((width,)), t((width,))
        self.fused = t((2 * int(c.get('moe_intermediate_size', 1)),))
        self.mlp_out, self.expert_out = t((self.hidden,)), t((self.hidden,))
        self.moe_sum = t((self.hidden,))
        self.zeros = t((self.hidden,))
        self.cos, self.sin = t((self.rot // 2,)), t((self.rot // 2,))
        if 'sparse' in m.mlp_types:
            experts = int(c['n_routed_experts'])
            k = int(c['num_experts_per_tok'])
            self.router_logits = t((experts,))
            self.route_ids, self.route_weights = t((k,), 'int32'), t((k,))
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

    def _linear(self, name, x, out, bias=False):
        self._matvec(self._weight(name + '.weight'), x, out)
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

    def _mlp(self, prefix, x, out):
        """down(SwiGLU(gate(x), up(x))) with both projections and the activation resident."""
        gate_w = self._weight(prefix + '.gate_proj.weight')
        width = gate_w.shape[0]
        gate, up, act = (t.view((width,)) for t in (self.gate, self.up, self.act))
        self._matvec(gate_w, x, gate)
        self._matvec(self._weight(prefix + '.up_proj.weight'), x, up)
        self._swiglu(gate, up, act, width)
        self._matvec(self._weight(prefix + '.down_proj.weight'), act, out)

    def _expert(self, prefix, expert, x, out):
        base = f'{prefix}.experts.{expert}'
        try:
            self.model._weight(base + '.gate_proj.weight')
        except KeyError:
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

    # ---------------------------------------------------------------- blocks
    def _index(self, layer, prefix, x, position):
        """DSA indexer for one layer: refresh its keys, score all causal keys, choose the top-k."""
        m = self.model
        tokens = position + 1
        self._linear(prefix + '.wq_b', self.q_res, self.iq)
        self._linear(prefix + '.wk', x, self.ik)
        self._launch('layernorm', 1, 256, [self.ik_norm, self.ik, self._weight(prefix + '.k_norm.weight'),
                                           self._weight(prefix + '.k_norm.bias'), I(1), I(self.index_dim), F(1e-6)])
        row = self._row(self.index_keys[layer], position, self.index_dim)
        self._copy(row, self.ik_norm, self.index_dim)
        self._rope(row, self.ik_norm, 1, self.rot, self.rot, self.rot)
        self._copy(self.iq_rot, self.iq, self.index_heads * self.index_dim)
        self._rope(self.iq_rot, self.iq, self.index_heads, self.rot, self.index_dim, self.index_dim)
        self._linear(prefix + '.weights_proj', x, self.head_weights_raw)
        self._launch('scale', self._grid(self.index_heads), _ELEMENTWISE_BLOCK,
                     [self.head_weights, self.head_weights_raw, F(_f32(self.index_heads ** -0.5, 'scale')), I(self.index_heads)])
        self._launch('index_scores', (tokens + 3) // 4, 128,
                     [self.scores, self.iq_rot, self.index_keys[layer], self.head_weights, I(tokens), I(self.index_heads),
                      I(self.index_dim), F(_f32(self.index_dim ** -0.5, 'scale'))])
        scores = np.frombuffer(self.scores.view((tokens,)).download(), dtype=np.float32)
        count = min(self.index_topk, tokens)
        # Stable by position, exactly as GlmMoeDsaModel._index without a device top-k.
        chosen = np.argsort(-scores, kind='stable')[:count].astype(np.int32)
        self.selected.view((count,)).upload(chosen)
        self.selected_count = count

    def _attention(self, layer, x, position, shared_ok):
        m, c = self.model, self.config
        prefix = f'model.layers.{layer}.self_attn'
        bias = bool(c.get('attention_bias', False))
        qk = self.nope + self.rot
        self._linear(prefix + '.q_a_proj', x, self.q_a, bias)
        self._rmsnorm(self.q_a, self._weight(prefix + '.q_a_layernorm.weight'), self.q_res, 1e-6)
        self._linear(prefix + '.q_b_proj', self.q_res, self.query)
        self._linear(prefix + '.kv_a_proj_with_mqa', x, self.compressed, bias)
        latent_row = self._row(self.latents[layer], position, self.rank)
        self._rmsnorm(self.compressed.view((self.rank,)), self._weight(prefix + '.kv_a_layernorm.weight'), latent_row, 1e-6)
        rope_row = self._row(self.rope_keys[layer], position, self.rot)
        self._rope(rope_row, self.compressed.view((self.rot,), offset=self.rank * 4), 1, self.rot, self.rot, self.rot)
        if m.indexer_types[layer] == 'full':
            self._index(layer, prefix + '.indexer', x, position)
        elif not shared_ok:
            raise ValueError('Shared DSA layer has no preceding full indexer selection')

        # Absorbed MLA. q_latent[h] = W_k[h]^T q_nope[h]; the transposed nope blocks are built once.
        kv_name = prefix + '.kv_b_proj.weight'
        kv_t_key = kv_name + ':nope_t'
        kv_t = self._weights.get(kv_t_key)
        if kv_t is None:
            kv = np.asarray(m._weight(kv_name), dtype=np.float32).reshape(self.heads, self.nope + self.value_dim, self.rank)
            kv_t = self._upload_weight(kv_t_key, np.ascontiguousarray(kv[:, :self.nope, :].transpose(0, 2, 1)))
        kv_b = self._weight(kv_name)
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
        self._matvec(self._weight(prefix + '.gate.weight'), x, self.router_logits)
        self._launch('router', 1, 32,
                     [self.route_ids, self.route_weights, self.router_logits,
                      self._weight(prefix + '.gate.e_score_correction_bias'), I(experts), I(k),
                      F(_f32(float(c.get('routed_scaling_factor', 1.0)), 'router scale'))])
        ids = np.frombuffer(self.route_ids.download(), dtype=np.int32)
        # Reduction order of the pinned official Experts: ascending expert id, whatever the top-k order.
        self._copy(self.moe_sum, self.zeros, self.hidden)
        for slot in np.argsort(ids, kind='stable'):
            self._expert(prefix, int(ids[slot]), x, self.expert_out)
            self._launch('scaled_add', self._grid(self.hidden), _ELEMENTWISE_BLOCK,
                         [self.moe_sum, self.expert_out, self.route_weights, I(int(slot)), I(self.hidden)])
        if int(c.get('n_shared_experts', 1)) > 0:
            self._mlp(prefix + '.shared_experts', x, self.mlp_out)
            self._add(self.moe_sum, self.moe_sum, self.mlp_out, self.hidden)
        self._copy(self.ffn_out, self.moe_sum, self.hidden)

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
        self.x.upload(np.ascontiguousarray(m._embedding(int(token_id)), dtype=np.float32))
        cosine, sine = _cached_rope_coefficients(self.rot, float(position), float(m.theta))
        self.cos.upload(np.ascontiguousarray(cosine, dtype=np.float32))
        self.sin.upload(np.ascontiguousarray(sine, dtype=np.float32))
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
        self.position += 1
        if not logits:
            return None
        self._rmsnorm(self.x, self._weight('model.norm.weight'), self.norm, m.eps)
        head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
        self._matvec(self._weight(head + '.weight'), self.norm, self.logits)
        return np.frombuffer(self.logits.download(), dtype=np.float32).copy()

    def prefill(self, token_ids):
        """Token-by-token prefill; returns the stacked logits of every position."""
        return np.stack([self.forward(t) for t in token_ids]) if len(token_ids) else np.empty((0, int(self.config['vocab_size'])), np.float32)
