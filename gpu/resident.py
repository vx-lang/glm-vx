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
``kv_b_proj`` is dequantized to float32 as for INT4, except with ``fp8_kernels='cvt'``: then both MLA
absorbs read its FP8 codes as stored (#42): the query one through fp8_matvec_cols_cvt (each head's nope
rows summed down the latent columns), the value one through fp8_matvec_heads_cvt (each head's value rows).

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

``fuse_router=True`` runs a MoE layer's BF16 router projection and the router in one launch, gpu/spawn's
router_bf16: a block a row summed in the hand bf16_matvec's order, then the selection, in one pass of
ranks, in the last block to finish. As with fused_attention, its sigmoid's exp is libdevice's expf, not
the hand router's ex2.approx, so on the GPU the routing weights, and the logits, differ from the default
in the last bits.

``fold_shared=True`` (with ``device_experts`` and ``fp8_kernels='cvt'``) runs a MoE layer's shared expert
inside the routed experts' two launches, as expert ``n_routed_experts`` with weight 1.0 in route slot k:
the same bits as its own three launches.

``moe_down2=True`` (with ``fp8_kernels='cvt'``) runs the routed experts' down projection through
fp8_moe_cvt_down2, which spreads the short rows of a tensor-parallel slice over more subgroups: the same
bits.

``moe_mma=True`` (or ``'f16'``, with ``fp8_kernels='cvt'``) runs it on the tensor cores instead
(gpu/spawn/moe_mma/fp8_moe_mma16_down_at, sm_89+): the FP8 weights unpacked to f16 in registers, the experts'
inputs in f16 (a power-of-two scale a group of 128), mma.sync.m16n8k16 with f32 sums; within 2e-4 of the CUDA
cores' result. ``moe_mma='e4m3'`` quantizes the inputs to E4M3 instead, as SGLang does before its FP8 GEMMs
(fp8_moe_mma_down_at, mma.sync e4m3: on sm_90 ptxas makes it the same f16 HMMA, so it is slower and 2e-2 off).
Not the CUDA cores' bits; where
k > 9, the expert width is not a multiple of 256, the hidden size not of 16 or the GPU is older than sm_89, the
down projection stays down2's or down's.

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
forward_many and mtp_step (#16) run as graphs of their own too: forward_many one for each token
count, with each token's cos, sin and step buffer uploaded beside the embedding rows; mtp_step one
with the indexer and one reusing the last selection. Each runs eagerly the first time and is captured
the second, and gives what the eager kernels give bit for bit.

``mla_kernel='chunks'`` runs the fused attention under graph replay through gpu/spawn's
mla_chunks_at (#33): 32 tokens a chunk, equal to mla_decode_at to rounding. ``mla_kernel='heads'``
runs mla_heads_at instead: four heads a block, so each selected latent row is read once for four
heads; each head's arithmetic is mla_chunks_at's, so the logits equal 'chunks' bit for bit. It needs
a head count that is a multiple of 4. ``mla_kernel='flash'`` runs mla_flash_at then
mla_flash_merge_at: flash decoding with each lane's query and sums in registers for the whole loop
over tokens (Vx #3475), up to 64 splits of about 8 tokens. Its scores are mla_chunks_at's bit for bit
and its softmax folds one token at a time, so the logits are close to 'chunks', not equal. It needs
GLM-5.3's MLA shape: heads a multiple of 16, rank 512 and rope 64. ``mla_kernel='tc'`` keeps the
latent and rope caches in bf16 (store_row_bf16_at) and runs mla_q_bf16_at, mla_tc_at and
mla_flash_merge_at: the scores and the weighted sum of latent rows on the tensor cores, bf16 products
with f32 sums, as SGLang's FlashMLA. Its logits agree with the f32 kernels' to bf16 rounding. It needs
the shape 'flash' needs and index_topk <= 2048 (64 splits of 32 tokens).

``fp8_kernels='x4'`` runs FP8 Linears through gpu/spawn's fp8_matvec4 and, with ``device_experts``,
the routed experts through fp8_moe4_gate_up/down (#30, #31): sixteen codes a lane step, one 16-byte
load. Their summation order differs from the hand kernels', so the logits are close to the default
path's, not equal; each kernel's own order is pinned by gpu/test_spawn_vx_host.py. Widths that are
not a multiple of 16 keep the default kernels.

``fp8_kernels='fma'`` is 'x4' with the dense FP8 Linears through fp8_matvec_fma (input width
FMA_LONG_ROWS or more) or fp8_matvec_fma_tiles (#30): one fma a code (fp8_matvec4 spends two
multiplies and an add) and two rows a warp, so not fp8_matvec4's bits; each is within 2e-6 of the
float64 sum of the dequantized weights and float32 activations, closer than fp8_matvec4. The routed
experts take fp8_moe_fma_gate_up and fp8_moe_fma_down (#31), the same dot products for each
expert's rows; forward_many keeps the x4 kernels.

``fp8_kernels='cvt'`` is 'fma' with the codes at their true values (glm-vx #38): the dense Linears
through fp8_matvec_cvt_tiles, q_a/kv_a/wk through fp8_matvec_cvt_tiles3, gate/up and the routed
experts through fp8_moe_cvt_*. On compute capability 8.9 and up they are the sm_90 builds, which
decode two codes an instruction with Hopper's e4m3x2 -> f16x2 conversion; elsewhere (an A100, the
test context) their _ref builds, an exact integer decode with the same bits.

``fp8_kernels='cublaslt'`` runs the dense FP8 Linears whose input width is a multiple of 128 through
cuBLASLt's block-scaled FP8 matmul (gpu/cublaslt.py, #29): x quantized to e4m3 per 128 columns
(gpu/fp8_quant.vx), the checkpoint's 128x128 weight scales, float32 accumulation. That is SGLang's
w8a8 block-FP8 scheme, so the logits are near the float32-activation kernels', not equal. The routed
experts and forward_many keep the x4 kernels; other widths keep the hand kernel. A test context
supplies fp8_block_matmul(m, n, k) in place of the library.

``matvec_kernels='spawn'`` runs INT4 and BF16 Linears through gpu/spawn's int4_matvec and
bf16_matvec instead of the hand kernels. gpu/test_spawn_hardware.py checks each against its hand
kernel bit for bit, so the logits do not change; the spawn kernels share device.vx's decode macros.
``matvec_kernels='x4'`` runs them through int4_matvec4 and bf16_matvec4 instead (#30): each lane reads
four consecutive u32 words a step (32 INT4 codes or 8 bfloat16 weights), the shape Vx #2362 makes one
16-byte load. Their summation order differs from the hand kernels', so the logits are close to the
default path's, not equal; gpu/test_spawn_vx_host.py pins each order. INT4 widths that are not a
multiple of 128 and BF16 widths that are not a multiple of 8 keep the hand kernels.

``tp`` (a gpu/tp.TensorParallel, #34) makes this object one rank of a tensor-parallel job on its slice of
the model: its heads, intermediate columns and vocabulary rows (gpu/tp.py has the split). After o_proj and
after each MLP / MoE the ranks' partial sums are all-gathered and added in rank order, so every rank
keeps the same residual stream bit for bit and n GPUs give the bits of n ranks emulated on one;
``one_shot`` does that sum in one launch over CUDA IPC peer buffers (gpu/spawn/tp_all_reduce). The
logits are gathered from the ranks' vocabulary rows. forward() only so far: no forward_many or MTP.

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
# float32 matrices of at most this many rows (the router's gate, the indexer's weights_proj) run
# through gpu/matvec_rows.ptx, one block a row with its products staged in shared memory, when
# that is built: the hand matvec's bits without its 24 memory round trips a 6144-column row (#29).
MATVEC_ROWS_MAX = 1024
MATVEC_ROWS_COLS = 8192  # matvec_rows.vx's shared floats
TOPK_SORT_TOKENS = 2048  # gpu/spawn/topk_sort.vx sorts contexts up to this many tokens (#32)
# fp8_kernels='fma': the input width from which fp8_matvec_fma beats fp8_matvec_fma_tiles (H200: o_proj's
# 16384 and mlp.down's 12288; at 6144 and below, fp8_matvec_fma_tiles).
FMA_LONG_ROWS = 12288
MANY_TOKENS = 4  # forward_many's limit: fp8_matvec4_rows takes up to four vectors (#16)
PREFILL_TOKENS = 2048  # prefill_chunk's limit a call (#63, item 5b)
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
                 topk='auto', long_topk='position', mla_kernel='decode', matvec_kernels='hand', mtp=False,
                 norm_block=256, fuse_glue=True, index_kernel='mma',
                 topk_blocks_from=65536, fold_shared=False, fuse_router=False, moe_down2=False, moe_mma=False, fuse_mla_q=False,
                 device_decode=False, prefill_rows=0, prefill_gemm='rows', prefill_sum='nccl', prefill_moe='tokens', prefill_attention='tokens',
                 fuse_tp_norm=0, tp=None):
        if getattr(model, 'packed_weights', False):
            raise NotImplementedError('ResidentGLM supports expanded float32 weights only')
        if getattr(model, 'expert_placement', None) is not None:
            raise NotImplementedError('ResidentGLM does not support expert placement')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        # tp (a gpu/tp.TensorParallel, #34): this object is one rank of a tensor-parallel job and holds
        # that rank's slice of the model (gpu/tp.ShardedModel): its heads, its intermediate columns and
        # its vocabulary rows. After o_proj and after the MLP / MoE every rank all-gathers the partial
        # sums and adds them in rank order (_tp_sum), so the ranks' residual streams stay identical bit
        # for bit; the logits are gathered from the ranks' vocabulary rows. A size of 1 is no split.
        self.tp = tp if tp is not None and tp.size > 1 else None
        if self.tp is not None:
            if mtp:
                raise NotImplementedError('tensor parallelism does not run the MTP layer yet')
            from .tp import ShardedModel
            model = ShardedModel(model, self.tp.rank, self.tp.size)
        self.model, self.config = model, model.config
        self.max_tokens = max_tokens
        # mtp: also the checkpoint's MTP layer (model.layers.{num_hidden_layers}, #16), for mtp_step.
        self.mtp = bool(mtp)
        if self.mtp and model.mtp_layers < 1:
            raise ValueError('mtp=True needs a config with num_nextn_predict_layers >= 1')
        self.device_topk = bool(device_topk)
        self.device_experts = bool(device_experts)
        self.fused_attention = bool(fused_attention)
        self.graph = bool(graph)
        if fp8_kernels not in ('hand', 'x4', 'fma', 'cvt', 'cublaslt'):
            raise ValueError("fp8_kernels must be 'hand', 'x4', 'fma', 'cvt' or 'cublaslt'")
        self.fp8_kernels = fp8_kernels
        if matvec_kernels not in ('hand', 'spawn', 'x4'):
            raise ValueError("matvec_kernels must be 'hand', 'spawn' or 'x4'")
        self.matvec_kernels = matvec_kernels
        # Threads of each RMSNorm block. 256 is the hybrid backend's, so the float32 path matches it bit
        # for bit; 1024 reads a 6144-wide row in one batch of loads where 256 takes three (a thread's
        # sum of squares then covers other elements, so other bits).
        if norm_block not in (256, 512, 1024):
            raise ValueError('norm_block must be 256, 512 or 1024')
        self.norm_block = norm_block
        # fuse_glue (#39): the step's small launches between the big ones fused, each with the bits of
        # the launches it replaces, so the logits are the same either way (an A/B of the time only):
        #   - the post-attention residual_rmsnorm as residual_rmsnorm8 (its row in registers), and
        #     each layer's closing add with the next layer's input RMSNorm as one residual_rmsnorm8,
        #     while the hidden width fits eight values a thread of norm_block; a MoE layer's output
        #     is read from moe_sum, not copied to ffn_out first;
        #   - the q_a and kv_a RMSNorms as one rmsnorm_pair launch of two blocks;
        #   - under graph replay, the indexer's LayerNorm, copy, RoPE and store_row as one
        #     index_key_at, and with mla_kernel='tc' the bf16 latent and rope rows (store, RoPE,
        #     store) as one kv_rows_bf16_at.
        # On by default since the H200 A/B (evidence/h200-20261116-r: about 13 us a layer); False keeps
        # the separate launches, for an A/B or to check a fusion against them.
        self.fuse_glue = bool(fuse_glue)
        # fuse_mla_q: the query's rope (rope_strided) inside the bf16 query kernel, gpu/spawn/mla_q_rope_bf16_at:
        # the same bits for mla_tc_at, one launch fewer (#63, item 7). Taken where mla_kernel='tc' runs fused in a
        # graph with rank 512 and rope 64.
        self.fuse_mla_q = bool(fuse_mla_q)
        # prefill_rows (#63, item 5b): the most tokens prefill_chunk takes in one call, and with N > 0 the
        # server's prompts go through it (ResidentSequenceModel.prefill_tokens); 0 keeps them a token at a
        # time and prefill_chunk at PREFILL_TOKENS.
        if type(prefill_rows) is not int or prefill_rows < 0:
            raise ValueError('prefill_rows must be a non-negative integer')
        self.prefill_rows = prefill_rows
        # prefill_chunk (#63, item 5b): its Linears over the chunk's rows through cuBLASLt's block-FP8 matmul
        # ('cublaslt': each FP8 Linear whose input width is a multiple of 128) or each row's _linear ('rows',
        # forward()'s kernels); under tensor parallelism its sums as one NCCL all-reduce over the rows
        # ('nccl') or each row's _tp_sum ('rows', forward()'s bits).
        if prefill_gemm not in ('rows', 'cublaslt'):
            raise ValueError("prefill_gemm must be 'rows' or 'cublaslt'")
        if prefill_sum not in ('nccl', 'rows'):
            raise ValueError("prefill_sum must be 'nccl' or 'rows'")
        self.prefill_gemm, self.prefill_sum = prefill_gemm, prefill_sum
        # prefill_moe: prefill_chunk's routed experts a token at a time with forward()'s kernels ('tokens'), or
        # all the chunk's (token, expert) pairs grouped by expert through one block-FP8 GEMM each for gate, up and
        # down ('grouped': gpu/spawn/fp8_gemm's grouped GEMM; FP8 experts whose widths are multiples of 128).
        if prefill_moe not in ('tokens', 'grouped'):
            raise ValueError("prefill_moe must be 'tokens' or 'grouped'")
        self.prefill_moe = prefill_moe
        # prefill_attention: prefill_chunk's attention a token at a time with forward()'s kernels ('tokens'), or in the
        # MHA form DeepSeek's and SGLang's prefill use ('mha': kv_b_proj expands every cached key's latent to its keys
        # and values, then one causal mha_prefill launch for the chunk) while the context fits in index_topk, where
        # every token attends to every earlier one (#43).
        if prefill_attention not in ('tokens', 'mha'):
            raise ValueError("prefill_attention must be 'tokens' or 'mha'")
        self.prefill_attention = prefill_attention
        if topk not in ('auto', 'sort', 'radix'):
            raise ValueError("topk must be 'auto', 'sort' or 'radix'")
        self._topk_choice = topk
        # Past index_topk: 'position' (the default) writes the stable top-k set in position order with
        # one topk_select launch (#32), the order the selection has within index_topk (#43) and the
        # reference model's; 'merge' and 'select' write it in score order (topk_chunks + topk_merge, or
        # topk_radix), as torch.topk does, for an A/B of the order's rounding.
        # long_topk='warp' (#32): 'position''s set and order from topk_select_warp_at under graph replay, the
        # same select with warp instructions in mlir! blocks (gpu/spawn/index_mma): per-warp histograms
        # with one add a distinct digit, shuffle scans, the digit by a ballot count. GPU only (no host
        # build); without graph replay it is 'position'.
        # long_topk='ilp' (#32): the same set and order from topk_select_ilp_at, three passes over the
        # scores (a histogram of the keys' top 12 bits, the threshold bin's keys listed in shared
        # memory, the write by warp), not topk_select_at's six. 'ilp_hist' also folds the first pass
        # into the scores kernel (index_scores_mma_hist_at / index_scores_fp8_hist_at count the bins
        # into score_bins, which topk_select_ilp_hist_at reads and zeroes): two passes. Where the scores
        # are not those kernels' (no graph, other shapes or index kernels), 'ilp_hist' is 'ilp'. GPU
        # only; without graph replay both are 'position'.
        if long_topk not in ('merge', 'select', 'position', 'warp', 'ilp', 'ilp_hist'):
            raise ValueError("long_topk must be 'merge', 'select', 'position', 'warp', 'ilp' or 'ilp_hist'")
        self.long_topk = long_topk
        # With 'ilp_hist', a step of more than topk_blocks_from tokens selects with the multi-block pair
        # topk_mb_list_at + topk_mb_write_at (8192 positions a block) instead of one block; the step graphs
        # are kept by that range too, so a request crossing it captures once more. Either kernel gives
        # the same set and order, so the choice is speed only.
        self.topk_blocks_from = int(topk_blocks_from)
        # fold_shared: a MoE layer's shared expert runs inside the routed experts' two launches as one more
        # expert, stacked last (id n_routed_experts, weight 1.0 in a ninth route slot the router never
        # writes), so its gate_up, down and the add after them are not launches of their own. Ascending
        # id order puts it last, where the add put it: for fp8_kernels='cvt' the same bits (the routed
        # kernels' rows are fp8_matvec_cvt_tiles' and fp8_moe_cvt_gate_up's). Needs device_experts, one
        # shared expert, and its Linears in the routed experts' format and shapes.
        self.fold_shared = bool(fold_shared)
        if self.fold_shared and not (self.device_experts and fp8_kernels == 'cvt'):
            raise ValueError("fold_shared needs device_experts=True and fp8_kernels='cvt'")
        # moe_down2: the routed experts' down projection through gpu/spawn/fp8_moe_cvt_down2 (two rows and two
        # experts a subgroup), the bits of fp8_moe_cvt_down: for the short expert rows of tensor parallelism.
        self.moe_down2 = bool(moe_down2)
        if self.moe_down2 and not (self.device_experts and fp8_kernels == 'cvt'):
            raise ValueError("moe_down2 needs device_experts=True and fp8_kernels='cvt'")
        # moe_mma: the down projection on the FP8 tensor cores (gpu/spawn/moe_mma, inline PTX: no CPU copy, so
        # only on a GPU of sm_89 or later). Changes bits: validation/moe_mma_check.py, then GSM8K.
        if moe_mma not in (False, True, 'f16', 'e4m3'):
            raise ValueError(f"moe_mma is False, True, 'f16' or 'e4m3', not {moe_mma!r}")
        self.moe_mma = {True: 'f16', False: False}.get(moe_mma, moe_mma)
        if self.moe_mma and not (self.device_experts and fp8_kernels == 'cvt'):
            raise ValueError("moe_mma needs device_experts=True and fp8_kernels='cvt'")
        # fuse_router: a MoE layer's router projection and the router in one launch (gpu/spawn/router_bf16:
        # a block a row in the hand bf16_matvec's order, then the router's choice in the last block to
        # finish). For a BF16 gate, as the official checkpoints store it; another gate keeps the two
        # launches. Its logits are the hand bf16_matvec's (so matvec_kernels='x4' changes its bits, as
        # bf16_matvec4 sums in another order). The hand router's exp is ex2.approx, router_bf16's
        # libdevice's expf: on the GPU the weights differ in the last bits (as fused_attention's); on the
        # CPU emulator the logits equal the hand path's bit for bit.
        self.fuse_router = bool(fuse_router)
        # device_decode (#63; docs/TP-SERVING-PLAN.md item 2): device_step, a greedy decode step whose inputs
        # (gpu/spawn/decode_inputs) and token (argmax_pair, under tensor parallelism the ranks' pairs
        # gathered, then decode_next) are made on the device inside the step's graph, so the host can
        # launch a step before it has read the token of the step before. The arithmetic is forward()'s.
        self.device_decode = bool(device_decode)
        if self.device_decode and not graph:
            raise ValueError('device_decode needs graph=True')
        # fuse_tp_norm (#63; docs/TP-SERVING-PLAN.md item 3): a block count > 0 sums a layer's partials across
        # the ranks together with the residual RMSNorm that follows (gpu/spawn/tp_sum_rmsnorm in that many
        # blocks: each value pushed to every rank over NVLink with the call's epoch, polled locally), in place
        # of tp_all_reduce then residual_rmsnorm8, with their bits. True is 8 blocks. Needs one-shot sums
        # (TensorParallel(one_shot=True)), fuse_glue=True and norm_block=1024.
        self.fuse_tp_norm = 8 if fuse_tp_norm is True else int(fuse_tp_norm)
        if self.fuse_tp_norm:
            if tp is None or not tp.one_shot:
                raise ValueError('fuse_tp_norm needs tensor parallelism with one-shot sums')
            if not fuse_glue or norm_block != 1024:
                raise ValueError('fuse_tp_norm needs fuse_glue=True and norm_block=1024')
            if not 1 <= self.fuse_tp_norm <= 1024 or 1024 % self.fuse_tp_norm:
                raise ValueError('fuse_tp_norm must be 0 or a block count dividing 1024')
        if mla_kernel not in ('decode', 'chunks', 'heads', 'flash', 'tc'):
            raise ValueError("mla_kernel must be 'decode', 'chunks', 'heads', 'flash' or 'tc'")
        self.mla_kernel = mla_kernel
        # index_kernel (#32): the indexer's scores past index_topk. 'simt' is index_scores(_at), in f32,
        # bit for bit the reference's; 'tc' (under graph replay) is index_scores_tc_at: q and the key
        # rows rounded to bf16, the products on the tensor cores with f32 sums, so the scores agree to
        # bf16 rounding. SGLang's indexer rounds both to FP8 (DeepGEMM fp8_paged_mqa_logits), so 'tc'
        # is at its precision or above. 'mma' (graph replay, 32 heads of dim 128 only) is the same
        # arithmetic with the warp's work in inline PTX (gpu/spawn/index_mma): key and query rows
        # loaded as float4s and rounded to bf16 in registers, no shared memory. It has no host build,
        # so only the GPU runs it. 'fp8' (graph replay, 32 heads of dim 128, sm_89 and up) is SGLang
        # 0.5.22's indexer exactly: the key rows quantized to E4M3 with one scale a row as they are
        # cached (index_key_fp8_at, as indexer_k.cuh), each head's q with one scale a head
        # (fused_q_indexer_rope_first_quant), mma.sync e4m3 with f32 sums (index_scores_fp8_at). It
        # matches SGLang's precision; it is slower than 'mma' until q is quantized once a step (in the
        # RoPE) and the key row in index_key_at, not in each warp and a launch of its own. 'mma' (bf16,
        # above SGLang's precision) is the default: 3.9 us at 3001 tokens on the H200 against
        # 'simt''s 11.2 and 'fp8''s 4.2 + 1.4 (evidence/h200-20261117-v, -w). Off its shape, or without
        # graph replay, 'mma' and 'fp8' take 'simt''s path, so smaller models keep the reference's bits.
        # 'mma16' (not yet GPU-run) is 'mma' over a bf16 copy of the key rows (index_key_bf16_at, by the
        # cvt.rn.bf16x2.f32 'mma' rounds them with in registers): the same scores bit for bit from half
        # the bytes (index_scores_mma16_at).
        if index_kernel not in ('simt', 'tc', 'mma', 'fp8', 'mma16'):
            raise ValueError("index_kernel must be 'simt', 'tc', 'mma', 'fp8' or 'mma16'")
        self.index_kernel = index_kernel
        if mla_kernel != 'decode' and not (self.graph and self.fused_attention):
            raise ValueError(f"mla_kernel={mla_kernel!r} needs graph=True and fused_attention=True")
        if self.graph and not (self.device_topk and self.device_experts):
            raise ValueError('graph=True needs device_topk and device_experts: a captured step cannot read device memory')
        # The step graph, one for steps whose every cached token is selected (no indexer scores or
        # top-k, #43) and one for longer contexts: each runs eagerly first, is captured the second time.
        self._step_graphs, self._step_warm = {}, set()
        # False: graph=True's steps always run their launches eagerly (the warm path, the replay's bits),
        # never captured. For a reference whose collective cannot be captured (validation/tp_decode_check).
        self._capture = True
        # device_step's buffers (_device_setup), and the position its device state holds (None: the host
        # must give the step's token, as after reset, rewind or a forward()).
        self._device, self._device_position = None, None
        # forward_many's graphs by token count, mtp_step's by reuse_selection: run eagerly the first
        # time, captured the second, replayed after (_replay).
        self._graphs, self._warmed = {}, set()
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
            # fp8_kernels='fma'/'cvt': the one expert id (0) that runs a dense gate/up through the MoE kernel.
            self._expert_zero = (self._tensor((1,), 'int32', np.zeros(1, np.int32).tobytes())
                                 if fp8_kernels in ('fma', 'cvt') else None)
            # fp8_kernels='cvt': the sm_90 builds (Hopper's conversion) from compute capability 8.9, else _ref.
            capability = tuple(getattr(self.ctx, 'compute_capability', None) or (0, 0))
            self._cvt_suffix = '' if capability >= (8, 9) else '_ref'
            self._kernels = {}
            self._spawn_kernels = {}
            self._lt, self._lt_matmuls, self._lt_inputs, self._lt_outputs = None, {}, {}, {}  # fp8_kernels='cublaslt'
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
            # The graphs first: NCCL's ncclCommDestroy waits until every graph holding its
            # operations is gone, so a tensor-parallel rank must close before collective.finish().
            for graph in [*getattr(self, '_step_graphs', {}).values(), *getattr(self, '_graphs', {}).values()]:
                graph.close()
            if getattr(self, '_tp_peers', None) is not None:
                self._tp_peers.close()
            for matmul in getattr(self, '_lt_matmuls', {}).values():
                matmul.close()
            if getattr(self, '_lt', None) is not None:
                self._lt.close()
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
        # mla_heads: the heads mla_flash_at and mla_tc_at run, a multiple of 16 (a tile's rows). A rank of a
        # tensor-parallel job may hold fewer (GLM-5.3 over 8 GPUs: 8): its query rows are padded with heads
        # of zeros, whose results nothing reads (#57). mla_tc_at's tile is 16 rows either way, so the padding
        # costs it no loads; mla_flash_at runs the padded heads as real ones.
        self.mla_heads = self.heads
        if self.mla_kernel in ('flash', 'tc'):
            self.mla_heads = -(-self.heads // 16) * 16
            if (self.mla_heads != self.heads and self.tp is None) or m.rank != 512 or m.rot != 64:
                raise ValueError(f"mla_kernel={self.mla_kernel!r} needs a multiple of 16 heads, rank 512 and rope 64 (or "
                                 f"tensor parallelism, which pads the heads); the model has {self.heads} heads, rank "
                                 f"{m.rank}, rope {m.rot}")
        if self.mla_kernel == 'tc' and m.index_topk > 2048:
            raise ValueError(f"mla_kernel='tc' needs index_topk <= 2048 (64 splits of 32), the model has {m.index_topk}")
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
            # With mtp=True the MTP step's hidden state follows them, so mtp_step uploads once too.
            inputs = self.hidden + 2 * half + 4
            self._inputs = t((inputs + (self.hidden if self.mtp else 0),))
            self.step_inputs = self._inputs.view((inputs,))
            if self.mtp:
                self.mtp_hidden = self._inputs.view((self.hidden,), offset=inputs * 4)
            self.x = self.step_inputs.view((self.hidden,))
            self.cos = self.step_inputs.view((half,), offset=self.hidden * 4)
            self.sin = self.step_inputs.view((half,), offset=(self.hidden + half) * 4)
            self.step = self.step_inputs.view((4,), 'int32', offset=(self.hidden + 2 * half) * 4)
            # forward_many's inputs: the [MANY_TOKENS, hidden] embedding rows, then each token's cos,
            # sin and step buffer, a 256-byte aligned block a token.
            block = -(-(2 * half + 4) // 64) * 64
            self._many_inputs = t((MANY_TOKENS * (self.hidden + block),))
            self._many_buffers = {('x', self.hidden): self._many_inputs.view((MANY_TOKENS, self.hidden))}
            base = MANY_TOKENS * self.hidden * 4
            self._many_steps = [(self._many_inputs.view((half,), offset=base + i * block * 4),
                                 self._many_inputs.view((half,), offset=base + (i * block + half) * 4),
                                 self._many_inputs.view((4,), 'int32', offset=base + (i * block + 2 * half) * 4))
                                for i in range(MANY_TOKENS)]
            self._many_block = block
        else:
            self.x = t((self.hidden,))
            self.cos, self.sin = t((half,)), t((half,))
            if self.mtp:
                self.mtp_hidden = t((self.hidden,))
        self.x_next, self.norm = t((self.hidden,)), t((self.hidden,))
        self.attn_out, self.ffn_out = t((self.hidden,)), t((self.hidden,))
        self.q_a, self.q_res = t((q_rank,)), t((q_rank,))
        self.query = t((self.heads * qk,))
        self.compressed = t((self.rank + self.rot,))
        hp = self.mla_heads  # the padded heads' query rows stay zero: nothing writes them
        self.latent_q = t((hp, self.rank), 'float32', bytes(hp * self.rank * 4))
        self.rot_q = t((hp, self.rot), 'float32', bytes(hp * self.rot * 4))
        # mla_flash_at takes up to 64 splits of about 8 tokens (evidence/h200-20261115-i); the others 16 of 256.
        self.splits_max = 64 if self.mla_kernel in ('flash', 'tc') else 16
        if self.mla_kernel == 'tc':  # the step's queries in bf16, in mla_tc_at's tile layout (mla_q_bf16_at)
            self.q_t, self.qr_t = t((hp // 16, 1024, 8), 'uint16'), t((hp // 16, 128, 8), 'uint16')
        self.numerator = t((hp, self.splits_max, self.rank))
        self.stats = t((hp, self.splits_max, 2))
        self.attended = t((hp, self.rank))
        if self.fused_attention:  # mla_decode's per-head arrival counts: zero once, its last splits re-zero them
            self.arrivals = t((self.heads,), 'int32', np.zeros(self.heads, np.int32))
        self.heads_out = t((self.heads * self.value_dim,))
        self.iq, self.iq_rot = t((self.index_heads * self.index_dim,)), t((self.index_heads * self.index_dim,))
        self.ik, self.ik_norm = t((self.index_dim,)), t((self.index_dim,))
        if self.fp8_kernels in ('fma', 'cvt'):
            # One buffer for q_a, kv_a and the indexer's wk outputs, as fp8_matvec_fma_tiles3 writes
            # them (each matrix's rows from a multiple of 8): one launch for all three (#30).
            pad = lambda n: -(-n // 8) * 8
            spans = (q_rank, self.rank + self.rot, self.index_dim)
            self._qkv_a = t((sum(pad(n) for n in spans),))
            self.q_a = self._qkv_a.view((q_rank,))
            self.compressed = self._qkv_a.view((spans[1],), offset=pad(q_rank) * 4)
            self.ik = self._qkv_a.view((self.index_dim,), offset=(pad(q_rank) + pad(spans[1])) * 4)
        self.head_weights_raw, self.head_weights = t((self.index_heads,)), t((self.index_heads,))
        self.scores = t((cap,))
        # long_topk='ilp_hist': the scores' bins of 4096, zero between layers (the top-k zeroes it).
        self.score_bins = t((4096,), 'uint32', np.zeros(4096, np.uint32)) if self.long_topk == 'ilp_hist' else None
        self._hist_counted = False
        # The multi-block top-k's scratch (topk_mb_list_at): each warp's keys above the threshold bin, the
        # bin's keys and positions, their count (zero between layers), each warp's first slots, K and left,
        # and the two launches' arrival counts. Its offsets hold 4096 warps of 256 positions.
        self._mb = None
        if self.score_bins is not None and cap > self.topk_blocks_from:
            if cap > 4096 * 256:
                raise ValueError(f'the multi-block top-k takes up to {4096 * 256} tokens, not {cap}')
            z = lambda n, dtype: t((n,), dtype, np.zeros(n, dtype))  # dtype by name, as CUDAContext.tensor takes it
            self._mb = dict(wcnt=t((4096,), 'int32'), ckey=t((16384,), 'uint32'), cpos=t((16384,), 'int32'),
                            count=z(1, 'uint32'), offs=t((8192,), 'int32'), sel=z(4, 'uint32'),
                            list_arrivals=z(1, 'int32'), write_arrivals=z(1, 'int32'))
        self.selected = t((self.index_topk,), 'int32')
        # 0, 1, ..., index_topk - 1: the selection while the context fits in index_topk (#43).
        self.positions = t((self.index_topk,), 'int32', np.arange(self.index_topk, dtype=np.int32))
        if self.device_topk:
            from .topk import scratch_ints
            self.topk_scratch = t((scratch_ints(cap, self.index_topk),), 'int32')
            if cap > TOPK_SORT_TOKENS and self.index_topk <= TOPK_CHUNK:
                # topk_chunks' sorted rows and topk_merge's, swapped each round (#32).
                chunks = -(-cap // TOPK_CHUNK)
                self.topk_lists = [t((chunks, TOPK_CHUNK), 'int64') for _ in range(2)]
                # topk_radix's counts, the threshold so far, the gathered values, their count and
                # its last block's arrivals; hist, count and arrivals zero once, round 8 and each
                # last block re-zero them (#32).
                self.topk_hist = t((256,), 'uint32', np.zeros(256, np.uint32))
                self.topk_state = t((5,), 'int64')
                self.topk_vals = t((TOPK_CHUNK,), 'int64')
                self.topk_count = t((1,), 'uint32', np.zeros(1, np.uint32))
                self.topk_arrivals = t((1,), 'int32', np.zeros(1, np.int32))
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
        if 'sparse' in m.mlp_types or self.mtp:  # the MTP layer is a MoE layer
            experts = int(c['n_routed_experts'])
            k = int(c['num_experts_per_tok'])
            self.router_logits = t((experts,))
            self.router_arrivals = t((1,), 'int32', np.zeros(1, np.int32))  # fuse_router's last-block count
            # fold_shared: slot k is the shared expert, id `experts` and weight 1.0, written once here.
            slots = k + 1 if self.fold_shared else k
            self.route_ids = t((slots,), 'int32', np.array([0] * k + [experts] * (slots - k), np.int32))
            self.route_weights = t((slots,), 'float32', np.array([0.0] * k + [1.0] * (slots - k), np.float32))
            if self.device_experts:  # each routed expert's SwiGLU output, in ascending id order
                self.moe_act = t((slots * int(c['moe_intermediate_size']),))
        vocab = int(c['vocab_size'])
        self.logits = t((vocab,))
        if self.tp is not None:
            # The rank's vocabulary rows of the logits, gathered into self.logits; the partial sums of
            # every rank, gathered for _tp_sum.
            self.logits_local = t((vocab // self.tp.size,))
            # forward(logits='greedy'): this rank's (largest logit, its index) and every rank's, gathered.
            self._greedy_pair = t((2,))
            self._greedy_pairs = t((2 * self.tp.size,))
            self._tp_gathered = t((self.tp.size * self.hidden,))
            self._tp_peers = None
            if self.tp.one_shot:
                from .tp import PeerSum, handles_by_collective
                if self.fuse_tp_norm and (self.hidden > 8192 or self.fuse_tp_norm < -(-self.hidden // 1024)):
                    raise ValueError(f'fuse_tp_norm: a hidden size of {self.hidden} needs at most 8192 floats and '
                                     f'at least {-(-self.hidden // 1024)} blocks')
                self._tp_peers = PeerSum(self.ctx, self.tp.rank, self.tp.size, self.hidden,
                                         handles_by_collective(self.tp.collective, self.ctx, self.stream),
                                         norm_blocks=self.fuse_tp_norm or 8)
        self.token_ids = t((MANY_TOKENS,), 'int32')  # argmax_rows' greedy tokens (greedy=True)
        # Per-layer caches, all rows the caller has produced so far stay on the device.
        # With mtp=True, one more of each for the MTP layer, at index n_layers.
        layers = m.n_layers + (1 if self.mtp else 0)
        cache = 'uint16' if self.mla_kernel == 'tc' else 'float32'  # mla_kernel='tc': bf16 bits
        self.latents = [t((cap, self.rank), cache) for _ in range(layers)]
        self.rope_keys = [t((cap, self.rot), cache) for _ in range(layers)]
        self.index_keys = [t((cap, self.index_dim)) if m.indexer_type(i) == 'full' else None
                           for i in range(layers)]
        # index_kernel='fp8': each full indexer's key rows also as E4M3 codes (128 bytes, 32 words a row)
        # and a scale a row, written by index_key_fp8_at as the rows are cached.
        self.index_fp8 = (self.graph and self.index_kernel == 'fp8' and self.index_heads == 32
                          and self.index_dim == 128)
        if self.index_fp8:
            self.index_codes = [t((cap, 32), 'int32') if k is not None else None for k in self.index_keys]
            self.index_key_scales = [t((cap,)) if k is not None else None for k in self.index_keys]
        # index_kernel='mma16': each full indexer's key rows also in bf16 (64 words a row), written by
        # index_key_bf16_at as the rows are cached.
        self.index_bf16 = (self.graph and self.index_kernel == 'mma16' and self.index_heads == 32
                           and self.index_dim == 128)
        if self.index_bf16:
            self.index_words = [t((cap, 64), 'int32') if k is not None else None for k in self.index_keys]
        if self.mtp:
            self.mtp_input = t((2 * self.hidden,))  # [enorm(embedding), hnorm(hidden)], eh_proj's input

    def reset(self):
        """Start a new sequence. Device caches are overwritten as positions are reused."""
        self.position = 0
        self.selected_count = 0
        self._device_position = None

    def rewind(self, position):
        """Drop the steps from `position` on (at most the current position), as a step launched ahead
        and not wanted: the caches' rows from there are overwritten as the positions are reused."""
        if not 0 <= int(position) <= self.position:
            raise ValueError(f'cannot rewind to {position} from {self.position}')
        self.position = int(position)
        self.selected_count = min(self.index_topk, self.position)
        self._device_position = None

    # ---------------------------------------------------------------- op helpers
    def _matvec(self, weight, x, out):
        rows, cols = weight.shape
        rows_kernel = rows <= MATVEC_ROWS_MAX and cols <= MATVEC_ROWS_COLS and self._matvec_rows()
        if rows_kernel:
            # One block a row: 1024 threads load a 6144-column row in one batch each. The CPU
            # emulator's kernels ask for a warp (gpu/testing/context.py).
            block = getattr(self._kernels['matvec_rows'], 'rows_block', 1024)
            self._launch('matvec_rows', rows, block, [out, weight, x, I(rows), I(cols)])
        else:
            self._launch('matvec', (rows + 3) // 4, 128, [out, weight, x, I(rows), I(cols)])
        return out

    def _matvec_rows(self):
        """Whether matvec_rows.ptx is built (gpu/build.sh); its kernel, loaded once, is then in _kernels."""
        if 'matvec_rows' not in self._kernels:
            path = self._ptx_dir / 'matvec_rows.ptx'
            if not path.exists():
                return False
            module = self._modules['matvec_rows'] = self.ctx.load_ptx(path)
            self._kernels['matvec_rows'] = module.kernel('glm_vx_gpu_matvec_rows')
        return True

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
            if self.fp8_kernels == 'cublaslt' and cols % 128 == 0:
                self._check_shape(prefix, fmt, rows, cols, x, out)
                return self._fp8_cublaslt(prefix, codes, rows, cols, x, out)
            if self.fp8_kernels in ('x4', 'fma', 'cvt') and cols % 16 == 0:
                if out.shape != (rows,) or x.shape != (cols,):
                    raise ValueError(f'{prefix}: FP8 weight is {rows}x{cols}, got x {x.shape} -> out {out.shape}')
                if self.fp8_kernels == 'cvt':
                    kernel = 'fp8_matvec_cvt_tiles' + self._cvt_suffix
                else:
                    kernel = 'fp8_matvec4' if self.fp8_kernels == 'x4' else (
                        'fp8_matvec_fma' if cols >= FMA_LONG_ROWS else 'fp8_matvec_fma_tiles')
                self._spawn(kernel, out=out, weight=codes.view((rows * cols // 4,), 'uint32'),
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

    def _fp8_cublaslt(self, prefix, codes, rows, cols, x, out):
        """out = W x through cuBLASLt (fp8_kernels='cublaslt'): x to e4m3 codes and per-128 scales in
        scratch kept per width, then one block-scaled FP8 matmul on the stream."""
        from .cublaslt import pack_weight_scales
        key = prefix + ':fp8:cublaslt-scales'
        scales = self._weights.get(key)
        if scales is None:  # BLK128x128_32F's layout: each row block's scales padded to a multiple of 4
            scales = self._weights[key] = self._upload_weight(key, pack_weight_scales(self.model.weights.block_fp8(prefix)[1]))
        # cuBLASLt multiplies 4 tokens (gpu/cublaslt.py); this one is row 0, rows 1-3 stay zero codes.
        inputs = self._lt_inputs.get(cols)
        if inputs is None:
            inputs = self._lt_inputs[cols] = (self._tensor((4, cols), 'uint8', np.zeros((4, cols), np.uint8)),
                                              self._tensor((cols // 128, 4), 'float32', np.ones((cols // 128, 4), np.float32)))
        x_codes, x_scales = inputs
        if 'fp8_quant_groups' not in self._kernels:
            module = self._modules['fp8_quant'] = self.ctx.load_ptx(self._ptx_dir / 'fp8_quant.ptx')
            self._kernels['fp8_quant_groups'] = module.kernel('glm_vx_gpu_fp8_quant_groups')
        self._launch('fp8_quant_groups', -(-cols // 128 // 4), 128, [x_codes, x_scales, x, I(1), I(cols), I(4)])
        matmul = self._lt_matmuls.get((rows, cols))
        if matmul is None:
            factory = getattr(self.ctx, 'fp8_block_matmul', None)
            if factory is not None:
                matmul = factory(1, rows, cols)
            else:
                from .cublaslt import CublasLt, Fp8BlockMatmul
                if self._lt is None:
                    self._lt = CublasLt(self.ctx)
                matmul = Fp8BlockMatmul(self._lt, 1, rows, cols)
            self._lt_matmuls[(rows, cols)] = matmul
        lt_out = self._lt_outputs.get(rows)
        if lt_out is None:
            lt_out = self._lt_outputs[rows] = self._tensor((4, rows))
        matmul(lt_out, x_codes, x_scales, codes, scales, stream=self.stream)
        out.copy_from(lt_out.view((rows,)), stream=self.stream)  # token 0's outputs, the first rows floats
        self.launches['cublaslt_fp8'] = self.launches.get('cublaslt_fp8', 0) + 1
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
        rounds, or with long_topk='select' its topk_radix rounds (the same ids
        as topk_stable): with 'sort', when they are built and index_topk <= 2048. The chunk is
        topk_sort's limit, so past it there are always two chunks or more."""
        names = ('topk_chunks', 'topk_merge') if self.long_topk == 'merge' else ('topk_radix',)
        if self.graph:
            names = tuple(n + '_at' for n in names)
        return (self._topk_sort() and hasattr(self, 'topk_lists') and TOPK_SORT_TOKENS == TOPK_CHUNK
                and all((self._ptx_dir / 'spawn' / (n + '.ptx')).exists() for n in names))

    def _topk_chunks_and_merge(self, tokens):
        """topk_chunks, then topk_merge rounds until one row is left; the last writes selected.
        long_topk='select': topk_radix rounds 0 .. 8; round 8 writes it."""
        if self.long_topk == 'select':
            return self._topk_radix(tokens)
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

    def _topk_radix(self, tokens):
        size = dict(step=self.step, max_tokens=self.max_tokens) if self.graph else dict(tokens=tokens)
        scores = self.scores.view(self.scores.shape if self.graph else (tokens,), 'int32')
        for r in range(9):
            # Rounds 0 .. 7 find a byte of the threshold each; round 8 gathers, sorts and writes selected.
            self._spawn('topk_radix_at' if self.graph else 'topk_radix', out_ids=self.selected, hist=self.topk_hist,
                        state=self.topk_state, vals=self.topk_vals, count=self.topk_count, scores=scores,
                        k=self.index_topk, round=r, arrivals=self.topk_arrivals, **size)

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

    def _tp_sum(self, partial):
        """With tp: partial (this rank's part of a sum over the ranks, [hidden]) becomes the whole sum,
        ((p0 + p1) + p2) + .., the same bits on every rank and on any number of GPUs (gpu/tp.py)."""
        if self.tp is None:
            return
        if self._tp_peers is not None:
            self._spawn('tp_all_reduce', **self._tp_peers.arguments(partial))
            return
        n, ranks = partial.shape[0], self.tp.size
        gathered = self._tp_gathered.view((ranks * n,))
        self.tp.collective.all_gather(partial, gathered, stream=self.stream)
        self.launches['tp_all_gather'] = self.launches.get('tp_all_gather', 0) + 1
        if self.spawn_built('sum_ranks'):  # one launch, the adds' bits
            self._spawn('sum_ranks', out=partial, gathered=gathered, ranks=ranks, n=n)
            return
        rows = [gathered.view((n,), offset=r * n * 4) for r in range(ranks)]
        self._add(partial, rows[0], rows[1], n)
        for row in rows[2:]:
            self._add(partial, partial, row, n)

    def _copy(self, out, x, n):
        self._launch('copy', self._grid(n), _ELEMENTWISE_BLOCK, [out, x, I(n)])

    def _rmsnorm(self, x, weight, out, eps):
        n = x.shape[0]
        self._launch('rmsnorm', 1, self.norm_block, [out, x, weight, I(n), F(_f32(eps, 'epsilon', True))])

    def _residual_rmsnorm(self, x, residual, weight, out, residual_out, eps):
        """residual_out = x + residual and out = RMSNorm(residual_out) in one launch: the same bits as
        _add then _rmsnorm (each thread sums the same squares in the same order)."""
        n = x.shape[0]
        self._launch('residual_rmsnorm', 1, self.norm_block,
                     [out, residual_out, x, residual, weight, I(1), I(n), F(_f32(eps, 'epsilon', True))])

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
        if self.graph and cache.dtype == 'uint16':  # mla_kernel='tc''s bf16 caches
            self._spawn('store_row_bf16_at', cache=cache, row=scratch.view((width,)), step=self.step, width=width)
        elif self.graph:
            self._launch('store_row', self._grid(width), _ELEMENTWISE_BLOCK, [cache, scratch, self.step, I(width)])

    def _mlp(self, prefix, x, out):
        """down(SwiGLU(gate(x), up(x))) with both projections and the activation resident."""
        width = self._out_features(prefix + '.gate_proj')
        gate, up, act = (t.view((width,)) for t in (self.gate, self.up, self.act))
        if not self._fma_gate_up(prefix, x, act, width):
            self._project(prefix + '.gate_proj', x, gate)
            self._project(prefix + '.up_proj', x, up)
            self._swiglu(gate, up, act, width)
        self._project(prefix + '.down_proj', act, out)

    def _fma_gate_up(self, prefix, x, act, width):
        """act = SwiGLU(gate(x), up(x)) in one launch for fp8_kernels='fma' (#30): fp8_moe_fma_gate_up
        with one expert, so each x is read once for both projections and the activation needs no
        launch of its own. False (nothing launched) for other kernels, formats or widths."""
        gate_p, up_p = prefix + '.gate_proj', prefix + '.up_proj'
        if self.fp8_kernels not in ('fma', 'cvt') or self._format(gate_p) != 'fp8' or self._format(up_p) != 'fp8':
            return False
        (g_codes, g_scales, (rows, cols)), (u_codes, u_scales, up_shape) = self._fp8_weight(gate_p), self._fp8_weight(up_p)
        if (rows, cols) != tuple(up_shape) or rows != width or cols % 16 or x.shape != (cols,):
            return False
        words = lambda t: t.view((int(np.prod(t.shape)) // 4,), 'uint32')
        flat = lambda t: t.view((int(np.prod(t.shape)),))
        kernel = 'fp8_moe_cvt_gate_up' + self._cvt_suffix if self.fp8_kernels == 'cvt' else 'fp8_moe_fma_gate_up'
        self._spawn(kernel, act=act, gate=words(g_codes), gate_scales=flat(g_scales), up=words(u_codes),
                    up_scales=flat(u_scales), x=x, ids=self._expert_zero, k=1, experts=1, inter=rows, up_row=0,
                    stride=rows, cols=cols)
        return True

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

    def _stacked_experts(self, prefix, name, shared=False):
        """All routed experts' Linear `name` ('gate_proj', 'up_proj', 'down_proj', or 'gate_up' /
        'down' of the fused layout) back to back on the device, uploaded once:
        (format, tensors, (rows, cols), extra arguments). shared: the shared expert's `name` after
        them, as expert n_routed_experts (fold_shared; separate Linears only)."""
        key = f'{prefix}.experts:{name}:stacked' + (':shared' if shared else '')
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
        if shared:
            linears.append(f'{prefix}.shared_experts.{name}')
        fmt = self._format(linears[0])
        mixed = [p for p in linears if self._format(p) != fmt]
        if mixed:
            raise ValueError(f'device_experts needs one format per expert projection: {linears[0]} is {fmt}, '
                             f'{mixed[0]} is {self._format(mixed[0])}')
        w = self.model.weights
        if shared:  # before any upload: _upload_stacked takes the parts' shapes as the first's
            shape_of = {'int4': lambda p: w.compressed_int4(p)[3], 'fp8': lambda p: w.block_fp8(p)[3],
                        'bf16': lambda p: w.bf16(p)[1]}.get(fmt, lambda p: self.model._weight(p + '.weight').shape)
            want, got = tuple(shape_of(linears[0])), tuple(shape_of(linears[-1]))
            if got != want:
                raise ValueError(f"fold_shared needs {linears[-1]} in the routed experts' shape {want}, got {got}")
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

    def _folds_shared(self, prefix):
        """fold_shared applies to this MoE layer: one shared expert and separate expert Linears."""
        return (self.fold_shared and int(self.config.get('n_shared_experts', 1)) == 1
                and not self._fused_experts(f'{prefix}.experts.0'))

    def _device_moe(self, prefix, x, k):
        """moe_sum = the routed experts' weighted sum in two launches, the ids read on the device:
        gpu/spawn's moe_gate_up (every expert's gate, up and SwiGLU) then moe_down (each output
        row sums its k terms in ascending id order). The same arithmetic as the per-expert
        kernels (*_expert_matvec, swiglu, expert_scaled_add from zeros), to the last bit. With
        fold_shared, the shared expert is the last of k + 1 (id n_routed_experts, weight 1.0)."""
        experts = int(self.config['n_routed_experts'])
        if self._folds_shared(prefix):
            gate, up, down = (self._stacked_experts(prefix, n, shared=True) for n in ('gate_proj', 'up_proj', 'down_proj'))
            inter, up_row = gate[2][0], 0
            stride = inter
            experts, k = experts + 1, k + 1
        elif self._fused_experts(f'{prefix}.experts.0'):
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
        suffix = ''
        if fmt == 'fp8' and self.fp8_kernels in ('x4', 'fma', 'cvt', 'cublaslt') and hidden % 16 == 0 and inter % 16 == 0:
            # the codes as u32: sixteen a lane step (#31), one fma a code ('fma'), the codes' true values ('cvt')
            kernel = {'fma': 'fp8_moe_fma_', 'cvt': 'fp8_moe_cvt_'}.get(self.fp8_kernels, 'fp8_moe4_')
            suffix = self._cvt_suffix if self.fp8_kernels == 'cvt' else ''
            words = lambda t: t.view((int(np.prod(t.shape)) // 4,), 'uint32')
            gate_args, down_args = dict(gate=words(gate[1][0]), up=words(up[1][0])), dict(down=words(down[1][0]))
        if fmt in ('int4', 'fp8'):
            gate_args.update(gate_scales=gate[1][1], up_scales=up[1][1])
            down_args.update(scales=down[1][1])
        act = self.moe_act.view((k * inter,))
        self._spawn(kernel + 'gate_up' + suffix, act=act, x=x, inter=inter, up_row=up_row, stride=stride, cols=hidden,
                    **gate_args, **common)
        down_kernel = kernel + ('down2' if self.moe_down2 and kernel == 'fp8_moe_cvt_' else 'down') + suffix
        if self.moe_mma and kernel == 'fp8_moe_cvt_' and not suffix and k <= 9 and hidden % 16 == 0 and inter % 256 == 0:
            down_kernel = 'moe_mma/fp8_moe_mma' + ('16' if self.moe_mma == 'f16' else '') + '_down_at'
        self._spawn(down_kernel, out=self.moe_sum, act=act, weights=self.route_weights, rows=hidden,
                    cols=inter, **down_args, **common)

    # ---------------------------------------------------------------- blocks
    def _index_project(self, prefix, x, q_res, iq, ik, head_weights_raw, key_done=False, key_only=False):
        """The indexer's three projections: its queries from q_res, its key and head weights from x.
        key_done: ik is already written (_fma_qkv_a). key_only: the context fits in index_topk, so
        _index_core stores the key row but scores nothing, and the queries and head weights are not
        needed (#43)."""
        if not key_only:
            self._linear(prefix + '.wq_b', q_res, iq)
        if not key_done:
            self._linear(prefix + '.wk', x, ik)
        if not key_only:
            self._linear(prefix + '.weights_proj', x, head_weights_raw)

    def _fma_qkv_a(self, prefix, x, full):
        """q_a, kv_a and, for a layer whose indexer runs, the indexer's wk of x in one launch for
        fp8_kernels='fma' (fp8_matvec_fma_tiles3 into self._qkv_a, whose views self.q_a,
        self.compressed and self.ik are; each row fp8_matvec_fma_tiles'). False, nothing launched, for
        other kernels, formats or widths."""
        if self.fp8_kernels not in ('fma', 'cvt'):
            return False
        names = [prefix + '.q_a_proj', prefix + '.kv_a_proj_with_mqa'] + ([prefix + '.indexer.wk'] if full else [])
        if any(self._format(n) != 'fp8' for n in names):
            return False
        mats = [self._fp8_weight(n) for n in names]
        cols = mats[0][2][1]
        rows = [m[2][0] for m in mats]
        if any(m[2][1] != cols for m in mats) or cols % 16 or x.shape != (cols,):
            return False
        if rows[0] != self.q_a.shape[0] or rows[1] != self.compressed.shape[0] or (full and rows[2] != self.ik.shape[0]):
            return False
        words = lambda t: t.view((int(np.prod(t.shape)) // 4,), 'uint32')
        flat = lambda t: t.view((int(np.prod(t.shape)),))
        third = mats[2] if full else mats[1]  # rows_c = 0: never read
        kernel = 'fp8_matvec_cvt_tiles3' + self._cvt_suffix if self.fp8_kernels == 'cvt' else 'fp8_matvec_fma_tiles3'
        self._spawn(kernel, out=self._qkv_a, x=x, cols=cols,
                    weight_a=words(mats[0][0]), scales_a=flat(mats[0][1]), rows_a=rows[0],
                    weight_b=words(mats[1][0]), scales_b=flat(mats[1][1]), rows_b=rows[1],
                    weight_c=words(third[0]), scales_c=flat(third[1]), rows_c=rows[2] if full else 0)
        return True

    def _index_key(self, layer, prefix, position, ik):
        """The indexer's key row at `position` from its projection ik: LayerNorm, RoPE and the cache row (and the
        FP8 or bf16 copies its scores kernel reads)."""
        if self.fuse_glue and self.graph and self.rot // 2 <= 256:
            # LayerNorm, RoPE and the cache row at the step's position in one launch (#39).
            self._launch('index_key_at', 1, 256, [self.index_keys[layer], ik, self._weight(prefix + '.k_norm.weight'),
                                                  self._weight(prefix + '.k_norm.bias'), self.cos, self.sin, self.step,
                                                  I(self.index_dim), I(self.rot), F(1e-6)])
        else:
            self._launch('layernorm', 1, 256, [self.ik_norm, ik, self._weight(prefix + '.k_norm.weight'),
                                               self._weight(prefix + '.k_norm.bias'), I(1), I(self.index_dim), F(1e-6)])
            row = self._new_row(self.index_keys[layer], position, self.index_dim, self.index_key_new)
            self._copy(row, self.ik_norm, self.index_dim)
            self._rope(row, self.ik_norm, 1, self.rot, self.rot, self.rot)
            self._store_row(self.index_keys[layer], row, self.index_dim)
        if self.index_fp8:
            self._spawn('index_mma/index_key_fp8_at', codes=self.index_codes[layer], scales=self.index_key_scales[layer],
                        keys=self.index_keys[layer], step=self.step, max_tokens=self.index_keys[layer].shape[0],
                        dim=self.index_dim)
        if self.index_bf16:
            self._spawn('index_mma/index_key_bf16_at', words=self.index_words[layer], keys=self.index_keys[layer],
                        step=self.step, max_tokens=self.index_keys[layer].shape[0], dim=self.index_dim)

    def _index_core(self, layer, prefix, position, iq, ik, head_weights_raw):
        """The indexer after its projections: the key row, the rotated queries, scores, top-k."""
        tokens = position + 1
        self._index_key(layer, prefix, position, ik)
        if tokens <= self.index_topk:
            # Every cached token is selected, in position order, as SGLang's DSA top-k does (topk_v2.cuh,
            # trivial_transform when seq_len <= topk) and GlmMoeDsaModel._index too (#43): the key row is
            # stored for later steps, but no scores or top-k run.
            self._copy(self.selected.view((self.index_topk,), 'float32'), self.positions.view((self.index_topk,), 'float32'),
                       self.index_topk)
            self.selected_count = tokens
            return
        self._copy(self.iq_rot, iq, self.index_heads * self.index_dim)
        self._rope(self.iq_rot, iq, self.index_heads, self.rot, self.index_dim, self.index_dim)
        self._launch('scale', self._grid(self.index_heads), _ELEMENTWISE_BLOCK,
                     [self.head_weights, head_weights_raw, F(_f32(self.index_heads ** -0.5, 'scale')), I(self.index_heads)])
        scale = F(_f32(self.index_dim ** -0.5, 'scale'))
        hist = {} if self.score_bins is None else {'hist': self.score_bins}
        self._hist_counted = False
        if self.index_fp8:
            self._hist_counted = bool(hist)
            self._spawn('index_mma/index_scores_fp8' + ('_hist' if hist else '') + '_at', out=self.scores, **hist,
                        q=self.iq_rot.view((self.index_heads, self.index_dim)), codes=self.index_codes[layer],
                        scales=self.index_key_scales[layer], weights=self.head_weights, step=self.step,
                        max_tokens=self.index_keys[layer].shape[0], heads=self.index_heads, dim=self.index_dim,
                        scale=_f32(self.index_dim ** -0.5, 'scale'))
        elif self.index_bf16:
            self._hist_counted = bool(hist)
            self._spawn('index_mma/index_scores_mma16' + ('_hist' if hist else '') + '_at', out=self.scores, **hist,
                        q=self.iq_rot.view((self.index_heads, self.index_dim)), keys=self.index_words[layer],
                        weights=self.head_weights, step=self.step, max_tokens=self.index_keys[layer].shape[0],
                        heads=self.index_heads, dim=self.index_dim, scale=_f32(self.index_dim ** -0.5, 'scale'))
        elif self.graph and self.index_kernel == 'mma' and self.index_heads == 32 and self.index_dim == 128:
            self._hist_counted = bool(hist)
            self._spawn('index_mma/index_scores_mma' + ('_hist' if hist else '') + '_at', out=self.scores, **hist,
                        q=self.iq_rot.view((self.index_heads, self.index_dim)), keys=self.index_keys[layer],
                        weights=self.head_weights, step=self.step, max_tokens=self.index_keys[layer].shape[0],
                        heads=self.index_heads, dim=self.index_dim, scale=_f32(self.index_dim ** -0.5, 'scale'))
        elif self.graph and self.index_kernel == 'tc' and self.index_heads <= 32 and self.index_dim in (64, 128):
            self._spawn('index_scores_tc_at', out=self.scores, q=self.iq_rot.view((self.index_heads, self.index_dim)),
                        keys=self.index_keys[layer], weights=self.head_weights, step=self.step,
                        max_tokens=self.index_keys[layer].shape[0], heads=self.index_heads, dim=self.index_dim,
                        scale=_f32(self.index_dim ** -0.5, 'scale'))
        elif self.graph:
            self._launch('index_scores_at', (self.max_tokens + 3) // 4, 128,
                         [self.scores, self.iq_rot, self.index_keys[layer], self.head_weights, self.step,
                          I(self.index_heads), I(self.index_dim), scale])
        else:
            self._launch('index_scores', (tokens + 3) // 4, 128,
                         [self.scores, self.iq_rot, self.index_keys[layer], self.head_weights, I(tokens), I(self.index_heads),
                          I(self.index_dim), scale])
        self._select(tokens)

    def _multi_block_topk(self, tokens):
        """Whether a step of `tokens` selects with the multi-block top-k (long_topk='ilp_hist' past
        topk_blocks_from)."""
        return self._mb is not None and tokens > self.topk_blocks_from

    def _select(self, tokens):
        """selected[:min(index_topk, tokens)] = the indexer's top-k of scores[:tokens], stable by position."""
        count = min(self.index_topk, tokens)
        if self.long_topk in ('position', 'warp', 'ilp', 'ilp_hist') and self.device_topk:
            # One block: the threshold by radix select in shared memory, then the set in position
            # order (#32). Under replay it reads tokens from the step buffer; any tokens, any k.
            if self.graph and self.long_topk == 'ilp_hist' and self._hist_counted and self._multi_block_topk(tokens):
                mb, size = self._mb, dict(max_tokens=self.scores.shape[0], k=self.index_topk)
                self._spawn('index_mma/topk_mb_list_at', wcnt=mb['wcnt'], ckey=mb['ckey'], cpos=mb['cpos'],
                            count=mb['count'], offs=mb['offs'], sel=mb['sel'],
                            scores=self.scores.view(self.scores.shape, 'int32'), hist=self.score_bins, step=self.step,
                            arrivals=mb['list_arrivals'], **size)
                self._spawn('index_mma/topk_mb_write_at', out_ids=self.selected,
                            scores=self.scores.view(self.scores.shape, 'int32'), hist=self.score_bins, offs=mb['offs'],
                            sel=mb['sel'], step=self.step, arrivals=mb['write_arrivals'], **size)
            elif self.graph and self.long_topk == 'ilp_hist' and self._hist_counted:
                self._spawn('index_mma/topk_select_ilp_hist_at', out_ids=self.selected,
                            scores=self.scores.view(self.scores.shape, 'int32'), hist=self.score_bins, step=self.step,
                            k=self.index_topk)
            elif self.graph and self.long_topk in ('ilp', 'ilp_hist'):
                self._spawn('index_mma/topk_select_ilp_at', out_ids=self.selected,
                            scores=self.scores.view(self.scores.shape, 'int32'), step=self.step, k=self.index_topk)
            elif self.graph and self.long_topk == 'warp':
                self._spawn('index_mma/topk_select_warp_at', out_ids=self.selected,
                            scores=self.scores.view(self.scores.shape, 'int32'), step=self.step, k=self.index_topk)
            elif self.graph:
                self._spawn('topk_select_at', out_ids=self.selected, scores=self.scores.view(self.scores.shape, 'int32'),
                            step=self.step, k=self.index_topk)
            else:
                self._spawn('topk_select', out_ids=self.selected, scores=self.scores.view((tokens,), 'int32'),
                            tokens=tokens, k=self.index_topk)
            self.selected_count = count
            return
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
            if self.long_topk in ('position', 'warp', 'ilp', 'ilp_hist'):
                chosen = np.sort(chosen)
            self.selected.view((count,)).upload(chosen)
        self.selected_count = count

    def _fp8_absorb(self, prefix):
        """With fp8_kernels='cvt' and a block-FP8 kv_b_proj (128 x 128 blocks, rank a multiple of 128): its
        codes (as uint32 words) and scales as stored, uploaded once, for both MLA absorbs (#42). Else None
        (the float32 matvec_heads path). kv_b_proj's rows are nope + value per head (GLM-5.3: 192 + 256 =
        448), so a block of scales can cover two heads; both kernels read each row's own."""
        name = prefix + '.kv_b_proj'
        key = name + ':fp8_absorb'
        if key in self._weights:
            return self._weights[key]
        h, nope, value, rank = self.heads, self.nope, self.value_dim, self.rank
        entry = None
        if self.fp8_kernels == 'cvt' and self._is_fp8(name) and rank % 128 == 0 and value % 2 == 0:
            codes, scales, block, shape = self.model.weights.block_fp8(name)
            if tuple(block) == (128, 128) and tuple(shape) == (h * (nope + value), rank):
                codes = np.ascontiguousarray(np.asarray(codes, np.uint8).reshape(-1))
                scales = np.ascontiguousarray(np.asarray(scales, np.float32).reshape(-1))
                entry = (self._upload_weight(key + ':codes', codes, 'uint8').view((codes.size // 4,), 'uint32'),
                         self._upload_weight(key + ':scales', scales, 'float32'))
        self._weights[key] = entry
        return entry

    def _kv_nope_t(self, prefix):
        kv_name = prefix + '.kv_b_proj.weight'
        kv_t = self._weights.get(kv_name + ':nope_t')
        if kv_t is None:
            kv = np.asarray(self.model._weight(kv_name), dtype=np.float32).reshape(self.heads, self.nope + self.value_dim, self.rank)
            kv_t = self._upload_weight(kv_name + ':nope_t', np.ascontiguousarray(kv[:, :self.nope, :].transpose(0, 2, 1)))
        return kv_t

    def _attention(self, layer, x, position, shared_ok, reuse_selection=False, defer_sum=False):
        prefix = f'model.layers.{layer}.self_attn'
        bias = bool(self.config.get('attention_bias', False))
        full = self.model.indexer_type(layer) == 'full' and not reuse_selection
        stacked = not bias and self._fma_qkv_a(prefix, x, full)
        if not stacked:
            self._linear(prefix + '.q_a_proj', x, self.q_a, bias)
        if self.fuse_glue:
            # Both norms in one launch of two blocks (#39): kv_a's projection first.
            if not stacked:
                self._linear(prefix + '.kv_a_proj_with_mqa', x, self.compressed, bias)
            latent_row = self._new_row(self.latents[layer], position, self.rank, self.latent_new)
            self._launch('rmsnorm_pair', 2, self.norm_block,
                         [self.q_res, self.q_a, self._weight(prefix + '.q_a_layernorm.weight'), I(self.q_a.shape[0]),
                          latent_row, self.compressed.view((self.rank,)), self._weight(prefix + '.kv_a_layernorm.weight'),
                          I(self.rank), F(1e-6)])
        else:
            self._rmsnorm(self.q_a, self._weight(prefix + '.q_a_layernorm.weight'), self.q_res, 1e-6)
        self._linear(prefix + '.q_b_proj', self.q_res, self.query)
        if not stacked and not self.fuse_glue:
            self._linear(prefix + '.kv_a_proj_with_mqa', x, self.compressed, bias)
        if full:
            self._index_project(prefix + '.indexer', x, self.q_res, self.iq, self.ik, self.head_weights_raw,
                                key_done=stacked, key_only=position + 1 <= self.index_topk)
        self._attention_core(layer, position, shared_ok, self.query, self.compressed,
                             (self.iq, self.ik, self.head_weights_raw) if full else None, kv_normed=self.fuse_glue)
        self._linear(prefix + '.o_proj', self.heads_out, self.attn_out, bias)
        if not defer_sum:  # else the caller sums it with the residual RMSNorm after it (fuse_tp_norm)
            self._tp_sum(self.attn_out)

    def _attention_core(self, layer, position, shared_ok, query, compressed, index, kv_normed=False):
        """One token's attention after its projections: its cache rows, the indexer's top-k (index:
        its (queries, key, head weights), for a full indexer), MLA over the selection, and the
        per-head values in self.heads_out. The caller runs o_proj. kv_normed: the latent row already
        holds kv_a's RMSNorm (fuse_glue's rmsnorm_pair)."""
        prefix = f'model.layers.{layer}.self_attn'
        qk = self.nope + self.rot
        latent_row = self._new_row(self.latents[layer], position, self.rank, self.latent_new)
        if not kv_normed:
            self._rmsnorm(compressed.view((self.rank,)), self._weight(prefix + '.kv_a_layernorm.weight'), latent_row, 1e-6)
        rope_x = compressed.view((self.rot,), offset=self.rank * 4)
        if self.fuse_glue and self.graph and self.latents[layer].dtype == 'uint16' and self.rope_keys[layer].dtype == 'uint16':
            # mla_kernel='tc''s bf16 rows, stored and RoPE'd in one launch (#39).
            self._spawn('kv_rows_bf16_at', latents=self.latents[layer], rope_keys=self.rope_keys[layer],
                        latent_row=latent_row, rope_x=rope_x, cosine=self.cos, sine=self.sin, step=self.step,
                        rank=self.rank, rot=self.rot)
        else:
            self._store_row(self.latents[layer], latent_row, self.rank)
            rope_row = self._new_row(self.rope_keys[layer], position, self.rot, self.rope_new)
            self._rope(rope_row, rope_x, 1, self.rot, self.rot, self.rot)
            self._store_row(self.rope_keys[layer], rope_row, self.rot)
        if index is not None:
            self._index_core(layer, prefix + '.indexer', position, *index)
        elif not shared_ok:
            raise ValueError('Shared DSA layer has no preceding full indexer selection')

        # Absorbed MLA. q_latent[h] = W_k[h]^T q_nope[h]; the transposed nope blocks are built once.
        warps = 4
        absorb = self._fp8_absorb(prefix)
        fuse_q = (self.fuse_mla_q and self.fused_attention and self.graph and self.mla_kernel == 'tc'
                  and self.rank == 512 and self.rot == 64)
        if absorb is not None:
            # kv_b_proj's FP8 codes, laid out once per head (#42): a quarter of the float32 copy's bytes.
            codes, scales = absorb
            self._spawn('fp8_matvec_cols_cvt' + self._cvt_suffix, out=self.latent_q.view((self.heads * self.rank,)),
                        weight=codes, scales=scales, x=query.view((self.heads * qk,)), groups=self.heads,
                        cols=self.rank, depth=self.nope, period=self.nope + self.value_dim, offset=0, x_stride=qk)
        else:
            kv_t = self._kv_nope_t(prefix)
            self._launch('matvec_heads', (self.heads * self.rank + warps - 1) // warps, 128,
                         [self.latent_q, kv_t, query, I(self.heads), I(self.rank), I(self.nope),
                          I(self.rank * self.nope), I(qk), I(self.rank)])
        if not fuse_q:
            self._rope(self.rot_q, query.view((self.heads * qk - self.nope,), offset=self.nope * 4),
                       self.heads, self.rot, qk, self.rot)
        count = self.selected_count
        splits = self._splits(count)
        block = ((self.rank + 31) // 32) * 32
        scale = F(_f32(qk ** -0.5, 'MLA scale', True))
        tokens = position + 1
        if self.fused_attention and self.graph and self.mla_kernel == 'tc':
            groups = self.mla_heads // 16
            numerator = self.numerator.view((groups, self.splits_max, 16 * self.rank))
            stats = self.stats.view((groups, self.splits_max, 32))
            if fuse_q:
                self._spawn('mla_q_rope_bf16_at', q_t=self.q_t, qr_t=self.qr_t, q_latent=self.latent_q,
                            x=query.view((self.heads * qk,)), cosine=self.cos, sine=self.sin, heads=self.mla_heads,
                            real=self.heads, rank=self.rank, rope_dim=self.rot, x_stride=qk, rope_at=self.nope)
            else:
                self._spawn('mla_q_bf16_at', q_t=self.q_t, qr_t=self.qr_t, q_latent=self.latent_q, q_rope=self.rot_q,
                            heads=self.mla_heads, rank=self.rank, rope_dim=self.rot)
            self._spawn('mla_tc_at', numerator=numerator.view((groups * self.splits_max * 16, self.rank)),
                        stats=stats.view((groups * self.splits_max, 32)), q_t=self.q_t.view((groups * 1024, 8)),
                        qr_t=self.qr_t.view((groups * 128, 8)), cache_latent=self.latents[layer],
                        cache_rope=self.rope_keys[layer], selected=self.selected, step=self.step,
                        heads=self.mla_heads, rank=self.rank, rope_dim=self.rot, selection_stride=0, splits_max=self.splits_max,
                        scale=scale.value)
            self._spawn('mla_flash_merge_at', out=self.attended, numerator=numerator, stats=stats, step=self.step,
                        heads=self.mla_heads, rank=self.rank, splits_max=self.splits_max)
        elif self.fused_attention and self.graph and self.mla_kernel == 'flash':
            # The per-head buffers viewed per group of 16 heads (mla_flash_at.vx).
            groups = self.mla_heads // 16
            numerator = self.numerator.view((groups, self.splits_max, 16 * self.rank))
            stats = self.stats.view((groups, self.splits_max, 32))
            self._spawn('mla_flash_at', numerator=numerator, stats=stats, q_latent=self.latent_q, q_rope=self.rot_q,
                        cache_latent=self.latents[layer], cache_rope=self.rope_keys[layer], selected=self.selected,
                        step=self.step, heads=self.mla_heads, rank=self.rank, rope_dim=self.rot, selection_stride=0,
                        splits_max=self.splits_max, scale=scale.value)
            self._spawn('mla_flash_merge_at', out=self.attended, numerator=numerator, stats=stats, step=self.step,
                        heads=self.mla_heads, rank=self.rank, splits_max=self.splits_max)
        elif self.fused_attention and self.graph and self.mla_kernel == 'heads':
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
        if absorb is not None:
            codes, scales = absorb
            self._spawn('fp8_matvec_heads_cvt' + self._cvt_suffix, out=self.heads_out.view((self.heads * self.value_dim,)),
                        weight=codes, scales=scales, x=self.attended.view((self.heads * self.rank,)),
                        rows=self.heads * self.value_dim, cols=self.rank, group=self.value_dim, x_stride=self.rank,
                        period=self.nope + self.value_dim, offset=self.nope)
            return
        kv_b = self._weight(prefix + '.kv_b_proj.weight')
        value_w = kv_b.view((self.heads * (self.nope + self.value_dim) - self.nope, self.rank), offset=self.nope * self.rank * 4)
        self._launch('matvec_heads', (self.heads * self.value_dim + warps - 1) // warps, 128,
                     [self.heads_out, value_w, self.attended, I(self.heads), I(self.value_dim), I(self.rank),
                      I((self.nope + self.value_dim) * self.rank), I(self.rank), I(self.value_dim)])

    def _route(self, prefix, x):
        """The MoE layer's router on x: route_ids and route_weights (slots 0..k-1), router_logits."""
        c = self.config
        experts, k = int(c['n_routed_experts']), int(c['num_experts_per_tok'])
        bias = self._weight(prefix + '.gate.e_score_correction_bias')
        scale = _f32(float(c.get('routed_scaling_factor', 1.0)), 'router scale')
        if self.fuse_router and self._format(prefix + '.gate') == 'bf16':
            bits, (rows, cols) = self._bf16_weight(prefix + '.gate')
            if rows == experts and cols % 8 == 0:
                self._check_shape(prefix + '.gate', 'bf16', rows, cols, x, self.router_logits)
                self._spawn('router_bf16', indices=self.route_ids, weights=self.route_weights, logits=self.router_logits,
                            gate=bits.view((rows * cols // 2,), 'uint32'), x=x, bias=bias, experts=experts, cols=cols,
                            k=k, scale=scale, arrivals=self.router_arrivals)
                return
        self._project(prefix + '.gate', x, self.router_logits)
        self._launch('router', 1, 32, [self.route_ids, self.route_weights, self.router_logits, bias, I(experts), I(k),
                                       F(scale)])

    def _feed_forward(self, layer, x):
        """The layer's MLP or MoE of x; returns the buffer that holds it: ffn_out, or with fuse_glue a
        MoE layer's moe_sum (no copy)."""
        m, c = self.model, self.config
        prefix = f'model.layers.{layer}.mlp'
        if m.mlp_type(layer) == 'dense':
            self._mlp(prefix, x, self.ffn_out)
            return self.ffn_out
        experts = int(c['n_routed_experts'])
        k = int(c['num_experts_per_tok'])
        self._route(prefix, x)
        # Reduction order of the pinned official Experts: ascending expert id, whatever the top-k order.
        if self.device_experts:
            self._device_moe(prefix, x, k)
        else:
            self._copy(self.moe_sum, self.zeros, self.hidden)
            self._host_experts(prefix, x, k)
        if int(c.get('n_shared_experts', 1)) > 0 and not self._folds_shared(prefix):
            self._mlp(prefix + '.shared_experts', x, self.mlp_out)
            self._add(self.moe_sum, self.moe_sum, self.mlp_out, self.hidden)
        if self.fuse_glue:
            return self.moe_sum
        self._copy(self.ffn_out, self.moe_sum, self.hidden)
        return self.ffn_out

    def _host_experts(self, prefix, x, k):
        """The routed experts with the ids read back to the host to pick each expert's weights."""
        ids = np.frombuffer(self.route_ids.download(), dtype=np.int32)
        for slot in np.argsort(ids, kind='stable'):
            self._expert(prefix, int(ids[slot]), x, self.expert_out)
            self._launch('scaled_add', self._grid(self.hidden), _ELEMENTWISE_BLOCK,
                         [self.moe_sum, self.expert_out, self.route_weights, I(int(slot)), I(self.hidden)])

    # ---------------------------------------------------------------- public
    def forward(self, token_id, *, logits=True):
        """Append one token and return next-token logits (None with logits=False). logits='greedy'
        returns the next token instead, numpy.argmax of the logits when none is NaN, chosen on the
        device: under tensor parallelism each rank takes the largest of its vocabulary rows
        (gpu/spawn/argmax_pair) and the ranks gather one pair each, in place of every logit; on one GPU,
        argmax_rows. A largest logit that is not finite raises ValueError."""
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
        self._device_position = None
        embedding = np.ascontiguousarray(m._embedding(int(token_id)), dtype=np.float32)
        cosine, sine = _cached_rope_coefficients(self.rot, float(position), float(m.theta))
        cosine, sine = (np.ascontiguousarray(v, dtype=np.float32) for v in (cosine, sine))
        if self.graph:
            count = min(self.index_topk, position + 1)
            short = position + 1 <= self.index_topk  # every cached token selected: no scores or top-k (#43)
            if self._multi_block_topk(position + 1):
                short = 'blocks'  # its own graph: the multi-block top-k's launches
            self.step_inputs.upload(np.concatenate([embedding, cosine, sine, self._step_buffer(position, count)]))
            graph = self._step_graphs.get(short)
            if graph is not None:
                graph.launch(self.stream)
                self.replays += 1
                self.selected_count = count
            elif short not in self._step_warm or not self._capture:
                self._layers(position)  # uploads every weight and loads every kernel the capture needs
                self._step_warm.add(short)
            else:
                with self.stream.capture() as graph:
                    self._layers(position)
                self._step_graphs[short] = graph
                graph.launch(self.stream)
                self.replays += 1
        else:
            self.x.upload(embedding)
            self.cos.upload(cosine)
            self.sin.upload(sine)
            self._layers(position)
        self.position += 1
        return self._head(logits)

    def _head(self, logits):
        """forward()'s output for the token whose final residual self.x holds: None with logits=False, the
        next token with 'greedy', else the logits."""
        m, c = self.model, self.config
        if not logits:
            return None
        self._rmsnorm(self.x, self._weight('model.norm.weight'), self.norm, m.eps)
        head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
        greedy = logits == 'greedy'
        if self.tp is None:
            self._project(head, self.norm, self.logits)
            if greedy:
                token = int(self._argmax(self.logits.view((1, self.logits.shape[0])), 1)[0])
                if not np.isfinite(np.frombuffer(self.logits.download(), dtype=np.float32)[token]):
                    raise ValueError('nonfinite logits')
                return token
        else:
            self._project(head, self.norm, self.logits_local)
            if greedy:
                rows = self.logits_local.shape[0]
                self._spawn('argmax_pair', out=self._greedy_pair, x=self.logits_local, dim=rows,
                            offset=self.tp.rank * rows, threads=1024)
                self.tp.collective.all_gather(self._greedy_pair, self._greedy_pairs, stream=self.stream)
                pairs = np.frombuffer(self._greedy_pairs.download(), dtype=np.float32).reshape(self.tp.size, 2)
                best = int(np.argmax(pairs[:, 0]))  # the first largest: the lower rank, so the lower id, on a tie
                if not np.isfinite(pairs[best, 0]):
                    raise ValueError('nonfinite logits')
                return int(pairs[best, 1:].view(np.int32)[0])
            self.tp.collective.all_gather(self.logits_local, self.logits, stream=self.stream)
        return np.frombuffer(self.logits.download(), dtype=np.float32).copy()

    DEVICE_SLOTS = 4  # device_step tickets that can be outstanding: the host slots their tokens land in

    def device_step(self, token_id=None, *, readback=True):
        """One greedy decode step (device_decode=True) launched without waiting for anything: its inputs
        come from the device state [token, position], which the step before left (decode_next) or, given
        token_id, uploaded here; its token is chosen on the device and left in that state for the next.
        Returns a ticket for device_token(); with readback=False (a tensor-parallel follower, which never
        reads the token) None. The token is forward(token_id, logits='greedy')'s: the same kernels on the
        same inputs."""
        m, c = self.model, self.config
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        if not self.device_decode:
            raise RuntimeError('device_step needs device_decode=True')
        position = self.position
        if position >= self.max_tokens:
            raise ValueError(f'sequence exceeds max_tokens={self.max_tokens}')
        if position >= int(c.get('max_position_embeddings', 2 ** 31 - 1)):
            raise ValueError('Request exceeds max_position_embeddings')
        d = self._device_setup()
        if token_id is None:
            if self._device_position != position:
                raise RuntimeError('device_step without a token must follow a device_step')
        else:
            if not isinstance(token_id, (int, np.integer)) or not 0 <= int(token_id) < int(c['vocab_size']):
                raise ValueError('token_id must be an integer in the vocabulary')
            d['state'].upload(np.array([int(token_id), position], np.int32))
        count = min(self.index_topk, position + 1)
        short = position + 1 <= self.index_topk  # as forward(): every cached token selected (#43)
        if self._multi_block_topk(position + 1):
            short = 'blocks'
        body = lambda: self._device_body(position)
        if self._capture:
            self._replay(('device', short), body)
        else:
            body()
        self.selected_count = count
        self.position += 1
        self._device_position = self.position
        if not readback:
            return None
        number = d['tickets']
        d['tickets'] += 1
        slot = number % self.DEVICE_SLOTS
        d['state'].download_async(d['host'], stream=self.stream, offset=8 * slot)
        d['events'][slot].record(self.stream)
        return number, position

    def device_token(self, ticket):
        """The token of device_step's `ticket`, waiting for that step only. A largest logit that is not
        finite raises ValueError, as forward(logits='greedy') does."""
        number, position = ticket
        d = self._device
        if d is None or not d['tickets'] - self.DEVICE_SLOTS <= number < d['tickets']:
            raise RuntimeError(f'device_step ticket {number} is no longer held')
        slot = number % self.DEVICE_SLOTS
        d['events'][slot].synchronize()
        token, after = (int(v) for v in np.frombuffer(d['host'].read(8 * slot, 8), dtype=np.int32))
        if after != position + 1:
            raise RuntimeError(f'device_step at {position} left position {after}')
        if token < 0:
            raise ValueError('nonfinite logits')
        return token

    def _device_setup(self):
        """device_step's buffers, made once: the embedding table on the device (its bfloat16 bits as the
        checkpoint stores them, else float32), every position's cos, sin and step buffer (the host's
        values, as forward() uploads them), the state [token, position], and the host slots the state is
        read back into."""
        if self._device is not None:
            return self._device
        m, c = self.model, self.config
        if c.get('tie_word_embeddings', False) and self.tp is not None:
            raise NotImplementedError('device_decode under tensor parallelism needs an embedding not tied to lm_head')
        t = self._tensor
        if self._is_bf16('model.embed_tokens'):
            embed16, _ = self._bf16_weight('model.embed_tokens')
            embed32, wide = t((1,), 'int32'), 0
        else:
            table = np.ascontiguousarray(m._weight('model.embed_tokens.weight'), dtype=np.float32)
            embed16, wide = t((1,), 'uint16'), 1
            embed32 = self._upload_weight('model.embed_tokens:decode', table.view(np.int32), 'int32')
        rows = []
        for position in range(self.max_tokens):
            cosine, sine = self._rope_inputs(position)
            rows.append(np.concatenate([cosine, sine, self._step_buffer(position, min(self.index_topk, position + 1))]))
        rows = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
        if self.tp is None:
            self._greedy_pair = t((2,))
        self._device = dict(
            embed16=embed16, embed32=embed32, wide=wide, width=rows.shape[1],
            positions=t(rows.shape, 'int32', rows.view(np.int32).tobytes()), state=t((2,), 'int32'),
            host=self.ctx.host_buffer(8 * self.DEVICE_SLOTS),
            events=[self.ctx.event(timing=False) for _ in range(self.DEVICE_SLOTS)], tickets=0)
        return self._device

    def _device_body(self, position):
        """device_step's launches, what its graph holds: decode_inputs, the layers, the final norm and
        lm_head, argmax_pair (each rank over its vocabulary rows; the pairs gathered), decode_next."""
        m, c, d = self.model, self.config, self._device
        vocab = int(c['vocab_size'])
        self._spawn('decode_inputs', inputs=self.step_inputs, embed16=d['embed16'], embed32=d['embed32'],
                    table=d['positions'].view((d['positions'].shape[0] * d['width'],), 'int32'), state=d['state'],
                    hidden=self.hidden, width=d['width'], vocab=vocab, positions=self.max_tokens, wide=d['wide'])
        self._layers(position)
        self._rmsnorm(self.x, self._weight('model.norm.weight'), self.norm, m.eps)
        head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
        if self.tp is None:
            self._project(head, self.norm, self.logits)
            self._spawn('argmax_pair', out=self._greedy_pair, x=self.logits, dim=vocab, offset=0, threads=1024)
            pairs, ranks = self._greedy_pair, 1
        else:
            self._project(head, self.norm, self.logits_local)
            rows = self.logits_local.shape[0]
            self._spawn('argmax_pair', out=self._greedy_pair, x=self.logits_local, dim=rows,
                        offset=self.tp.rank * rows, threads=1024)
            self.tp.collective.all_gather(self._greedy_pair, self._greedy_pairs, stream=self.stream)
            pairs, ranks = self._greedy_pairs, self.tp.size
        self._spawn('decode_next', state=d['state'], pairs=pairs.view((2 * ranks,), 'int32'), ranks=ranks)

    def _splits(self, count):
        """The MLA splits of a step that selects `count` tokens: about 8 tokens a split for mla_flash_at,
        32 (one chunk) for mla_tc_at, 256 for the others, at most splits_max."""
        per = {'flash': 8, 'tc': 32}.get(self.mla_kernel, 256)
        return min(self.splits_max, max(1, (count + per - 1) // per))

    def _step_buffer(self, position, count):
        """The step buffer the *_at kernels read, [position, tokens, count, splits], as float32 bits."""
        splits = self._splits(count)
        return np.array([position, position + 1, count, splits], np.int32).view(np.float32)

    def _rope_inputs(self, position):
        cosine, sine = _cached_rope_coefficients(self.rot, float(position), float(self.model.theta))
        return [np.ascontiguousarray(v, dtype=np.float32) for v in (cosine, sine)]

    def _replay(self, key, body):
        """With graph=True, body's launches as the graph `key`: run eagerly the first time (every weight
        uploaded, every kernel and scratch buffer made), captured and launched the second, replayed
        after. Everything that changes per call comes from the inputs uploaded before."""
        if key in self._graphs:
            self._graphs[key].launch(self.stream)
            self.replays += 1
        elif key not in self._warmed:
            body()
            self._warmed.add(key)
        else:
            with self.stream.capture() as graph:
                body()
            self._graphs[key] = graph
            graph.launch(self.stream)
            self.replays += 1

    def _layers(self, position):
        """Every layer of one decode step, from self.x (the embedding row) to the final residual
        in self.x; with graph=True, what the step graph holds."""
        n = self.model.n_layers
        fuse = self._fuse_norms()
        for layer in range(n):
            # fuse_glue: a layer's closing add takes the next layer's input RMSNorm along (#39).
            following = f'model.layers.{layer + 1}.input_layernorm.weight' if fuse and layer + 1 < n else None
            self._layer(layer, position, layer > 0, normed=fuse and layer > 0, next_norm=following)

    def _fuse_norms(self):
        """Whether fuse_glue's residual_rmsnorm8 holds the hidden row: eight values a thread."""
        return self.fuse_glue and self.hidden <= 8 * self.norm_block

    def _residual_rmsnorm8(self, x, residual, weight, out, residual_out, eps):
        """_residual_rmsnorm's bits (and _add then _rmsnorm's) with the row in registers (#39)."""
        n = x.shape[0]
        self._launch('residual_rmsnorm8', 1, self.norm_block,
                     [out, residual_out, x, residual, weight, I(n), F(_f32(eps, 'epsilon', True))])

    def _layer(self, layer, position, have_selection, reuse_selection=False, *, normed=False, next_norm=None):
        """One decoder layer from self.x to self.x: attention and the feed-forward, each with its
        norm and residual. reuse_selection: attend to self.selected without running a full indexer.
        normed: self.norm already holds this layer's input RMSNorm of self.x; next_norm: the next
        layer's input RMSNorm weight, applied with the closing add (fuse_glue, #39)."""
        m = self.model
        prefix = f'model.layers.{layer}'
        if not normed:
            self._rmsnorm(self.x, self._weight(prefix + '.input_layernorm.weight'), self.norm, m.eps)
        fused = self._fused_tp_norm()
        self._attention(layer, self.norm, position, have_selection or reuse_selection, reuse_selection, defer_sum=fused)
        post = (self._tp_sum_rmsnorm if fused else self._residual_rmsnorm8 if self._fuse_norms()
                else self._residual_rmsnorm)
        post(self.attn_out, self.x, self._weight(prefix + '.post_attention_layernorm.weight'), self.norm, self.x_next, m.eps)
        self.x, self.x_next = self.x_next, self.x
        out = self._feed_forward(layer, self.norm)
        if next_norm is not None and fused:
            self._tp_sum_rmsnorm(out, self.x, self._weight(next_norm), self.norm, self.x_next, m.eps)
        else:
            self._tp_sum(out)
            if next_norm is not None:
                self._residual_rmsnorm8(self.x, out, self._weight(next_norm), self.norm, self.x_next, m.eps)
            else:
                self._add(self.x_next, self.x, out, self.hidden)
        self.x, self.x_next = self.x_next, self.x

    def _fused_tp_norm(self):
        """Whether a layer's cross-rank sums go through tp_sum_rmsnorm with the RMSNorm after them."""
        return bool(self.fuse_tp_norm) and getattr(self, '_tp_peers', None) is not None and self._fuse_norms()

    def _tp_sum_rmsnorm(self, partial, residual, weight, out, residual_out, eps):
        """_tp_sum(partial), then _residual_rmsnorm8(partial, residual, weight, out, residual_out, eps), in one
        launch with their bits (a + b and b + a are the same float); partial is left unsummed."""
        self._spawn('tp_sum_rmsnorm', **self._tp_peers.norm_arguments(partial, residual, weight, out, residual_out,
                                                                      _f32(eps, 'epsilon', True)))

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
        if planned is None and self.device_decode:
            self._device_setup()
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

        for layer in range(m.n_layers + (1 if self.mtp else 0)):
            p = f'model.layers.{layer}'
            if layer == m.n_layers:  # the MTP layer's own inputs and head norm
                linear(p + '.eh_proj')
                for n in ('enorm', 'hnorm', 'shared_head.norm'):
                    self._weight(f'{p}.{n}.weight')
            self._weight(p + '.input_layernorm.weight')
            self._weight(p + '.post_attention_layernorm.weight')
            a = p + '.self_attn'
            for n in ('q_a_proj', 'kv_a_proj_with_mqa', 'o_proj'):
                linear(f'{a}.{n}', bias)
            linear(a + '.q_b_proj')
            self._weight(a + '.q_a_layernorm.weight')
            self._weight(a + '.kv_a_layernorm.weight')
            if self._fp8_absorb(a) is None:  # else kv_b_proj's FP8 codes, laid out per head (#42)
                self._kv_nope_t(a)
                self._weight(a + '.kv_b_proj.weight')
            if m.indexer_type(layer) == 'full':
                for n in ('wq_b', 'wk', 'weights_proj'):
                    linear(f'{a}.indexer.{n}')
                self._weight(a + '.indexer.k_norm.weight')
                self._weight(a + '.indexer.k_norm.bias')
            f = p + '.mlp'
            if m.mlp_type(layer) == 'dense':
                mlp(f)
                continue
            linear(f + '.gate')
            self._weight(f + '.gate.e_score_correction_bias')
            if self.device_experts:
                fused = self._fused_experts(f + '.experts.0')
                for n in (('gate_up', 'down') if fused else ('gate_proj', 'up_proj', 'down_proj')):
                    self._stacked_experts(f, n, shared=self._folds_shared(f))
            for e in range(0 if self.device_experts else int(c['n_routed_experts'])):
                if self._fused_experts(f'{f}.experts.{e}'):
                    self._weight(f + '.experts.gate_up_proj')
                    self._weight(f + '.experts.down_proj')
                else:
                    mlp(f'{f}.experts.{e}')
            if int(c.get('n_shared_experts', 1)) > 0 and not self._folds_shared(f):
                mlp(f + '.shared_experts')
        self._weight('model.norm.weight')
        linear('model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head')
        if self.device_topk:
            self._topk_kernel()

    def forward_hidden(self, token_id):
        """forward(), and the final hidden state after model.norm: (logits, hidden), MTP's input."""
        logits = self.forward(token_id)
        return logits, np.frombuffer(self.norm.download(), dtype=np.float32).copy()

    def mtp_step(self, token_id, hidden, position, *, reuse_selection=False, greedy=False):
        """The MTP layer (mtp=True; glm_vx/speculative.py's MTPDrafter.step on the device): from the
        token at position + 1 and the hidden state at position, its logits for the token at
        position + 2 and its own hidden state, (logits, hidden). It writes its cache rows at
        `position`, so stepping again at a position replaces what was there; reuse_selection
        attends to the positions the last step selected (index_share_for_mtp_iteration).

        hidden is an array, or a device row a step returned with greedy=True (copied on the
        device). greedy=True returns (token, hidden): the argmax of the logits, chosen on the device
        (gpu/spawn's argmax_rows, numpy.argmax's token), and the hidden state as a device row, valid
        until the next step; only the token is downloaded.

        With graph=True the step is a graph of its own, one for each reuse_selection: one upload
        of [embedding, cos, sin, step buffer, hidden], one launch, then the head."""
        m, c = self.model, self.config
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        if not self.mtp:
            raise ValueError('mtp_step needs ResidentGLM(..., mtp=True)')
        if not 0 <= position < self.max_tokens or not 0 <= int(token_id) < int(c['vocab_size']):
            raise ValueError('mtp_step: position or token out of range')
        embedding = np.zeros(self.hidden, np.float32) if position == 0 else m._embedding(int(token_id))
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)
        on_device = hasattr(hidden, 'download')  # a device row from a greedy step
        if on_device:
            if tuple(hidden.shape) != (self.hidden,):
                raise ValueError(f'mtp_step: a device hidden state is [{self.hidden}], got {list(hidden.shape)}')
            device_hidden, hidden = hidden, np.zeros(self.hidden, np.float32)
        hidden = np.ascontiguousarray(hidden, dtype=np.float32)
        cosine, sine = self._rope_inputs(position)
        reuse_selection = bool(reuse_selection)
        if self.graph:
            # Reusing, the step attends to the last selection: its count, not this position's.
            count = self.selected_count if reuse_selection else min(self.index_topk, position + 1)
            inputs = [embedding, cosine, sine, self._step_buffer(position, count)]
            if on_device:
                self.step_inputs.upload(np.concatenate(inputs))
                self._copy(self.mtp_hidden, device_hidden, self.hidden)
            else:
                self._inputs.upload(np.concatenate(inputs + [hidden]))
            short = None if reuse_selection else position + 1 <= self.index_topk  # #43
            self._replay(('mtp', reuse_selection, short), lambda: self._mtp_layer(position, reuse_selection))
            self.selected_count = count
        else:
            self.x.upload(embedding)
            if on_device:
                self._copy(self.mtp_hidden, device_hidden, self.hidden)
            else:
                self.mtp_hidden.upload(hidden)
            self.cos.upload(cosine)
            self.sin.upload(sine)
            self._mtp_layer(position, reuse_selection)
        prefix = f'model.layers.{m.n_layers}'
        self._rmsnorm(self.x, self._weight(prefix + '.shared_head.norm.weight'), self.norm, m.eps)
        head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
        self._project(head, self.norm, self.logits)
        if greedy:
            return int(self._argmax(self.logits.view((1, self.logits.shape[0])), 1)[0]), self.norm
        return (np.frombuffer(self.logits.download(), dtype=np.float32).copy(),
                np.frombuffer(self.norm.download(), dtype=np.float32).copy())

    def spawn_built(self, name):
        """Whether gpu/spawn kernel `name` is built in ptx_dir/spawn."""
        return (self._ptx_dir / 'spawn' / (name + '.ptx')).exists()

    def _argmax(self, logits, count):
        """numpy.argmax of each of the first `count` rows of the [rows, vocab] logits, on the device."""
        self._spawn('argmax_rows', out=self.token_ids, x=logits, rows=count, dim=logits.shape[1], threads=1024)
        return np.frombuffer(self.token_ids.download(), dtype=np.int32)[:count].copy()

    def _mtp_layer(self, position, reuse_selection):
        """From the embedding in self.x and the hidden state in self.mtp_hidden to the MTP layer's
        residual in self.x: eh_proj([enorm(embedding), hnorm(hidden)]), then its decoder layer."""
        m = self.model
        prefix = f'model.layers.{m.n_layers}'
        halves = [self.mtp_input.view((self.hidden,), offset=i * self.hidden * 4) for i in (0, 1)]
        self._rmsnorm(self.x, self._weight(prefix + '.enorm.weight'), halves[0], m.eps)
        self._rmsnorm(self.mtp_hidden, self._weight(prefix + '.hnorm.weight'), halves[1], m.eps)
        self._project(prefix + '.eh_proj', self.mtp_input, self.x)
        self._layer(m.n_layers, position, False, reuse_selection)

    def forward_many(self, token_ids, *, greedy=False):
        """Append one to four tokens as one step: (logits, hidden), each [len(token_ids), ...], the
        logits and the final hidden state (after model.norm) of every position, bit for bit what
        forward() gives one token at a time. MTP verification (glm_vx/speculative.py, #16).

        The projections run once for all the tokens: with fp8_kernels='x4', FP8 Linears through
        fp8_matvec4_rows, which reads each weight once for up to four vectors; other formats run
        their kernel per token. Everything else runs per token, in order, so each token's attention
        and indexer see the cache rows of the tokens before it, and each token keeps its own
        top-k for the shared-indexer layers.

        greedy=True returns (tokens, hiddens): each position's argmax token, chosen on the device
        (argmax_rows), and its hidden state as a device row (mtp_step takes it), valid until the
        next forward_many; only the tokens are downloaded.

        With graph=True the layers are one graph for each token count: one upload of the
        embedding rows and each token's cos, sin and step buffer, one launch, then the head."""
        m, c = self.model, self.config
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        if self.tp is not None:
            raise NotImplementedError('tensor parallelism runs forward() only so far')
        tokens = [int(t) for t in token_ids]
        if not 1 <= len(tokens) <= MANY_TOKENS or any(not 0 <= t < int(c['vocab_size']) for t in token_ids):
            raise ValueError(f'forward_many takes 1 to {MANY_TOKENS} tokens of the vocabulary')
        start, count = self.position, len(tokens)
        if start + count > self.max_tokens:
            raise ValueError(f'sequence exceeds max_tokens={self.max_tokens}')
        self._device_position = None
        if start + count > int(c.get('max_position_embeddings', 2 ** 31 - 1)):
            raise ValueError('Request exceeds max_position_embeddings')
        hidden = self.hidden
        x, norm = self._many('x', hidden), self._many('norm', hidden)
        embeddings = np.zeros((MANY_TOKENS, hidden), np.float32)
        for i, token in enumerate(tokens):
            embeddings[i] = m._embedding(token)
        if self.graph:
            steps = np.zeros((MANY_TOKENS, self._many_block), np.float32)
            for i in range(count):
                cosine, sine = self._rope_inputs(start + i)
                steps[i, :2 * len(cosine) + 4] = np.concatenate(
                    [cosine, sine, self._step_buffer(start + i, min(self.index_topk, start + i + 1))])
            self._many_inputs.upload(np.concatenate([embeddings.ravel(), steps.ravel()]))
            shorts = tuple(start + i + 1 <= self.index_topk for i in range(count))  # #43
            self._replay(('many', count, shorts), lambda: self._layers_many(start, count))
            self.selected_count = min(self.index_topk, start + count)
        else:
            x.upload(embeddings)
            self._layers_many(start, count)
        self.position += count
        for i in range(count):
            self._rmsnorm(self._many_row(x, i), self._weight('model.norm.weight'), self._many_row(norm, i), m.eps)
        vocab = int(c['vocab_size'])
        logits = self._many('logits', vocab)
        head = 'model.embed_tokens' if c.get('tie_word_embeddings', False) else 'lm_head'
        self._project_many(head, norm, logits, count)
        if greedy:
            return self._argmax(logits, count), [self._many_row(norm, i) for i in range(count)]
        rows = lambda t, width: np.frombuffer(t.download(), dtype=np.float32).reshape(MANY_TOKENS, width)[:count].copy()
        return rows(logits, vocab), rows(norm, hidden)

    def _layers_many(self, start, count):
        """Every layer for forward_many's tokens, from the embedding rows in x to the final residuals
        in x; with graph=True, what its graph holds."""
        m, hidden = self.model, self.hidden
        x, x_next, norm = self._many('x', hidden), self._many('x_next', hidden), self._many('norm', hidden)
        selected, counts = self.selected, [0] * count
        try:
            for layer in range(m.n_layers):
                prefix = f'model.layers.{layer}'
                for i in range(count):
                    self._rmsnorm(self._many_row(x, i), self._weight(prefix + '.input_layernorm.weight'),
                                  self._many_row(norm, i), m.eps)
                attended = self._attention_many(layer, norm, start, count, layer > 0, counts)
                for i in range(count):
                    self._add(self._many_row(x_next, i), self._many_row(x, i), self._many_row(attended, i), hidden)
                    self._rmsnorm(self._many_row(x_next, i), self._weight(prefix + '.post_attention_layernorm.weight'),
                                  self._many_row(norm, i), m.eps)
                ffn = self._feed_forward_many(layer, norm, count)
                for i in range(count):
                    self._add(self._many_row(x, i), self._many_row(x_next, i), self._many_row(ffn, i), hidden)
        finally:
            self.selected = selected
        self.selected_count = counts[-1]

    def _many(self, name, width):
        """A [MANY_TOKENS, width] scratch matrix of forward_many, allocated on first use."""
        buffers = self.__dict__.setdefault('_many_buffers', {})
        if (name, width) not in buffers:
            buffers[name, width] = self._tensor((MANY_TOKENS, width))
        return buffers[name, width]

    @staticmethod
    def _many_row(matrix, i):
        width = matrix.shape[1]
        return matrix.view((width,), offset=i * width * 4)

    def _project_many(self, prefix, x, out, count, bias=False):
        """out[i] = W x[i] for i < count, each row what _linear(prefix, x[i], out[i], bias) writes."""
        fmt = self._format(prefix)
        rows, cols = out.shape[1], x.shape[1]
        if fmt == 'fp8' and self.fp8_kernels in ('x4', 'fma', 'cvt', 'cublaslt') and cols % 16 == 0:
            codes, scales, shape = self._fp8_weight(prefix)
            if tuple(shape) != (rows, cols):
                raise ValueError(f'{prefix}: FP8 weight is {shape[0]}x{shape[1]}, got x [{cols}] -> out [{rows}]')
            self._spawn('fp8_matvec4_rows', out=out, weight=codes.view((rows * cols // 4,), 'uint32'),
                        scales=scales.view((int(np.prod(scales.shape)),)), x=x.view((MANY_TOKENS * cols,)),
                        rows=rows, cols=cols, m=count)
            if bias:
                for i in range(count):
                    self._add(self._many_row(out, i), self._many_row(out, i), self._weight(prefix + '.bias'), rows)
            return out
        for i in range(count):
            self._linear(prefix, self._many_row(x, i), self._many_row(out, i), bias)
        return out

    def _attention_many(self, layer, x, start, count, shared_ok, counts):
        """_attention for forward_many's tokens: o_proj's outputs, one row a token."""
        prefix = f'model.layers.{layer}.self_attn'
        bias = bool(self.config.get('attention_bias', False))
        q_rank, qk = int(self.config['q_lora_rank']), self.nope + self.rot
        q_a, q_res = self._many('q_a', q_rank), self._many('q_res', q_rank)
        query, compressed = self._many('query', self.heads * qk), self._many('compressed', self.rank + self.rot)
        self._project_many(prefix + '.q_a_proj', x, q_a, count, bias)
        for i in range(count):
            self._rmsnorm(self._many_row(q_a, i), self._weight(prefix + '.q_a_layernorm.weight'),
                          self._many_row(q_res, i), 1e-6)
        self._project_many(prefix + '.q_b_proj', q_res, query, count)
        self._project_many(prefix + '.kv_a_proj_with_mqa', x, compressed, count, bias)
        full = self.model.indexer_type(layer) == 'full'
        if full:
            iq = self._many('iq', self.index_heads * self.index_dim)
            ik, head_weights = self._many('ik', self.index_dim), self._many('head_weights', self.index_heads)
            self._project_many(prefix + '.indexer.wq_b', q_res, iq, count)
            self._project_many(prefix + '.indexer.wk', x, ik, count)
            self._project_many(prefix + '.indexer.weights_proj', x, head_weights, count)
        heads_out = self._many('heads_out', self.heads * self.value_dim)
        inputs = (self.cos, self.sin, self.step) if self.graph else None
        try:
            for i in range(count):
                position = start + i
                if self.graph:  # this token's cos, sin and step buffer, uploaded with the embeddings
                    self.cos, self.sin, self.step = self._many_steps[i]
                else:
                    cosine, sine = self._rope_inputs(position)
                    self.cos.upload(cosine)
                    self.sin.upload(sine)
                # Each token's own top-k: a full indexer writes it, the shared layers after it read it.
                self.selected = self._many_selection(i)
                self.selected_count = counts[i]
                index = tuple(self._many_row(t, i) for t in (iq, ik, head_weights)) if full else None
                self._attention_core(layer, position, shared_ok, self._many_row(query, i), self._many_row(compressed, i),
                                     index)
                counts[i] = self.selected_count
                self._copy(self._many_row(heads_out, i), self.heads_out, self.heads * self.value_dim)
        finally:
            if self.graph:
                self.cos, self.sin, self.step = inputs
        attended = self._many('attn_out', self.hidden)
        self._project_many(prefix + '.o_proj', heads_out, attended, count, bias)
        return attended

    def _many_selection(self, i):
        selections = self.__dict__.setdefault('_many_selections', {})
        if i not in selections:
            selections[i] = self._tensor((self.index_topk,), 'int32')
        return selections[i]

    def _feed_forward_many(self, layer, x, count):
        """_feed_forward for forward_many's tokens, one row a token: the dense MLP and the shared
        experts once for all of them, the router and the routed experts per token."""
        c = self.config
        prefix = f'model.layers.{layer}.mlp'
        out = self._many('ffn_out', self.hidden)
        if self.model.mlp_type(layer) == 'dense':
            return self._mlp_many(prefix, x, out, count)
        experts, k = int(c['n_routed_experts']), int(c['num_experts_per_tok'])
        moe = self._many('moe_sum', self.hidden)
        for i in range(count):
            row = self._many_row(x, i)
            self._route(prefix, row)
            if self.device_experts:
                self._device_moe(prefix, row, k)
            else:
                self._copy(self.moe_sum, self.zeros, self.hidden)
                self._host_experts(prefix, row, k)
            self._copy(self._many_row(moe, i), self.moe_sum, self.hidden)
        if int(c.get('n_shared_experts', 1)) > 0 and not self._folds_shared(prefix):
            shared = self._mlp_many(prefix + '.shared_experts', x, self._many('mlp_out', self.hidden), count)
            for i in range(count):
                self._add(self._many_row(moe, i), self._many_row(moe, i), self._many_row(shared, i), self.hidden)
        for i in range(count):
            self._copy(self._many_row(out, i), self._many_row(moe, i), self.hidden)
        return out

    def _mlp_many(self, prefix, x, out, count):
        width = self._out_features(prefix + '.gate_proj')
        gate, up, act = (self._many(name, width) for name in ('gate', 'up', 'act'))
        self._project_many(prefix + '.gate_proj', x, gate, count)
        self._project_many(prefix + '.up_proj', x, up, count)
        for i in range(count):
            self._swiglu(self._many_row(gate, i), self._many_row(up, i), self._many_row(act, i), width)
        self._project_many(prefix + '.down_proj', act, out, count)
        return out

    # ---------------------------------------------------------------- prefill (#63, item 5b)
    def prefill_chunk(self, token_ids, *, logits=False):
        """Append the prompt tokens token_ids (1 to PREFILL_TOKENS of them) in one pass (#63, item 5b's
        first milestone): the caches that forward(t, logits=False) for each of them leaves, then what
        forward() returns for the last one (logits False, True or 'greedy').

        Each row-wise step runs once over the chunk's rows: the RMSNorms, the residual adds, SwiGLU, and
        the Linears (_prefill_linear: with prefill_gemm='cublaslt' cuBLASLt's block-FP8 matmul, each row's
        _linear otherwise). Each token's attention (its cache rows, the indexer, MLA) and routed experts
        still run one token at a time, in order, with forward()'s kernels, so token i sees the cache rows
        of the tokens before it. Under tensor parallelism the sums after o_proj and the FFN run over the
        rows: one NCCL all-reduce (prefill_sum='nccl', the same bits on every rank) or each row's _tp_sum
        ('rows', forward()'s bits).

        With prefill_gemm='rows' and prefill_sum='rows', and forward() on its plain kernels (no fuse_glue,
        fp8_kernels='hand'), the caches and logits are forward()'s bit for bit
        (gpu/test_resident_prefill.py). cuBLASLt quantizes the Linears' inputs to e4m3 per 128 columns, as
        SGLang's block-FP8 prefill does: close to forward(), not equal."""
        m, c = self.model, self.config
        if self._closed:
            raise RuntimeError('ResidentGLM is closed')
        tokens = [int(t) for t in token_ids]
        if not 1 <= len(tokens) <= self._prefill_capacity() or any(not 0 <= t < int(c['vocab_size']) for t in tokens):
            raise ValueError(f'prefill_chunk takes 1 to {self._prefill_capacity()} tokens of the vocabulary')
        start, count = self.position, len(tokens)
        if start + count > self.max_tokens:
            raise ValueError(f'sequence exceeds max_tokens={self.max_tokens}')
        if start + count > int(c.get('max_position_embeddings', 2 ** 31 - 1)):
            raise ValueError('Request exceeds max_position_embeddings')
        if self.mtp:
            raise NotImplementedError('prefill_chunk does not keep the MTP layer\'s inputs yet')
        self._device_position = None
        hidden = self.hidden
        x = self._prefill_rows('x', hidden)
        embeddings = np.zeros((count, hidden), np.float32)
        for i, token in enumerate(tokens):
            embeddings[i] = m._embedding(token)
        x.view((count * hidden,)).upload(embeddings.ravel())
        self._prefill_upload_steps(start, count)
        x_next, norm = self._prefill_rows('x_next', hidden), self._prefill_rows('norm', hidden)
        selected, counts = self.selected, [0] * count
        inputs = (self.cos, self.sin, getattr(self, 'step', None))
        try:
            for layer in range(m.n_layers):
                prefix = f'model.layers.{layer}'
                self._rmsnorm_rows(x, self._weight(prefix + '.input_layernorm.weight'), norm, count, m.eps)
                attended = self._prefill_attention(layer, norm, start, count, layer > 0, counts)
                self._prefill_sum_rows(attended, count)
                self._launch('residual_rmsnorm', count, self.norm_block,
                             [norm, x_next, attended, x, self._weight(prefix + '.post_attention_layernorm.weight'),
                              I(count), I(hidden), F(_f32(m.eps, 'epsilon', True))])
                ffn = self._prefill_feed_forward(layer, norm, count)
                self._prefill_sum_rows(ffn, count)
                self._add(x.view((count * hidden,)), x_next.view((count * hidden,)), ffn.view((count * hidden,)),
                          count * hidden)
        finally:
            self.selected = selected
            self.cos, self.sin = inputs[0], inputs[1]
            if inputs[2] is not None:
                self.step = inputs[2]
        # prefill_attention='mha' (a dense context) sets no counts: every token is selected
        self.selected_count = counts[-1] or min(self.index_topk, start + count)
        self.position += count
        self._copy(self.x, self._prefill_row(x, count - 1), hidden)
        return self._head(logits)

    def _prefill_capacity(self):
        """prefill_chunk's rows: prefill_rows (PREFILL_TOKENS if 0), or max_tokens if fewer, rounded up to
        cuBLASLt's 4."""
        return -(-min(self.prefill_rows or PREFILL_TOKENS, self.max_tokens) // 4) * 4

    @staticmethod
    def _prefill_bucket(count):
        """The rows a cuBLASLt matmul of count tokens multiplies: a power of two up to 64, then a multiple of
        64, so a server's prompt lengths need a few dozen matmul plans a Linear, not one each."""
        padded = -(-count // 4) * 4
        if padded <= 64:
            return 1 << (padded - 1).bit_length()
        return -(-padded // 64) * 64

    def _prefill_buffer_rows(self):
        """prefill_chunk's scratch rows: its tokens, or with prefill_attention='mha' every key of a dense prefill (up to
        index_topk), in cuBLASLt's bucket so a matmul's padding rows are in it."""
        rows = self._prefill_capacity()
        if self.prefill_attention == 'mha':
            rows = max(rows, min(self.index_topk, self.max_tokens))
        return self._prefill_bucket(rows)

    def _prefill_rows(self, name, width, dtype='float32'):
        """A [_prefill_buffer_rows(), width] scratch matrix of prefill_chunk, allocated on first use."""
        buffers = self.__dict__.setdefault('_prefill_buffers', {})
        if (name, width, dtype) not in buffers:
            buffers[name, width, dtype] = self._tensor((self._prefill_buffer_rows(), width), dtype)
        return buffers[name, width, dtype]

    @staticmethod
    def _prefill_row(matrix, i):
        """Row i of a float32 [rows, width] matrix."""
        width = matrix.shape[1]
        return matrix.view((width,), offset=i * width * 4)

    def _prefill_upload_steps(self, start, count):
        """Each token's cos, sin and step buffer (forward()'s step inputs), one upload for the chunk:
        _prefill_steps(i) are token i's views, which its attention takes as self.cos, self.sin and self.step."""
        half = self.rot // 2
        block = -(-(2 * half + 4) // 64) * 64
        capacity = self._prefill_capacity()
        if getattr(self, '_prefill_inputs', None) is None:
            self._prefill_inputs = self._tensor((capacity * block,))
            self._prefill_views = {}
        steps = np.zeros((count, block), np.float32)
        for i in range(count):
            cosine, sine = self._rope_inputs(start + i)
            position = start + i
            steps[i, :2 * half + 4] = np.concatenate(
                [cosine, sine, self._step_buffer(position, min(self.index_topk, position + 1))])
        self._prefill_inputs.view((count * block,)).upload(steps.ravel())
        self._prefill_block = block

    def _prefill_steps(self, i):
        views = self._prefill_views.get(i)
        if views is None:
            half, block, inputs = self.rot // 2, self._prefill_block, self._prefill_inputs
            views = self._prefill_views[i] = (inputs.view((half,), offset=i * block * 4),
                                              inputs.view((half,), offset=(i * block + half) * 4),
                                              inputs.view((4,), 'int32', offset=(i * block + 2 * half) * 4))
        return views

    def _rmsnorm_rows(self, x, weight, out, count, eps):
        """_rmsnorm of each of the first count rows (gpu/kernels.vx's rmsnorm_rows: the same bits a row)."""
        self._launch('rmsnorm_rows', count, self.norm_block,
                     [out, x, weight, I(count), I(x.shape[1]), F(_f32(eps, 'epsilon', True))])

    def _prefill_sum_rows(self, partial, count):
        """_tp_sum over the first count rows of partial: one all-reduce, or each row's _tp_sum."""
        if self.tp is None:
            return
        if self.prefill_sum == 'rows':
            for i in range(count):
                self._tp_sum(self._prefill_row(partial, i))
            return
        width = partial.shape[1]
        self.tp.collective.all_reduce_sum(partial.view((count * width,)), stream=self.stream)
        self.launches['tp_all_reduce_rows'] = self.launches.get('tp_all_reduce_rows', 0) + 1

    def _prefill_linear(self, prefix, x, out, count):
        """out[i] = W x[i] for the first count rows (no bias): cuBLASLt with prefill_gemm='cublaslt' for an
        FP8 Linear whose input width is a multiple of 128, else _linear a row."""
        if bool(self.config.get('attention_bias', False)) and prefix.endswith(('q_a_proj', 'kv_a_proj_with_mqa', 'o_proj')):
            raise NotImplementedError('prefill_chunk does not add attention biases yet')
        rows, cols = out.shape[1], x.shape[1]
        if self.prefill_gemm == 'cublaslt' and self._format(prefix) == 'fp8' and cols % 128 == 0:
            codes, _, shape = self._fp8_weight(prefix)
            if tuple(shape) != (rows, cols):
                raise ValueError(f'{prefix}: FP8 weight is {shape[0]}x{shape[1]}, got x [{cols}] -> out [{rows}]')
            return self._prefill_cublaslt(prefix, codes, rows, cols, x, out, count)
        for i in range(count):
            self._linear(prefix, self._prefill_row(x, i), self._prefill_row(out, i))
        return out

    def _prefill_cublaslt(self, prefix, codes, rows, cols, x, out, count):
        """out[:count] = x[:count] W^T through cuBLASLt's block-FP8 matmul: the rows quantized to e4m3 with
        per-128 scales (gpu/fp8_quant.vx), padded to _prefill_bucket(count) tokens; out's rows past count get
        the padding's outputs (of stale codes) and are not used."""
        from .cublaslt import pack_weight_scales
        key = prefix + ':fp8:cublaslt-scales'
        scales = self._weights.get(key)
        if scales is None:  # BLK128x128_32F's layout: each row block's scales padded to a multiple of 4
            scales = self._weights[key] = self._upload_weight(key, pack_weight_scales(self.model.weights.block_fp8(prefix)[1]))
        padded = self._prefill_bucket(count)
        x_codes = self._prefill_rows('lt_codes', cols, 'uint8')
        x_scales = self._prefill_rows('lt_scales', cols // 128)  # used as [cols / 128, padded]
        if 'fp8_quant_groups' not in self._kernels:
            module = self._modules['fp8_quant'] = self.ctx.load_ptx(self._ptx_dir / 'fp8_quant.ptx')
            self._kernels['fp8_quant_groups'] = module.kernel('glm_vx_gpu_fp8_quant_groups')
        codes_view = x_codes.view((padded * cols,), 'uint8')
        scales_view = x_scales.view((cols // 128 * padded,))
        self._launch('fp8_quant_groups', -(-count * cols // 128 // 4), 128,
                     [codes_view, scales_view, x.view((count * cols,)), I(count), I(cols), I(padded)])
        matmul = self._lt_matmuls.get((padded, rows, cols))
        if matmul is None:
            factory = getattr(self.ctx, 'fp8_block_matmul', None)
            if factory is not None:
                matmul = factory(padded, rows, cols)
            else:
                from .cublaslt import CublasLt, Fp8BlockMatmul
                if self._lt is None:
                    self._lt = CublasLt(self.ctx)
                if getattr(self, '_prefill_workspace', None) is None:  # one for every prefill matmul: one stream
                    self._prefill_workspace = self._tensor((32 << 20,), 'uint8')
                matmul = Fp8BlockMatmul(self._lt, padded, rows, cols, workspace=self._prefill_workspace)
            self._lt_matmuls[(padded, rows, cols)] = matmul
        matmul(out.view((padded * rows,)), codes_view, scales_view, codes, scales, stream=self.stream)
        self.launches['cublaslt_fp8_rows'] = self.launches.get('cublaslt_fp8_rows', 0) + 1
        return out

    def _prefill_attention(self, layer, x, start, count, shared_ok, counts):
        """The attention of prefill_chunk's tokens: the projections over the rows, then each token's
        _attention_core in order (or _prefill_mha); returns o_proj's outputs, one row a token (this rank's part
        under TP)."""
        if self.prefill_attention == 'mha' and start + count <= self.index_topk:
            return self._prefill_mha(layer, x, start, count)
        prefix = f'model.layers.{layer}.self_attn'
        q_rank, qk = int(self.config['q_lora_rank']), self.nope + self.rot
        rows = self._prefill_rows
        q_a, q_res = rows('q_a', q_rank), rows('q_res', q_rank)
        query, compressed = rows('query', self.heads * qk), rows('compressed', self.rank + self.rot)
        self._prefill_linear(prefix + '.q_a_proj', x, q_a, count)
        self._rmsnorm_rows(q_a, self._weight(prefix + '.q_a_layernorm.weight'), q_res, count, 1e-6)
        self._prefill_linear(prefix + '.q_b_proj', q_res, query, count)
        self._prefill_linear(prefix + '.kv_a_proj_with_mqa', x, compressed, count)
        full = self.model.indexer_type(layer) == 'full'
        if full:
            iq = rows('iq', self.index_heads * self.index_dim)
            ik, head_weights = rows('ik', self.index_dim), rows('head_weights', self.index_heads)
            self._prefill_linear(prefix + '.indexer.wk', x, ik, count)
            if start + count > self.index_topk:  # some token scores the keys (#43)
                self._prefill_linear(prefix + '.indexer.wq_b', q_res, iq, count)
                self._prefill_linear(prefix + '.indexer.weights_proj', x, head_weights, count)
        heads_out = rows('heads_out', self.heads * self.value_dim)
        for i in range(count):
            self.cos, self.sin, step = self._prefill_steps(i)
            if self.graph:
                self.step = step
            # Each token's own top-k: a full indexer writes it, the shared layers after it read it.
            self.selected = self._prefill_selection(i, start + i)
            self.selected_count = counts[i]
            index = tuple(self._prefill_row(t, i) for t in (iq, ik, head_weights)) if full else None
            self._attention_core(layer, start + i, shared_ok, self._prefill_row(query, i), self._prefill_row(compressed, i),
                                 index)
            counts[i] = self.selected_count
            self._copy(self._prefill_row(heads_out, i), self.heads_out, self.heads * self.value_dim)
        attended = rows('attn_out', self.hidden)
        self._prefill_linear(prefix + '.o_proj', heads_out, attended, count)
        return attended

    def _prefill_mha(self, layer, x, start, count):
        """prefill_attention='mha' while every token attends to every earlier one (start + count <= index_topk): the
        projections over the rows; the chunk's cache rows in one launch (kv_rows_prefill: forward()'s latent and
        rope rows); the indexer's key rows a token at a time; the queries with RoPE (q_rows_prefill); kv_b_proj over
        every cached latent row, 0 to start + count, for the keys' k_nope and values; one causal mha_prefill launch;
        o_proj. The same attention as the absorbed form forward() runs, in float32 here, summed in another order."""
        prefix = f'model.layers.{layer}.self_attn'
        c = self.config
        q_rank, qk = int(c['q_lora_rank']), self.nope + self.rot
        rows = self._prefill_rows
        q_a, q_res = rows('q_a', q_rank), rows('q_res', q_rank)
        query, compressed = rows('query', self.heads * qk), rows('compressed', self.rank + self.rot)
        self._prefill_linear(prefix + '.q_a_proj', x, q_a, count)
        self._rmsnorm_rows(q_a, self._weight(prefix + '.q_a_layernorm.weight'), q_res, count, 1e-6)
        self._prefill_linear(prefix + '.q_b_proj', q_res, query, count)
        self._prefill_linear(prefix + '.kv_a_proj_with_mqa', x, compressed, count)
        steps = self._prefill_inputs.view((self._prefill_capacity(), self._prefill_block))
        bf16 = self.latents[layer].dtype == 'uint16'
        self._spawn('kv_rows_prefill_bf16' if bf16 else 'kv_rows_prefill', latents=self.latents[layer],
                    rope_keys=self.rope_keys[layer], compressed=compressed, weight=self._weight(prefix + '.kv_a_layernorm.weight'),
                    steps=steps, start=start, rows=count, rank=self.rank, rot=self.rot, eps=_f32(1e-6, 'epsilon', True),
                    threads=self.norm_block)
        if self.model.indexer_type(layer) == 'full':  # the key rows later steps score
            ik = rows('ik', self.index_dim)
            self._prefill_linear(prefix + '.indexer.wk', x, ik, count)
            inputs = (self.cos, self.sin, getattr(self, 'step', None))
            try:
                for i in range(count):
                    self.cos, self.sin, step = self._prefill_steps(i)
                    if self.graph:
                        self.step = step
                    self._index_key(layer, prefix + '.indexer', start + i, self._prefill_row(ik, i))
            finally:
                self.cos, self.sin = inputs[0], inputs[1]
                if inputs[2] is not None:
                    self.step = inputs[2]
        q = rows('q_heads', self.heads * qk)
        self._spawn('q_rows_prefill', q_out=q.view((q.shape[0], self.heads, qk)), query=query, steps=steps, rows=count,
                    heads=self.heads, nope=self.nope, rot=self.rot)
        keys = start + count
        if bf16:
            latent, rope = rows('latent_f32', self.rank), rows('rope_f32', self.rot)
            self._spawn('rows_bf16_f32', out=latent, x=self.latents[layer], rows=keys, cols=self.rank)
            self._spawn('rows_bf16_f32', out=rope, x=self.rope_keys[layer], rows=keys, cols=self.rot)
        else:
            latent, rope = self.latents[layer], self.rope_keys[layer]
        width = self.nope + self.value_dim
        kv = rows('kv_heads', self.heads * width)
        self._prefill_linear(prefix + '.kv_b_proj', latent, kv, keys)
        out = rows('heads_out', self.heads * self.value_dim)
        self._spawn('mha_prefill', o=out.view((out.shape[0], self.heads, self.value_dim)),
                    q=q.view((q.shape[0], self.heads, qk)), kv=kv.view((kv.shape[0], self.heads, width)), k_rope=rope,
                    start=start, queries=count, heads=self.heads, nope=self.nope, rot=self.rot, value=self.value_dim,
                    scale=_f32(qk ** -0.5, 'MLA scale', True))
        attended = rows('attn_out', self.hidden)
        self._prefill_linear(prefix + '.o_proj', out, attended, count)
        return attended

    def _prefill_selection(self, i, position):
        """Token i's selection: within index_topk every token takes all the cached tokens in position
        order (#43), so one buffer serves them all; past it, a buffer a token."""
        if position + 1 <= self.index_topk:
            return self._prefill_short_selection()
        selections = self.__dict__.setdefault('_prefill_selections', {})
        if i not in selections:
            selections[i] = self._tensor((self.index_topk,), 'int32')
        return selections[i]

    def _prefill_short_selection(self):
        if getattr(self, '_prefill_short', None) is None:
            self._prefill_short = self._tensor((self.index_topk,), 'int32', np.arange(self.index_topk, dtype=np.int32))
        return self._prefill_short

    def _prefill_feed_forward(self, layer, x, count):
        """The FFN of prefill_chunk's tokens, one row a token (this rank's part under TP): a dense MLP and
        the shared experts over the rows; the router and the routed experts a token at a time."""
        c = self.config
        prefix = f'model.layers.{layer}.mlp'
        if self.model.mlp_type(layer) == 'dense':
            return self._prefill_mlp(prefix, x, self._prefill_rows('ffn_out', self.hidden), count)
        k = int(c['num_experts_per_tok'])
        if self._prefill_grouped_ok(prefix):
            moe = self._prefill_moe_grouped(prefix, x, count)
            if int(c.get('n_shared_experts', 1)) > 0 and not self._folds_shared(prefix):
                shared = self._prefill_mlp(prefix + '.shared_experts', x, self._prefill_rows('mlp_out', self.hidden), count)
                n = count * self.hidden
                self._add(moe.view((n,)), moe.view((n,)), shared.view((n,)), n)
            return moe
        moe = self._prefill_rows('moe_sum', self.hidden)
        for i in range(count):
            row = self._prefill_row(x, i)
            self._route(prefix, row)
            if self.device_experts:
                self._device_moe(prefix, row, k)
            else:
                self._copy(self.moe_sum, self.zeros, self.hidden)
                self._host_experts(prefix, row, k)
            self._copy(self._prefill_row(moe, i), self.moe_sum, self.hidden)
        if int(c.get('n_shared_experts', 1)) > 0 and not self._folds_shared(prefix):
            shared = self._prefill_mlp(prefix + '.shared_experts', x, self._prefill_rows('mlp_out', self.hidden), count)
            n = count * self.hidden
            self._add(moe.view((n,)), moe.view((n,)), shared.view((n,)), n)
        return moe

    def _prefill_grouped_ok(self, prefix):
        """prefill_moe='grouped' applies to this MoE layer: FP8 experts as separate Linears on the device, with
        widths the grouped GEMM takes (multiples of 128)."""
        if self.prefill_moe != 'grouped' or not self.device_experts or self._fused_experts(f'{prefix}.experts.0'):
            return False
        gate = self._stacked_experts(prefix, 'gate_proj', shared=self._folds_shared(prefix))
        return gate[0] == 'fp8' and gate[2][0] % 128 == 0 and gate[2][1] % 128 == 0

    def _prefill_route(self, prefix, x, count, k):
        """Each of the first count rows' experts and weights, [count, k] each, on the host: router_logits_rows and
        router_topk_rows for a BF16 gate of at most 256 experts (router_bf16's bits), else _route a row."""
        c = self.config
        experts = int(c['n_routed_experts'])
        ids, weights = self._prefill_rows('route_ids', k, 'int32'), self._prefill_rows('route_weights', k)
        if self._format(prefix + '.gate') == 'bf16' and experts <= 256 and k <= 8:
            bits, (rows, cols) = self._bf16_weight(prefix + '.gate')
            if rows == experts and cols % 8 == 0 and cols <= 8192:
                logits = self._prefill_rows('router_logits', experts)
                scale = _f32(float(c.get('routed_scaling_factor', 1.0)), 'router scale')
                self._spawn('router_logits_rows', logits=logits, gate=bits.view((rows * cols // 2,), 'uint32'), x=x,
                            tokens=count, experts=experts, cols=cols)
                self._spawn('router_topk_rows', indices=ids, weights=weights, logits=logits,
                            bias=self._weight(prefix + '.gate.e_score_correction_bias'), tokens=count, experts=experts,
                            k=k, scale=scale)
                return (np.frombuffer(ids.view((count * k,), 'int32').download(), np.int32).reshape(count, k).copy(),
                        np.frombuffer(weights.view((count * k,)).download(), np.float32).reshape(count, k).copy())
        out_ids, out_weights = np.zeros((count, k), np.int32), np.zeros((count, k), np.float32)
        for i in range(count):
            self._route(prefix, self._prefill_row(x, i))
            out_ids[i] = np.frombuffer(self.route_ids.download(), np.int32)[:k]
            out_weights[i] = np.frombuffer(self.route_weights.download(), np.float32)[:k]
        return out_ids, out_weights

    def _grouped_rows(self, name, width, dtype='float32'):
        """A [rows, width] matrix for prefill_chunk's grouped (token, expert) rows: the capacity's bucket times the
        most slots a token (k, and the shared expert's)."""
        buffers = self.__dict__.setdefault('_grouped_buffers', {})
        if (name, width, dtype) not in buffers:
            slots = int(self.config['num_experts_per_tok']) + 1
            buffers[name, width, dtype] = self._tensor((self._prefill_bucket(self._prefill_capacity()) * slots, width), dtype)
        return buffers[name, width, dtype]

    def _prefill_moe_grouped(self, prefix, x, count):
        """The routed experts of the first count rows of x, grouped by expert (prefill_moe='grouped'): each row's k
        experts (and the shared expert last, with fold_shared), the (token, expert) pairs sorted by expert on the
        host, the rows gathered and quantized to e4m3 per 128 columns (fp8_quant_rows), gate and up as grouped
        block-FP8 GEMMs, SwiGLU, the activations quantized, down, and each token's rows weighted and added in
        ascending expert id (moe_combine_rows: moe_down's order). Returns this rank's part, one row a token."""
        c = self.config
        experts, k = int(c['n_routed_experts']), int(c['num_experts_per_tok'])
        shared = self._folds_shared(prefix)
        kk, total = (k + 1, experts + 1) if shared else (k, experts)
        gate, up, down = (self._stacked_experts(prefix, n, shared=shared) for n in ('gate_proj', 'up_proj', 'down_proj'))
        inter, hidden = gate[2]
        ids, weights = self._prefill_route(prefix, x, count, k)
        if shared:  # the shared expert: id n_routed_experts, weight 1.0, as _device_moe's slot k
            ids = np.concatenate([ids, np.full((count, 1), experts, np.int32)], 1)
            weights = np.concatenate([weights, np.ones((count, 1), np.float32)], 1)
        rows = count * kk
        order = np.argsort(ids.ravel(), kind='stable')  # by expert, then token: the grouped rows
        src = (order // kk).astype(np.int32)
        slot_of = np.empty(rows, np.int32)
        slot_of[order] = np.arange(rows, dtype=np.int32)
        by_id = np.argsort(ids, axis=1, kind='stable')
        tokens = np.arange(count)[:, None]
        slots, ordered = slot_of.reshape(count, kk)[tokens, by_id], weights[tokens, by_id]
        offsets = np.searchsorted(ids.ravel()[order], np.arange(total + 1)).astype(np.int32)
        capacity = self._grouped_rows('y', hidden).shape[0]
        index = self._grouped_rows('index', 1, 'int32').view((capacity,), 'int32')
        index.view((rows,), 'int32').upload(src.tobytes())
        table = self._prefill_rows('combine_slots', kk + 1, 'int32')  # [count, kk] slots, then the offsets
        table.view((count * kk,), 'int32').upload(slots.astype(np.int32).tobytes())
        offsets_dev = self._prefill_rows('group_offsets', 1, 'int32').view((total + 1,), 'int32') if total + 1 <= table.shape[0] \
            else self._tensor((total + 1,), 'int32')
        offsets_dev.upload(offsets.tobytes())
        combine_weights = self._prefill_rows('combine_weights', kk)
        combine_weights.view((count * kk,)).upload(ordered.astype(np.float32).tobytes())
        a_codes, a_scales = self._grouped_rows('a_codes', hidden, 'uint8'), self._grouped_rows('a_scales', hidden // 128)
        self._spawn('fp8_quant_rows', codes=a_codes, scales=a_scales, x=x, src=index, rows=rows, cols=hidden, gather=1)
        g, u, act = (self._grouped_rows(n, inter) for n in ('gate', 'up', 'act'))
        self._grouped_gemm(a_codes, a_scales, gate[1], g, offsets, offsets_dev, rows, inter, hidden, total)
        self._grouped_gemm(a_codes, a_scales, up[1], u, offsets, offsets_dev, rows, inter, hidden, total)
        n = rows * inter
        self._swiglu(g.view((n,)), u.view((n,)), act.view((n,)), n)
        q_codes, q_scales = self._grouped_rows('q_codes', inter, 'uint8'), self._grouped_rows('q_scales', inter // 128)
        self._spawn('fp8_quant_rows', codes=q_codes, scales=q_scales, x=act, src=index, rows=rows, cols=inter, gather=0)
        y = self._grouped_rows('y', hidden)
        self._grouped_gemm(q_codes, q_scales, down[1], y, offsets, offsets_dev, rows, hidden, inter, total)
        moe = self._prefill_rows('moe_sum', self.hidden)
        self._spawn('moe_combine_rows', out=moe, y=y, slots=table.view((count, kk), 'int32'),
                    weights=combine_weights.view((count, kk)), tokens=count, cols=hidden, kk=kk)
        self.launches['grouped_moe'] = self.launches.get('grouped_moe', 0) + 1
        return moe

    def _grouped_gemm(self, a, a_scales, stack, out, offsets, offsets_dev, rows, n, k, experts):
        """out[r] = W[e] a[r] for the grouped rows r of each expert e (offsets[e] <= r < offsets[e + 1]): a's e4m3
        codes and row-major per-128 scales against the stacked experts' codes and 128x128 block scales. A test
        context's fp8_grouped_gemm, else gpu/spawn/fp8_gemm's grouped GEMM (its tile table planned on the device
        from offsets_dev; the tensor maps over the buffers' whole rows, so their few shapes are cached)."""
        codes, scales = stack
        factory = getattr(self.ctx, 'fp8_grouped_gemm', None)
        if factory is not None:
            factory()(a, a_scales, codes, scales, out, offsets, rows=rows, n=n, k=k, experts=experts, stream=self.stream)
            return out
        from .fp8_gemm import FP8GroupedGEMM, grouped_tiles_max
        if getattr(self, '_grouped', None) is None:
            self._grouped = FP8GroupedGEMM(self.ctx, self._ptx_dir / 'spawn' / 'fp8_gemm')
        capacity = out.shape[0]
        tiles = self.__dict__.setdefault('_grouped_tiles', {}).get(experts)
        if tiles is None:
            tiles = self._grouped_tiles[experts] = self._tensor((1 + 3 * grouped_tiles_max(capacity, experts),), 'int32')
        self._grouped.plan(offsets_dev, tiles, rows=capacity, experts=experts, stream=self.stream)
        flat = lambda tensor, dtype='float32': tensor.view((int(np.prod(tensor.shape)),), dtype)  # its parameters are 1-D
        self._grouped.gemm(flat(a, 'uint8'), flat(a_scales), flat(codes, 'uint8'), flat(scales), flat(out), tiles,
                           rows=capacity, n=n, k=k, experts=experts, stream=self.stream)
        return out

    def _prefill_mlp(self, prefix, x, out, count):
        """_mlp of each of the first count rows: with prefill_gemm='cublaslt' the three projections over the
        rows and one SwiGLU, else _mlp a row (forward()'s kernels, its fused gate_up included)."""
        if self.prefill_gemm != 'cublaslt':
            for i in range(count):
                self._mlp(prefix, self._prefill_row(x, i), self._prefill_row(out, i))
            return out
        width = self._out_features(prefix + '.gate_proj')
        gate, up, act = (self._prefill_rows(name, width) for name in ('gate', 'up', 'act'))
        self._prefill_linear(prefix + '.gate_proj', x, gate, count)
        self._prefill_linear(prefix + '.up_proj', x, up, count)
        n = count * width
        self._swiglu(gate.view((n,)), up.view((n,)), act.view((n,)), n)
        self._prefill_linear(prefix + '.down_proj', act, out, count)
        return out

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

    With a ResidentGLM(device_decode=True), forward_greedy runs one step ahead (greedy_ahead): it
    launches the step after the one whose token it returns before it reads that token, so the GPUs
    never wait for the host between greedy steps. A step launched ahead and not used (the request
    stopped, or the next call gives another token) is rewound; at most one is in flight.
    """

    def __init__(self, model, resident):
        self.model, self.resident, self.backend = model, resident, resident
        self.config = dict(model.config)
        limit = int(self.config.get('max_position_embeddings', resident.max_tokens))
        self.config['max_position_embeddings'] = min(limit, int(resident.max_tokens))
        self._current = None
        # The scheduler passes forward_greedy last= when this is set (glm_vx/scheduler.py).
        self.greedy_ahead = bool(getattr(resident, 'device_decode', False))
        self._ahead = None  # (the token it starts from, its device_step ticket)
        # A ResidentGLM(prefill_rows=N) takes the prompt's tokens before the last in chunks of up to N
        # (prefill_chunk, #63 item 5b): the scheduler calls prefill_tokens when the model has it.
        if getattr(resident, 'prefill_rows', 0):
            self.prefill_tokens = self._prefill_tokens

    def new_cache(self):
        self._ahead = None  # reset() drops it with every other position
        self.resident.reset()
        self._current = object()
        return self._current

    def _settle(self):
        """Rewind the step launched ahead, before a step that does not start from its token."""
        if self._ahead is not None:
            self._ahead = None
            self.resident.rewind(self.resident.position - 1)

    def _check(self, cache):
        if cache is None or cache is not self._current:
            raise RuntimeError('ResidentGLM holds one sequence and this request no longer owns it')

    def prefill_token(self, token_id, cache):
        self._check(cache)
        self._settle()
        self.resident.forward(token_id, logits=False)

    def _prefill_tokens(self, token_ids, cache):
        """prefill_token for each of token_ids, as prefill_chunk calls of up to prefill_rows tokens."""
        self._check(cache)
        self._settle()
        tokens, rows = [int(t) for t in token_ids], self.resident.prefill_rows
        for start in range(0, len(tokens), rows):
            self.resident.prefill_chunk(tokens[start:start + rows])

    def forward(self, token_id, cache):
        self._check(cache)
        self._settle()
        # Logits come back so the scheduler samples on the host; device-side sampling is #17.
        return self.resident.forward(token_id)

    def forward_greedy(self, token_id, cache, *, last=False):
        """The next token of a greedy request (temperature 0), chosen on the device: numpy.argmax of
        forward's logits, without them coming back (the scheduler calls it in place of forward).
        With greedy_ahead, the step after it is launched first unless `last` (the request ends with
        this token) or the caches are full."""
        self._check(cache)
        r = self.resident
        if not self.greedy_ahead:
            return r.forward(token_id, logits='greedy')
        ahead, self._ahead = self._ahead, None
        if ahead is not None and ahead[0] != int(token_id):
            r.rewind(r.position - 1)
            ahead = None
        ticket = ahead[1] if ahead is not None else r.device_step(int(token_id))
        following = None
        if not last and r.position < self.config['max_position_embeddings']:
            following = r.device_step()  # from the token the device chose: launched before it is read
        try:
            token = r.device_token(ticket)
        except BaseException:
            if following is not None:
                r.rewind(r.position - 1)
            raise
        if following is not None:
            self._ahead = (token, following)
        return token

    def close(self):
        self.resident.close()
