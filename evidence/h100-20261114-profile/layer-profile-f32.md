# One decode step, kernel by kernel: f32, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 805 | - | - | - | - |
| layer 1 | 31 | 798 | - | - | - | - |
| layer 2 | 31 | 802 | - | - | - | - |
| layer 3 | 26 | 1010 | - | - | - | - |
| head | 2 | 1051 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| matvec | self_attn.o_proj | 128.3 | - | - | - | - |  x  |  |
| matvec | mlp.down_proj | 99.1 | - | - | - | - |  x  |  |
| matvec | mlp.gate_proj | 97.1 | - | - | - | - |  x  |  |
| matvec | mlp.up_proj | 96.7 | - | - | - | - |  x  |  |
| matvec | self_attn.q_b_proj | 48.6 | - | - | - | - |  x  |  |
| matvec | self_attn.q_a_proj | 33.3 | - | - | - | - |  x  |  |
| index_scores | attention_core.index_core | 32.9 | - | - | - | - |  x  |  |
| matvec | self_attn.kv_a_proj_with_mqa | 25.1 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.4 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 22.1 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wk | 22.0 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 20.9 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wq_b | 20.3 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.0 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 18.3 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 17.0 | - | - | - | - |  x  |  |
| topk_sort | index_core.select | 10.4 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 7.8 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 6.1 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 6.0 | - | - | - | - |  x  |  |
| layernorm | attention_core.index_core | 5.6 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.6 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.4 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.3 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.3 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.2 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.2 | - | - | - | - |  x  |  |
| scale | attention_core.index_core | 4.2 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.1 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| moe_gate_up | mlp.experts | 281.9 | - | - | - | - |  x  |  |
| moe_down | mlp.experts | 198.9 | - | - | - | - |  x  |  |
| matvec | self_attn.o_proj | 128.9 | - | - | - | - |  x  |  |
| matvec | self_attn.q_b_proj | 49.2 | - | - | - | - |  x  |  |
| matvec | self_attn.q_a_proj | 33.6 | - | - | - | - |  x  |  |
| matvec | mlp.shared_experts.up_proj | 33.5 | - | - | - | - |  x  |  |
| matvec | mlp.shared_experts.gate_proj | 33.1 | - | - | - | - |  x  |  |
| router | feed_forward | 30.6 | - | - | - | - |  x  |  |
| matvec | mlp.shared_experts.down_proj | 25.4 | - | - | - | - |  x  |  |
| matvec | self_attn.kv_a_proj_with_mqa | 24.7 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 23.9 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 22.6 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.8 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.5 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 17.0 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.7 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.4 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 6.1 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 5.7 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 4.9 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.7 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| add | feed_forward.add | 4.4 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.3 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 4.2 | - | - | - | - |  x  |  |
