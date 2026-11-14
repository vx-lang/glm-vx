# One decode step, kernel by kernel: bf16, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 2195 | - | - | - | - |
| layer 1 | 31 | 2202 | - | - | - | - |
| layer 2 | 31 | 2192 | - | - | - | - |
| layer 3 | 26 | 1860 | - | - | - | - |
| head | 2 | 2651 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| bf16_matvec | self_attn.o_proj | 380.5 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.down_proj | 287.7 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.up_proj | 283.9 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.gate_proj | 283.2 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.indexer.weights_proj | 152.8 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_a_proj | 143.6 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.indexer.wk | 143.5 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.kv_a_proj_with_mqa | 141.9 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_b_proj | 113.2 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.indexer.wq_b | 52.6 | - | - | - | - |  x  |  |
| index_scores | attention_core.index_core | 33.3 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.3 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 20.1 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 19.0 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.7 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.7 | - | - | - | - |  x  |  |
| topk_sort | index_core.select | 10.4 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.1 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 6.0 | - | - | - | - |  x  |  |
| layernorm | attention_core.index_core | 5.6 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.5 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.1 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.7 | - | - | - | - |  x  |  |
| add | layer.add | 4.6 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.5 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.5 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.4 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.2 | - | - | - | - |  x  |  |
| scale | attention_core.index_core | 4.2 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| bf16_matvec | self_attn.o_proj | 381.2 | - | - | - | - |  x  |  |
| bf16_moe_gate_up | mlp.experts | 265.1 | - | - | - | - |  x  |  |
| bf16_moe_down | mlp.experts | 148.8 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.shared_experts.gate_proj | 144.7 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.shared_experts.up_proj | 144.4 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_a_proj | 143.9 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.gate | 142.6 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.kv_a_proj_with_mqa | 141.5 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_b_proj | 113.5 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.shared_experts.down_proj | 54.6 | - | - | - | - |  x  |  |
| router | feed_forward | 30.8 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 25.3 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.5 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.6 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.9 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.4 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 9.2 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.8 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 5.7 | - | - | - | - |  x  |  |
| add | layer.add | 4.6 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 4.6 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.5 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 4.5 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | feed_forward.add | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
