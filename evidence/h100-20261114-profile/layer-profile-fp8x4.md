# One decode step, kernel by kernel: fp8, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 596 | - | - | - | - |
| layer 1 | 31 | 586 | - | - | - | - |
| layer 2 | 31 | 590 | - | - | - | - |
| layer 3 | 26 | 635 | - | - | - | - |
| head | 2 | 1050 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_matvec4 | self_attn.o_proj | 91.8 | - | - | - | - |  x  |  |
| fp8_matvec4 | mlp.down_proj | 69.3 | - | - | - | - |  x  |  |
| fp8_matvec4 | mlp.up_proj | 57.6 | - | - | - | - |  x  |  |
| fp8_matvec4 | mlp.gate_proj | 57.5 | - | - | - | - |  x  |  |
| index_scores | attention_core.index_core | 33.0 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.q_b_proj | 29.0 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.4 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 22.0 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 20.6 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 18.7 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.5 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 17.2 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.q_a_proj | 16.7 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.kv_a_proj_with_mqa | 14.1 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.indexer.wk | 13.7 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.indexer.wq_b | 11.3 | - | - | - | - |  x  |  |
| topk_sort | index_core.select | 10.6 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.1 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 6.0 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.8 | - | - | - | - |  x  |  |
| layernorm | attention_core.index_core | 5.6 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.1 | - | - | - | - |  x  |  |
| add | layer.add | 4.6 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.5 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.5 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.5 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| scale | attention_core.index_core | 4.2 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.2 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.1 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe4_gate_up | mlp.experts | 142.3 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.o_proj | 91.6 | - | - | - | - |  x  |  |
| fp8_moe4_down | mlp.experts | 89.7 | - | - | - | - |  x  |  |
| router | feed_forward | 32.2 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.q_b_proj | 28.6 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.3 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 22.9 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.3 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.2 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 17.7 | - | - | - | - |  x  |  |
| fp8_matvec4 | mlp.shared_experts.gate_proj | 17.3 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.9 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.q_a_proj | 16.5 | - | - | - | - |  x  |  |
| fp8_matvec4 | mlp.shared_experts.up_proj | 16.2 | - | - | - | - |  x  |  |
| fp8_matvec4 | mlp.shared_experts.down_proj | 15.9 | - | - | - | - |  x  |  |
| fp8_matvec4 | self_attn.kv_a_proj_with_mqa | 13.9 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.7 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 6.1 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 6.0 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 4.8 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.5 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| add | feed_forward.add | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 4.3 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.2 | - | - | - | - |  x  |  |
