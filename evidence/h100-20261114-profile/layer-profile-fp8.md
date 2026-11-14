# One decode step, kernel by kernel: fp8, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 1368 | - | - | - | - |
| layer 1 | 31 | 1363 | - | - | - | - |
| layer 2 | 31 | 1366 | - | - | - | - |
| layer 3 | 26 | 1435 | - | - | - | - |
| head | 2 | 1052 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_matvec | self_attn.o_proj | 242.0 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.down_proj | 187.6 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.up_proj | 175.8 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.gate_proj | 174.2 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_a_proj | 83.8 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.kv_a_proj_with_mqa | 82.3 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.indexer.wk | 78.8 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_b_proj | 78.2 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.indexer.wq_b | 33.7 | - | - | - | - |  x  |  |
| index_scores | attention_core.index_core | 32.3 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 23.9 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 21.0 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.6 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.5 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 18.5 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.9 | - | - | - | - |  x  |  |
| topk_sort | index_core.select | 10.4 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.3 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 6.0 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.8 | - | - | - | - |  x  |  |
| layernorm | attention_core.index_core | 5.3 | - | - | - | - |  x  |  |
| add | layer.add | 5.1 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.1 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.6 | - | - | - | - |  x  |  |
| add | layer.add | 4.6 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.5 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.4 | - | - | - | - |  x  |  |
| scale | attention_core.index_core | 4.3 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.2 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.2 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.1 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe_gate_up | mlp.experts | 365.2 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.o_proj | 241.0 | - | - | - | - |  x  |  |
| fp8_moe_down | mlp.experts | 178.2 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.shared_experts.up_proj | 83.6 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_a_proj | 83.4 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.shared_experts.gate_proj | 83.2 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.kv_a_proj_with_mqa | 81.7 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_b_proj | 79.1 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.shared_experts.down_proj | 37.5 | - | - | - | - |  x  |  |
| router | feed_forward | 31.6 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.6 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 22.7 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 20.5 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.5 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.4 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.2 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.7 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 6.0 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.6 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 4.9 | - | - | - | - |  x  |  |
| add | layer.add | 4.6 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 4.5 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.4 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| add | feed_forward.add | 4.4 | - | - | - | - |  x  |  |
