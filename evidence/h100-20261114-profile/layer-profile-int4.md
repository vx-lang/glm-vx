# One decode step, kernel by kernel: int4, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 1048 | - | - | - | - |
| layer 1 | 31 | 1043 | - | - | - | - |
| layer 2 | 31 | 1040 | - | - | - | - |
| layer 3 | 26 | 1156 | - | - | - | - |
| head | 2 | 1051 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| int4_matvec | self_attn.o_proj | 193.0 | - | - | - | - |  x  |  |
| int4_matvec | mlp.down_proj | 148.6 | - | - | - | - |  x  |  |
| int4_matvec | mlp.gate_proj | 133.5 | - | - | - | - |  x  |  |
| int4_matvec | mlp.up_proj | 133.1 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_b_proj | 60.6 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_a_proj | 52.7 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.kv_a_proj_with_mqa | 50.2 | - | - | - | - |  x  |  |
| index_scores | attention_core.index_core | 32.3 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.6 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 22.0 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wk | 21.8 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wq_b | 20.5 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 20.2 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.2 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 18.5 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.5 | - | - | - | - |  x  |  |
| topk_sort | index_core.select | 10.6 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.1 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.8 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 5.7 | - | - | - | - |  x  |  |
| layernorm | attention_core.index_core | 5.3 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.1 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 5.0 | - | - | - | - |  x  |  |
| scale | attention_core.index_core | 4.5 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.2 | - | - | - | - |  x  |  |
| copy | index_core.copy | 4.2 | - | - | - | - |  x  |  |
| rope_strided | index_core.rope | 4.1 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| int4_moe_gate_up | mlp.experts | 319.4 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.o_proj | 192.5 | - | - | - | - |  x  |  |
| int4_moe_down | mlp.experts | 145.0 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_b_proj | 60.6 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_a_proj | 53.1 | - | - | - | - |  x  |  |
| int4_matvec | mlp.shared_experts.gate_proj | 52.1 | - | - | - | - |  x  |  |
| int4_matvec | mlp.shared_experts.up_proj | 52.0 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.kv_a_proj_with_mqa | 50.6 | - | - | - | - |  x  |  |
| router | feed_forward | 31.9 | - | - | - | - |  x  |  |
| int4_matvec | mlp.shared_experts.down_proj | 29.0 | - | - | - | - |  x  |  |
| mla_partial | attention.attention_core | 24.1 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 23.3 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 19.5 | - | - | - | - |  x  |  |
| matvec_heads | attention.attention_core | 18.8 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.7 | - | - | - | - |  x  |  |
| rmsnorm | layer.rmsnorm | 16.7 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 8.4 | - | - | - | - |  x  |  |
| mla_merge | attention.attention_core | 5.7 | - | - | - | - |  x  |  |
| rmsnorm | attention_core.rmsnorm | 5.6 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 4.9 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.7 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 4.5 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| add | feed_forward.add | 4.4 | - | - | - | - |  x  |  |
| add | layer.add | 4.4 | - | - | - | - |  x  |  |
| rope_strided | attention_core.rope | 4.2 | - | - | - | - |  x  |  |
