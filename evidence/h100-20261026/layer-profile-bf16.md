# One decode step, kernel by kernel: bf16, full width, 4 layers, context 64

CUDA events around each launch of a warm step (ncu had no counters): GPU time only, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 3413 | - | - | - | - |
| layer 1 | 31 | 3293 | - | - | - | - |
| layer 2 | 31 | 3495 | - | - | - | - |
| layer 3 | 26 | 3630 | - | - | - | - |
| head | 2 | 2599 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| bf16_matvec | self_attn.o_proj | 377.9 | - | - | - | - |  x  |  |
| topk_stable | index | 361.5 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.up_proj | 291.0 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.gate_proj | 283.5 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.down_proj | 282.6 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.kv_a_proj_with_mqa | 159.1 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_a_proj | 149.6 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.indexer.wk | 148.2 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.indexer.weights_proj | 142.3 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_b_proj | 125.1 | - | - | - | - |  x  |  |
| mla_merge | attention | 74.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 73.7 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.indexer.wq_b | 72.7 | - | - | - | - |  x  |  |
| mla_partial | attention | 70.6 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 57.6 | - | - | - | - |  x  |  |
| matvec_heads | attention | 57.6 | - | - | - | - |  x  |  |
| index_scores | index | 57.5 | - | - | - | - |  x  |  |
| matvec_heads | attention | 52.9 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 51.9 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 50.1 | - | - | - | - |  x  |  |
| add | layers.add | 45.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 45.8 | - | - | - | - |  x  |  |
| layernorm | index | 45.5 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 43.9 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 43.4 | - | - | - | - |  x  |  |
| add | layers.add | 43.3 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 43.2 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 41.5 | - | - | - | - |  x  |  |
| scale | index | 41.3 | - | - | - | - |  x  |  |
| copy | index.copy | 39.5 | - | - | - | - |  x  |  |
| copy | index.copy | 38.9 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| bf16_moe_gate_up | mlp.experts | 1119.3 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.o_proj | 374.0 | - | - | - | - |  x  |  |
| bf16_moe_down | mlp.experts | 359.4 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.kv_a_proj_with_mqa | 183.3 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_a_proj | 156.9 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.shared_experts.up_proj | 145.1 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.gate | 143.0 | - | - | - | - |  x  |  |
| bf16_matvec | self_attn.q_b_proj | 139.4 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.shared_experts.gate_proj | 133.2 | - | - | - | - |  x  |  |
| bf16_matvec | mlp.shared_experts.down_proj | 72.4 | - | - | - | - |  x  |  |
| mla_merge | attention | 67.6 | - | - | - | - |  x  |  |
| mla_partial | attention | 65.8 | - | - | - | - |  x  |  |
| matvec_heads | attention | 59.9 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 59.4 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 58.5 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 57.3 | - | - | - | - |  x  |  |
| router | feed_forward | 57.3 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 50.4 | - | - | - | - |  x  |  |
| matvec_heads | attention | 49.0 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 48.1 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 41.6 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 38.5 | - | - | - | - |  x  |  |
| add | layers.add | 38.0 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 38.0 | - | - | - | - |  x  |  |
| add | feed_forward.add | 37.7 | - | - | - | - |  x  |  |
| add | layers.add | 36.9 | - | - | - | - |  x  |  |
