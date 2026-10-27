# One decode step, kernel by kernel: fp8, full width, 4 layers, context 64

CUDA events around each launch of a warm step (ncu had no counters): GPU time only, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 2657 | - | - | - | - |
| layer 1 | 31 | 2645 | - | - | - | - |
| layer 2 | 31 | 2574 | - | - | - | - |
| layer 3 | 26 | 2770 | - | - | - | - |
| head | 2 | 1261 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| topk_stable | index | 357.8 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.o_proj | 245.8 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.down_proj | 193.2 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.gate_proj | 182.7 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.up_proj | 181.4 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_a_proj | 110.2 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.kv_a_proj_with_mqa | 106.6 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_b_proj | 102.1 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.indexer.wk | 98.5 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 74.0 | - | - | - | - |  x  |  |
| scale | index | 68.3 | - | - | - | - |  x  |  |
| mla_partial | attention | 66.1 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.indexer.wq_b | 65.5 | - | - | - | - |  x  |  |
| index_scores | index | 56.9 | - | - | - | - |  x  |  |
| matvec_heads | attention | 51.5 | - | - | - | - |  x  |  |
| matvec_heads | attention | 50.1 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 50.0 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 46.1 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 45.9 | - | - | - | - |  x  |  |
| layernorm | index | 45.8 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 45.8 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 45.7 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 44.9 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 44.1 | - | - | - | - |  x  |  |
| mla_merge | attention | 42.6 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 41.0 | - | - | - | - |  x  |  |
| copy | index.copy | 40.6 | - | - | - | - |  x  |  |
| add | layers.add | 40.2 | - | - | - | - |  x  |  |
| copy | index.copy | 39.5 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 37.9 | - | - | - | - |  x  |  |
| add | layers.add | 36.5 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe_gate_up | mlp.experts | 690.2 | - | - | - | - |  x  |  |
| fp8_moe_down | mlp.experts | 429.3 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.o_proj | 247.7 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.shared_experts.gate_proj | 105.8 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.shared_experts.up_proj | 104.7 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.kv_a_proj_with_mqa | 104.2 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_a_proj | 103.5 | - | - | - | - |  x  |  |
| fp8_matvec | self_attn.q_b_proj | 96.7 | - | - | - | - |  x  |  |
| router | feed_forward | 74.2 | - | - | - | - |  x  |  |
| add | layers.add | 69.2 | - | - | - | - |  x  |  |
| mla_partial | attention | 65.8 | - | - | - | - |  x  |  |
| fp8_matvec | mlp.shared_experts.down_proj | 61.4 | - | - | - | - |  x  |  |
| add | feed_forward.add | 49.4 | - | - | - | - |  x  |  |
| matvec_heads | attention | 48.9 | - | - | - | - |  x  |  |
| matvec_heads | attention | 48.6 | - | - | - | - |  x  |  |
| mla_merge | attention | 48.1 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 47.5 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 47.1 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 46.0 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 43.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 42.3 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 40.5 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 39.6 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 39.5 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 38.5 | - | - | - | - |  x  |  |
| add | layers.add | 37.4 | - | - | - | - |  x  |  |
