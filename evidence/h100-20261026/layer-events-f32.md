# One decode step, kernel by kernel: f32, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 1224 | - | - | - | - |
| layer 1 | 31 | 1223 | - | - | - | - |
| layer 2 | 31 | 1224 | - | - | - | - |
| layer 3 | 26 | 1185 | - | - | - | - |
| head | 2 | 1220 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| topk_stable | index | 333.2 | - | - | - | - |  x  |  |
| matvec | self_attn.o_proj | 145.5 | - | - | - | - |  x  |  |
| matvec | mlp.down_proj | 113.0 | - | - | - | - |  x  |  |
| matvec | mlp.gate_proj | 111.9 | - | - | - | - |  x  |  |
| matvec | mlp.up_proj | 111.3 | - | - | - | - |  x  |  |
| matvec | self_attn.q_b_proj | 56.3 | - | - | - | - |  x  |  |
| matvec | self_attn.q_a_proj | 35.3 | - | - | - | - |  x  |  |
| index_scores | index | 32.0 | - | - | - | - |  x  |  |
| matvec | self_attn.kv_a_proj_with_mqa | 26.7 | - | - | - | - |  x  |  |
| mla_partial | attention | 23.9 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wq_b | 23.0 | - | - | - | - |  x  |  |
| matvec_heads | attention | 22.2 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wk | 22.2 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 21.8 | - | - | - | - |  x  |  |
| matvec_heads | attention | 20.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 18.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 17.5 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 9.6 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 7.3 | - | - | - | - |  x  |  |
| mla_merge | attention | 6.9 | - | - | - | - |  x  |  |
| layernorm | index | 6.9 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 6.4 | - | - | - | - |  x  |  |
| add | layers.add | 6.1 | - | - | - | - |  x  |  |
| add | layers.add | 6.0 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 5.8 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.6 | - | - | - | - |  x  |  |
| copy | index.copy | 5.6 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 5.5 | - | - | - | - |  x  |  |
| scale | index | 5.5 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.5 | - | - | - | - |  x  |  |
| copy | index.copy | 5.5 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| moe_gate_up | mlp.experts | 369.4 | - | - | - | - |  x  |  |
| moe_down | mlp.experts | 233.1 | - | - | - | - |  x  |  |
| matvec | self_attn.o_proj | 147.4 | - | - | - | - |  x  |  |
| matvec | self_attn.q_b_proj | 56.7 | - | - | - | - |  x  |  |
| matvec | mlp.shared_experts.gate_proj | 35.8 | - | - | - | - |  x  |  |
| matvec | mlp.shared_experts.up_proj | 35.4 | - | - | - | - |  x  |  |
| matvec | self_attn.q_a_proj | 35.1 | - | - | - | - |  x  |  |
| router | feed_forward | 30.0 | - | - | - | - |  x  |  |
| matvec | mlp.shared_experts.down_proj | 28.2 | - | - | - | - |  x  |  |
| matvec | self_attn.kv_a_proj_with_mqa | 26.0 | - | - | - | - |  x  |  |
| mla_partial | attention | 23.9 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 23.6 | - | - | - | - |  x  |  |
| matvec_heads | attention | 22.6 | - | - | - | - |  x  |  |
| matvec_heads | attention | 20.6 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 16.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 16.8 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 9.7 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 6.9 | - | - | - | - |  x  |  |
| mla_merge | attention | 6.8 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 5.9 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.8 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.7 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.7 | - | - | - | - |  x  |  |
| add | layers.add | 5.7 | - | - | - | - |  x  |  |
| add | feed_forward.add | 5.6 | - | - | - | - |  x  |  |
| add | layers.add | 5.6 | - | - | - | - |  x  |  |
