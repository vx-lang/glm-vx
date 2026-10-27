# One decode step, kernel by kernel: int4, full width, 4 layers, context 64

CUDA events around each launch, each after a 512 MB copy (ncu had no counters): GPU time with cold caches, no DRAM bytes.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 1327 | - | - | - | - |
| layer 1 | 31 | 1328 | - | - | - | - |
| layer 2 | 31 | 1325 | - | - | - | - |
| layer 3 | 26 | 1289 | - | - | - | - |
| head | 2 | 1220 | - | - | - | - |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| topk_stable | index | 330.6 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.o_proj | 172.3 | - | - | - | - |  x  |  |
| int4_matvec | mlp.down_proj | 133.5 | - | - | - | - |  x  |  |
| int4_matvec | mlp.gate_proj | 120.5 | - | - | - | - |  x  |  |
| int4_matvec | mlp.up_proj | 120.2 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_b_proj | 57.3 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_a_proj | 51.1 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.kv_a_proj_with_mqa | 49.2 | - | - | - | - |  x  |  |
| index_scores | index | 31.7 | - | - | - | - |  x  |  |
| mla_partial | attention | 24.5 | - | - | - | - |  x  |  |
| matvec_heads | attention | 23.5 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.weights_proj | 22.7 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wq_b | 22.6 | - | - | - | - |  x  |  |
| matvec | self_attn.indexer.wk | 22.4 | - | - | - | - |  x  |  |
| matvec_heads | attention | 20.9 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 19.3 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 17.6 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 9.6 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 7.1 | - | - | - | - |  x  |  |
| mla_merge | attention | 7.0 | - | - | - | - |  x  |  |
| layernorm | index | 6.8 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 6.3 | - | - | - | - |  x  |  |
| add | layers.add | 5.7 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 5.7 | - | - | - | - |  x  |  |
| rope_strided | index.rope | 5.6 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.6 | - | - | - | - |  x  |  |
| copy | index.copy | 5.6 | - | - | - | - |  x  |  |
| add | layers.add | 5.5 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.5 | - | - | - | - |  x  |  |
| scale | index | 5.5 | - | - | - | - |  x  |  |
| copy | index.copy | 5.4 | - | - | - | - |  x  |  |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| int4_moe_gate_up | mlp.experts | 353.9 | - | - | - | - |  x  |  |
| int4_moe_down | mlp.experts | 252.0 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.o_proj | 173.5 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_b_proj | 56.5 | - | - | - | - |  x  |  |
| int4_matvec | mlp.shared_experts.gate_proj | 52.0 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.q_a_proj | 51.8 | - | - | - | - |  x  |  |
| int4_matvec | mlp.shared_experts.up_proj | 51.5 | - | - | - | - |  x  |  |
| int4_matvec | self_attn.kv_a_proj_with_mqa | 49.0 | - | - | - | - |  x  |  |
| router | feed_forward | 31.1 | - | - | - | - |  x  |  |
| int4_matvec | mlp.shared_experts.down_proj | 28.5 | - | - | - | - |  x  |  |
| mla_partial | attention | 24.4 | - | - | - | - |  x  |  |
| matvec | mlp.gate | 24.3 | - | - | - | - |  x  |  |
| matvec_heads | attention | 22.4 | - | - | - | - |  x  |  |
| matvec_heads | attention | 20.1 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 17.6 | - | - | - | - |  x  |  |
| rmsnorm | layers.rmsnorm | 17.4 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 9.2 | - | - | - | - |  x  |  |
| rmsnorm | attention.rmsnorm | 7.0 | - | - | - | - |  x  |  |
| mla_merge | attention | 6.8 | - | - | - | - |  x  |  |
| swiglu | mlp.swiglu | 5.9 | - | - | - | - |  x  |  |
| add | layers.add | 5.8 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.7 | - | - | - | - |  x  |  |
| add | layers.add | 5.6 | - | - | - | - |  x  |  |
| add | feed_forward.add | 5.6 | - | - | - | - |  x  |  |
| copy | feed_forward.copy | 5.6 | - | - | - | - |  x  |  |
| rope_strided | attention.rope | 5.5 | - | - | - | - |  x  |  |
