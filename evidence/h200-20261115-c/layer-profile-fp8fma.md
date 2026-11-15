# One decode step, kernel by kernel: fp8, full width, 4 layers, context 64

ncu, each kernel alone with cold caches; not a step time.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 31 | 335 | 459.4 | 1372 | 29 | 95 |
| layer 1 | 31 | 336 | 459.4 | 1368 | 28 | 95 |
| layer 2 | 31 | 333 | 459.1 | 1379 | 29 | 95 |
| layer 3 | 26 | 472 | 565.9 | 1199 | 25 | 118 |
| head | 2 | 838 | 3810.4 | 4548 | 94 | 791 |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_matvec_fma | self_attn.o_proj | 46.2 | 103.51 | 2242 | 47 | 41 | 768 x 128 | 79 |
| fp8_matvec_fma | mlp.down_proj | 35.4 | 78.18 | 2211 | 46 | 41 | 768 x 128 | 79 |
| fp8_matvec_fma_tiles | mlp.up_proj | 33.7 | 78.20 | 2321 | 48 | 50 | 1536 x 128 | 87 |
| fp8_matvec_fma_tiles | mlp.gate_proj | 33.6 | 78.46 | 2333 | 49 | 50 | 1536 x 128 | 87 |
| mla_partial | attention.attention_core | 20.8 | 0.20 | 10 | 0 | 9 | 64 x 512 | 32 |
| fp8_matvec_fma_tiles | self_attn.q_b_proj | 18.0 | 34.13 | 1891 | 39 | 49 | 2048 x 128 | 87 |
| index_scores | attention_core.index_core | 15.6 | 0.06 | 4 | 0 | 1 | 17 x 128 | 32 |
| matvec_heads | attention.attention_core | 12.4 | 34.47 | 2791 | 58 | 32 | 4096 x 128 | 32 |
| matvec_heads | attention.attention_core | 12.3 | 25.26 | 2050 | 43 | 52 | 8192 x 128 | 32 |
| fp8_matvec_fma_tiles | self_attn.q_a_proj | 12.2 | 12.64 | 1034 | 22 | 23 | 256 x 128 | 87 |
| fp8_matvec_fma_tiles | self_attn.kv_a_proj_with_mqa | 10.6 | 3.59 | 338 | 7 | 7 | 72 x 128 | 87 |
| fp8_matvec_fma_tiles | self_attn.indexer.wk | 10.3 | 0.84 | 81 | 2 | 2 | 16 x 128 | 87 |
| topk_sort | index_core.select | 8.1 | 0.01 | 1 | 0 | 0 | 1 x 1024 | 22 |
| rmsnorm | layer.rmsnorm | 7.3 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| rmsnorm | layer.rmsnorm | 7.2 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| fp8_matvec_fma_tiles | self_attn.indexer.wq_b | 7.0 | 8.43 | 1197 | 25 | 32 | 512 x 128 | 87 |
| matvec_rows | self_attn.indexer.weights_proj | 4.7 | 0.82 | 175 | 4 | 4 | 32 x 1024 | 32 |
| rmsnorm | attention.rmsnorm | 4.4 | 0.03 | 6 | 0 | 0 | 1 x 256 | 32 |
| mla_merge | attention.attention_core | 4.0 | 0.14 | 36 | 1 | 2 | 64 x 256 | 32 |
| layernorm | attention_core.index_core | 3.3 | 0.03 | 8 | 0 | 0 | 1 x 256 | 32 |
| rmsnorm | attention_core.rmsnorm | 3.2 | 0.01 | 5 | 0 | 0 | 1 x 256 | 32 |
| swiglu | mlp.swiglu | 2.8 | 0.11 | 37 | 1 | 1 | 48 x 256 | 25 |
| rope_strided | attention_core.rope | 2.5 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
| add | layer.add | 2.5 | 0.06 | 24 | 0 | 1 | 24 x 256 | 32 |
| rope_strided | index_core.rope | 2.4 | 0.01 | 5 | 0 | 0 | 4 x 256 | 32 |
| rope_strided | attention_core.rope | 2.4 | 0.02 | 9 | 0 | 0 | 8 x 256 | 32 |
| add | layer.add | 2.4 | 0.06 | 24 | 0 | 1 | 24 x 256 | 32 |
| copy | index_core.copy | 2.4 | 0.01 | 3 | 0 | 0 | 1 x 256 | 32 |
| rope_strided | index_core.rope | 2.4 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
| copy | index_core.copy | 2.3 | 0.02 | 10 | 0 | 0 | 16 x 256 | 32 |
| scale | attention_core.index_core | 2.3 | 0.01 | 4 | 0 | 0 | 1 x 256 | 32 |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe4_gate_up | mlp.experts | 160.9 | 204.23 | 1269 | 26 | 35 | 4096 x 128 | 72 |
| fp8_moe4_down | mlp.experts | 91.7 | 103.43 | 1128 | 23 | 48 | 6144 x 256 | 56 |
| fp8_matvec_fma | self_attn.o_proj | 43.9 | 103.41 | 2354 | 49 | 43 | 768 x 128 | 79 |
| mla_partial | attention.attention_core | 20.3 | 0.20 | 10 | 0 | 9 | 64 x 512 | 32 |
| fp8_matvec_fma_tiles | self_attn.q_b_proj | 18.0 | 34.09 | 1892 | 39 | 49 | 2048 x 128 | 87 |
| matvec_heads | attention.attention_core | 12.7 | 34.24 | 2702 | 56 | 31 | 4096 x 128 | 32 |
| fp8_matvec_fma_tiles | self_attn.q_a_proj | 12.3 | 12.64 | 1026 | 21 | 23 | 256 x 128 | 87 |
| matvec_heads | attention.attention_core | 12.2 | 25.24 | 2065 | 43 | 52 | 8192 x 128 | 32 |
| fp8_matvec_fma_tiles | mlp.shared_experts.up_proj | 11.5 | 12.64 | 1104 | 23 | 25 | 256 x 128 | 87 |
| fp8_matvec_fma_tiles | mlp.shared_experts.gate_proj | 11.4 | 12.64 | 1107 | 23 | 25 | 256 x 128 | 87 |
| fp8_matvec_fma_tiles | self_attn.kv_a_proj_with_mqa | 10.5 | 3.59 | 341 | 7 | 7 | 72 x 128 | 87 |
| fp8_matvec_fma_tiles | mlp.shared_experts.down_proj | 10.0 | 12.62 | 1260 | 26 | 33 | 768 x 128 | 87 |
| rmsnorm | layer.rmsnorm | 7.3 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| rmsnorm | layer.rmsnorm | 7.2 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| router | feed_forward | 7.0 | 0.01 | 2 | 0 | 0 | 1 x 32 | 40 |
| matvec_rows | mlp.gate | 5.7 | 6.33 | 1111 | 23 | 24 | 256 x 1024 | 32 |
| rmsnorm | attention.rmsnorm | 4.0 | 0.03 | 7 | 0 | 0 | 1 x 256 | 32 |
| mla_merge | attention.attention_core | 3.8 | 0.14 | 38 | 1 | 3 | 64 x 256 | 32 |
| rmsnorm | attention_core.rmsnorm | 3.1 | 0.01 | 5 | 0 | 0 | 1 x 256 | 32 |
| swiglu | mlp.swiglu | 2.9 | 0.02 | 8 | 0 | 0 | 8 x 256 | 25 |
| rope_strided | attention_core.rope | 2.7 | 0.01 | 2 | 0 | 0 | 1 x 256 | 32 |
| rope_strided | attention_core.rope | 2.7 | 0.02 | 8 | 0 | 0 | 8 x 256 | 32 |
| add | layer.add | 2.7 | 0.06 | 22 | 0 | 1 | 24 x 256 | 32 |
| copy | feed_forward.copy | 2.6 | 0.03 | 12 | 0 | 0 | 24 x 256 | 32 |
| add | feed_forward.add | 2.6 | 0.06 | 23 | 0 | 1 | 24 x 256 | 32 |
| add | layer.add | 2.4 | 0.06 | 24 | 0 | 1 | 24 x 256 | 32 |
