# One decode step, kernel by kernel: fp8, full width, 4 layers, context 64

ncu, each kernel alone with cold caches; not a step time.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 26 | 321 | 455.8 | 1422 | 30 | 95 |
| layer 1 | 26 | 321 | 456.6 | 1423 | 30 | 95 |
| layer 2 | 26 | 320 | 456.3 | 1428 | 30 | 95 |
| layer 3 | 22 | 364 | 566.1 | 1556 | 32 | 118 |
| head | 2 | 854 | 3810.4 | 4461 | 93 | 791 |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe_fma_gate_up | mlp.fma_gate_up | 69.2 | 153.61 | 2220 | 46 | 55 | 3072 x 128 | 122 |
| fp8_matvec_fma | self_attn.o_proj | 44.3 | 103.29 | 2330 | 48 | 43 | 768 x 128 | 79 |
| fp8_matvec_fma | mlp.down_proj | 35.7 | 78.16 | 2191 | 46 | 40 | 768 x 128 | 79 |
| residual_rmsnorm | layer.residual_rmsnorm | 21.7 | 0.08 | 4 | 0 | 0 | 1 x 256 | 32 |
| mla_partial | attention.attention_core | 20.3 | 0.20 | 10 | 0 | 9 | 64 x 512 | 32 |
| fp8_matvec_fma_tiles | self_attn.q_b_proj | 17.9 | 34.04 | 1906 | 40 | 50 | 2048 x 128 | 87 |
| index_scores | attention_core.index_core | 14.3 | 0.06 | 4 | 0 | 1 | 17 x 128 | 32 |
| fp8_matvec_fma_tiles3 | attention.fma_qkv_a | 12.6 | 17.01 | 1345 | 28 | 31 | 344 x 128 | 122 |
| matvec_heads | attention.attention_core | 12.6 | 34.40 | 2735 | 57 | 32 | 4096 x 128 | 32 |
| matvec_heads | attention.attention_core | 12.3 | 25.24 | 2059 | 43 | 52 | 8192 x 128 | 32 |
| topk_sort | index_core.select | 7.8 | 0.01 | 1 | 0 | 0 | 1 x 1024 | 22 |
| rmsnorm | layer.rmsnorm | 7.3 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| fp8_matvec_fma_tiles | self_attn.indexer.wq_b | 6.6 | 8.43 | 1285 | 27 | 34 | 512 x 128 | 87 |
| matvec_rows | self_attn.indexer.weights_proj | 4.7 | 0.82 | 175 | 4 | 4 | 32 x 1024 | 32 |
| rmsnorm | attention.rmsnorm | 3.9 | 0.03 | 7 | 0 | 0 | 1 x 256 | 32 |
| mla_merge | attention.attention_core | 3.7 | 0.14 | 38 | 1 | 3 | 64 x 256 | 32 |
| layernorm | attention_core.index_core | 3.3 | 0.03 | 8 | 0 | 0 | 1 x 256 | 32 |
| rmsnorm | attention_core.rmsnorm | 2.9 | 0.02 | 5 | 0 | 0 | 1 x 256 | 32 |
| rope_strided | index_core.rope | 2.6 | 0.01 | 5 | 0 | 0 | 4 x 256 | 32 |
| add | layer.add | 2.6 | 0.06 | 22 | 0 | 1 | 24 x 256 | 32 |
| rope_strided | attention_core.rope | 2.6 | 0.02 | 8 | 0 | 0 | 8 x 256 | 32 |
| rope_strided | attention_core.rope | 2.5 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
| copy | index_core.copy | 2.5 | 0.01 | 3 | 0 | 0 | 1 x 256 | 32 |
| rope_strided | index_core.rope | 2.4 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
| copy | index_core.copy | 2.3 | 0.02 | 10 | 0 | 0 | 16 x 256 | 32 |
| scale | attention_core.index_core | 2.2 | 0.01 | 4 | 0 | 0 | 1 x 256 | 32 |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe_fma_gate_up | mlp.experts | 91.4 | 204.22 | 2234 | 46 | 57 | 4096 x 128 | 122 |
| fp8_moe_fma_down | mlp.experts | 65.1 | 103.54 | 1591 | 33 | 50 | 3072 x 256 | 95 |
| fp8_matvec_fma | self_attn.o_proj | 44.9 | 103.48 | 2303 | 48 | 42 | 768 x 128 | 79 |
| residual_rmsnorm | layer.residual_rmsnorm | 20.8 | 0.08 | 4 | 0 | 0 | 1 x 256 | 32 |
| mla_partial | attention.attention_core | 20.0 | 0.20 | 10 | 0 | 9 | 64 x 512 | 32 |
| fp8_matvec_fma_tiles | self_attn.q_b_proj | 18.1 | 34.17 | 1890 | 39 | 49 | 2048 x 128 | 87 |
| fp8_moe_fma_gate_up | mlp.fma_gate_up | 14.3 | 25.25 | 1765 | 37 | 45 | 512 x 128 | 122 |
| fp8_matvec_fma_tiles3 | attention.fma_qkv_a | 12.5 | 16.22 | 1293 | 27 | 30 | 328 x 128 | 122 |
| matvec_heads | attention.attention_core | 12.5 | 34.36 | 2739 | 57 | 32 | 4096 x 128 | 32 |
| matvec_heads | attention.attention_core | 12.1 | 25.24 | 2086 | 43 | 52 | 8192 x 128 | 32 |
| fp8_matvec_fma_tiles | mlp.shared_experts.down_proj | 9.6 | 12.62 | 1311 | 27 | 35 | 768 x 128 | 87 |
| rmsnorm | layer.rmsnorm | 7.0 | 0.06 | 9 | 0 | 0 | 1 x 256 | 32 |
| router | feed_forward | 6.8 | 0.02 | 2 | 0 | 0 | 1 x 32 | 40 |
| matvec_rows | mlp.gate | 5.6 | 6.33 | 1123 | 23 | 25 | 256 x 1024 | 32 |
| rmsnorm | attention.rmsnorm | 4.0 | 0.03 | 7 | 0 | 0 | 1 x 256 | 32 |
| mla_merge | attention.attention_core | 3.7 | 0.14 | 39 | 1 | 3 | 64 x 256 | 32 |
| rmsnorm | attention_core.rmsnorm | 3.1 | 0.01 | 5 | 0 | 0 | 1 x 256 | 32 |
| rope_strided | attention_core.rope | 2.6 | 0.02 | 8 | 0 | 0 | 8 x 256 | 32 |
| add | feed_forward.add | 2.5 | 0.06 | 23 | 0 | 1 | 24 x 256 | 32 |
| copy | feed_forward.copy | 2.4 | 0.03 | 13 | 0 | 1 | 24 x 256 | 32 |
| add | layer.add | 2.3 | 0.06 | 25 | 1 | 1 | 24 x 256 | 32 |
| rope_strided | attention_core.rope | 2.3 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
