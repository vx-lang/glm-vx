# One decode step, kernel by kernel: fp8, full width, 4 layers, context 64

ncu, each kernel alone with cold caches; not a step time.

| Group | Kernels | GPU us | DRAM MB | GB/s | % of peak BW | us at peak BW |
|---|---|---|---|---|---|---|
| layer 0 | 45 | 365 | 459.5 | 1259 | 26 | 95 |
| layer 1 | 45 | 369 | 460.1 | 1245 | 26 | 96 |
| layer 2 | 45 | 366 | 459.7 | 1257 | 26 | 95 |
| layer 3 | 38 | 511 | 566.3 | 1108 | 23 | 118 |
| head | 2 | 838 | 3810.4 | 4547 | 94 | 791 |

## Layer 0: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.o_proj | 37.3 | 103.28 | 2766 | 58 | 35 | 96 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_TNN | mlp.up_proj | 29.8 | 78.19 | 2627 | 55 | 33 | 96 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_TNN | mlp.gate_proj | 29.7 | 78.21 | 2634 | 55 | 33 | 96 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | mlp.down_proj | 28.4 | 78.22 | 2756 | 57 | 35 | 96 x 384 | 168 |
| mla_partial | attention.attention_core | 21.2 | 0.20 | 9 | 0 | 8 | 64 x 512 | 32 |
| nvjet_qqsss_128x128_128x6_2x1_v_bz_coopA_algo2_Ablk128_Bvec128_TNN | self_attn.q_b_proj | 16.2 | 33.79 | 2091 | 44 | 27 | 128 x 384 | 168 |
| index_scores | attention_core.index_core | 15.9 | 0.06 | 4 | 0 | 1 | 17 x 128 | 32 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.q_a_proj | 13.1 | 12.64 | 963 | 20 | 13 | 32 x 384 | 168 |
| matvec_heads | attention.attention_core | 12.8 | 34.44 | 2684 | 56 | 31 | 4096 x 128 | 32 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.kv_a_proj_with_mqa | 12.7 | 3.60 | 283 | 6 | 6 | 16 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_1x1_h_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.indexer.wk | 12.3 | 0.84 | 68 | 1 | 1 | 2 x 384 | 168 |
| matvec_heads | attention.attention_core | 12.3 | 25.24 | 2060 | 43 | 52 | 8192 x 128 | 32 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_TNN | self_attn.indexer.wq_b | 10.6 | 8.43 | 798 | 17 | 10 | 32 x 384 | 168 |
| topk_sort | index_core.select | 8.1 | 0.01 | 1 | 0 | 0 | 1 x 1024 | 22 |
| rmsnorm | layer.rmsnorm | 7.4 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| rmsnorm | layer.rmsnorm | 6.9 | 0.06 | 9 | 0 | 0 | 1 x 256 | 32 |
| matvec_rows | self_attn.indexer.weights_proj | 4.6 | 0.82 | 180 | 4 | 4 | 32 x 1024 | 32 |
| rmsnorm | attention.rmsnorm | 4.0 | 0.03 | 7 | 0 | 0 | 1 x 256 | 32 |
| mla_merge | attention.attention_core | 3.7 | 0.14 | 39 | 1 | 3 | 64 x 256 | 32 |
| fp8_quant_groups | self_attn.q_b_proj | 3.5 | 0.02 | 6 | 0 | 0 | 4 x 128 | 24 |
| fp8_quant_groups | self_attn.indexer.wk | 3.5 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| splitKreduce_kernel | self_attn.o_proj | 3.5 | 0.21 | 59 | 1 | 12 | 192 x 512 | 32 |
| fp8_quant_groups | mlp.gate_proj | 3.5 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| fp8_quant_groups | self_attn.o_proj | 3.5 | 0.08 | 23 | 0 | 2 | 32 x 128 | 24 |
| fp8_quant_groups | self_attn.indexer.wq_b | 3.4 | 0.02 | 6 | 0 | 0 | 4 x 128 | 24 |
| splitKreduce_kernel | mlp.down_proj | 3.4 | 0.21 | 60 | 1 | 12 | 192 x 512 | 32 |
| fp8_quant_groups | mlp.down_proj | 3.4 | 0.06 | 18 | 0 | 1 | 24 x 128 | 24 |
| fp8_quant_groups | self_attn.q_a_proj | 3.3 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| fp8_quant_groups | self_attn.kv_a_proj_with_mqa | 3.3 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| fp8_quant_groups | mlp.up_proj | 3.3 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| layernorm | attention_core.index_core | 3.2 | 0.03 | 8 | 0 | 0 | 1 x 256 | 32 |
| splitKreduce_kernel | self_attn.kv_a_proj_with_mqa | 3.2 | 0.03 | 9 | 0 | 1 | 18 x 512 | 32 |
| rmsnorm | attention_core.rmsnorm | 3.2 | 0.01 | 5 | 0 | 0 | 1 x 256 | 32 |
| splitKreduce_kernel | self_attn.q_a_proj | 3.1 | 0.07 | 24 | 0 | 4 | 64 x 512 | 32 |
| splitKreduce_kernel | self_attn.indexer.wk | 3.1 | 0.01 | 4 | 0 | 0 | 4 x 512 | 32 |
| swiglu | mlp.swiglu | 2.8 | 0.11 | 38 | 1 | 1 | 48 x 256 | 25 |
| rope_strided | attention_core.rope | 2.8 | 0.02 | 8 | 0 | 0 | 8 x 256 | 32 |
| add | layer.add | 2.7 | 0.06 | 22 | 0 | 1 | 24 x 256 | 32 |
| add | layer.add | 2.4 | 0.06 | 24 | 1 | 1 | 24 x 256 | 32 |
| rope_strided | index_core.rope | 2.4 | 0.01 | 5 | 0 | 0 | 4 x 256 | 32 |
| rope_strided | attention_core.rope | 2.3 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
| copy | index_core.copy | 2.3 | 0.01 | 4 | 0 | 0 | 1 x 256 | 32 |
| rope_strided | index_core.rope | 2.3 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
| copy | index_core.copy | 2.3 | 0.02 | 10 | 0 | 0 | 16 x 256 | 32 |
| scale | attention_core.index_core | 2.3 | 0.01 | 4 | 0 | 0 | 1 x 256 | 32 |

## Layer 3: kernels by GPU time

| Kernel | What | GPU us | DRAM MB | GB/s | DRAM % | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|
| fp8_moe4_gate_up | mlp.experts | 163.2 | 204.11 | 1251 | 26 | 35 | 4096 x 128 | 72 |
| fp8_moe4_down | mlp.experts | 91.8 | 103.38 | 1126 | 23 | 48 | 6144 x 256 | 56 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.o_proj | 36.8 | 103.27 | 2804 | 58 | 36 | 96 x 384 | 168 |
| mla_partial | attention.attention_core | 20.0 | 0.20 | 10 | 0 | 9 | 64 x 512 | 32 |
| nvjet_qqsss_128x128_128x6_2x1_v_bz_coopA_algo2_Ablk128_Bvec128_TNN | self_attn.q_b_proj | 16.2 | 33.85 | 2086 | 43 | 27 | 128 x 384 | 168 |
| matvec_heads | attention.attention_core | 13.6 | 34.45 | 2527 | 53 | 29 | 4096 x 128 | 32 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | mlp.shared_experts.up_proj | 13.1 | 12.64 | 964 | 20 | 13 | 32 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | mlp.shared_experts.gate_proj | 13.1 | 12.64 | 968 | 20 | 13 | 32 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.q_a_proj | 12.9 | 12.64 | 980 | 20 | 13 | 32 x 384 | 168 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_splitK_TNN | self_attn.kv_a_proj_with_mqa | 12.7 | 3.60 | 282 | 6 | 6 | 16 x 384 | 168 |
| matvec_heads | attention.attention_core | 12.3 | 25.25 | 2060 | 43 | 52 | 8192 x 128 | 32 |
| nvjet_qqsss_128x128_128x6_4x1_v_bz_coopA_algo2_Ablk128_Bvec128_TNN | mlp.shared_experts.down_proj | 10.7 | 12.63 | 1185 | 25 | 15 | 48 x 384 | 168 |
| rmsnorm | layer.rmsnorm | 7.3 | 0.06 | 8 | 0 | 0 | 1 x 256 | 32 |
| router | feed_forward | 7.0 | 0.01 | 2 | 0 | 0 | 1 x 32 | 40 |
| rmsnorm | layer.rmsnorm | 6.9 | 0.06 | 9 | 0 | 0 | 1 x 256 | 32 |
| matvec_rows | mlp.gate | 5.7 | 6.33 | 1117 | 23 | 24 | 256 x 1024 | 32 |
| mla_merge | attention.attention_core | 3.8 | 0.14 | 37 | 1 | 3 | 64 x 256 | 32 |
| rmsnorm | attention.rmsnorm | 3.7 | 0.03 | 7 | 0 | 0 | 1 x 256 | 32 |
| splitKreduce_kernel | self_attn.o_proj | 3.6 | 0.21 | 58 | 1 | 11 | 192 x 512 | 32 |
| fp8_quant_groups | self_attn.q_a_proj | 3.5 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| fp8_quant_groups | self_attn.q_b_proj | 3.4 | 0.02 | 6 | 0 | 0 | 4 x 128 | 24 |
| fp8_quant_groups | mlp.shared_experts.gate_proj | 3.4 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| fp8_quant_groups | self_attn.o_proj | 3.4 | 0.08 | 23 | 0 | 2 | 32 x 128 | 24 |
| fp8_quant_groups | self_attn.kv_a_proj_with_mqa | 3.3 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| fp8_quant_groups | mlp.shared_experts.up_proj | 3.3 | 0.04 | 11 | 0 | 1 | 12 x 128 | 24 |
| splitKreduce_kernel | mlp.shared_experts.up_proj | 3.3 | 0.07 | 23 | 0 | 4 | 64 x 512 | 32 |
| fp8_quant_groups | mlp.shared_experts.down_proj | 3.2 | 0.02 | 7 | 0 | 0 | 4 x 128 | 24 |
| splitKreduce_kernel | self_attn.kv_a_proj_with_mqa | 3.2 | 0.03 | 9 | 0 | 1 | 18 x 512 | 32 |
| splitKreduce_kernel | mlp.shared_experts.gate_proj | 3.2 | 0.07 | 24 | 0 | 4 | 64 x 512 | 32 |
| splitKreduce_kernel | self_attn.q_a_proj | 3.1 | 0.07 | 24 | 0 | 4 | 64 x 512 | 32 |
| rmsnorm | attention_core.rmsnorm | 3.1 | 0.01 | 5 | 0 | 0 | 1 x 256 | 32 |
| swiglu | mlp.swiglu | 2.7 | 0.02 | 9 | 0 | 0 | 8 x 256 | 25 |
| rope_strided | attention_core.rope | 2.6 | 0.02 | 8 | 0 | 0 | 8 x 256 | 32 |
| add | layer.add | 2.6 | 0.06 | 23 | 0 | 1 | 24 x 256 | 32 |
| add | feed_forward.add | 2.4 | 0.06 | 24 | 1 | 1 | 24 x 256 | 32 |
| copy | feed_forward.copy | 2.4 | 0.03 | 13 | 0 | 1 | 24 x 256 | 32 |
| add | layer.add | 2.4 | 0.06 | 24 | 1 | 1 | 24 x 256 | 32 |
| rope_strided | attention_core.rope | 2.3 | 0.00 | 2 | 0 | 0 | 1 x 256 | 32 |
