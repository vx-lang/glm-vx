# Layer timing gate

Baseline: `evidence/h100-20261026`. A group is a regression when it is more than 10% and 2 us slower than its reference. Targets are reported, not gated.

## bf16 (`layer-profile-bf16.json`)

### Against baseline layer-events-bf16.json: step 10888 -> 11101 us (+2%), 0 regressions

| Layer | Reference us | Run us | Change |
|---|---|---|---|
| 0 | 1897 | 2195 | +16% |
| 1 | 1898 | 2202 | +16% |
| 2 | 1900 | 2192 | +15% |
| 3 | 2640 | 1860 | -30% |
| head | 2555 | 2651 | +4% |

Layer 0 (dense), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| self_attn.o_proj | bf16_matvec | 358.2 | 380.5 | +6% |  |
| mlp.down_proj | bf16_matvec | 272.5 | 287.7 | +6% |  |
| mlp.up_proj | bf16_matvec | 267.5 | 283.9 | +6% |  |
| mlp.gate_proj | bf16_matvec | 267.6 | 283.2 | +6% |  |
| self_attn.indexer.weights_proj | bf16_matvec | 145.0 | 152.8 | +5% |  |
| self_attn.q_a_proj | bf16_matvec | 136.5 | 143.6 | +5% |  |
| self_attn.indexer.wk | bf16_matvec | 135.8 | 143.5 | +6% |  |
| self_attn.kv_a_proj_with_mqa | bf16_matvec | 133.7 | 141.9 | +6% |  |
| self_attn.q_b_proj | bf16_matvec | 105.0 | 113.2 | +8% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 69.1 | - |  |
| self_attn.indexer.wq_b | bf16_matvec | 52.2 | 52.6 | +1% |  |
| attention_core.index_core | index_scores, layernorm, scale | - | 43.1 | - |  |
| layer.rmsnorm | rmsnorm | - | 35.6 | - |  |
| index_core.select | topk_sort | - | 10.4 | - |  |
| attention_core.rope | rope_strided | - | 9.0 | - |  |
| index_core.copy | copy | - | 9.0 | - |  |
| layer.add | add | - | 9.0 | - |  |
| index_core.rope | rope_strided | - | 8.7 | - |  |
| attention.rmsnorm | rmsnorm | 16.4 | 8.1 | -50% |  |
| attention_core.rmsnorm | rmsnorm | - | 5.5 | - |  |
| mlp.swiglu | swiglu | 6.4 | 5.1 | -21% |  |

Layer 3 (sparse), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| mlp.experts | bf16_moe_down, bf16_moe_gate_up | 1376.3 | 413.9 | -70% |  |
| self_attn.o_proj | bf16_matvec | 359.9 | 381.2 | +6% |  |
| mlp.shared_experts.gate_proj | bf16_matvec | 136.4 | 144.7 | +6% |  |
| mlp.shared_experts.up_proj | bf16_matvec | 137.4 | 144.4 | +5% |  |
| self_attn.q_a_proj | bf16_matvec | 136.2 | 143.9 | +6% |  |
| mlp.gate | bf16_matvec | 134.8 | 142.6 | +6% |  |
| self_attn.kv_a_proj_with_mqa | bf16_matvec | 134.1 | 141.5 | +5% |  |
| self_attn.q_b_proj | bf16_matvec | 107.3 | 113.5 | +6% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 69.1 | - |  |
| mlp.shared_experts.down_proj | bf16_matvec | 52.1 | 54.6 | +5% |  |
| layer.rmsnorm | rmsnorm | - | 33.3 | - |  |
| feed_forward | router | 31.4 | 30.8 | -2% |  |
| attention.rmsnorm | rmsnorm | 16.7 | 9.2 | -45% |  |
| layer.add | add | - | 9.0 | - |  |
| attention_core.rope | rope_strided | - | 8.9 | - |  |
| attention_core.rmsnorm | rmsnorm | - | 5.8 | - |  |
| mlp.swiglu | swiglu | 5.7 | 4.6 | -20% |  |
| feed_forward.copy | copy | 5.9 | 4.5 | -24% |  |
| feed_forward.add | add | 5.8 | 4.4 | -25% |  |

### Targets

| Issue | Layer | What | Run us | us at peak | Value | Target | Met |
|---|---|---|---|---|---|---|---|
| #32 | 0 | topk at 64 tokens | 10.4 | - | 10.4 us | < 20 us | yes |

## f32 (`layer-profile-f32.json`)

### Against baseline layer-events-f32.json: step 4349 -> 4467 us (+3%), 0 regressions

| Layer | Reference us | Run us | Change |
|---|---|---|---|
| 0 | 690 | 805 | +17% |
| 1 | 692 | 798 | +15% |
| 2 | 692 | 802 | +16% |
| 3 | 1055 | 1010 | -4% |
| head | 1220 | 1051 | -14% |

Layer 0 (dense), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| self_attn.o_proj | matvec | 145.5 | 128.3 | -12% |  |
| mlp.down_proj | matvec | 113.0 | 99.1 | -12% |  |
| mlp.gate_proj | matvec | 111.9 | 97.1 | -13% |  |
| mlp.up_proj | matvec | 111.3 | 96.7 | -13% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 70.2 | - |  |
| self_attn.q_b_proj | matvec | 56.3 | 48.6 | -14% |  |
| attention_core.index_core | index_scores, layernorm, scale | - | 42.7 | - |  |
| layer.rmsnorm | rmsnorm | - | 35.3 | - |  |
| self_attn.q_a_proj | matvec | 35.3 | 33.3 | -6% |  |
| self_attn.kv_a_proj_with_mqa | matvec | 26.7 | 25.1 | -6% |  |
| self_attn.indexer.weights_proj | matvec | 21.8 | 22.1 | +1% |  |
| self_attn.indexer.wk | matvec | 22.2 | 22.0 | -1% |  |
| self_attn.indexer.wq_b | matvec | 23.0 | 20.3 | -12% |  |
| index_core.select | topk_sort | - | 10.4 | - |  |
| layer.add | add | - | 8.9 | - |  |
| attention_core.rope | rope_strided | - | 8.7 | - |  |
| index_core.copy | copy | - | 8.5 | - |  |
| index_core.rope | rope_strided | - | 8.5 | - |  |
| attention.rmsnorm | rmsnorm | 16.9 | 7.8 | -54% |  |
| attention_core.rmsnorm | rmsnorm | - | 6.1 | - |  |
| mlp.swiglu | swiglu | 6.4 | 5.4 | -16% |  |

Layer 3 (sparse), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| mlp.experts | moe_down, moe_gate_up | 602.5 | 480.8 | -20% |  |
| self_attn.o_proj | matvec | 147.4 | 128.9 | -13% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 68.0 | - |  |
| self_attn.q_b_proj | matvec | 56.7 | 49.2 | -13% |  |
| layer.rmsnorm | rmsnorm | - | 33.7 | - |  |
| self_attn.q_a_proj | matvec | 35.1 | 33.6 | -4% |  |
| mlp.shared_experts.up_proj | matvec | 35.4 | 33.5 | -5% |  |
| mlp.shared_experts.gate_proj | matvec | 35.8 | 33.1 | -7% |  |
| feed_forward | router | 30.0 | 30.6 | +2% |  |
| mlp.shared_experts.down_proj | matvec | 28.2 | 25.4 | -10% |  |
| self_attn.kv_a_proj_with_mqa | matvec | 26.0 | 24.7 | -5% |  |
| mlp.gate | matvec | 23.6 | 22.6 | -4% |  |
| attention_core.rope | rope_strided | - | 9.0 | - |  |
| layer.add | add | - | 8.7 | - |  |
| attention.rmsnorm | rmsnorm | 16.6 | 8.4 | -49% |  |
| attention_core.rmsnorm | rmsnorm | - | 6.1 | - |  |
| mlp.swiglu | swiglu | 5.8 | 4.9 | -16% |  |
| feed_forward.add | add | 5.6 | 4.4 | -22% |  |
| feed_forward.copy | copy | 5.9 | 4.2 | -29% |  |

### Targets

| Issue | Layer | What | Run us | us at peak | Value | Target | Met |
|---|---|---|---|---|---|---|---|
| #32 | 0 | topk at 64 tokens | 10.4 | - | 10.4 us | < 20 us | yes |

## fp8 (`layer-profile-fp8.json`)

### Against baseline layer-events-fp8.json: step 6245 -> 6585 us (+5%), 0 regressions

| Layer | Reference us | Run us | Change |
|---|---|---|---|
| 0 | 1109 | 1368 | +23% |
| 1 | 1107 | 1363 | +23% |
| 2 | 1106 | 1366 | +24% |
| 3 | 1705 | 1435 | -16% |
| head | 1218 | 1052 | -14% |

Layer 0 (dense), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| self_attn.o_proj | fp8_matvec | 224.3 | 242.0 | +8% |  |
| mlp.down_proj | fp8_matvec | 176.0 | 187.6 | +7% |  |
| mlp.up_proj | fp8_matvec | 160.4 | 175.8 | +10% |  |
| mlp.gate_proj | fp8_matvec | 161.9 | 174.2 | +8% |  |
| self_attn.q_a_proj | fp8_matvec | 80.1 | 83.8 | +5% |  |
| self_attn.kv_a_proj_with_mqa | fp8_matvec | 78.3 | 82.3 | +5% |  |
| self_attn.indexer.wk | fp8_matvec | 77.4 | 78.8 | +2% |  |
| self_attn.q_b_proj | fp8_matvec | 73.0 | 78.2 | +7% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 68.0 | - |  |
| attention_core.index_core | index_scores, layernorm, scale | - | 41.9 | - |  |
| layer.rmsnorm | rmsnorm | - | 35.4 | - |  |
| self_attn.indexer.wq_b | fp8_matvec | 33.3 | 33.7 | +1% |  |
| self_attn.indexer.weights_proj | matvec | 22.2 | 21.0 | -5% |  |
| index_core.select | topk_sort | - | 10.4 | - |  |
| layer.add | add | - | 9.7 | - |  |
| attention_core.rope | rope_strided | - | 8.7 | - |  |
| index_core.rope | rope_strided | - | 8.7 | - |  |
| index_core.copy | copy | - | 8.7 | - |  |
| attention.rmsnorm | rmsnorm | 16.1 | 8.3 | -48% |  |
| attention_core.rmsnorm | rmsnorm | - | 5.8 | - |  |
| mlp.swiglu | swiglu | 6.2 | 5.1 | -19% |  |

Layer 3 (sparse), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| mlp.experts | fp8_moe_down, fp8_moe_gate_up | 963.7 | 543.5 | -44% |  |
| self_attn.o_proj | fp8_matvec | 223.8 | 241.0 | +8% |  |
| mlp.shared_experts.up_proj | fp8_matvec | 80.2 | 83.6 | +4% |  |
| self_attn.q_a_proj | fp8_matvec | 80.5 | 83.4 | +4% |  |
| mlp.shared_experts.gate_proj | fp8_matvec | 80.6 | 83.2 | +3% |  |
| self_attn.kv_a_proj_with_mqa | fp8_matvec | 77.8 | 81.7 | +5% |  |
| self_attn.q_b_proj | fp8_matvec | 73.5 | 79.1 | +8% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 69.5 | - |  |
| mlp.shared_experts.down_proj | fp8_matvec | 36.3 | 37.5 | +4% |  |
| layer.rmsnorm | rmsnorm | - | 32.6 | - |  |
| feed_forward | router | 31.4 | 31.6 | +1% |  |
| mlp.gate | matvec | 23.4 | 22.7 | -3% |  |
| layer.add | add | - | 9.0 | - |  |
| attention_core.rope | rope_strided | - | 8.9 | - |  |
| attention.rmsnorm | rmsnorm | 16.4 | 8.7 | -47% |  |
| attention_core.rmsnorm | rmsnorm | - | 5.6 | - |  |
| mlp.swiglu | swiglu | 6.0 | 4.9 | -20% |  |
| feed_forward.copy | copy | 5.6 | 4.5 | -20% |  |
| feed_forward.add | add | 6.0 | 4.4 | -28% |  |

### Targets

| Issue | Layer | What | Run us | us at peak | Value | Target | Met |
|---|---|---|---|---|---|---|---|
| #30 | 0 | self_attn.o_proj | 242.0 | 25.6 | 11% | >= 60% | no |
| #30 | 0 | mlp.gate_proj | 174.2 | 19.2 | 11% | >= 60% | no |
| #30 | 0 | mlp.up_proj | 175.8 | 19.2 | 11% | >= 60% | no |
| #30 | 0 | mlp.down_proj | 187.6 | 19.2 | 10% | >= 60% | no |
| #30 | 0 | self_attn.q_a_proj | 83.8 | 3.2 | 4% | >= 60% | no |
| #31 | 3 | mlp.experts | 543.5 | 76.7 | 14% | >= 60% | no |
| #32 | 0 | topk at 64 tokens | 10.4 | - | 10.4 us | < 20 us | yes |

## fp8-x4 (`layer-profile-fp8x4.json`)

### Against baseline layer-events-fp8.json: step 6245 -> 3457 us (-45%), 0 regressions

| Layer | Reference us | Run us | Change |
|---|---|---|---|
| 0 | 1109 | 596 | -46% |
| 1 | 1107 | 586 | -47% |
| 2 | 1106 | 590 | -47% |
| 3 | 1705 | 635 | -63% |
| head | 1218 | 1050 | -14% |

Layer 0 (dense), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| self_attn.o_proj | fp8_matvec4 (was fp8_matvec) | 224.3 | 91.8 | -59% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 69.5 | - |  |
| mlp.down_proj | fp8_matvec4 (was fp8_matvec) | 176.0 | 69.3 | -61% |  |
| mlp.up_proj | fp8_matvec4 (was fp8_matvec) | 160.4 | 57.6 | -64% |  |
| mlp.gate_proj | fp8_matvec4 (was fp8_matvec) | 161.9 | 57.5 | -64% |  |
| attention_core.index_core | index_scores, layernorm, scale | - | 42.9 | - |  |
| layer.rmsnorm | rmsnorm | - | 35.9 | - |  |
| self_attn.q_b_proj | fp8_matvec4 (was fp8_matvec) | 73.0 | 29.0 | -60% |  |
| self_attn.indexer.weights_proj | matvec | 22.2 | 22.0 | -1% |  |
| self_attn.q_a_proj | fp8_matvec4 (was fp8_matvec) | 80.1 | 16.7 | -79% |  |
| self_attn.kv_a_proj_with_mqa | fp8_matvec4 (was fp8_matvec) | 78.3 | 14.1 | -82% |  |
| self_attn.indexer.wk | fp8_matvec4 (was fp8_matvec) | 77.4 | 13.7 | -82% |  |
| self_attn.indexer.wq_b | fp8_matvec4 (was fp8_matvec) | 33.3 | 11.3 | -66% |  |
| index_core.select | topk_sort | - | 10.6 | - |  |
| index_core.copy | copy | - | 9.0 | - |  |
| layer.add | add | - | 9.0 | - |  |
| attention_core.rope | rope_strided | - | 8.7 | - |  |
| index_core.rope | rope_strided | - | 8.5 | - |  |
| attention.rmsnorm | rmsnorm | 16.1 | 8.1 | -50% |  |
| attention_core.rmsnorm | rmsnorm | - | 5.8 | - |  |
| mlp.swiglu | swiglu | 6.2 | 5.1 | -18% |  |

Layer 3 (sparse), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| mlp.experts | fp8_moe4_down, fp8_moe4_gate_up (was fp8_moe_down, fp8_moe_gate_up) | 963.7 | 232.0 | -76% |  |
| self_attn.o_proj | fp8_matvec4 (was fp8_matvec) | 223.8 | 91.6 | -59% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 67.8 | - |  |
| layer.rmsnorm | rmsnorm | - | 34.6 | - |  |
| feed_forward | router | 31.4 | 32.2 | +2% |  |
| self_attn.q_b_proj | fp8_matvec4 (was fp8_matvec) | 73.5 | 28.6 | -61% |  |
| mlp.gate | matvec | 23.4 | 22.9 | -2% |  |
| mlp.shared_experts.gate_proj | fp8_matvec4 (was fp8_matvec) | 80.6 | 17.3 | -78% |  |
| self_attn.q_a_proj | fp8_matvec4 (was fp8_matvec) | 80.5 | 16.5 | -80% |  |
| mlp.shared_experts.up_proj | fp8_matvec4 (was fp8_matvec) | 80.2 | 16.2 | -80% |  |
| mlp.shared_experts.down_proj | fp8_matvec4 (was fp8_matvec) | 36.3 | 15.9 | -56% |  |
| self_attn.kv_a_proj_with_mqa | fp8_matvec4 (was fp8_matvec) | 77.8 | 13.9 | -82% |  |
| attention_core.rope | rope_strided | - | 8.7 | - |  |
| layer.add | add | - | 8.7 | - |  |
| attention.rmsnorm | rmsnorm | 16.4 | 8.7 | -47% |  |
| attention_core.rmsnorm | rmsnorm | - | 6.1 | - |  |
| mlp.swiglu | swiglu | 6.0 | 4.8 | -20% |  |
| feed_forward.add | add | 6.0 | 4.4 | -28% |  |
| feed_forward.copy | copy | 5.6 | 4.3 | -24% |  |

### Against this run, fp8 hand kernels: step 6585 -> 3457 us (-47%), 0 regressions

| Layer | Reference us | Run us | Change |
|---|---|---|---|
| 0 | 1368 | 596 | -56% |
| 1 | 1363 | 586 | -57% |
| 2 | 1366 | 590 | -57% |
| 3 | 1435 | 635 | -56% |
| head | 1052 | 1050 | -0% |

Layer 0 (dense), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| self_attn.o_proj | fp8_matvec4 (was fp8_matvec) | 242.0 | 91.8 | -62% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | 68.0 | 69.5 | +2% |  |
| mlp.down_proj | fp8_matvec4 (was fp8_matvec) | 187.6 | 69.3 | -63% |  |
| mlp.up_proj | fp8_matvec4 (was fp8_matvec) | 175.8 | 57.6 | -67% |  |
| mlp.gate_proj | fp8_matvec4 (was fp8_matvec) | 174.2 | 57.5 | -67% |  |
| attention_core.index_core | index_scores, layernorm, scale | 41.9 | 42.9 | +2% |  |
| layer.rmsnorm | rmsnorm | 35.4 | 35.9 | +2% |  |
| self_attn.q_b_proj | fp8_matvec4 (was fp8_matvec) | 78.2 | 29.0 | -63% |  |
| self_attn.indexer.weights_proj | matvec | 21.0 | 22.0 | +5% |  |
| self_attn.q_a_proj | fp8_matvec4 (was fp8_matvec) | 83.8 | 16.7 | -80% |  |
| self_attn.kv_a_proj_with_mqa | fp8_matvec4 (was fp8_matvec) | 82.3 | 14.1 | -83% |  |
| self_attn.indexer.wk | fp8_matvec4 (was fp8_matvec) | 78.8 | 13.7 | -83% |  |
| self_attn.indexer.wq_b | fp8_matvec4 (was fp8_matvec) | 33.7 | 11.3 | -66% |  |
| index_core.select | topk_sort | 10.4 | 10.6 | +2% |  |
| index_core.copy | copy | 8.7 | 9.0 | +3% |  |
| layer.add | add | 9.7 | 9.0 | -8% |  |
| attention_core.rope | rope_strided | 8.7 | 8.7 | -0% |  |
| index_core.rope | rope_strided | 8.7 | 8.5 | -3% |  |
| attention.rmsnorm | rmsnorm | 8.3 | 8.1 | -3% |  |
| attention_core.rmsnorm | rmsnorm | 5.8 | 5.8 | +0% |  |
| mlp.swiglu | swiglu | 5.1 | 5.1 | +1% |  |

Layer 3 (sparse), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| mlp.experts | fp8_moe4_down, fp8_moe4_gate_up (was fp8_moe_down, fp8_moe_gate_up) | 543.5 | 232.0 | -57% |  |
| self_attn.o_proj | fp8_matvec4 (was fp8_matvec) | 241.0 | 91.6 | -62% |  |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | 69.5 | 67.8 | -3% |  |
| layer.rmsnorm | rmsnorm | 32.6 | 34.6 | +6% |  |
| feed_forward | router | 31.6 | 32.2 | +2% |  |
| self_attn.q_b_proj | fp8_matvec4 (was fp8_matvec) | 79.1 | 28.6 | -64% |  |
| mlp.gate | matvec | 22.7 | 22.9 | +1% |  |
| mlp.shared_experts.gate_proj | fp8_matvec4 (was fp8_matvec) | 83.2 | 17.3 | -79% |  |
| self_attn.q_a_proj | fp8_matvec4 (was fp8_matvec) | 83.4 | 16.5 | -80% |  |
| mlp.shared_experts.up_proj | fp8_matvec4 (was fp8_matvec) | 83.6 | 16.2 | -81% |  |
| mlp.shared_experts.down_proj | fp8_matvec4 (was fp8_matvec) | 37.5 | 15.9 | -58% |  |
| self_attn.kv_a_proj_with_mqa | fp8_matvec4 (was fp8_matvec) | 81.7 | 13.9 | -83% |  |
| attention_core.rope | rope_strided | 8.9 | 8.7 | -2% |  |
| layer.add | add | 9.0 | 8.7 | -3% |  |
| attention.rmsnorm | rmsnorm | 8.7 | 8.7 | -0% |  |
| attention_core.rmsnorm | rmsnorm | 5.6 | 6.1 | +9% |  |
| mlp.swiglu | swiglu | 4.9 | 4.8 | -1% |  |
| feed_forward.add | add | 4.4 | 4.4 | +0% |  |
| feed_forward.copy | copy | 4.5 | 4.3 | -5% |  |

### Targets

| Issue | Layer | What | Run us | us at peak | Value | Target | Met |
|---|---|---|---|---|---|---|---|
| #30 | 0 | self_attn.o_proj | 91.8 | 25.6 | 28% | >= 60% | no |
| #30 | 0 | mlp.gate_proj | 57.5 | 19.2 | 33% | >= 60% | no |
| #30 | 0 | mlp.up_proj | 57.6 | 19.2 | 33% | >= 60% | no |
| #30 | 0 | mlp.down_proj | 69.3 | 19.2 | 28% | >= 60% | no |
| #30 | 0 | self_attn.q_a_proj | 16.7 | 3.2 | 19% | >= 60% | no |
| #31 | 3 | mlp.experts | 232.0 | 76.7 | 33% | >= 60% | no |
| #32 | 0 | topk at 64 tokens | 10.6 | - | 10.6 us | < 20 us | yes |

## int4 (`layer-profile-int4.json`)

### Against baseline layer-events-int4.json: step 4763 -> 5337 us (+12%), 13 regressions

| Layer | Reference us | Run us | Change |
|---|---|---|---|
| 0 | 795 | 1048 | +32% |
| 1 | 796 | 1043 | +31% |
| 2 | 794 | 1040 | +31% |
| 3 | 1158 | 1156 | -0% |
| head | 1220 | 1051 | -14% |

Layer 0 (dense), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| self_attn.o_proj | int4_matvec | 172.3 | 193.0 | +12% | REGRESSION |
| mlp.down_proj | int4_matvec | 133.5 | 148.6 | +11% | REGRESSION |
| mlp.gate_proj | int4_matvec | 120.5 | 133.5 | +11% | REGRESSION |
| mlp.up_proj | int4_matvec | 120.2 | 133.1 | +11% | REGRESSION |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 69.7 | - |  |
| self_attn.q_b_proj | int4_matvec | 57.3 | 60.6 | +6% |  |
| self_attn.q_a_proj | int4_matvec | 51.1 | 52.7 | +3% |  |
| self_attn.kv_a_proj_with_mqa | int4_matvec | 49.2 | 50.2 | +2% |  |
| attention_core.index_core | index_scores, layernorm, scale | - | 42.1 | - |  |
| layer.rmsnorm | rmsnorm | - | 35.1 | - |  |
| self_attn.indexer.weights_proj | matvec | 22.7 | 22.0 | -3% |  |
| self_attn.indexer.wk | matvec | 22.4 | 21.8 | -3% |  |
| self_attn.indexer.wq_b | matvec | 22.6 | 20.5 | -9% |  |
| index_core.select | topk_sort | - | 10.6 | - |  |
| attention_core.rope | rope_strided | - | 9.4 | - |  |
| layer.add | add | - | 8.7 | - |  |
| index_core.copy | copy | - | 8.4 | - |  |
| index_core.rope | rope_strided | - | 8.4 | - |  |
| attention.rmsnorm | rmsnorm | 16.8 | 8.1 | -52% |  |
| attention_core.rmsnorm | rmsnorm | - | 5.8 | - |  |
| mlp.swiglu | swiglu | 6.3 | 5.1 | -19% |  |

Layer 3 (sparse), by run time:

| What | Kernels | Reference us | Run us | Change | |
|---|---|---|---|---|---|
| mlp.experts | int4_moe_down, int4_moe_gate_up | 605.9 | 464.4 | -23% |  |
| self_attn.o_proj | int4_matvec | 173.5 | 192.5 | +11% | REGRESSION |
| attention.attention_core | matvec_heads, mla_merge, mla_partial | - | 68.1 | - |  |
| self_attn.q_b_proj | int4_matvec | 56.5 | 60.6 | +7% |  |
| self_attn.q_a_proj | int4_matvec | 51.8 | 53.1 | +3% |  |
| mlp.shared_experts.gate_proj | int4_matvec | 52.0 | 52.1 | +0% |  |
| mlp.shared_experts.up_proj | int4_matvec | 51.5 | 52.0 | +1% |  |
| self_attn.kv_a_proj_with_mqa | int4_matvec | 49.0 | 50.6 | +3% |  |
| layer.rmsnorm | rmsnorm | - | 33.4 | - |  |
| feed_forward | router | 31.1 | 31.9 | +3% |  |
| mlp.shared_experts.down_proj | int4_matvec | 28.5 | 29.0 | +2% |  |
| mlp.gate | matvec | 24.3 | 23.3 | -4% |  |
| attention_core.rope | rope_strided | - | 8.9 | - |  |
| layer.add | add | - | 8.7 | - |  |
| attention.rmsnorm | rmsnorm | 16.3 | 8.4 | -48% |  |
| attention_core.rmsnorm | rmsnorm | - | 5.6 | - |  |
| mlp.swiglu | swiglu | 5.9 | 4.9 | -18% |  |
| feed_forward.copy | copy | 5.6 | 4.5 | -20% |  |
| feed_forward.add | add | 5.6 | 4.4 | -22% |  |

Regressions in other layers:

- layer 1 self_attn.o_proj: 174.2 -> 192.1 us
- layer 1 mlp.gate_proj: 120.2 -> 132.6 us
- layer 1 mlp.up_proj: 120.6 -> 134.0 us
- layer 1 mlp.down_proj: 134.8 -> 148.8 us
- layer 2 self_attn.o_proj: 172.9 -> 191.6 us
- layer 2 mlp.gate_proj: 120.0 -> 132.2 us
- layer 2 mlp.up_proj: 120.5 -> 132.9 us
- layer 2 mlp.down_proj: 133.6 -> 148.0 us

### Targets

| Issue | Layer | What | Run us | us at peak | Value | Target | Met |
|---|---|---|---|---|---|---|---|
| #32 | 0 | topk at 64 tokens | 10.6 | - | 10.6 us | < 20 us | yes |

**13 regressions.**
