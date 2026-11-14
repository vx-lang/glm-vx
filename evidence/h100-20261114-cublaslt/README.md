# H100 NVL, 2026-11-14 evening: cuBLASLt block-FP8 for the dense FP8 Linears (#29)

Pod root@216.81.245.58 (NVIDIA H100 NVL, driver 595.91.07, system CUDA 12.8, glibc 2.39), toolchain
bundle vx-toolchain-89832571, repo aff72eb plus the working-tree changes committed after it. The pod
went down before the evidence files were copied off, so **this note is transcribed from the run's
terminal output** (the tables are as printed). There are no JSON files. The SGLang profile was not reached.

## What cuBLASLt needed (gpu/cublaslt.py)

* The system cuBLASLt (12.8.4) has no Hopper block scales. The loader skipped it and `build-cublaslt`
  installed the CUDA 13 wheel: `nvidia/cu13/lib/libcublasLt.so.13`, version 131000 (13.10).
* `cublasLtMatmulAlgoGetHeuristic` returned INVALID_VALUE for every shape until the scale pointers were
  set: "The scaling mode for A scale (VEC128_32F) requires a custom pointer for A scale"
  (CUBLASLT_LOG_LEVEL=5). The heuristic now sees a placeholder device pointer.
* With the pointers set, cuBLASLt refuses every token count except multiples of 4 with NOT_SUPPORTED
  (tried m = 1, 2, 3, 5 refused; 4, 8, 12 accepted; n = 2048 and 576, k = 6144). One token is
  multiplied as 4. Rows 1-3 are padding, and output column j depends only on token j.
* With the weights as operand B (`weights_b`), BF16 output is refused even at m = 4. `weights_a` takes
  both output types.
* Accuracy against float64 sums of the same quantized operands is about 1-3e-4 of the largest output,
  not ~1e-6. That fits Hopper's FP8 tensor-core accumulation within a 128-wide k block, which SGLang's
  kernels share. Tests and the profile check now allow 1e-3.

## Hardware tests

`test-hardware` (gpu/test_hardware.py ... gpu/test_cublaslt_hardware.py): 69 passed, 6 skipped. The skips
are the weights_b + uint16 (BF16) cases. gpu/test_cublaslt_hardware.py alone: 22 passed, 6 skipped.
That includes the quantizer bitwise against gpu/fp8_quant.py, one token's scales at stride 4, and the
matmul at m = 1, 2, 4 for both orientations.

## Dense FP8 matvec at GLM-5.3 shapes (validation/matvec_profile.py)

Peak DRAM 3939 GB/s; CUDA events, L2 flushed, median of 20. us (% of peak bandwidth). `cublaslt` is
gpu/fp8_quant.vx and then the matmul, at one token padded to 4. `cublaslt_mm` is the matmul alone.

| Projection | rows x cols | MB | us at peak | hand | spawn | matvec4 | cublaslt | cublaslt_mm |
|---|---|---|---|---|---|---|---|---|
| q_a_proj | 2048 x 6144 | 12.6 | 3.2 | 83.5 (4%) | 52.2 (6%) | 21.2 (15%) | 21.3 (15%) | 17.9 (18%) |
| q_b_proj | 16384 x 2048 | 33.6 | 8.5 | 78.0 (11%) | 57.9 (15%) | 37.0 (23%) | 26.3 (32%) | 23.1 (37%) |
| kv_a_proj | 576 x 6144 | 3.5 | 0.9 | 81.3 (1%) | 50.3 (2%) | 17.8 (5%) | 20.7 (4%) | 17.4 (5%) |
| o_proj | 6144 x 16384 | 100.7 | 25.6 | 238.6 (11%) | 134.5 (19%) | 131.5 (19%) | 53.6 (48%) | 50.2 (51%) |
| mlp.gate/up | 12288 x 6144 | 75.5 | 19.2 | 172.8 (11%) | 108.6 (18%) | 76.9 (25%) | 39.2 (49%) | 35.5 (54%) |
| mlp.down | 6144 x 12288 | 75.5 | 19.2 | 183.5 (10%) | 106.0 (18%) | 100.1 (19%) | 44.7 (43%) | 41.2 (46%) |
| shared_expert.gate/up | 2048 x 6144 | 12.6 | 3.2 | 83.7 (4%) | 52.4 (6%) | 21.0 (15%) | 21.2 (15%) | 18.1 (18%) |
| shared_expert.down | 6144 x 2048 | 12.6 | 3.2 | 36.6 (9%) | 24.0 (13%) | 21.9 (15%) | 17.3 (18%) | 13.9 (23%) |

In one dense layer's eight shapes, cuBLASLt with its quantizer takes 263 us against fp8_matvec4's
427 us. The small shapes sit at a 14-21 us floor for every kernel, which includes the event timer's
own overhead.

## BF16 and INT4 (same harness; lm_head is new)

| Projection | rows x cols | MB | us at peak | hand | spawn | matvec4 |
|---|---|---|---|---|---|---|
| q_a_proj | 2048 x 6144 | 25.2 | 6.4 | 146.5 (4%) | 50.4 (13%) | 20.9 (31%) |
| q_b_proj | 16384 x 2048 | 67.1 | 17.0 | 150.4 (11%) | 44.6 (38%) | 34.4 (50%) |
| kv_a_proj | 576 x 6144 | 7.1 | 1.8 | 146.4 (1%) | 47.0 (4%) | 17.2 (10%) |
| o_proj | 6144 x 16384 | 201.3 | 51.1 | 468.5 (11%) | 124.2 (41%) | 87.7 (58%) |
| mlp.gate/up | 12288 x 6144 | 151.0 | 38.3 | 331.0 (12%) | 98.4 (39%) | 60.3 (64%) |
| mlp.down | 6144 x 12288 | 151.0 | 38.3 | 354.2 (11%) | 96.2 (40%) | 69.0 (56%) |
| shared_expert.gate/up | 2048 x 6144 | 25.2 | 6.4 | 147.2 (4%) | 50.6 (13%) | 20.9 (31%) |
| shared_expert.down | 6144 x 2048 | 25.2 | 6.4 | 66.5 (10%) | 24.8 (26%) | 18.3 (35%) |
| **lm_head** | 154880 x 6144 | 1903.2 | 483.2 | 3447.6 (14%) | 860.3 (56%) | **595.7 (81%)** |

bf16_matvec4 runs the BF16 lm_head at 596 us, 81% of peak. The 2.65 ms/token on the 2026-11-14 morning
pod was the hand kernel. ResidentGLM(matvec_kernels='x4') takes this path.

| INT4 | rows x cols | MB | us at peak | hand | spawn | matvec4 |
|---|---|---|---|---|---|---|
| q_a_proj | 2048 x 6144 | 6.5 | 1.6 | 52.1 (3%) | 52.1 (3%) | 19.4 (8%) |
| q_b_proj | 16384 x 2048 | 17.3 | 4.4 | 60.0 (7%) | 54.0 (8%) | 45.9 (10%) |
| kv_a_proj | 576 x 6144 | 1.8 | 0.5 | 49.9 (1%) | 50.8 (1%) | 13.3 (3%) |
| o_proj | 6144 x 16384 | 51.9 | 13.2 | 190.4 (7%) | 142.3 (9%) | 145.6 (9%) |
| mlp.gate/up | 12288 x 6144 | 38.9 | 9.9 | 131.2 (8%) | 108.8 (9%) | 90.1 (11%) |
| mlp.down | 6144 x 12288 | 38.9 | 9.9 | 147.2 (7%) | 109.9 (9%) | 110.6 (9%) |
| shared_expert.gate/up | 2048 x 6144 | 6.5 | 1.6 | 52.0 (3%) | 51.9 (3%) | 19.2 (9%) |
| shared_expert.down | 6144 x 2048 | 6.5 | 1.6 | 29.0 (6%) | 23.3 (7%) | 27.3 (6%) |

## ResidentGLM(fp8_kernels='cublaslt') at full width

Every dense FP8 Linear of a 2-layer full-geometry model goes through `_project` on the GPU. Each is
compared with the float64 sum of its own quantized operands (the error column), and with the
float32-activation sum (the drift column), both divided by the largest output.

| Linear | rows x cols | error | quantization drift |
|---|---|---|---|
| layers.0.mlp.down_proj | 6144 x 12288 | 1.5e-04 | 2.6e-02 |
| layers.0.mlp.gate_proj | 12288 x 6144 | 1.2e-04 | 2.5e-02 |
| layers.0.mlp.up_proj | 12288 x 6144 | 1.5e-04 | 3.2e-02 |
| layers.0.self_attn.indexer.wk | 128 x 6144 | 1.1e-04 | 2.6e-02 |
| layers.0.self_attn.indexer.wq_b | 4096 x 2048 | 1.6e-04 | 2.6e-02 |
| layers.0.self_attn.kv_a_proj_with_mqa | 576 x 6144 | 1.5e-04 | 2.4e-02 |
| layers.0.self_attn.kv_b_proj | 28672 x 512 | 2.3e-04 | 2.5e-02 |
| layers.0.self_attn.o_proj | 6144 x 16384 | 1.3e-04 | 2.8e-02 |
| layers.0.self_attn.q_a_proj | 2048 x 6144 | 1.2e-04 | 2.3e-02 |
| layers.0.self_attn.q_b_proj | 16384 x 2048 | 1.7e-04 | 2.6e-02 |
| layers.1.* (10 Linears) | | 1.1e-04 to 1.9e-04 | 2.2e-02 to 3.5e-02 |

Whole-model logits (validation/resident_compare --fp8 --fp8-kernels cublaslt --geometry full --layers 2
--tokens 8) against the float32-activation x4 run:
* max |diff| 0.412 against max |logit| 4.71, and the top-1 token differs;
* 288 cuBLASLt launches.

The drift compounds through the layers. That is a property of quantized activations, not an error:
the per-Linear check above is the correctness gate, and resident_compare now uses it for cublaslt.
Whether 8.7% logit drift matters is the comparison with SGLang's own logits, on the same
quantization, still to run.

## One decode step, fp8 x4 (validation/layer_profile, CUDA events: ncu still ERR_NVGPUCTRPERM)

| Group | Kernels | GPU us |
|---|---|---|
| layer 0 (dense) | 31 | 677 |
| layer 1 | 31 | 675 |
| layer 2 | 31 | 675 |
| layer 3 (MoE) | 26 | 710 |
| head | 2 | 1053 (hand BF16 lm_head; x4 measures 596 above) |

Layer 0's largest items are the fp8_matvec4 projections:
* o_proj 131.9, down 100.0, gate 78.0, up 78.0, q_b 38.3, q_a 20.8, kv_a 17.7 us;
* the rest of the layer: mla_partial 23.6, two matvec_heads 20.2 and 18.3, index_scores 18.4, three
  rmsnorms 11.6, 9.5 and 5.7, and 4-5 us elementwise kernels.

Layer 3's largest: fp8_moe4_gate_up 182.9, o_proj 132.1, fp8_moe4_down 104.9.

The cuBLASLt layer profile (`layer-profile-fp8lt`) was running when the pod went down.
