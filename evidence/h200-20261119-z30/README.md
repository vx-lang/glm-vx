# Vx PR #3649 (`f8e8m0-mxfp4-scales` at 4d15f5315) on the H200: no change on Hopper

vx-7e asked for this run: the PR moves vx-expand-narrow-floats to the start of both pipelines, so every fp8
conversion becomes integer code before anything folds it. vxc was built on the pod and restored to
`~/glm-h200/vxtc-4d15` (`z30-vx.sh`, `z30b.sh`).

**Vx's tests:**

| test | GPU | CPU (`CUDA_VISIBLE_DEVICES=`) |
|---|---|---|
| `lane_matmuls_on_tensor_cores.vx` (bf16, f16) | `bf16: -3 -2 7 sum of all 2041`, `f16: -3 3 -2 sum of all -6` | the same lines |
| `lane_matmuls_of_fp8_tiles_give_the_sums_of_a_loop.vx` | `e4m3: -3 -3 7 sum of all 1020`, `e5m2: -5 2 5 sum of all -3` | the same lines |
| `fp8_and_fp4_conversions.vx` (a host program) | | matches its EXPECT lines |
| `f8e8m0_conversions.vx` (a host program) | | matches its EXPECT lines |

For the two matmul tests, nsys shows one CUDA kernel ran on the GPU (`*.nsys.txt`). The CPU runs print one extra
line, "no CUDA device".

**glm-vx's kernels:** all 113 spawn kernels were rebuilt with this vxc.
- No glm-vx kernel uses Vx's fp8 types: ours decode FP8 with integers, an LLVM intrinsic, or inline PTX.
- 100 PTX are identical to the Vx 3609 build. The 13 that differ (`spawn-ptx-diff-b.txt`) are the top-k and
  indexer kernels, `mla_tc_at`, `router_bf16`, `tp_sum_rmsnorm` and `tp_all_reduce`; none of them is an FP8
  conversion kernel.
- GLM-5.3's first 4 layers on the rebuilt kernels (`tp-real-4d15.json`) give the same 32 tokens and the same
  top-5 logits as the 3609 kernels (`../h200-20261119-z29/tp-real-plain.json`).
- `gpu/test_resident_graph_hardware.py` and `gpu/test_moe_mma_hardware.py` pass on the rebuilt kernels: 32
  tests, including the indexer and top-k past `index_topk` (`hw-tests-4d15.log`).

**One behaviour change found.**
- `mla_q_rope_bf16_at` wrote `qr_t[..] = (if live { a * c - b * s } else { 0.0 }) as bf16;`. Vx 3609 compiled
  it; 4d15f5315 refuses it with E3007, "If expression branches have incompatible types: f32 and bf16": the bare
  `0.0` gets bf16.
- A `let y: bf16 = (if c { x } else { 0.0 }) as bf16;` form (`if_literal_as_bf16.vx`) is refused by both, so the
  quirk was already there; what changed between 3609 and 4d15f5315 is that it now reaches this form too.
- glm-vx now uses typed f32 lets (4d1abf7), which give the same values.
