# The decode MoE down projection as a plain Vx matmul, on one H200 (2026-11-20)

`gpu/spawn/moe_mma/fp8_moe_vxmm16_down_at` (`ResidentGLM(moe_mma='vx16')`) does the arithmetic of
`fp8_moe_mma16_down_at`, the kernel glm-vx serves with (`moe_mma=True`), but its tensor-core part is a plain Vx
lane matmul, `acc[w] = (down[tile] as f32) @ (xs[w] as f32)`, where the inline PTX kernel has `mma.sync` and its
loads in an `mlir!` block. It needs Vx 10c6f478f or later (Vx #3694: a matmul tile in HBM; #3696: 16-byte loads).

## Claim

**11.90 µs against 13.61 µs (0.874x)** for one decode step's routed-expert down projection of GLM-5.3 in block FP8
at one rank of tensor parallelism 8 (6144 rows, an expert width of 256, 9 experts with the shared one), batch 1, on
one H200, against glm-vx's inline-PTX kernel `fp8_moe_mma16_down_at` built by the same vxc. The median of 10 rounds
each: the fastest and slowest rounds are 11.66 and 12.22 µs against 13.47 and 14.42 µs.

The decode step has one such launch for each of GLM-5.3's 75 MoE layers. This folder measures the kernel alone,
not a served token.

## Setup

What both sides share:

| | |
|---|---|
| GPU | one NVIDIA H200 (141 GB HBM3e), driver 580.173.02, CUDA 13.0 |
| machine | Nebius gpu-h200-sxm, 1gpu-16vcpu-200gb, Ubuntu 24.04.5, kernel 6.11.0-1016-nvidia |
| compiler | vxc of Vx main 10c6f478f; PTX 8.4 for sm_90a, checked by CUDA 13.1 ptxas, compiled by the driver at load |
| inputs | `validation/moe_mma_check.run`'s, seed 1: random e4m3 codes (no NaN) for 257 experts x 6144 x 256, block scales U(0.5e-3, 2e-3), act 0.05 N(0, 1), 9 distinct ids with the shared expert, router weights U(0.01, 0.3) and 1.0 |
| launch | (6144 + 15) / 16 = 384 blocks of 288 threads (9 warps, a warp an expert, 16 rows a block) |
| arithmetic | act in f16 after a power of two a group of 128 that puts the group's max in [2^14, 2^15); the FP8 weights as f16; f32 sums; each group's scales in f32 |
| bytes | the same 14.2 MB from DRAM a launch (ncu, z57): the 9 experts' 6144 x 256 codes and their scales |
| timing | each launch alone: a 512 MB copy flushes L2, then CUDA events; the median of 50 launches a round |
| rounds | 5 a run, the three kernels' order reversed every other round, the GPU idle before each run |

What differs:

| | `fp8_moe_mma16_down_at` (inline PTX) | `fp8_moe_vxmm16_down_at` (plain Vx) |
|---|---|---|
| the tensor-core part | `mma.sync.m16n8k16` and its loads in an `mlir!` block, act in registers | a Vx lane matmul: A = the expert's [16, 256] block of codes in HBM, read 16 bytes at once; B = act, [256, 8] f16, through shared memory |
| registers, shared memory | 72, 576 bytes | 52, 42,176 bytes |
| the order of the experts' sum | ascending id | ascending id |
| inline PTX left | the whole warp body | `slot_of_rank_warp`, the routing slot (shuffles, a ballot, `bfind`), the same as the PTX kernel's |

The output of both is within 2e-7 of a float64 NumPy model of the arithmetic (relative to max |out|), and 6.9e-8
of each other: the same terms and the same order of the experts, but another f32 order inside each dot
product (the matmul's K order) and in the scales' products. `gpu/test_moe_mma_hardware.py` passes with every
variant here (`logs/test_moe_mma_hardware-*.log`); it also checks `ResidentGLM(moe_mma='vx16')` against
`moe_mma=True` on a small model.

## Every run

Microseconds, the median of the run's rounds [fastest, slowest]. `mma` is `fp8_moe_mma_down_at`, the inline PTX
kernel with act in e4m3, for reference.

| `fp8_moe_vxmm16_down_at`'s routing slot | rounds | vxmm16 | mma16 | mma | vxmm16 / mma16 |
|---|---|---|---|---|---|
| **inline PTX** (`slot_of_rank_warp`; the kernel in glm-vx) | 10 | **11.90 [11.66-12.22]** | 13.61 [13.47-14.42] | 13.43 [13.30-13.55] | **0.874** |
| v1: plain Vx, ids in shared memory, 9 x 9 nested reads a lane | 5 | 13.44 [13.17-13.50] | 13.60 [13.52-13.65] | 13.33 [13.26-13.44] | 0.988 |
| v2: plain Vx, ranks once a block (two more block barriers) | 5 | 15.06 [14.83-15.12] | 13.70 [13.60-13.78] | 13.47 [13.44-13.50] | 1.099 |
| v3: plain Vx, ranks once a warp in shared memory (warp syncs) | 5 | 12.96 [12.91-13.06] | 13.66 [13.63-13.71] | 13.46 [13.34-13.50] | 0.948 |
| v4: plain Vx, 9 ids loaded into registers a lane, ranked there | 10 | 14.46 [14.24-14.61] | 13.66 [13.58-13.78] | 13.46 [13.42-13.60] | 1.059 |

`runs.csv` has every round of every run (105 rows); `logs/` has each run's JSON and output; `variants/` has the
four plain Vx routing slots.

## Where the gain comes from

- **The matmul in plain Vx is faster than the inline PTX.** With the same routing slot, 11.90 µs against 13.61.
  ncu of an earlier build (z57: the same kernel but for the scaling of act) read the same DRAM bytes with 37% fewer
  instructions: act is converted to f16 once a block into shared memory, where each lane of the inline PTX kernel
  quantizes its own copy in registers.
- **The routing slot is still cheapest as inline PTX.** Each of the four plain Vx versions costs 1 to 3 µs more:
  a warp shuffle, a ballot and `bfind` have no Vx form yet, and shared memory or block barriers stand in for them.
  So the kernel keeps that one small block of PTX, the same one the inline PTX kernel has.

## Limits

- One GPU and the kernel alone, after an L2 flush: not a decode step, a served token or tensor parallelism on 8
  GPUs. `moe_mma='vx16'` stays off by default until an 8-GPU W1 and GSM8K run.
- Only an expert width of 256 (one rank of 8 on GLM-5.3); other widths use `fp8_moe_mma16_down_at`.
- Random weights and act, one seed; one machine, one day.
- Runs from earlier sessions (z56-z58, 11.34-11.41 µs against 13.22-13.39) agree but are not in this folder's
  rounds.

## Commands

`run.sh <name>` is one run; `moe_down_bench.py` is its script; `env.txt` records the machine, the compiler and the
commits.
