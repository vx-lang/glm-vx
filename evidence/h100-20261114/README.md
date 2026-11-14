# H100 NVL run, 2026-11-14 (#37)

NVIDIA **H100 NVL** (94 GB HBM3, sm_90), driver 595.91.07, 144 vCPUs, Ubuntu 24.04 (glibc 2.39),
CUDA 12.8 ptxas. glm-vx `12cd455` plus the launcher fix 55a4e6c, and Vx main `89832571` from a
bundle built on the pod (`bundle-build.log`; the script fixes are dc55ea1). This is not the SXM of
`h100-20261026`: the NVL's HBM is faster (peak read from the device: about 3.9 TB/s against 3.35),
but its SM clock is lower (boost 1785 MHz against 1980), so compute-bound kernels run about 11%
slower here and memory-bound ones faster.

The run was cut short on purpose. Whole-model checks on one GPU say little about GLM, which runs
across several, so after the correctness steps the run was restarted with the bottleneck profiles
alone (`STEPS`, a166e7d) into `../h100-20261114-profile/`, and stopped after five layer profiles.
Not run: resident INT4 full4 onwards (`resident-int4-full4.txt` is a killed run), the graph
comparisons, the int4/bf16 spawn and x4 layer profiles, moe-, matvec-, topk- and mla-profile,
mla-hint and the stress pass.

## Correctness

| Step | Result |
| --- | --- |
| build-ptx, build-spawn, build-cpu-lib | ok; all 57 spawn kernels assemble for sm_90, `ld.global.nc.v4` 5 / 3 / 9 in fp8 / bf16 / int4_matvec4 |
| test-hardware / -quant / -stress | 28 / 38 (1 skipped) / 25 passed |
| test-spawn-hardware | 137 passed, 1 xfailed: the x4 matvecs, fp8_matvec4_rows, argmax_rows at 154880 columns, topk_chunks/merge and the radix select to 2^20 tokens, mla_heads_at |
| test-spawn-vx-host / -graph | 200 / 8 passed |
| test-spawn-vx-launch, and with VX_CUDA_SYNC | 9 of 59 failed in the run (`test-spawn-vx-launch*.txt`): `vx_launch.py` had no `u32` element, and argmax_rows' case passed a flat x. With 55a4e6c, rerun on the pod: **59 / 59 passed** both ways |
| test-resident-device-experts / -fused-attention / -graph | 5 / 1 / 7 passed |
| test-resident-many | 6 passed: `forward_many` bit for bit `forward` one token at a time, fp8 x4 included, and under `graph=True` |
| test-speculative-resident | **12 passed**: MTP drafting on the device with `forward_many` verifying gives plain greedy decoding's tokens, position and cache rows, k = 1 and 3, host and device experts, `graph=True`, and the tokens chosen on the device (#16) |
| resident f32 small / full4 | bitwise (max abs diff 0), top-1 equal over 48 tokens |
| resident INT4 small | top-1 equal, max abs diff 5.2e-6, as on the SXM and the A100 |

## Bottlenecks: one decode step, kernel by kernel

`../h100-20261114-profile/layer-profile-*.md`: four full-width layers (0 dense, 3 MoE) and the
head, context 64, CUDA events with an L2 flush before each launch (ncu has no counters in this
container either: ERR_NVGPUCTRPERM). `layer-gate.md` compares them with `h100-20261026`.

| Format | Dense layer (us) | MoE layer (us) | Head (us) |
| --- | --- | --- | --- |
| f32 | 805 | 1010 | 1051 |
| int4 (hand) | 1048 | 1156 | 1051 |
| fp8 (hand) | 1368 | 1435 | 1052 |
| **fp8 x4** | **596** | **635** | 1050 |
| bf16 (hand) | 2195 | 1860 | 2651 |

- **fp8 x4 (#30, #31) more than halves the FP8 step**: 6585 -> 3457 us for the four layers and
  the head (-47%). o_proj 242 -> 92 us, gate/up 175 -> 58, down 188 -> 69, the routed experts
  544 -> 232 us. Still 28 to 33% of peak bandwidth against the 60% target (o_proj 25.6 us at
  peak, the experts 76.7).
- **The quantized hand kernels are compute bound, not memory bound**: f32 reads 4x FP8's bytes
  and is still the faster format. The INT4 hand kernels are 11 to 12% slower than on the SXM with
  byte-identical PTX (`ptx.sha256`), the NVL/SXM clock ratio; the layer gate counts these as its
  13 regressions, but no code changed.
- **What is left in an fp8 x4 layer (596 us)**: the FP8 projections about 360 us; MLA attention
  (`mla_partial`, `mla_merge`, two `matvec_heads`) 68 us at 64 tokens; the indexer's scores and
  norms 42 us; the two layer rmsnorms 35 us for 6144 values (about 2 us at peak); the router 32 us
  (one warp) and its f32 gate matvec 23 us; the indexer's f32 weights_proj 22 us; top-k 10.6 us
  (#32's < 20 us met at this context). Each small kernel costs at least about 4 us here.
- **The head**: lm_head, 154880 x 6144, takes 1050 us in f32 (at peak) and 2651 us through the
  BF16 hand kernel. GLM-5.3's checkpoint keeps lm_head in BF16, so at batch 1 the head alone is
  2.65 ms a token today against about 0.49 ms at peak: bf16_matvec4 (#30) on the head is the
  largest single item measured here.
