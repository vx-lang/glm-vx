# H100 run, 2026-10-26/27

NVIDIA H100 80GB HBM3 (SXM), driver 595.91.07, 160 vCPUs, Ubuntu 24.04 (glibc 2.39), CUDA 12.8
ptxas. glm-vx `e69ef71` and Vx main `371c2bae` from the bundle, restored with pod_restore.sh. The
LLVM 22 runtime libraries came from apt.llvm.org (see c4c2f6e). The PTX target is sm_80, which the
driver JITs on sm_90; ptxas checked it for sm_90. This was the first non-A100 run.

`summary.txt` is `scripts/pod_validate.sh`: **33 steps, 0 failed.**

## Correctness: same as the A100

| Step | Result |
| --- | --- |
| test-hardware / -quant / -stress | 28 / 39 / 25 passed. The quant suite includes the real INT4 checkpoint Linears. |
| test-spawn-hardware | 91 passed, 1 xfailed |
| test-spawn-vx-host / -launch / -launch with VX_CUDA_SYNC / -graph | 103 / 42 / 42 / 8 passed |
| test-resident-device-experts / -fused-attention / -graph | 5 / 1 / 7 passed |
| resident f32, FP8, BF16, small and full4 | bitwise. f32 full4 is bitwise against the hybrid path over 20 tokens. |
| resident INT4 small / full4 | top-1 equal, max abs diff 5.2e-6 / 9.1e-6, the same values as on the A100 |
| graph replay f32, INT4, FP8, BF16 (small) | bitwise |
| stress, 8 processes | 0 mismatches |

## Per-kernel GPU time

ncu could not read the GPU's counters in this container (ERR_NVGPUCTRPERM, kept as
`*.ncu.failed.txt`). Every profile here is therefore timed with CUDA events: GPU time only, no
DRAM bytes. Each kernel runs after a 512 MB copy, which leaves its data out of L2 (cold caches, as
under ncu). The peak DRAM bandwidth, read from the device, is 3352 GB/s.

`layer-profile-*` are **superseded**. They timed each launch between events on an idle GPU, so
the host's time preparing a launch landed inside the measurement (every small kernel showed
about 40 us). `layer-events-*` are the corrected rerun (5da66fa: an L2 flush runs before each
launch and keeps the GPU busy while the host queues it). The MoE profile and mla-hint already
flushed this way, so their numbers stand.

### One decode layer, full width (`layer-events-<format>.md`), us

| Format | Dense layer (0) | MoE layer (3) | Output head |
| --- | --- | --- | --- |
| f32 | 1224 | 1185 | 1220 |
| INT4 | 1327 | 1289 | 1220 |
| FP8 | 1642 | 1837 | 1218 |
| BF16 | 2432 | 2773 | 2555 |

**f32 is the fastest format, even though it reads 2 to 8 times the bytes.** The quantized
matvec and expert kernels are limited by their per-element decode, not by memory:
- FP8 `o_proj` reads about 100 MB, which is 30 us at peak bandwidth. It takes 224 us.
- The BF16 output head takes 2555 us, against 1220 us for f32.

The largest kernels, by layer:
- **Dense layer:** `topk_stable`, the indexer's top-k, takes about 330 us in every format, 20 to
  27% of the layer, at only 64 tokens of context. Next come `o_proj` and the three MLP matvecs.
- **MoE layer:** the two fused expert kernels take 600 us in f32, 606 us in INT4, 964 us in FP8
  and 1376 us in BF16. Next comes `o_proj`.
- Attention itself (`mla_partial` + `mla_merge` + `matvec_heads`) takes about 75 us at this
  context.

### Fused MoE expert kernels alone (`moe-profile.md`)

These run at GLM-5.3's shapes: hidden 6144, expert width 2048, 256 experts. At k = 8 routed
experts (GLM-5.3's setting):

| Format | gate_up + down, us | Weight bytes, MB | us at peak bandwidth | % of peak |
| --- | --- | --- | --- | --- |
| f32 | 368 + 232 | 1208 | 360 | 65 / 52 |
| INT4 | 351 + 257 | 156 | 46 | 9 / 6 |
| FP8 | 554 + 406 | 302 | 90 | 11 / 7 |
| BF16 | 1042 + 469 | 604 | 180 | 12 / 13 |

FP8 has 10x headroom, and INT4 13x. At k = 1, gate_up still takes 93 to 265 us, so a single
expert also does not fill the GPU.

### mla_decode_at with `.minnctapersm 1` (`mla-hint.md`), for Vx

On sm_90 ptxas gives the kernel 32 registers and an 8 B spill. The hint gives it 64 registers
and no spill, at 1 block per SM. Outputs match bit for bit. Median us:

| Selected (splits) | As built | With hint |
| --- | --- | --- |
| 256 (1) | 451.5 | 454.1 |
| 1024 (4) | 501.9 | 906.8 |
| 2048 (8) | 997.3 | 1675.6 |

The hint is up to 1.8x slower, so Vx will not emit it. Separately, `mla_decode_at` takes 1 ms
for 2048 selected tokens (64 heads, rank 512). That is a target of its own (#12).

## Vx memory-model litmus tests (`memory-model/results.txt`), for Vx #2291

These are vx-21's programs (Vx PR #2293), built with the bundle's vxc for `fleet/h100-sxm.vx`.
Each run is 100000 rounds between two one-lane blocks. The stress variants add 14 blocks.

All 10 programs printed 0 in all 10 runs. That includes the two controls (mp_relaxed and
sb_rel_acq), which the model allows to be weak. So this setup does not yet provoke weak
behaviour on the H100, and the zeros in the forbidden tests (mp_rel_acq, sb_seq_cst,
lb_relaxed) are not strong evidence. vx-21 is redesigning the tests.

A process note: vxc ignores `-o` and runs the program at once (Vx #2296). So the "build" at 23:28
UTC already ran each program once on the GPU, during this run's INT4 layer profile
(`layer-profile-int4`, superseded anyway). No correctness step was running at that time.
