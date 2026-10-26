# A100 runs, 2026-10-26

NVIDIA A100 80GB PCIe, driver 595.91.07, 252 vCPUs, 944 GB host RAM, Ubuntu 24.04 (glibc 2.39),
CUDA 12.8 ptxas, LLVM 22.1.8 from apt.llvm.org. glm-vx `f687b4f`. Vx was built on the pod.

## First run: Vx main `e6722f78` (this folder)

`summary.txt` is `scripts/pod_validate.sh`: 4 failed steps, all for one Vx bug.

- **The failure.** `mla_decode_at` has a 1024-thread block, but its PTX did not declare that to
  ptxas, which gave it 80 registers per thread. At 80 registers the kernel can run at most 768
  threads, so the runtime refused the launch ("block exceeds kernel/device thread limit").
- **Failed cases:** 2 in test-spawn-hardware, 1 in each of test-spawn-vx-launch and -sync, and 1
  in test-resident-graph (the fused case).
- **The fix:** Vx #2246, fixed by #2247 (merged as `371c2bae`). A block of more than 256 threads
  now gets `.maxntid`, and `mla_decode_at` uses 32 registers with no spills.
- **Everything else passed,** including all 8 graph comparisons, each bit for bit against the
  per-launch path.

The `graph-*` files also hold the first graph timings. Per-launch vs graph replay, in ms per token:
- Small model: 10.3 vs 2.0 (launch overhead removed).
- Full width, 4 layers: 10.1 vs 9.6 (f32), 13.6 vs 13.0 (FP8), 18.2 vs 18.1 (BF16), 10.7 vs
  10.2 (INT4). At this width the kernels themselves dominate.

Whole-model timings on a 4-layer slice compare with nothing, so they are no longer reported. The
per-layer profiles below replace them.

## Rerun: Vx main `371c2bae` (#2247)

Copied from the run's output before the pod became unreachable (06:40 UTC). The raw files stayed on
the pod, so these numbers are transcribed here.

| Step | Result |
| --- | --- |
| test-hardware / -quant / -stress | 28 / 39 / 25 passed |
| test-spawn-hardware | 91 passed, 1 xfailed, including mla_decode_at (2 cases) |
| test-spawn-vx-host | 103 passed |
| test-spawn-vx-launch, and with VX_CUDA_SYNC | 42 passed each, including mla_decode_at |
| test-spawn-vx-graph | 8 passed |
| test-resident-device-experts / -fused-attention / -graph | 5 / 1 / 7 passed |
| resident f32, FP8, BF16, small and full4 | bitwise |
| resident f32 full4 against the hybrid path | bitwise over 20 reference tokens (new: REFERENCE_TOKENS) |
| resident INT4 small / full4 | top-1 equal, max abs diff 5.2e-6 / 9.1e-6, as on every run since 2026-10-20 |
| graph f32, INT4, FP8, BF16 small; f32 full4 | bitwise |
| stress, 8 processes | 0 mismatches |

Register counts (ptxas -arch=sm_80 -v) on the same pod, for Vx #2247 before it merged:

| Kernel | Before (e6722f78) | #2247 first commit (ba3d9d1b) | Merged form (8c3bec86) |
| --- | --- | --- | --- |
| mla_decode_at (block 1024) | 80, cap 768 threads | 32, `.maxntid 1024` | 32, `.maxntid 1024` |
| mla_decode (block 1024) | 31 | 31 | 31, `.maxntid 1024` |
| fp8_moe_down (block 128) | 56 | 48 + 20 B spill | 56, no `.maxntid` |
| fp8_moe_gate_up (block 128) | 68 | 64 + 8 B spill | 68, no `.maxntid` |

No spawn kernel spills at 8c3bec86. On 8c3bec86 test-spawn-hardware, test-spawn-vx-launch, the
resident device-experts, fused-attention and graph tests passed (146 passed, 1 xfailed), as did
the vx-launch run with sync (42 passed) and test-spawn-vx-graph (8 passed).

Step times compared with the first run:
- resident-f32-full4: 17 min, down from 35. Its hybrid reference covers 20 tokens, not 48.
- resident-fp8-full4: 10.5 min, down from 20. FP8 encoding is now vectorized and threaded,
  with bit-for-bit identical codes.

Not finished: the first `validation/layer_profile.py` runs failed, because the driver did not
bind `cuProfilerStart` (fixed in the commit with this folder). The pod went away during
`validation/moe_profile.py`. Both profiles are for the next pod.
