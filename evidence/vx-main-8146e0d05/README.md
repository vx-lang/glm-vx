# Vx main 8146e0d05 (after #3646 to #3652): glm-vx's kernels unchanged from the H200-tested #3649 head

Aditya merged the Vx FP8 and FP4 PRs on 2026-11-19 (#3646, #3647, #3648, #3649, #3650, #3652). Since 4d15f5315,
the head of #3649 that `evidence/h200-20261119-z30` ran on the H200, main has added #3424 (a semaphore), #3648,
#3649 itself, #3650 and #3652.

**Run** (`vxmain-8146.sh`, on the dev box, no GPU): vxc built from 8146e0d05 and from 4d15f5315. glm-vx's spawn
kernels at c3ae595 were built with each of them, and with the Vx 3609 vxc that `deploy15` (the next 8x H200 run)
uses. ptxas 13.3 checked every kernel for sm_90: 111 PTX each, no errors (`vxmain-8146.txt`).

**Result** (`ptx-same.txt`, from `ptx_same.sh`):

| builds | identical PTX | differ |
|---|---|---|
| 4d15f5315 vs main 8146e0d05 | 111 | 0 |
| 3609 vs main 8146e0d05 | 110 | `mla_tc_at` |

- **A plain `cmp` overstates the difference.** It reports 11 and 12 differing kernels. The PTX embeds the path of
  Vx's `stdlib/core/sync/atomic.vx` in the assert messages of every kernel that uses atomics, and that path names
  the build directory (`vx-4d15`, `vx-8146`, `vx-3609`). `ptx_same.sh` normalizes it.
- **So main changes none of glm-vx's kernels.** z30 ran the 4d15f5315 kernels on the H200: GLM-5.3's first 4 layers
  gave the same tokens and top-5 as the 3609 kernels, and 32 GPU tests passed. Those results hold for main.
- **z30's "13 differ" was mostly this path.** The one real difference is `mla_tc_at`. z30's list also had
  `router_bf16`, whose source changed between the two builds it compared (af6d4d2).
