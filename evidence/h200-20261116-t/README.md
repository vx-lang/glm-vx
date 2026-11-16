# H200: the tested Vx moves to a4045882 (Vx main after #3523 to #3538)

The same H200 VM as h200-20261116-m to -s. The toolchain bundle was built on the pod from Vx main
a4045882 (`scripts/pod_build_bundle.sh`) and restored into `~/glm-h200/vxtc-a404`. The repo is
branch vx-64ff (4501cfb) in its own checkout. The steps are in `t.sh`.

Between 64ff9cd4 and a4045882, Vx merged #3523 to #3538. Among them is #3531: `lanes::copy` checks
the whole tile once, not each element. Its test, `lane_copy_into_and_out_of_a_tile.vx`, ran on
this GPU for the Vx session (`lane3531/`). vxc and FileCheck (`--implicit-check-not=Error`) both
exit 0:

| run | target |
|---|---|
| the test's RUN lines: flat, `-j 2`, `--legacy-codegen` (`CUDA_VISIBLE_DEVICES=`, the CPU copy) | `fleet/h100-sxm.vx` |
| the same three on the H200 | `fleet/h200.vx` |
| flat on the H200 | `fleet/h100-sxm.vx` |

The CPU copies print `[Vx CUDA] no CUDA device ...; kernels will run on the host`. The GPU runs
don't, and their output is otherwise the same.

`scripts/pod_validate.sh` with `STEPS='^(build-|test-)'`: every build and GPU test step,
0 failed (summary.txt). Each step's counts are those of h200-20261116-s.

| step | result |
|---|---|
| test-spawn-hardware | 162 passed, 1 xfailed |
| test-spawn-vx-host | 315 passed |
| test-spawn-vx-launch, -sync | 84, 84 passed |
| test-spawn-vx-graph | ok |
| test-hardware | 69 passed, 6 skipped |
| test-hardware-stress | 25 passed |
| test-quant-hardware | 38 passed, 1 skipped |
| test-resident-graph | 11 passed |
| test-speculative-resident | 12 passed |
| test-resident-many | 6 passed |
| test-resident-device-experts | 5 passed |
| test-resident-fused-attention | 1 passed |

The PTX (`ptx.sha256` against h200-20261116-s's): every file is the same except
`topk_radix.ptx` and `topk_radix_at.ptx`. In those two, only an assert message changed: it names
the path of the compiler's stdlib source (`.../vx-64ff9cd4/stdlib/core/sync/atomic.vx` became
`.../vx-a404/...`). The host `.so` files change with every compiler.

The step profiles (`validation/step_profile.py`), against h200-20261116-s's on 64ff9cd4. The
`ab-*.txt` files are per-kernel sums from `abdiff.py`.

| step (us) | 64ff9cd4 | a4045882 |
|---|---|---|
| 1 dense layer, 3001 tokens, `merge` | 268.9 | 269.6 |
| 1 dense layer, 3001 tokens, `position` + `fuse_glue` | 215.3 | 214.7 |
| dense + MoE layer, context 1000, `fuse_glue` | 399.6 | 401.2 |

Registers: `ptxas -arch=sm_90` on all 82 spawn PTX files. None changed from 64ff9cd4
(`registers-sm90.txt`, `registers-sm90-64ff.txt`; `registers-changed.txt` is empty).
