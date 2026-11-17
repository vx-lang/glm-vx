# H200: the tested Vx moves to 64ff9cd4 (vx-lang/Vx#3517)

The same H200 VM as h200-20261116-m to -r. The toolchain bundle was built on the pod from Vx main
64ff9cd4 (`scripts/pod_build_bundle.sh`) and restored into `~/glm-h200/vxtc-64ff`. The repo is branch
vx-64ff (c1ec543, `index_key_at` with its `eps <= 0.0` guard back) in its own checkout. The steps are
in `s.sh`.

`scripts/pod_validate.sh` with `STEPS='^(build-|test-)'`: every build and GPU test step, 0 failed
(summary.txt).

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

The skips are not Vx's: six are cuBLASLt finding no algorithm for a weights_b bf16 shape
(CUBLAS_STATUS_NOT_SUPPORTED); one is the INT4 test of a real checkpoint, which needs
GLM_VX_INT4_CHECKPOINT.

The step profiles again (`validation/step_profile.py`), against h200-20261116-r's on 4f966fd6
(`ab-*.txt` are per-kernel sums from `abdiff.py`):

| step (us) | 4f966fd6 | 64ff9cd4 |
|---|---|---|
| 1 dense layer, 3001 tokens, `merge` | 269.6 | 268.9 |
| 1 dense layer, 3001 tokens, `position` + `fuse_glue` | 214.7 | 215.3 |
| dense + MoE layer, context 1000, `fuse_glue` | 400.1 | 399.6 |

Registers: ptxas -arch=sm_90 on all 82 spawn PTX files; none changed from 4f966fd6
(`registers-sm90.txt`, `registers-sm90-4f96.txt`, `registers-changed.txt` empty).
