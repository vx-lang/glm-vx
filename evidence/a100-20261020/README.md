# A100 run, 2026-10-20

NVIDIA A100-SXM4-80GB, driver 595.91.07, 192 vCPUs, Ubuntu 24.04 (glibc 2.39). vxc built on the pod
from Vx `390c2534`, LLVM 22.1.8, ptxas 12.8. The checkout was `e597ecd` copied without `.git` (hence
the git noise in `host.txt`), plus the two fixes committed with this folder.

`validate.log` is `scripts/pod_validate.sh`; the pod was shut down during `resident-int4-full4`, so the
full-width INT4, FP8 and BF16 comparisons did not run. The small FP8 and BF16 comparisons and the
8-process stress pass were run by hand with the script's own commands.

| Step | Result |
| --- | --- |
| build-ptx | 7 PTX files, byte-identical to the local build (`ptx.sha256`), sm_80 cubins, 0 spills |
| test-hardware | 28 passed |
| test-quant-hardware | first run 12 failed: missing import in the BF16 tests (test bug). `test-quant-hardware-rerun.txt`: 39 passed, incl. real compressed-tensors INT4 Linears |
| test-hardware-stress | 25 passed |
| resident-f32-small | bitwise equal, 9.2 ms/token resident vs 60 ms hybrid |
| resident-f32-full4 | bitwise equal (hidden 6144, 256 experts, 4 layers), 10.8 ms/token vs 29 s hybrid, 19 GiB resident |
| resident-int4-small | top-1 equal, max abs diff 5.2e-6 (gate 4.9e-4), 3.8x fewer weight bytes |
| resident-fp8-small | bitwise equal |
| resident-bf16-small | bitwise equal |
| stress-1..8 | 0 mismatches |

Random weights throughout; timings are single runs of the harness, not benchmarks.
