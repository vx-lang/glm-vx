# A100 run, 2026-10-25

NVIDIA A100 80GB PCIe, driver 595.91.07, 128 vCPUs, Ubuntu 24.04 (glibc 2.39), CUDA 12.8 ptxas.
vxc built on the pod from Vx main `c3716538`, LLVM 22.1.8. The checkout was `97821af` copied without
`.git`, plus the fixes committed with this folder (`host.txt` says `97821af+fixes`).

This is the first GPU run of the gpu/spawn kernels (Vx `across` loops), of Vx's own launches and
CUDA graphs around them, of the fused MoE kernels in resident decode, of mla_decode, and of the
full-width FP8 and BF16 resident comparisons.

`summary.txt` is `scripts/pod_validate.sh`. It failed five steps, all test or launcher bugs,
fixed in the commit with this folder; `rerun-*.txt` are those five steps run again from that tree:

| Step | First run | Cause | Rerun |
| --- | --- | --- | --- |
| test-spawn-hardware | 3 failed | matvec launched with block 64 for a kernel built for 128 (rows 2, 3, 6 unwritten); MLA sums near zero over rtol 1e-5 from the exp difference | 89 passed, 1 xfailed |
| test-spawn-vx-launch (and -sync) | mla_decode faulted, the rest failed after it | Vx's library launch gives mla_decode a host arrival count (Vx #2231, PR #2232); the case is skipped | 40 passed, 1 skipped |
| test-resident-device-experts | 5 failed | expected bitwise, but moe_gate_up's expf is libdevice's and the hand swiglu's is ex2.approx: up to 1.2e-7 | 5 passed |
| test-resident-fused-attention | 1 failed | the tiny config's 512 positions, under 700 steps | 1 passed |

| Step | Result |
| --- | --- |
| build-ptx, build-spawn | 7 hand PTX files and 39 spawn kernels, sm_80, no spills |
| test-hardware | 28 passed |
| test-quant-hardware | 39 passed, incl. real compressed-tensors INT4 Linears |
| test-spawn-hardware | all 39 spawn kernels against the hand kernels; mla_decode bitwise against spawn mla_partial + mla_merge over 3 launches on one arrivals buffer, counts back at 0 each time |
| test-spawn-vx-host | 101 passed (Vx's CPU path) |
| test-spawn-vx-launch | each kernel through its library's pub fn, bitwise against our launch of the same PTX |
| test-spawn-vx-graph | Vx launches on our stream, captured in a CUDA graph: replays bitwise; a 4-kernel chain 104 us eager, 15.7 us per replay |
| test-hardware-stress | 25 passed |
| test-resident-device-experts | routed experts in two launches against the host loop, 5 formats, within 1e-5 / 1e-6, same argmax |
| test-resident-fused-attention | 700 steps with mla_decode (up to 3 splits) against mla_partial + mla_merge, within 1e-4, same argmax |
| resident-f32 small / full4 | bitwise; 9.8 / 10.9 ms per token resident (hybrid 66 ms / 33.5 s) |
| resident-fp8 small / full4 | bitwise; full4 13.5 ms per token, 7.4 GiB vs 18.2 GiB |
| resident-bf16 small / full4 | bitwise; full4 18.8 ms per token, 9.9 GiB vs 19.4 GiB |
| resident-int4 small / full4 | top-1 equal, max abs diff 5.2e-6 / 9.1e-6 (as on 2026-10-20); full4 15.7 ms per token, 6.0 GiB vs 20.0 GiB |
| stress-1..8 | 0 mismatches |

The quantized formats are slower per token than float32 here, with a third of the weight bytes:
these SIMT kernels are not bandwidth-bound at 4 layers, and every step still launches each kernel
eagerly. Random weights throughout; timings are single runs of the harness, not benchmarks.

Not in this folder: a first attempt with the pod's newer Vx (main + #2223) failed to build the hand
PTX and to load six spawn libraries, because of Vx #2215's vx_stderr_write (Vx #2229, PR #2230; the
hand-PTX side is fixed in build_ptx.py).
