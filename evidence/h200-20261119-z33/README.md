# vx-94's block-FP8 GEMMs on one H200: the grouped GEMM prefill's experts use, and the dense one (#63, 5b; #10)

`gpu/spawn/fp8_gemm` (vx-94, branch vx94-grouped-fp8-gemm; built with Vx PR #3656's vxc e37f1c06e): the dense
wgmma m64n128k32 e4m3 GEMM written for #10 and never run before, and `fp8_gemm_grouped_m64`, one launch for every
routed expert's rows (offsets on the device, `fp8_gemm_grouped_plan`). GPU 0 of the 8x H200 (`z33.sh`, then reruns).

1. **First run (3e27413): every test failed** at `cuTensorMapEncodeTiled`: CUDA_ERROR_INVALID_CONTEXT
   (`test_fp8_gemm*_hardware.log`). The tensor maps were encoded with no current context. The node's copy was
   patched with `with self.context._activate():`; vx-94's b8b547b is the same fix.
2. **Rerun** (`*.rerun.log`):
   - dense: 3 passed, 1 failed. `fp8_gemm_m128` cannot launch: its 9 warps leave 170 registers a thread on one
     SM sub-partition, and it needs ~220. Dropped in 6c0f790. The dense Linears use cuBLASLt, so it is not on
     prefill's path.
   - grouped: 7 passed, 1 failed. All values were right; 24 of 113M elements fell outside an error bound modeled
     too tight for K = 256 (max relative error 4.1e-4). The bound is fixed in 6c0f790.
3. **bf16 C (f6fbe98, the merged prefill branch, `test_fp8_gemm_grouped_hardware.bf16.log`, `grouped-bf16.json`):
   16 passed**, f32 and bf16 C. TP8 shapes, a rank:

| grouped GEMM | 2304 rows (256-token chunk) | 18432 rows (2048-token chunk) |
|---|---|---|
| gate_up, n 512, k 6144, f32 C | 126 us | 312 us |
| gate_up, bf16 C | 119 us | 301 us |
| down, n 6144, k 256, f32 C | 261 us | 676 us |
| down, bf16 C | 249 us | **547 us** |

The 18432-row gate_up first measured 821 us with bf16 C. Three reruns (`grouped-rerun-*.json`, with
`smi-rerun.csv`: SM clock 1980 MHz throughout) gave 312 us (f32) and 301 us (bf16), so that was a slow run.
prefill_chunk keeps gate and up in f32 and takes down's C in bf16 (04a2b8c).
