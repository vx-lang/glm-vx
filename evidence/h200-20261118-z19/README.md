# Vx PR #3646 (FP8 tile matmul on the tensor cores) on one H200

vx-94's draft PR, branch `fp8-tensor-core-matmul` at 66ff5d17e, run at their request: no GPU had run it.

**Build.** vxc was built on the pod (`TC=66ff5d17e pod_go.sh HOST bundle`, `scripts/pod_build_bundle.sh`) and
restored to `~/glm-h200/vxtc-66ff`, beside glm-vx's own toolchain. Script: `z19-vx.sh`.

**The PR's value test:**

```
vxc tests/optimizations/pass/lane_matmuls_of_fp8_tiles_give_the_sums_of_a_loop.vx --host default --machine fleet/h200.vx --run
```

- It prints the two expected lines exactly (`run.log`):
  - `e4m3: -3 -3 7 sum of all 1020`
  - `e5m2: -5 2 5 sum of all -3`
- It ran on the GPU: nsys lists one CUDA kernel, `vx_npu_kernel_0`, 7.3 us (`run-nsys.log`).
- `--emit-ptx` refuses this test, because its kernel uses values that are not parameters of `main`
  (`emit-ptx.log`). So the SASS of the running kernel comes from ncu: `ncu --page source --print-source sass`,
  `ncu-sass.csv`.

**The SASS has 8 `HMMA.16816.F32` and no FP8 MMA.** That is ptxas's lowering on sm_90, not the PR's
(`sass-lowering.txt`): one `mma.sync.m16n8k32.row.col.f32.e4m3.e4m3.f32` assembles to
- sm_89: one `QMMA.16832.F32.E4M3.E4M3`;
- sm_90a: 12 `F2FP.F16.E4M3.UNPACK_B` + 2 `HMMA.16816.F32`.

So FP8 tiles through `mma.sync` give the right values on Hopper but run on its f16 tensor cores. Hopper's FP8
tensor cores are only reachable through `wgmma`. Reported to vx-94.
