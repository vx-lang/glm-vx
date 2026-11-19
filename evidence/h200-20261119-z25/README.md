# Vx PR #3648 (f16 x FP8 tile matmul) on one H200

vx-94's PR, branch `mixed-f16-by-fp8-tiles` at 7933f6af1, on top of #3646. Run at their request. vxc was built on
the pod and restored to `~/glm-h200/vxtc-7933` (`z25-vx.sh`).

**Value test.** `lane_matmuls_of_fp8_tiles_give_the_sums_of_a_loop.vx --host default --machine fleet/h200.vx --run`
prints the four expected lines exactly (`run.log`):

```
e4m3: -3 -3 7 sum of all 1020
e5m2: -5 2 5 sum of all -3
e4m3 by f16: 30 30 22 sum of all 9
f16 by e5m2: 0 -7 -18 sum of all -16
```

The last two show that `cvt.rn.f16x2.e4m3x2` / `e5m2x2` put the first FP8 byte in the low half.

**It ran on the GPU:** nsys lists one CUDA kernel, `vx_npu_kernel_0`, 8.7 us (`run-nsys.log`).

**SASS of the running kernel** (`ncu-sass.csv`, from ncu's source page):
- 12 `HMMA.16816.F32`;
- 24 `F2FP.F16.E4M3.UNPACK_B` and 28 `F2FP.F16.E5M2.UNPACK_B`;
- 2 `LDSM.16.M88.4` and 2 `LDSM.16.MT88.2`, ldmatrix for the f16 tiles.

`--emit-ptx` refuses the test, as on #3646.

(The script's echo once said z19-done; its copy here is corrected.)
