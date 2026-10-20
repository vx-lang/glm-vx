# Kernels through Vx's own GPU path (`spawn on`)

The kernels in `gpu/*.vx` reach the GPU through a hand pipeline: `vxc --emit-llvm`, then
`gpu/toolchain/build_ptx.py` swaps `vx_gpu_*` externs for NVVM, then `llc`. The kernels here
are the same kernels written as a `spawn on(Topology::Device)` region. vxc itself outlines
the region, lowers it through the gpu dialect and NVVM, and puts the PTX in the dispatch
payload. `build.py` only takes that PTX out, gives the entry a stable name and checks its
parameters. There is no extern adapter and no `llc` step of ours.

Python still launches the kernels (`launch.py`, on `gpu/runtime.py`), with our grid and
block, because Vx's runtime cannot do that yet (see below).

| Kernel | Replaces | Registers (sm_80) |
| --- | --- | --- |
| `add.vx` | `glm_vx_gpu_add` | 14 |
| `copy.vx` | `glm_vx_gpu_copy` | 10 |
| `scale.vx` | `glm_vx_gpu_scale` | 12 |
| `scaled_add.vx` | `glm_vx_gpu_scaled_add` | 14 |
| `rope_glm.vx` | `glm_vx_gpu_rope_glm` | 27 |
| `rope_strided.vx` | `glm_vx_gpu_rope_strided` | 27 |
| `matvec.vx` | `glm_vx_gpu_matvec` | 26 |
| `matvec_heads.vx` | `glm_vx_gpu_matvec_heads` | 32 |
| `index_scores.vx` | `glm_vx_gpu_index_scores` | 26 |
| `bf16_matvec.vx` | `glm_vx_gpu_bf16_matvec` | 26 |
| `bf16_matmul.vx` | `glm_vx_gpu_bf16_matmul` | 31 |
| `int4_matvec.vx` | `glm_vx_gpu_int4_matvec` | 31 |
| `int4_matmul.vx` | `glm_vx_gpu_int4_matmul` | 37 |
| `fp8_matvec.vx` | `glm_vx_gpu_fp8_matvec` | 30 |
| `fp8_matmul.vx` | `glm_vx_gpu_fp8_matmul` | 38 |
| `gemm_f32.vx` | `glm_vx_gpu_gemm_f32` | 32 |
| `topk.vx` | `glm_vx_gpu_topk_stable` | 38 |

All build with vxc 390c2534 and assemble with ptxas for sm_80 with no spills; none has run on
a GPU yet. Not ported: the kernels that need a correctly rounded sqrt (rmsnorm, rmsnorm_rows,
residual_rmsnorm, layernorm: #1944), the ones that use exp (softmax, softmax_rows, swiglu,
router, mla_partial, mla_merge: vxc gives `ex2.approx.f32` where the hand kernels use
`ex2.approx.ftz.f32`), and gemm_tf32 (`mma.sync m16n8k8`: #350).

`gpu/test_spawn_hardware.py` checks each against the hand kernel, bit for bit, on a GPU.

## Build

```sh
VXC=/path/to/vxc PTXAS=/usr/local/cuda/bin/ptxas gpu/spawn/build.sh   # -> gpu/build/spawn/
```

Each file holds one `spawn on` region and one declaration line:

```
// spawn-include: device.vx
// spawn-kernel: glm_vx_spawn_add(n: i32, a: f32[?], b: f32[?], out: mut f32[?])
```

The parameters are the region's captures in first-use order. An assignment reads its right
side first and an index before the tensor it indexes, so `out[i] = a[i] + b[i]` captures `a`,
`b`, then `out`, and `out[h * out_stride + r] = total` captures `out_stride` before `out`. A rank-1 tensor becomes
five u64 (base, aligned pointer, offset, size, stride). `build.py` refuses a declaration whose
scalar widths or descriptor layout differ from the PTX, or whose `mut` tensors are not the
ones the kernel stores to, or whose element types differ from the width of the loads made
through each tensor (address arithmetic is traced from the descriptor's pointer). Two inputs
of the same type in each other's place still pass; the hardware tests catch those.
`device.vx` holds thread-id and shuffle helpers and the bf16, binary16, INT4 and FP8 decodes.

## What Vx is missing

| Need | Today | Vx issue |
| --- | --- | --- |
| Thread ids, warp shuffles | `mlir!` gpu-dialect helpers in `device.vx` | #260 |
| A stable entry name and documented parameter order | `build.py` renames `vx_npu_kernel_N` and checks the order | #251 |
| Correctly rounded sqrt / div (rmsnorm, softmax parity) | only `sqrt.approx` / `rsqrt.approx` | #1944 |
| exp as the hand kernels compute it, `ex2.approx.ftz.f32(x * log2 e)` | `math.exp2` gives `ex2.approx.f32` (no ftz), so a subnormal exp(x) differs | comment on #1944 |
| Bit casts (`f32::from_bits`) | crash; bf16 decode uses exact arithmetic instead | #1943 |
| Scalars passed by value | E6003 unless `machine.vx` lists CPU_DRAM as visible | #1947 |
| Raw pointers in a region | silent host fallback; kernels take `[?]` tensors | #1945 |
| Shared memory `::new()` | zero-fill races with other warps; use `unsafe { ::uninit() }` | #1946 |
| A Vx build of the same file (exe or .so) | fails on `gpu.shuffle` / `gpu.thread_id` in the host copy | #1942 |
| Grid and block chosen by the program | runtime picks from literal trip counts, else 1x1x1 | #252 |
| Launch without a device sync each time | `cudaDeviceSynchronize` after every launch | #25 |
| TF32 `mma.sync m16n8k8` | WMMA m16n16k8 only | #350 |

Once #1944 lands, rmsnorm and softmax follow; once #252 and #25 land, Vx can launch these
itself instead of Python.
