# Kernels through Vx's own GPU path (`spawn on`)

The kernels in `gpu/*.vx` reach the GPU through a hand pipeline: `vxc --emit-llvm`, then
`gpu/toolchain/build_ptx.py` swaps `vx_gpu_*` externs for NVVM, then `llc`. The kernels here
are the same kernels written as a `spawn on(Topology::Device)` region. vxc itself outlines
the region, lowers it through the gpu dialect and NVVM, and `vxc --emit-ptx` writes the PTX
with a manifest of its parameters. `build.py` runs that and checks the manifest against the
PTX. There is no extern adapter and no `llc` step of ours.

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
| `rmsnorm_rows.vx` | `glm_vx_gpu_rmsnorm_rows`, and `glm_vx_gpu_rmsnorm` as rows = 1 | 16 |
| `residual_rmsnorm.vx` | `glm_vx_gpu_residual_rmsnorm` | 18 |
| `layernorm.vx` | `glm_vx_gpu_layernorm` | 21 |
| `softmax_rows.vx` | `glm_vx_gpu_softmax_rows`, and `glm_vx_gpu_softmax` as rows = 1 | 18 |
| `swiglu.vx` | `glm_vx_gpu_swiglu` | 17 |
| `router.vx` | `glm_vx_gpu_router` | 22 |
| `mla_partial.vx` | `glm_vx_gpu_mla_partial` | 25 |
| `mla_merge.vx` | `glm_vx_gpu_mla_merge` | 32 |

All assemble with ptxas for sm_80 with no spills; none has run on a GPU yet. The three norms
need a correctly rounded sqrt, which needs #1944 (PR #1955), and the five kernels with exp use
`fast_exp` (device.vx), `ex2.approx.ftz.f32(x * log2 e)` exactly as the hand kernels do, which
needs nvvm intrinsics in the device image, #1945 (PR #1959). Built and checked with Vx main
plus PRs #1952, #1954, #1955, #1958, #1959, #1960, #1961, #1962 and #1964 (`--emit-ptx`). Barrier counts match the hand kernels (softmax 4, router
1, mla_partial 2). Not ported: gemm_tf32 (`mma.sync m16n8k8`: #350).

`gpu/test_spawn_hardware.py` checks each against the hand kernel, bit for bit, on a GPU.

## Build

```sh
VXC=/path/to/vxc PTXAS=/usr/local/cuda/bin/ptxas gpu/spawn/build.sh   # -> gpu/build/spawn/
```

Each file holds one `pub fn glm_vx_spawn_NAME` with one `spawn on` region, and may start with
`// spawn-include: device.vx` to paste the helpers in front. The PTX entry has the function's
name, and its parameters are the function's parameters, in declared order (Vx
`docs/lang/abi.md` section 3.5); a rank-1 tensor is five u64 (base, aligned pointer, offset,
size, stride). `launch.py` takes the arguments by name.

`build.py` does not take vxc's manifest on trust. It refuses one whose scalar widths or
descriptor layout differ from the PTX, whose mutable tensors are not the ones the kernel
stores to, or whose element types differ from the width of the loads made through each tensor
(address arithmetic is traced from the descriptor's pointer).
`device.vx` holds thread-id and shuffle helpers, warp sum and max, `fast_exp` and `sigmoid`, and
the bf16, binary16, INT4 and FP8 decodes.

## What Vx is missing

| Need | Today | Vx issue |
| --- | --- | --- |
| Thread ids, warp shuffles | `mlir!` gpu-dialect helpers in `device.vx` | #260 |
| A stable entry name and documented parameter order | `vxc --emit-ptx` (PR #1964): the entry is the function's name, the parameters its own in declared order | #251 |
| Correctly rounded sqrt / div (the norms) | `sqrt.approx` / `rsqrt.approx` in 390c2534; `sqrt.rn` + `rcp.rn` with PR #1955 | #1944 |
| A shared tensor passed to a device function | `memref.cast` dropped address space 3 and the fallback panicked; PR #1961 fixes it, and `block_sum` / `block_max` live in device.vx | #1956 |
| `barrier()` inside a device function | gave no device image, no warning; PR #1962 fixes it | #1957 |
| exp as the hand kernels compute it, `ex2.approx.ftz.f32(x * log2 e)` | `llvm.call_intrinsic "llvm.nvvm.ex2.approx.ftz.f"` in an mlir! block, with PR #1959 | #1945 |
| Bit casts (`f32::from_bits`) | crash; bf16 decode uses exact arithmetic instead | #1943 |
| Scalars passed by value | E6003 unless `machine.vx` lists CPU_DRAM as visible | #1947 |
| Raw pointers in a region | silent host fallback before PR #1959, which takes them and warns on any other fallback; kernels take `[?]` tensors | #1945 |
| Shared memory `::new()` | zero-fill races with other warps; use `unsafe { ::uninit() }` | #1946 |
| A Vx build of the same file (exe or .so) | fails on `gpu.shuffle` / `gpu.thread_id` in the host copy (fix in review: a CPU stub that aborts) | #1942 |
| Grid and block chosen by the program | runtime picks from literal trip counts, else 1x1x1 | #252 |
| Launch without a device sync each time | `cudaDeviceSynchronize` after every launch | #25 |
| TF32 `mma.sync m16n8k8` | WMMA m16n16k8 only | #350 |

Once #252 and #25 land, Vx can launch these
itself instead of Python.
