# Kernels through Vx's own GPU path (`spawn on`)

The kernels in `gpu/*.vx` reach the GPU through a hand pipeline: `vxc --emit-llvm`, then
`gpu/toolchain/build_ptx.py` swaps `vx_gpu_*` externs for NVVM, then `llc`. The kernels here
are the same kernels written as a `spawn on(Topology::Device)` region. vxc itself outlines
the region, lowers it through the gpu dialect and NVVM, and `vxc --emit-ptx` writes the PTX
with a manifest of its parameters. `build.py` runs that and checks the manifest against the
PTX. There is no extern adapter and no `llc` step of ours.

Each region gets its launch shape from its `across` loops (Vx #252, merged as #2095 to #2097):
`for blk in 0..(n - 1) / 256 + 1 across blocks { for t in 0..256 across lanes { .. } }` is a grid
of `(n - 1) / 256 + 1` blocks of 256 lanes, which the manifest's `launch` records. The block
loop strides by the grid and a lane loop by the block, so any grid and block the caller passes
also cover every pass. A kernel whose rows have a run-time length writes a rank-2 tensor
(`out[row][col]`), because the race proof (L4) reads an index only as a sum of loop indices
times numbers; `launch.py` passes a rank-2 `DeviceTensor` as its 7-word descriptor.

Two vxc builds are in use until every kernel is on `across`:

- **Vx main** (b151bb8c on): `add`, `copy`, `scale`, `scaled_add`, `rope_glm`, `rope_strided` and
  `gemm_f32`, on `across`. Main no longer parses `grid([..]) block(n)`.
- **The old launch stack** (Vx 7608e6f7 plus the closed PRs #2065 to #2068 and #2064): the other
  18 kernels, `probes/` and `fp8_gemm/`, still on `grid([..]) block(n)`. The proof refuses what
  they need in an `across` loop: warp shuffles, `barrier()` and `fast_exp` are `mlir!` blocks or
  calls it cannot see into, and the `mlir!` blocks of the tensor-core kernels wait for declared
  writes (Vx #2076).

`ONLY=across` or `ONLY=grid` makes `build.sh` build one kind (it greps each source for
`across blocks`), so each vxc builds its own; `scripts/pod_validate.sh` takes the main vxc as
`VXC_ACROSS`. Each `lib<kernel>.so` links the dispatch library of the vxc that built it, by path,
so a process that loads both kinds holds two Vx runtimes, each with its own stream setting.

Two launches use the same PTX:

- **Vx launches it.** `build.py --library` also builds `lib<kernel>.so` from the same file, and
  `vx_launch.py` calls its `pub fn`: Vx's CUDA runtime launches the kernel with that shape, on
  the legacy default stream, without waiting for the GPU. The tensors must come from
  `CUDAContext(primary=True)`, the context Vx's runtime uses; they are passed with no copy.
  `VxKernel(...)(..., stream=s)` launches on our stream `s` instead (Vx's `vx_cuda_set_stream`,
  #2100 on main, PR #2068 on the old stack, set for the call; the stream set before is put back after it), so `s.capture()` can
  record Vx's launches, with ours, into a CUDA graph. Call each kernel once before capturing.
- **Python launches it.** `launch.py` (on `gpu/runtime.py`) launches the PTX with the manifest's
  shape, or with any grid and block the caller passes (the hardware tests sweep them).

| Kernel | Replaces | Registers (sm_80) |
| --- | --- | --- |
| `add.vx` | `glm_vx_gpu_add` | 20 |
| `copy.vx` | `glm_vx_gpu_copy` | 17 |
| `scale.vx` | `glm_vx_gpu_scale` | 18 |
| `scaled_add.vx` | `glm_vx_gpu_scaled_add` | 20 |
| `rope_glm.vx` | `glm_vx_gpu_rope_glm` | 28 |
| `rope_strided.vx` | `glm_vx_gpu_rope_strided` | 28 |
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
plus PRs #1952, #1954, #1955, #1958, #1959, #1960, #1961, #1962, #1964 (`--emit-ptx`), #1972
(launch shapes) and #1974 (no wait after each launch); the libraries for streams and graphs
also take #1987, #1976 and #1994 (`vx_cuda_set_stream`, which `nm -D` shows exported). Barrier counts match the hand kernels (softmax 4, router
1, mla_partial 2). Not ported: gemm_tf32 (`mma.sync m16n8k8`: #350).
`probes/` holds tensor-core probes (mma.sync on sm_80, wgmma bf16 and block-FP8 on sm_90a) with
register-tensor accumulators (Vx PR #2058, was #1991); see `probes/README.md`.
`fp8_gemm/` holds the block-FP8 GEMM for H100 (TMA, mbarrier ring, wgmma e4m3; sm_90a, its own
`build.sh` and machine file; Vx PRs #2058 (was #1991), #1995, #1996); see `gpu/FP8-GEMM.md`.

`gpu/test_spawn_hardware.py` checks each against the hand kernel, bit for bit, on a GPU, and
`gpu/test_spawn_vx_launch.py` checks Vx's launch of each against Python's launch of the same
PTX with the same shape, bit for bit (run it also with `VX_CUDA_SYNC=1`).
`gpu/test_spawn_vx_graph.py` runs a chain of Vx launches on our stream and as a replayed CUDA
graph (also mixed with Python launches), bit for bit against the legacy stream, and checks that
Vx stops a capture on `VX_CUDA_SYNC=1` or `VX_TIME_KERNEL`; it needs the libraries built with
#2100 (main) or PR #2068 (the old stack).

## Build

```sh
VXC=/path/to/vxc PTXAS=/usr/local/cuda/bin/ptxas gpu/spawn/build.sh   # -> gpu/build/spawn/
# and the libraries Vx launches from (vxc needs llvm-config on PATH):
VX_DISPATCH_LIB=/path/to/libvx_cuda_dispatch.so VXC=... gpu/spawn/build.sh
```

A launch size is a number, a parameter, or computed from parameters with `+ - * / %`
(truncating, in i32), and the manifest records which. `launch.py` evaluates a computed size as
Vx does. The library's `pub fn` takes every parameter of the function, so each must be one the
kernel uses, and `build.py` refuses a library whose embedded kernel is not the PTX it checked.

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
| A Vx build of the same file (exe or .so) | the host copy of a GPU-only helper stops the program; PR #1958 | #1942 |
| Grid and block chosen by the program | `across` loops (merged, #2095 to #2097); `grid([..]) block(..)` on the old stack (PR #2065, closed) | #252 |
| A reduction over lanes (warp sum, block sum) inside `across` | none: shuffles are `mlir!`, and `barrier()` is refused; asked on Vx | #252 |
| Launch without a device sync each time | waits only before a copy to the host, a free, or exit; merged as #2098 (was #2066) | #25 |
| A stream of the caller, and CUDA graphs | `vx_cuda_set_stream` (merged as #2100, was #2068): launches, cuBLAS and copies on our stream, temporaries stream-ordered, a stop with a message on a wait during a capture; `gpu/test_spawn_vx_graph.py` | #1986 |
| Futures the compiler waits on | no waits are emitted; the runtime waits before a host copy, a free, or exit | #25 |
| TF32 `mma.sync m16n8k8` | WMMA m16n16k8 only | #350 |

Nothing has run on a GPU yet: the next pod runs both test files (`scripts/pod_validate.sh`).
