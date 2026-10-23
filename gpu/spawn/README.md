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
times numbers; `launch.py` passes a rank-2 `DeviceTensor` as its 7-word descriptor. A row's
index may then be anything, as in topk's `hist[t][digit]`.

All 25 kernels build with one vxc, from Vx main (a6efd340 on: `device.vx` uses `if` values that read
a tensor, #2112), which no longer parses
`grid([..]) block(n)`. What the proof takes in a loop shapes them:

- **No calls it cannot read.** A function call (other than the stdlib's pure scalar ones, such
  as `f32::sqrt` and `f32::fast_exp`), an `mlir!` block and `barrier()` are refused in an
  `across` loop. So `device.vx` holds macros, which expand in the loop: the decoders, `sigmoid`,
  and the reductions.
- **Reductions in shared memory.** A warp sum is five lane loops on shared memory (offsets 16, 8,
  4, 2, 1, as vxk_warp_sum), each reading one tensor and writing another, since a pass may not
  read an element that another pass of the same loop writes. Each lane loop ends with a barrier,
  so a warp sum costs five barriers where the hand kernels shuffle: correct first, until
  reductions become library routines (Vx #260).
- **Results that do not depend on the launch.** A block reduction follows the hand kernel
  launched with `threads` threads, which the norms, softmax and mla_partial take as a parameter:
  a 1024-lane loop in which lane t < threads sums d = t, t + threads, .. and the rest add zeros.
- **Uniform control around lane loops.** A condition or loop bound around a lane loop may read
  only block indices and host values, so data-dependent state (mla_partial's m and l, topk's key
  prefix) lives in shared memory, and a one-lane loop updates it after the lanes have read it.
- **exp is libdevice's.** `f32::fast_exp` is expf from libdevice, where the hand kernels use
  `ex2.approx.ftz.f32(x * log2 e)` in an `mlir!` block; the five kernels with exp differ from
  them in the last bits (the hardware tests compare them within 1e-5) until an `across` loop takes
  an `mlir!` block that declares it writes nothing (Vx #2076 D1 to D3).

`probes/` and `fp8_gemm/` (wgmma, TMA, mbarriers) are not built: their `mlir!` blocks wait for
declared writes (Vx #2076).

Two launches use the same PTX:

- **Vx launches it.** `build.py --library` also builds `lib<kernel>.so` from the same file, and
  `vx_launch.py` calls its `pub fn`: Vx's CUDA runtime launches the kernel with that shape, on
  the legacy default stream, without waiting for the GPU. The tensors must come from
  `CUDAContext(primary=True)`, the context Vx's runtime uses; they are passed with no copy.
  `VxKernel(...)(..., stream=s)` launches on our stream `s` instead (Vx's `vx_cuda_set_stream`,
  #2100, set for the call; the stream set before is put back after it), so `s.capture()` can
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
| `matvec.vx` | `glm_vx_gpu_matvec` | 38 |
| `matvec_heads.vx` | `glm_vx_gpu_matvec_heads` (out is [heads, out_stride]) | 48 |
| `index_scores.vx` | `glm_vx_gpu_index_scores` | 43 |
| `bf16_matvec.vx` | `glm_vx_gpu_bf16_matvec` | 47 |
| `bf16_matmul.vx` | `glm_vx_gpu_bf16_matmul` (out is [M, N]) | 53 |
| `int4_matvec.vx` | `glm_vx_gpu_int4_matvec` | 50 |
| `int4_matmul.vx` | `glm_vx_gpu_int4_matmul` (out is [M, N]) | 56 |
| `fp8_matvec.vx` | `glm_vx_gpu_fp8_matvec` | 61 |
| `fp8_matmul.vx` | `glm_vx_gpu_fp8_matmul` (out is [M, N]) | 62 |
| `gemm_f32.vx` | `glm_vx_gpu_gemm_f32` | 32 |
| `topk.vx` | `glm_vx_gpu_topk_stable` (its own scratch: `gpu.topk.spawn_buffers`) | 32 |
| `rmsnorm_rows.vx` | `glm_vx_gpu_rmsnorm_rows`, and `glm_vx_gpu_rmsnorm` as rows = 1 | 38 |
| `residual_rmsnorm.vx` | `glm_vx_gpu_residual_rmsnorm` | 36 |
| `layernorm.vx` | `glm_vx_gpu_layernorm` | 40 |
| `softmax_rows.vx` | `glm_vx_gpu_softmax_rows`, and `glm_vx_gpu_softmax` as rows = 1 | 40 |
| `swiglu.vx` | `glm_vx_gpu_swiglu` | 25 |
| `router.vx` | `glm_vx_gpu_router` | 28 |
| `mla_partial.vx` | `glm_vx_gpu_mla_partial` (numerator [tasks, rank], stats [tasks, 2]) | 40 |
| `mla_merge.vx` | `glm_vx_gpu_mla_merge` (out is [heads, rank]) | 32 |

All assemble with ptxas for sm_80 with no spills; none has run on a GPU yet. The norms, softmax
and mla_partial launch 1024 lanes; a warp-per-row kernel has four rows per block of 128.
topk keeps gpu/topk.vx's radix select, then, because the hand kernel's compaction and rank write
at computed positions, puts each lane's candidates in its own column of scratch, gathers them,
and sorts them by (key, position) as one i64 with a bitonic sort (a numpy model of it matches
`np.argsort(-scores, kind='stable')`). Not ported: gemm_tf32 (`mma.sync m16n8k8`: #350).
`probes/` holds tensor-core probes (mma.sync on sm_80, wgmma bf16 and block-FP8 on sm_90a) with
register-tensor accumulators; see `probes/README.md`. `fp8_gemm/` holds the block-FP8 GEMM for
H100 (TMA, mbarrier ring, wgmma e4m3; sm_90a, its own `build.sh` and machine file); see
`gpu/FP8-GEMM.md`. Both are still written with `grid([..]) block(n)` and wait for Vx #2076.

`gpu/test_spawn_hardware.py` checks each against the hand kernel, bit for bit (within 1e-5 for
the five with exp), on a GPU, and
`gpu/test_spawn_vx_launch.py` checks Vx's launch of each against Python's launch of the same
PTX with the same shape, bit for bit (run it also with `VX_CUDA_SYNC=1`).
`gpu/test_spawn_vx_graph.py` runs a chain of Vx launches on our stream and as a replayed CUDA
graph (also mixed with Python launches), bit for bit against the legacy stream, and checks that
Vx stops a capture on `VX_CUDA_SYNC=1` or `VX_TIME_KERNEL`; it needs the libraries built with
#2100.

`gpu/test_spawn_vx_host.py` needs no GPU: with no device visible, Vx runs each region in order
on the CPU (`gpu/spawn/host.py` calls a library with NumPy arrays, in a worker process per
library). It checks all 25 against NumPy, bit for bit where the reference follows the hand
kernel's order (lane-strided partials, the warp tree 16, 8, 4, 2, 1, the block's second stage),
which is every kernel without exp; the others to 2e-6 (1e-5 for MLA). That pins the order Vx
#260's ordered reductions must keep. Run it after `build.sh` (with `VX_DISPATCH_LIB`).

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
`device.vx` holds macros: the bf16, binary16, INT4 and FP8 decodes, `sigmoid`, and the warp and
block sums and maxima on shared memory.

## What Vx is missing

| Need | Today | Vx issue |
| --- | --- | --- |
| Reductions (sum, max) in an `across` loop | lane loops on shared memory, five barriers per warp sum; to become library routines, OpenMP style | #260 |
| Device functions in an `across` loop | refused (the proof does not read their bodies); the helpers are macros | #2076 D4 |
| `mlir!` blocks in an `across` loop (exp as the hand kernels, wgmma, TMA) | refused; exp is libdevice expf, the tensor-core kernels are not built | #2076 D1 to D3 |
| A stable entry name and documented parameter order | `vxc --emit-ptx` (PR #1964): the entry is the function's name, the parameters its own in declared order | #251 |
| Correctly rounded sqrt / div (the norms) | `sqrt.approx` / `rsqrt.approx` in 390c2534; `sqrt.rn` + `rcp.rn` with PR #1955 | #1944 |
| Bit casts (`f32::from_bits`) | not a pure call to the proof; bf16 decode uses exact arithmetic instead | #1943 |
| Several Vx libraries in one process | every library's kernel is `vx_npu_kernel_0`: the host path finds it with `dlsym(RTLD_DEFAULT, ..)` and the GPU path caches it by name, so the second library runs the first one's kernel; `host.py` uses a process per library, and `test_spawn_vx_launch.py` / `test_spawn_vx_graph.py` fail until the fix | #2114 |
| Scalars passed by value | E6003 unless `machine.vx` lists CPU_DRAM as visible | #1947 |
| Raw pointers in a region | silent host fallback before PR #1959, which takes them and warns on any other fallback; kernels take `[?]` tensors | #1945 |
| Shared memory `::new()` | zero-fill races with other warps; use `unsafe { ::uninit() }` | #1946 |
| A Vx build of the same file (exe or .so) | the host copy of a GPU-only helper stops the program; PR #1958 | #1942 |
| Grid and block chosen by the program | `across` loops (merged, #2095 to #2097) | #252 |
| Launch without a device sync each time | waits only before a copy to the host, a free, or exit; merged as #2098 (was #2066) | #25 |
| A stream of the caller, and CUDA graphs | `vx_cuda_set_stream` (merged as #2100, was #2068): launches, cuBLAS and copies on our stream, temporaries stream-ordered, a stop with a message on a wait during a capture; `gpu/test_spawn_vx_graph.py` | #1986 |
| Futures the compiler waits on | no waits are emitted; the runtime waits before a host copy, a free, or exit | #25 |
| TF32 `mma.sync m16n8k8` | WMMA m16n16k8 only | #350 |

Nothing has run on a GPU yet: the next pod runs both test files (`scripts/pod_validate.sh`).
