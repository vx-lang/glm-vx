# Tensor-core probes (glm-vx #10, plan #2)

Can Vx's own GPU path (`spawn on`, `vxc --emit-ptx`) hold a tensor-core accumulator in registers
across a k loop? These probes check that with register tensors, `Tensor<f32, [32], Memory::REG>`
(Vx #1983, PR #1991). They are not among the 25 kernels: `gpu/spawn/build.sh` builds
`gpu/spawn/*.vx` only, and this directory has its own `build.sh`.

| Probe | What it computes | Chip |
| --- | --- | --- |
| `mma_bf16_sm80.vx` | one warp, a 16 x 64 tile of `A B^T` (bf16 -> f32), 8 `mma.sync.m16n8k16` per k16 step | sm_80 |
| `wgmma_bf16_sm90a.vx` | one warpgroup, a 64 x 64 tile, 4 `wgmma.mma_async.m64n64k16` per 64-k block, A and B in shared memory through matrix descriptors (no swizzle) | sm_90a |
| `wgmma_fp8_sm90a.vx` | the block-FP8 tile: `m64n64k32 .e4m3.e4m3`, and each 128-k block's partial times `a_scale[m, kb] * b_scale[n / 128, kb]`, in a plain loop over the REG tensors | sm_90a |
| `wgmma_fp8_fn_sm90a.vx` | the same, with the scale loop in a device function that takes `&mut Tensor<f32, [32], Memory::REG>` (the known inlining gap of PR #1991) | sm_90a |

Each file writes the fragment layout it uses in comments, with the PTX ISA section:
9.7.14.5.8 for the mma.m16n8k16 A, B and C fragments; 9.7.15.5.1 for the shared memory layout and
the matrix descriptor; 9.7.15.5.2 for the wgmma D fragment. `gen_wgmma.py` writes the mlir! block
that issues the wgmma instructions (32 accumulator operands); the probes hold its output.

## Results

`vxc` from Vx main (c041a98d, with #1964 `--emit-ptx` and #1989 the chip from the machine file)
merged with PRs #1954, #1991 (register tensors, with #1959 under it), #1960, #1961, #1962, #1972
(launch shapes) and #1974. ptxas from CUDA 13.1, `-v`, for the probe's chip. The .text size is
the cubin's `.text.<kernel>` section (no cuobjdump here); a SASS instruction is 16 bytes.

| Probe | Registers | Stack | Spills | `__local_depot` | Shared | SASS .text |
| --- | --- | --- | --- | --- | --- | --- |
| `mma_bf16_sm80` | 56 | 0 | 0 | none | 0 | 3,584 B (224) |
| `wgmma_bf16_sm90a` | 72 | 0 | 0 | none | 16 KiB | 10,368 B (648) |
| `wgmma_fp8_sm90a` | 110 | 0 | 0 | none | 16 KiB | 12,160 B (760) |
| `wgmma_fp8_fn_sm90a` | 110 | 0 | 0 | none | 16 KiB | 12,160 B (760) |

- The k loop stays a loop with a trip count from a parameter; its body has the 8 `mma.sync` (or 4
  `wgmma`) on the same 32 registers, and the loops over the REG tensors are unrolled.
- `mma_strip`, `wgmma_*` and `scale_accumulate` take a REG tensor and were inlined each time,
  also with two call sites of each in one loop pass (a test copy of the fp8 probe: 166
  registers, 8 wgmma, no call). The inlining gap did not show here: no `__local_depot` came from a
  device function that takes a REG tensor.
- None has run on a GPU yet. `gpu/test_tensor_core_probes.py` checks each against numpy on a GPU
  (mma on sm_80 or newer, wgmma on compute capability 9.0 only), and replays each probe's index
  math against the PTX ISA layouts in numpy on any machine (those tests catch a swapped LBO and
  SBO, or a wrong C or D layout). The descriptor fields (LBO is the k step between core matrices,
  SBO the 8-row step, for K-major without swizzle) follow the ISA's description and CUTLASS's
  `make_gmma_desc`; only a GPU run settles them.

## Vx findings

1. **A `let mut` declared inside an `if` of a spawn region can stay in local memory.** Its alloca
   is not in the kernel's entry block, and LLVM's SROA promotes only entry-block allocas; later
   passes usually promote it, not always. `repro_reg_in_if.vx` (`vxc repro_reg_in_if.vx --host
   default --machine fleet/a100-40.vx --action emit-llvm -o /dev/null`) leaves `acc[31]` of a
   REG tensor in `__local_depot8[128]` (ptxas removes it). A copy of the fp8 probe with two
   k blocks per loop pass left its loop counter `kb` in `__local_depot26[8]`, which ptxas kept
   (8 bytes of stack, stores only). Declaring them before the `if` removes both, so the probes do.
   The fix would put the region's allocas in the entry block.
2. **An mlir! block that takes a shared tensor needs `memref<N x T, 3>`.** With `memref<2048xi32>`
   vxc says `'memref.cast' op operand type 'memref<2048xi32, 3>' and result type
   'memref<2048xi32>' are cast incompatible`, with no position and no mention of the mlir!
   block, then falls back to the AST path, which panics: `panicked at
   src/codegen/lower/expr.rs:3567:13: Function barrier not found`. `repro_smem_mlir.vx`.
   `memref.extract_aligned_pointer_as_index` of the `, 3` memref gives the shared::cta address
   that a wgmma descriptor needs (`mov.b64 %rd, <smem symbol>`).
3. The fleet files of PR #1989 have no `Memory::REG`, so these probes use their own machine files
   (`machine-sm80.vx`, `machine-sm90a.vx`); `gpu/spawn/machine.vx` is unchanged.

## Build and test

```
VXC=/path/to/vxc PTXAS=/usr/local/cuda/bin/ptxas gpu/spawn/probes/build.sh   # -> gpu/build/spawn/probes/
python -m pytest gpu/test_tensor_core_probes.py
```

## Next, for a real block-FP8 GEMM

- Load A and B with TMA (`cp.async.bulk.tensor`) into a 3-4 stage shared ring with mbarriers, a
  producer warp and two consumer warpgroups, instead of every thread staging words and two
  `bar.sync` per block. Vx needs: a `CUtensorMap` kernel parameter (`.param .align 64` by value)
  in the `--emit-ptx` manifest, mbarrier ops, and dynamic shared memory past 48 KiB (launch.py
  and Vx's launch must set the attribute).
- 128-byte swizzle (descriptor mode 1): a shared tensor aligned to 1024 bytes, which Vx cannot
  ask for yet (shared tensors are `.align 16`).
- Larger tiles: m64n128 or n256 per warpgroup (64-128 accumulator registers), `setmaxnreg`.
- Keep one wgmma group in flight (`wait_group 1`) and apply the block scale to the previous
  block's partial while the next runs, with two REG accumulators (as here: `acc` and `part`).
- Write bf16 output (`stmatrix` or a TMA store) and fuse the activation's per-row scale load.
- Vx: allocas in the entry block (finding 1), and a guarantee that a device function taking a REG
  tensor is inlined.
