# Block-FP8 GEMM on H100 (glm-vx #10)

The first tensor-core GEMM of the model path: a block-FP8 GEMM for sm_90a written in Vx as a
`spawn on` kernel, built with `vxc --emit-ptx`. It uses TMA, an mbarrier ring, and wgmma
m64n128k32 e4m3. This is v1: correct and pipelined, and laid out so it can be tuned. It is not a
DeepGEMM clone yet. It has not run on a GPU; `gpu/test_fp8_gemm_hardware.py` is the pod test.

```
C[m, n] = sum_kb a_scale[m, kb] * b_scale[n / 128, kb] * sum_{k in block kb} A[m, k] * B[n, k]
```

| Operand | Layout |
| --- | --- |
| A | e4m3 [M, K], row-major (the activations) |
| a_scale | f32 [M, ceil(K/128)], per-token groups of 1 x 128 (DeepGEMM, SGLang W8A8 block) |
| B | e4m3 [N, K], row-major (the checkpoint's FP8 weight) |
| b_scale | f32 [ceil(N/128), ceil(K/128)], the checkpoint's `weight_scale_inv` (dequant = code * scale) |
| C | f32 or bfloat16 [M, N] |

K must be a multiple of 16, since TMA's row stride is K bytes. Other M, N and K tails are
handled. lm_head (154880 x 6144) is BF16 in the checkpoint and does not use this kernel.

## Files

| File | What it is |
| --- | --- |
| `gpu/spawn/fp8_gemm/fp8_gemm.vx.in` | The kernel template, with the design notes in its header |
| `gpu/spawn/fp8_gemm/gen_fp8_gemm.py` | Writes `fp8_gemm_m64.vx`, `fp8_gemm_m128.vx` and `wgmma_m64n128k32.vx` (`--check` verifies them) |
| `gpu/spawn/fp8_gemm/sm90.vx` | mbarrier, TMA, tensormap fence, wgmma descriptor and bf16 helpers (inline PTX in `mlir!`) |
| `gpu/spawn/fp8_gemm/fp8_gemm_reduce.vx` | The split-K reduction |
| `gpu/spawn/fp8_gemm/build.sh`, `machine-sm90a.vx` | The build, into `gpu/build/spawn/fp8_gemm/`. Not part of `gpu/spawn/build.sh` |
| `gpu/fp8_gemm.py` | Host side: `plan`, `FP8GEMM`, tensor maps, `reference` / `error_bound`, and `simulate` (a NumPy replay of the kernel) |
| `gpu/runtime.py` | `encode_tensor_map_2d` (`cuTensorMapEncodeTiled`, an optional driver symbol) |
| `gpu/test_fp8_gemm.py` | Offline tests: tensor maps with a fake driver, the replay against the reference, descriptor mutations, the bound, the wrapper's launches and argument checks |
| `gpu/test_fp8_gemm_hardware.py` | H100 test. Runs the model shapes at M in {1, 7, 64, 128, 256} plus tails. Writes time, TFLOP/s, GB/s and the measured error to JSON. `scripts/pod_validate.sh` runs it on compute capability 9.0 |

## Design

**CTA and threads.** One CTA computes a `BLOCK_M x 128` tile of C over one range of 128-k blocks.
There are two variants:

- **m64**: one consumer warpgroup, 160 threads. Used when M <= 64.
- **m128**: two consumer warpgroups that share each B tile, 288 threads.

Threads `0 .. 128*c - 1` are the consumer warpgroups. A warpgroup must start at a warp index that
is a multiple of 4 (PTX ISA 9.7.15.1). One producer warp follows them, and one thread of it issues
all the TMA. The CTA order is `bid = (n_tile * splits + split) * tiles_m + m_tile`, so CTAs that
read the same B tile are next to each other.

**Shared memory ring.** Each of the 4 stages holds one A box (`BLOCK_M` rows x 128 bytes) and one
B box (128 x 128 bytes). Both use 128-byte swizzle in 1024-aligned buffers
(`Tensor<i32, [..], Memory::SMEM, aligned(1024)>`). Each stage has two mbarriers:

- `full[s]`: 1 arrival plus the stage's TMA bytes.
- `empty[s]`: one arrival from lane 0 of each consumer warp.

Dynamic shared memory per variant. Vx lays the tensors out in declaration order: the A ring at 0,
the B ring after it (both 1024-aligned) and the barriers last.

| Variant | Dynamic shared | CTAs per SM |
| --- | --- | --- |
| m64 | 98,368 B | 2 |
| m128 | 131,136 B | 1 |

**Producer loop.** For each k block:

1. Wait on `empty[s]` with parity `p ^ 1`. A fresh barrier passes this wait.
2. `mbarrier.arrive.expect_tx(full[s], bytes)`.
3. Issue two `cp.async.bulk.tensor.2d` loads, at `{k, m0}` for A and `{k, n0}` for B.

TMA fills elements outside A or B with zeros. That is the whole tail story for M, N and K: a zero
row or zero k contributes nothing, and the epilogue masks rows and columns past C.

**Consumer loop.** For each k block:

1. Load this block's three scales (`b_scale[n0/128, kb]` and `a_scale` of the thread's two rows)
   before waiting on `full[s]`, so their latency hides behind the wait.
2. Issue 4 x `wgmma.mma_async.m64n128k32.f32.e4m3.e4m3` into the REG tensor `part` (64 f32
   registers). The first step has scale-d = 0. Then `commit_group` and `wait_group 0`.
3. Lane 0 of each warp arrives on `empty[s]`.
4. Promote: `acc += part * (a_scale * b_scale)` in f32 on the CUDA cores, per 128-k block.

Waiting for 0 groups before the promotion is DeepGEMM's order too. At m64n128, the two 64-register
accumulators leave no room for a third one to keep a group in flight. The TMA loads of the next
stages are in flight meanwhile.

**Descriptors** (`sw128_desc`). The descriptor sets: start address >> 4, LBO 1 (unused for a
swizzled K-major tile one swizzle row wide), SBO 1024 (one 8-row group to the next), and swizzle
mode 1 (128B). The base offset is 0 because the buffers are 1024-aligned. Each k32 step adds 32
bytes (+2 encoded), the way DeepGEMM and CUTLASS advance inside a swizzle row. Warpgroup g's A
starts 8192 * g bytes in.

**Epilogue.** Register `4j + q` of a thread holds row `16*warp + lane/4 + 8*(q >> 1)` and column
`8j + 2*(lane%4) + (q & 1)` (PTX ISA 9.7.15.5.2). It is stored as f32, or as bfloat16 via
`cvt.rn.bf16.f32`, masked to `row < M` and `col < N`.

**Split-K** (decode). M = 1 and N = 2048 is only 16 tiles. `plan` splits K until the CTAs fill
the GPU's CTA slots (132 SMs x CTAs per SM), with at least 2 k blocks per split. It picks the best
wave efficiency, then the fewest splits. Each split writes f32 partials to its own
`[splits, M, N]` slice. `fp8_gemm_reduce.vx` sums them in order, so the result is deterministic,
and writes f32 or bf16. The plans at the model's shapes:

| M, N, K | Variant | Splits | Grid |
| --- | --- | --- | --- |
| 1, 2048, 6144 | m64 | 16 | 256 |
| 1, 576, 6144 | m64 | 24 | 120 |
| 1, 6144, 2048 | m64 | 4 | 192 |
| 1, 12288, 6144 | m64 | 5 | 480 |
| 1, 6144, 12288 | m64 | 11 | 528 |
| 128, 2048, 6144 | m128 | 8 | 128 |
| 256, 2048, 6144 | m128 | 4 | 128 |
| 256, 12288, 6144 | m128 | 1 | 192 |

**Tensor maps in global memory.** Vx has no `CUtensorMap` kernel parameter (`.param .align 64`),
so the host passes the two maps in device memory instead. `FP8GEMM.tensor_maps` encodes them with
`cuTensorMapEncodeTiled`:

- uint8 elements, box {128, rows}, 128-byte swizzle, L2 promotion 256B, out-of-bounds fill zero.
- A's map at byte 0 and B's at byte 128 of one 256-byte allocation, so each is 64-byte aligned.
- They are uploaded with a blocking copy and cached by (A, B, shape, variant).

PTX allows the `tensorMap` operand of `cp.async.bulk.tensor` to be the generic address of a map
in `.global` memory (PTX ISA 5.5.8 "Tensor-map"). But the tensormap proxy is *not* acquired from
the generic proxy at kernel start (PTX ISA 8.9.4, "Memory synchronization"). The CUDA programming
guide (4.12.2.2) therefore fences a map in global memory once per thread block, with system scope
because the host wrote it, before first use. The producer thread is the only reader of the maps,
so it runs `fence.proxy.tensormap::generic.acquire.sys [map], 128` on each one, then
`prefetch.tensormap`.

That fence needs PTX ISA 8.3, and vxc pins sm_90a to 8.0. ptxas reports `Feature 'fence.proxy
with unidirection proxy fence' requires PTX ISA .version 8.3`. So `build.sh` passes
`build.py --ptx-version 8.3`:

- It raises the `.version` line and changes nothing else, since PTX 8.3 accepts every 8.0
  instruction.
- The manifest records `ptx_version: {vxc: 8.0, raised_to: 8.3, vxc_ptx_sha256}`.
- PTX 8.3 needs a CUDA 12.3 or newer driver.

The proper fix is for Vx to emit PTX 8.3 or newer for sm_90a, or to let the machine file name the
version. A `CUtensorMap` parameter would remove the fence altogether.

Rules for the maps:

- Rewrite a map only between launches that read it.
- Upload maps outside a CUDA graph capture. Call once before capturing, so the cached maps
  already exist.
- Never write a map from a kernel or from an async copy on another stream without
  `fence.proxy.tensormap` on both sides.

## Numerical contract

Each block's sum of e4m3 products is exact in f64. A product has at most 8 significant bits,
between 2^-18 and 2^17.6, so 128 of them need fewer than 53 bits. `reference` is therefore the true
C up to f64 rounding of the scale products.

Hopper's FP8 wgmma does not accumulate in full f32. DeepSeek-V3 (arXiv 2412.19437, section 3.3.2)
measured "around 14 bits", which is why the kernel promotes to f32 every 128 k, as DeepGEMM does.
The bound used for every element is:

```
|C_kernel - C| <= EPS_MMA * sum_kb |s_kb| S_kb
                + (KB + splits + 2) * 2^-24 * sum_kb |s_kb| (|P_kb| + EPS_MMA * S_kb)
                [+ 2^-8 * (|C| + bound) for bf16 output]

where
  EPS_MMA = 2^-10
  s_kb    = a_scale * b_scale
  P_kb    = the block's exact sum
  S_kb    = sum_k |a_k b_k|
```

Where EPS_MMA comes from: each 128-k block is 4 k32 steps, and each step sums 32 products and adds
them to the running block sum. That is at most 8 roundings, each at most 2^-13 of a partial sum
bounded by S_kb, so 8 x 2^-13 = 2^-10.

The second term covers the f32 rounding of `s`, of `part * s`, of the accumulation and of the
split-K sums. The bf16 term is round-to-nearest-even with 8 significant bits.

`gpu/test_fp8_gemm.py` checks the bound against two emulated accumulators:

- It holds for one that keeps 14 bits, truncated, at every step and every accumulation.
- It fails for one that keeps 9 bits, so the bound is not vacuous.

The hardware test records `effective_eps_mma`, the measured max |err| / sum |s| S. This is the
H100's actual accumulation precision for these data.

## PTX ISA sections relied on

- **9.7.15.1:** a warpgroup is 4 contiguous warps whose first warp index is a multiple of 4.
- **9.7.15.5.1 / 9.7.15.5.1.2:** the shared-memory matrix layouts (128B swizzle, K-major) and the
  matrix descriptor format.
- **9.7.15.5.2:** the register fragment of D for m64nNk32 .f32.
- **9.7.15.7.1-3:** `wgmma.fence`, `commit_group`, `wait_group`.
- **9.7.15.5 (e4m3 rows):** `.e4m3` wgmma takes K-major A and B only, with no imm-trans.
- **5.5.8:** the tensor-map (128 bytes; `.param`, `.const` or `.global`).
- **8.9.4:** the tensormap proxy is not acquired at kernel start.
- **`fence.proxy.tensormap::generic` (PTX 8.3+)** and `prefetch.tensormap`.
- **`cp.async.bulk.tensor`** (the tile mode, `.mbarrier::complete_tx::bytes`). A shared::cta
  address is a valid shared::cluster address of the executing CTA.
- **mbarrier:** `init`, `arrive`, `arrive.expect_tx`, `try_wait.parity`, and
  `fence.mbarrier_init`.
- **`cvt.rn.bf16.f32`.**
- **cuda.h `cuTensorMapEncodeTiled`:** the host map 64-byte aligned, the global address 16-byte
  aligned, strides a multiple of 16, a box row a multiple of 16 bytes and at most the swizzle span.

## Vx PRs required

Built with a vxc made from Vx `c041a98d` (main with #1989, sm_90a) merged with these PRs. Since
then #1995, #1996 and #2058 are on Vx main (#2058 replaced #1991: same commits, checked with
`git range-diff`), and #2013/#2014 replaced #1972/#1974:

| PR | What it gives this kernel |
| --- | --- |
| #1954, #1960, #1961, #1962, #1972, #1974 | The earlier spawn fixes and launch shapes, as for the 25 kernels |
| #1991, through 2731ec40 | REG tensors, with allocas in the entry block |
| #1995 | Dynamic shared memory over 48 KiB; `dynamic_shared_bytes` in the manifest |
| #1996 | `aligned(N)` and the machine file's `alignment:` |

This is the scratchpad worktree `vxfp8`. Its two merges on top of the earlier `vxreg` merge
resolve conflicts in `abi.md`, `fleet/README.md`, `cuda_dispatch.cpp` (launch shape plus dynamic
shared size), `diagnostic.rs`, `VxLowering.cpp` and the error index.

The 25 sm_80 spawn kernels with this vxc:

- #1995 and #1996 change none of them.
- 2731ec40, the alloca move in #1991, reorders instructions in 10 of them: bf16_matmul,
  fp8_matmul, gemm_f32, index_scores, int4_matmul, int4_matvec, mla_merge, mla_partial,
  softmax_rows, topk.
- 5 of those 10 differ only in instruction order. All 25 still assemble with 0 spills, and
  registers are the same except topk, which goes from 38 to 32.

## Numbers

ptxas (CUDA 13.1, `-arch=sm_90a -v`), with a vxc from Vx main (`dfae3065`, which has #1995 and
#1996) merged with the #2013-#2016 launch stack, #2058 and #2002. The vxc of `c041a98d` gives the
same registers, stack and shared memory for this source.

| Kernel | Registers | Stack | Spills | Dynamic shared | Block | SASS |
| --- | --- | --- | --- | --- | --- | --- |
| `fp8_gemm_m64` | 158 | 0 | 0 | 98,368 B | 160 | 19,840 B (1,240 instructions) |
| `fp8_gemm_m128` | 158 | 0 | 0 | 131,136 B | 288 | 19,840 B (1,240 instructions) |
| `fp8_gemm_reduce` | 32 | 0 | 0 | 0 | 256 | 3,712 B (232 instructions) |

Neither variant has a `__local_depot`. Register use fits both block sizes: 158 x 288 = 45.5k of 64k.

GPU numbers have not been measured: there was no H100 here. `scripts/pod_validate.sh` on an
H100 writes `fp8_gemm_hardware.json` with time, TFLOP/s and GB/s per shape.

## Vx findings

1. **Two `if` statements in a row in a spawn region fail with E3048.** The first `if` is reported
   as "this `if` has no `else`, so it has no value". The same code in a host function compiles.
   This happens with the old vxc too. Repro (`--machine` any GPU machine file, `--emit-ptx`):

   ```
   pub fn repro(out: &mut Tensor<f32, [?], Memory::HBM>, n: i32) -> void {
     spawn on(Topology::Device) grid([1]) block(64) {
       let tid: i32 = thread_id();
       let mut i: i32 = 0;
       if tid == 32 {
         i = n;
       }
       if tid < 32 {
         out[tid as i64] = i as f32;
       }
     }
     return;
   }
   ```

   A statement between the two `if`s, or an `else`, avoids it. The kernel uses `else`.
2. **`Tensor<f32, [64], Memory::REG>::fill(0.0)` stays a loop with a variable index.** That puts
   the tensor in local memory: 528 bytes of stack, and every promotion goes through `ld.local`. A
   `for r in 0..64 { acc[r] = 0.0; }` loop is unrolled and fixes it. `::fill` of a REG tensor
   should unroll, or be refused, like any other variable index (E6033).
3. **Fixed by 1341c84c (in #2058):** #1995's views no longer kept declaration order once
   2731ec40 was merged. 2731ec40 moves each late alloca to `entry.begin()` in walk order, which reverses them. #1995 then lays the views
   out in that reversed order. Here the barriers (declared last) land at offset 0, B at 1024 and A
   at 66560, which wastes 960 bytes of padding. It is harmless for this kernel, which takes each
   tensor's address. The fix is to move each slot after the previous one.
4. **vxc writes PTX 8.0 for sm_90a.** `fence.proxy.tensormap` needs 8.3, so a tensor map in global
   memory needs the `--ptx-version` workaround described above. The asks, in order:
   - emit PTX 8.3 or newer for sm_90a;
   - a `CUtensorMap` kernel parameter (`.param .align 64 .b8 [128]`), which removes the fence;
   - `setmaxnreg`.
5. **Over 48 KiB of shared memory declared after a loop crashes vxc** (SIGSEGV in "Lowering Vx to
   Standard"; Vx #1995 together with #2058, the register tensors: plain main before #2058 compiles
   it). Two shared tensors that are both used, after a `for` or `while` in the spawn body, are
   enough. The kernel declares its shared tensors before the accumulator loop, which avoids it.
   Repro sent to vx-21:

   ```vx
   pub fn k() -> void {
     spawn on(Topology::Device) {
       for _r in 0..1 {
       }
       let mut ring = unsafe { Tensor<i32, [16384], Memory::SMEM>::uninit() };
       let mut bars = unsafe { Tensor<i32, [1], Memory::SMEM>::uninit() };
       ring[0] = 0;
       bars[0] = 0;
     }
     return;
   }
   ```

## What is left for DeepGEMM-level speed

- **setmaxnreg** (producer about 40, consumers 232). Then m64n256 or two m64n128 accumulators can
  keep one wgmma group in flight while the previous block is promoted, and BLOCK_N = 256 halves
  the A traffic per FLOP.
- **TMA multicast across a 2-CTA cluster** for the shared operand, and a persistent grid with a
  tile scheduler (stream-K for decode) in place of split-K plus a second launch. 256 x 12288
  currently runs 1.45 waves.
- **A TMA-loaded a_scale** in shared memory (DeepGEMM keeps it MN-major), so scale loads do not
  sit in the consumer's instruction stream. The epilogue also needs work: `stmatrix` or a TMA
  store of bf16, not 2-byte scattered stores.
- **Prefill sizes (M = 1k-16k):** BLOCK_M = 128 x 2 warpgroups, larger stages and a tuned stage
  count.
- **On the host side:** maps cached per buffer are fine for resident decode. A changing activation
  pointer needs `cuTensorMapReplaceAddress` or an in-kernel `tensormap.replace`, so as not to
  re-upload.
- **Measure, then tune**, against DeepGEMM on the same H100: issue #10's bar is 90%.

## Build and test

```
VXC=<vxfp8 vxc> PTXAS=/usr/local/cuda/bin/ptxas gpu/spawn/fp8_gemm/build.sh
python -m pytest -q gpu/test_fp8_gemm.py                    # any machine
OMP_NUM_THREADS=16 python -m pytest -q gpu/test_fp8_gemm_hardware.py   # H100; JSON in gpu/build/
```

The pod's toolchain bundle (vxc 390c2534) lacks these PRs. Build the vxc on the pod from Vx main
(which has #2058) with #2013 (and #2002 for PTX 8.3), or bring its PTX: the PTX is portable, and only
`build.sh`'s checks need vxc.
