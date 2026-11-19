# A pipelined tensor-core `gate_up`, with x in shared memory: closer, still not faster (#63, item 4)

`gpu/spawn/moe_mma/fp8_moe_mma16p_gate_up_at`, from `gate_up16p_ptx` in `gen_moe_mma.py`, is z22's kernel with each
warp's whole range (6 groups of 128 columns) in one PTX call:
- The weights of the first two groups are issued first.
- x is loaded in one round trip (24 floats a lane), put in f16 with a power-of-two scale a group, and stored to
  the warp's region of a Vx SMEM tensor. The PTX gets the region's shared-window address from
  `memref.extract_aligned_pointer_as_index` on the `memref<8x384xi32, 3>`; the PTX shows
  `mov.b64 %rd, vx_npu_kernel_0_smem_1` with no `cvta`.
- Each group's B fragments come from shared memory. Group j + 2's weights are issued once group j is done.
- 128 registers (20 bytes spilled), 13 KB of shared memory, 144 blocks of 8 warps.

**Correctness** (`mma-check.json`): within 4.1e-7 of max |act| of the NumPy model at inter 256 and 2048, the same
as z22's kernel. So the shared-memory address from Vx works in inline PTX.

**Time** (ncu, cold, TP8 shapes):

| gate_up | k = 7 | k = 8 | k = 9 | instructions (k = 9) | warps active |
|---|---|---|---|---|---|
| `fp8_moe_cvt_gate_up` (served) | 12.5 us | 12.5 | 13.3 | 4.42 M | 26% |
| `fp8_moe_mma16_gate_up_at` (z22) | | | 15.3 | 1.78 M | 14% |
| `fp8_moe_mma16p_gate_up_at` | 10.5 | 12.6 | 14.4 | 1.23 M | 14% |

Files: `moe-tp8.md`, `moe-tp8-k.md`.

- Pipelining takes 1.1 us off z22's kernel, but it is still behind the CUDA-core kernel at k = 9.
- Its time grows with k (the bytes), while `cvt`'s barely moves. With about 8 warps an SM and one group of weights
  in flight a warp while the last one computes, it moves about 2 TB/s.
- More would need a third group in flight (about 160 registers, so one block an SM and 144 blocks in two waves),
  or the K range split across blocks with a cross-block sum.
- Parked. `fp8_moe_cvt_gate_up` stays in use, and `ResidentGLM(moe_mma=True)` changes only the down projection.
