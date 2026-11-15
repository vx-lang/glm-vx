# MLA decode on the tensor cores with a bf16 cache, H200 (#33)

These runs cover `mla_kernel='tc'`: `mla_q_bf16_at`, then `mla_tc_at`, then `mla_flash_merge_at`, with
the latent and rope caches in bf16. The step is GLM-5.3's MLA shape: 64 heads, rank 512, rope 64, and
`selected` a random permutation of a 4096-token cache. The kernels were built with Vx PR #3487/#3490's
vxc (84a6688fc, which is now part of main d75d4c60). `lanes::copy_rows_async` gathers the rows with
cp.async.

The pod was stopped before the logs were copied back, so this README records the numbers as the
scripts printed them. The scripts are in this folder; run them from the repository root with
`PYTHONPATH=.` and a folder of the built `.ptx`/`.json` files as the argument.

## The whole step as a CUDA graph (`graph_sweep.py`)

These times are what ResidentGLM replays. Each graph launch is timed with CUDA events after an L2
flush, median of 50; an almost empty graph (one 32-element add) takes 3.1 us. The error column is the
max error against float64 attention, over the bf16 cache and queries for tc and over the f32 values
for flash.

| selected tokens | flash: mla_flash_at + merge (f32) | tc: prep + mla_tc_at + merge (bf16) | SGLang FlashMLA |
|---|---|---|---|
| 256 | 21.0 us (4.4e-8 .. 1.1e-7) | 22.8 us (4.9e-4) | |
| 1000 | 27.7 us | 28.3 us (2.7e-4) | |
| 2048 | 39.3 us | 30.5 us (1.8e-4) | 13.9 us |

In ResidentGLM on a 16-head, rank-512 model, 600 replayed steps of `'tc'` give logits within 6.7e-3
(relative to the largest) of `'flash'`, and the argmax is the same at every step. `'flash'` against
`'chunks'` is within 4.7e-7. `gpu/test_resident_graph_hardware.py` checks both.

## The steps that got there (kernel alone, `tc_sweep.py`, CUDA events, L2 flushed)

| version | 256 | 2048 | what changed |
|---|---|---|---|
| first: 4 subgroups, gather through registers, queries converted in each block | 33.3 us | 36.4 us | |
| ids loaded first, gather unrolled | 30.5 | 33.5 | the row loads no longer wait on each id |
| queries unrolled | 27.3 | 30.6 | |
| queries from mla_q_bf16_at with lanes::copy_async; blocks past the splits return | 26.7 | 30.6 | prep: 11 us at first, 5 us with 64 blocks |
| 8 subgroups | 23.0 | 42.0 | 226 registers: one block an SM, two waves at 2048 |
| lanes::copy_rows_async | 12.4 | 21.4 | cp.async, 16 bytes a lane; 140 registers |
| .maxnreg 128 | 12.5 | 16.9 | two blocks an SM |

ncu, before copy_rows_async: 74% of the stall samples were in the gather. Each bf16 element was one
st.shared.b16 with PRMT/SHF and address arithmetic around it, about 1,600 SASS instructions a warp
before the first mma, at 2 warps a scheduler.

## Where the time goes now (`phase_time.py`, the kernel cut after each phase, as graphs)

| | 1 split (4 blocks) | 64 splits (256 blocks) |
|---|---|---|
| ids and gather | 6.3 us | 7.0 us |
| + scores (mma) | 7.3 | 8.4 |
| + softmax | 7.9 | 9.2 |
| + P @ rows and the store of numerator (full kernel) | 10.8 | 16.6 |

The merge then reads the 8 MB f32 numerator back. As its own graph it takes 11 us at 8 splits and
19.8 us at 64 (L2 cold); in the step it adds about 9 us. The prep kernel takes about 5 us as its own
graph.

## Tried and dropped

- **All 64 heads a block (`mla_tc64_at` + its own merge, 16 subgroups, 128 registers with 200 B of
  spills):** each chunk's rows are gathered once instead of four times, but a block takes about 22 us.
  The step took 32.1 / 35.7 / 36.9 us at 256 / 1000 / 2048 tokens.
- **Two chunks a block (one max over 64 tokens, no rescaling; half the splits and half the
  numerator):** 30.4 / 35.5 / 37.1 us, slower at every count. Four chunks do not fit in shared memory.
- **Merge with 16 column blocks a head instead of 4 (`chunks_sweep.py` era):** 42.2 us against 40.6 us
  with the flash kernel at 2048.

## Next

- **The store of numerator and the merge** cost about 16 us of the 30.5 us. Options: wider stores from
  the mma registers, a smaller numerator (fewer splits with lanes::scale_rows, Vx #3491), or folding
  the merge in.
- **The prep launch** could be folded into the kernel that writes the queries (matvec_heads).
- **Vx #3488 (wide shared stores of bf16) and #3489 (`as bf16` as one cvt.rn.bf16.f32)** trim the
  rope and query copies and the P conversion.
