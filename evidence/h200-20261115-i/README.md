# MLA flash decoding on an H200 (#33), vxc 243ea31c

These runs time `mla_flash_at` + `mla_flash_merge_at` against `mla_chunks_at` and `mla_heads_at`. The
step is GLM-5.3's decode MLA: 64 heads, rank 512, rope 64, and `selected` a random permutation of a
4096-token cache. Timing uses CUDA events with L2 flushed, median of 30
(`validation.moe_profile.event_us`), and the flash rows time the flash kernel and the merge together.
The output check is the max |diff| against `mla_chunks_at`, and the error against float64 attention
relative to its largest value. The kernels were built with vxc from Vx PR #3475 (daf984ac), and
243ea31c, its merge, gives the same PTX for them.

The four files trace the kernel through its versions. The names in them are of the variants tried:

| file | flash kernel | merge |
|---|---|---|
| `flash_sweep.txt` | `flash_at`: four heads a subgroup, 199 registers; `flash2_at`: two heads, 112 | one block a head, a `while` over the splits |
| `flash_sweep2.txt` | the same | the same, with a branch-free `for s in 0..64` |
| `flash_sweep3.txt` | `flash3_at`: `flash2_at` loading the next token's row while it scores one (139 registers); `flash3r_at`: the same under `.maxnreg 128` | as in 2 |
| `flash_sweep4.txt` | as in 3 | four subgroups of 16 splits, four columns a lane, then a block merge |

The committed kernels are `flash3r_at` (as `mla_flash_at.vx`) and the merge of `flash_sweep4.txt`.

| selected tokens | `mla_heads_at`, best split | first flash (2 heads) | committed | max \|diff\| vs chunks | rel. err vs f64 |
|---|---|---|---|---|---|
| 256 | 52.3 us | 32.2 us | 22.1 us (8 a split) | 1.5e-7 | 2.2e-7 |
| 1000 | 81.6 us | 56.1 us | 28.9 us (16 a split) | 1.0e-7 | 3.1e-7 |
| 2048 | 119.8 us | 73.8 us | 40.2 us (8 a split, capped at 64 splits) | 7.5e-8 | 2.2e-7 |

The flash kernel and the merge, timed apart (`flash_split.py`, 2048 tokens, 64 splits):

| kernel | before | after | what changed |
|---|---|---|---|
| flash | 41.0 us | 29.2 us | prefetching the next token's row and the id after it, with a 128-register budget |
| merge | 53–59 us | 19.8 us | ptxas would not unroll the branchy version, so its loads ran one at a time; it now uses constant-length, branch-free loops and a subgroup per 16 splits |

ncu, before the merge rewrite:
- `mla_flash2_at` took 44 us, with 25% occupancy at 112 registers and SM throughput at 42%.
- The merge took 57 us. It read 8.4 MB at 148 GB/s, with 12% achieved occupancy.

SGLang's FlashMLA takes 13.9 us for this step (evidence/h200-20261115-sglang). What remains is the
separate merge launch, which reads the 8 MB f32 numerator back, and the flash kernel's chain of
dependent steps for each token.

`mla-profile.md` / `.json` come from `validation/mla_profile.py` (vxc 243ea31c built on the pod,
`~/glm-h200/vxtc-243e`). Each kernel runs at ResidentGLM's split policy for it: 16 splits of 256
tokens for the others, 64 of 8 for flash, which takes 21 to 40 us against `mla_heads_at`'s 149 to
172. The same tree passed `gpu/test_resident_graph_hardware.py` on this H200 (8 passed), including
`mla_kernel='flash'` against `'chunks'` through 600 replayed steps, and `gpu/test_spawn_vx_launch.py -k mla`.
