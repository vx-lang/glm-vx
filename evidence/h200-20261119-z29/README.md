# `fuse_mla_q`: the query's rope inside its bf16 kernel, one launch fewer, the same bits (#63, item 7)

`gpu/spawn/mla_q_rope_bf16_at` is `mla_q_bf16_at` that rotates the queries' rope parts itself: `rope_strided`'s
arithmetic, read from the query, with no float32 `rot_q` in between. `ResidentGLM(fuse_mla_q=True)` uses it on the
`mla_kernel='tc'` path in a graph (rank 512, rope 64).

**Bits.**
- `gpu/test_resident_graph_hardware.py::test_fuse_mla_q_gives_the_same_bits_on_the_gpu` (16 and 32 heads, FP8
  absorb, past `index_topk`) passes, with fewer `rope_strided` launches.
- The emulator gives the same bits.
- On GLM-5.3's first 4 trained layers (`tp_real --ranks 1`), the 32 tokens and every step's top-5 logits are
  equal (`tp-real-qrope.json` against `tp-real-plain.json`).

**Time** (nsys, a decode step at 1000 tokens, `steps-*.txt`):

| | the query's bf16 tiles |
|---|---|
| before | `rope_strided` 1.3 us + `mla_q_bf16_at` 1.8 us |
| `fuse_mla_q` | `mla_q_rope_bf16_at` 2.0 us |

About −1.1 us a layer, or −0.09 ms a token over TP8's 78 layers.

## Shelved: the absorb, rope and bf16 in one launch

`shelved-mla_q_absorb_at.vx` (not in `gpu/spawn`) also put `fp8_matvec_cols_cvt`'s absorb in the same launch:
- its blocks were cols_cvt's own, writing bf16 straight into `mla_tc_at`'s tiles;
- (gq, c) was merged into one block axis so that the race proof sees linear write indices;
- `build.py` had to take rank-4/5 tensor parameters.

The bits were the same on the emulator and the GPU, but it was slower: 11.3-14.8 us against 8.4 for the three
kernels (`ncu-*.csv`).
- Unbudgeted, ptxas gave it 104 registers against cols_cvt's 32 for the same loop: one block an SM.
- At a budget of 32 it spilled 168 bytes and ran 3.25 M instructions against 2.12 M.

Not pursued; the absorb stays its own launch.
