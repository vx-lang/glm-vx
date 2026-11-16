# MLA decode on the GPU

How `ResidentGLM` runs GLM-5.3's MLA attention during decode. This covers the kernels, their data
layouts, how to test and time them, and where they stand against SGLang. Tracking issue: #33.
Follow-ups: #40 (like-for-like profile), #41 (`mla_tc_at`'s critical path), #42 (the absorb
matvecs) and #43 (skipping top-k for short contexts). The work that led here is in
[journal/2026-11-15-mla-flash-and-tensor-cores.md](journal/2026-11-15-mla-flash-and-tensor-cores.md).

## The step

For each layer and decode step, after the projections:

1. The new token's latent row (rank 512, RMSNorm'd) and rope key (64) are written into the layer's
   caches.
2. The DSA indexer selects up to `index_topk` (2048) cached tokens, the step's `selected` ids.
   While the context fits in `index_topk`, every cached token is selected. They are taken in position
   order, as SGLang's top-k does (its `trivial_transform`), and the indexer only stores its key row:
   no scores or top-k run (#43), and its query projections (`wq_b`, `weights_proj`) do not run either.
   Past index_topk the stable top-k set is selected, also in position order, with one
   `topk_select_at` launch (#32, `long_topk='position'`, the default); `long_topk='merge'` gives it
   in score order, as `torch.topk` does, for an A/B. `GlmMoeDsaModel._index` and the expanded oracles
   take the same position order, so ResidentGLM still matches the reference bit for bit. The set of
   tokens is the official one; only the order of the attention sums differs from `torch.topk`'s.
3. Each head's query is absorbed into the latent space: a 512-wide latent query and a 64-wide rope
   query, 64 heads. With `fp8_kernels='cvt'` and an FP8 `kv_b_proj`, this and step 5's value absorb read
   its FP8 codes as stored (#42): `fp8_matvec_cols_cvt` sums each head's 192 nope rows down the 512
   latent columns, and `fp8_matvec_heads_cvt` takes each head's 256 value rows. A head is 448 rows, so a
   128-row block of scales can hold two heads; both kernels read each row's own scale. Otherwise
   `matvec_heads` on a float32 copy.
4. **MLA decode.** For each head:
   - score = (latent query · latent row + rope query · rope row) × scale, for each selected token;
   - then softmax over the scores;
   - then the weighted sum of the latent rows, one 512-wide vector per head.
5. The value projection maps that vector back out (`matvec_heads`), then `o_proj`.

Step 4 is what the kernels below do. The work is split over *splits* (groups of selected tokens),
each with its own max m, sum of weights l, and weighted-sum numerator. A merge kernel combines them:
out = sum over splits of exp(m_s - m) · numerator_s, divided by sum of exp(m_s - m) · l_s.

## `mla_kernel` options

`ResidentGLM(..., graph=True, fused_attention=True, mla_kernel=...)`. Every option except `'decode'`
needs graph replay: those kernels read the step buffer `[position, tokens, count, splits]` on the
device.

| option | kernels | cache | splits | notes |
|---|---|---|---|---|
| `'decode'` | `mla_decode_at` (with fused_attention) or `mla_partial_at` + `mla_merge_at` | f32 | up to 16 of 256 tokens | the original; one token at a time a block |
| `'chunks'` | `mla_chunks_at` | f32 | up to 16 of 256 | 32 tokens a chunk |
| `'heads'` | `mla_heads_at` | f32 | up to 16 of 256 | four heads a block, each row read once for them; equal to `'chunks'` bit for bit |
| `'flash'` | `mla_flash_at` + `mla_flash_merge_at` | f32 | up to 64 of about 8 tokens | flash decoding with each lane's state in registers |
| `'tc'` | `mla_q_bf16_at` + `mla_tc_at` + `mla_flash_merge_at`, `store_row_bf16_at` | **bf16** | up to 64 of 32 | tensor cores (mma.sync); index_topk <= 2048 |

`'flash'` and `'tc'` need GLM-5.3's MLA shape: heads a multiple of 16, rank 512, rope 64. The
constructor refuses other shapes, and `'tc'` also refuses an index_topk above 2048.

### Accuracy

- `'flash'` agrees with `'chunks'` to 4.7e-7 in the logits (relative to the largest), over 600 replayed
  steps on a 16-head, rank-512 model. Its scores are `mla_chunks_at`'s, and only the softmax folds
  one token at a time.
- `'tc'` uses bf16 for the cache, the queries and P, with f32 sums, as SGLang's FlashMLA does. Its
  logits agree with `'flash'` to 6.7e-3 relative, with the same argmax at every step. Against
  float64 attention on the same bf16 values, the attention output is within 2e-4 to 1e-3.

Both are checked on the GPU by `gpu/test_resident_graph_hardware.py`
(`test_mla_flash_matches_mla_chunks_through_its_splits`,
`test_mla_tc_matches_mla_flash_to_bf16_rounding`). That shape is too large for the CPU emulator, so
`gpu/test_resident_graph.py` only checks that other shapes are refused.

## The kernels

### `mla_flash_at` (`gpu/spawn/mla_flash_at.vx`)

Flash decoding on the f32 cache.

- **Work split:** block (group of 16 heads, split), eight subgroups of two heads each. Split s takes
  the selected tokens s, s + splits, s + 2·splits, and so on.
- **Lane state:** each lane holds its elements `e * 32 + lane` of its two heads in registers for the
  whole token loop: the queries (16 latent + 2 rope each), the sums of weighted latent rows (16
  each), and the online softmax's max and sum. These tiles are written in shared memory, but every
  access is at `[w][..][lane]`, so Vx keeps them in registers (Vx #3469, PR #3475).
- **Per token:** the lane dots its 18 elements of the row with both queries, then `lanes::sum` gives
  each score. The softmax folds the score in, and a head's sums are rescaled only when its max rises.
- **Prefetch:** the next token's row and the id two tokens on are loaded while the current token is
  scored, so the loop does not wait on two dependent global loads for every token.
- **Registers:** `.maxnreg 128` (`// spawn-maxnreg:`), so two 256-thread blocks fit on an SM. Without
  it, ptxas takes 139 registers.

### `mla_tc_at` (`gpu/spawn/mla_tc_at.vx`)

Tensor cores on the bf16 cache.

- **Work split:** block b = group × splits_max + split takes 16 heads and one chunk of 32 selected
  tokens. GLM-5.3 selects at most 2048 tokens, so 64 splits always suffice, and a block never needs
  to rescale. It has eight subgroups.
- **Phase 1, load.** Each subgroup copies its query tiles with `lanes::copy_async`, from
  `mla_q_bf16_at`'s layout. It gathers latent columns 64·w .. 64·w + 63 of the chunk's tokens with
  `lanes::copy_rows_async` (Vx #3490: cp.async, 16 bytes a lane, rows picked by an i32 id tile, zero
  rows for ids out of range). Subgroups 0 and 1 also gather the rope rows.
- **Phase 2, scores.** S[token][head] = row · query, as subgroup tile matmuls (`acc + (a as f32) @
  (b as f32)` on bf16 shared-memory tiles, which Vx lowers to ldmatrix + mma.sync m16n8k16). Each
  subgroup takes a [16 tokens × 8 heads] piece over half of the latent columns. Subgroups 0-3 also
  add the rope part.
- **Phase 3, softmax.** Two heads a subgroup, one token a lane: the halves are added, then
  `lanes::max`, `exp`, and `lanes::sum`. P is written in bf16 for the next product.
- **Phase 4, O = P @ latent rows.** Each subgroup computes 64 latent columns with mma.sync, and the
  accumulator goes straight from the mma registers to `numerator` with `lanes::copy`.
- **Resources:** `.maxnreg 128`, 115 registers, 68 KB of dynamic shared memory, two blocks an SM.

Layouts: `numerator` is [heads/16 × splits_max × 16, 512] and `stats` is [heads/16 × splits_max, 32].
These are views of `mla_flash_merge_at`'s [heads/16, splits_max, 16 × 512] and [.., 32].

### `mla_q_bf16_at`

Converts the step's queries to bf16 once, in `mla_tc_at`'s tile layout:
`q_t[g][(c·2 + hh)·128 + k][h]` is head 16·g + 8·hh + h's latent query at 128·c + k, and `qr_t` holds
the rope queries. It runs on 64 blocks, one per head (group, head half, head), one lane a column.

### `store_row_bf16_at`

Writes the new cache row in bf16 at the step buffer's position. With `'tc'`, the latent and rope
caches are uint16 tensors that hold bf16 bits.

With `ResidentGLM(fuse_glue=True)` (#39) the latent row and the rope key go to the caches in one
`kv_rows_bf16_at` launch (store, RoPE, store, with their bits), the latent's RMSNorm runs with q_a's
in one `rmsnorm_pair`, and the indexer's key row is one `index_key_at` (LayerNorm, RoPE, store).

### `mla_flash_merge_at`

Merges the splits of `'flash'` and `'tc'`. Block (head, 128 columns):

1. one lane per split reads m and l, and `lanes::max` gives the head's max;
2. the factors exp(m_s - m) go to shared memory, and `lanes::sum` gives the denominator;
3. each lane sums factor · numerator over the step's splits only (the loop bound comes from the step
   buffer).

Working the factors out once a block, instead of in every lane for all 64 split slots, took it from
about 10.8 us (4.6 M warp instructions) to about 3-7 us in the step.

## Speed (H200, the step as a CUDA graph)

From `evidence/h200-20261116/graph_split.py`: one CUDA graph a step, CUDA events after an L2 flush,
median of 50. An almost empty graph takes 3.1 us.

| selected tokens | `'heads'` (event-timed) | `'flash'` | `'tc'` |
|---|---|---|---|
| 256 | 149 | 15.6 | 19.0 |
| 1000 | 162 | 24.2 | 22.2 |
| 2048 | 172 | 35.9 | 26.2 |

These include the graph floor and a cold L2.

### Against SGLang, like for like (#40)

Both sides are nsys kernel durations inside a real batch-1 decode step (warm cache), from
`evidence/h200-20261116/step-profile`. SGLang's attention is three kernels: `prepare_varlen_num_blocks`
(split scheduling), `FlashAttnFwdSm90` and `FlashAttnFwdCombine`. The 13.9 us quoted before was the
middle one alone.

| context | ours `'tc'` (bf16) | ours `'flash'` (f32) | SGLang |
|---|---|---|---|
| 1000 | 1.9 + 12.2 + 4.4 = **18.5 us** | 14.3 + 6.5 = 20.8 us | 3.3 + 14.4 + 5.8 = **23.5 us** |
| 2048 | 1.8 + 13.7 + 6.7 = **22.2 us** | 24.4 + 6.4 = 30.8 us | 3.2 + 16.1 + 5.9 = **25.2 us** |

The rest of the attention block, at context 1000 (in brackets: past index_topk, our 3001 tokens,
SGLang's 2048), from evidence/h200-20261116-k (2026-11-16: after #42 and #43). Before them the absorbs
were 11.5 + 11.7 us (float32) and context 1000 ran scores (10.5 us) and top-k (25.6 us):

| kernel | ours | SGLang |
|---|---|---|
| absorb matvecs (#42: `fp8_matvec_cols_cvt` + `fp8_matvec_heads_cvt`) | **5.4 + 6.3 us** | 5.9 + 6.5 us |
| indexer scores | **none** (#43) (11.4 us) | 3.6 (3.7) us |
| indexer top-k (#32) | **none** (#43) (0.9 + 51.0 + 5.8 us; `long_topk='position'`: **13.2 us**) | 1.7 (6.0) us |

With `long_topk='position'` (now the default) the dense layer at 3001 tokens is 226.5 us (270.5 with
`'merge'`), at 8192 tokens 240.8 (291.2); at context 1000 it is 185.3 us, without the indexer's query
projections (#32, evidence/h200-20261116-m to -q). `fuse_glue` (#39, evidence/h200-20261116-r) takes
about 10 us more off a layer, the same bits: 186.1 -> 175.4 us at context 1000, 224.3 -> 215.1 at 3001
tokens, 243.1 -> 231.4 at 8192, and 425.9 -> 400.1 for a dense layer and a MoE layer together, where
a layer boundary's add and next RMSNorm are one launch and the MoE output is not copied. At 131072 tokens the one-block `topk_select_at` takes 270 us against
the chunk sort and merge's 91: long contexts need it over more blocks. The scores, 11.4 us at 3001
tokens and 75 at 32768, are the other gap: a warp a token, about 13 instructions a multiply-add.

Where `'tc'`'s 26.2 us goes at 2048 tokens:

| part | time |
|---|---|
| graph floor | about 3 us |
| `mla_q_bf16_at` | about 1.6 us |
| `mla_tc_at` | about 14.6 us |
| merge | about 7 us |

Inside `mla_tc_at`, per phase (evidence/h200-20261115-j):

| phase | time |
|---|---|
| gather (ids, then rows) | about 4 us |
| scores | about 1.4 us |
| softmax | about 0.8 us |
| P @ rows and the store | about 7 us |

## Testing and timing

- **CPU** (the kernels' own CPU libraries, built by `gpu/spawn/build.sh` with `VX_DISPATCH_LIB`):
  `gpu/test_spawn_vx_host.py -k "mla"`. It checks against `mla_chunks_at` and float64 attention, with
  bf16-rounded inputs for `'tc'`.
- **GPU:** `gpu/test_spawn_vx_launch.py -k mla` (Vx's launch against ours), and
  `gpu/test_resident_graph_hardware.py`.
- **Timing:** `validation/mla_profile.py` (each kernel at ResidentGLM's split policy).
  - `evidence/h200-20261116/graph_split.py`: whole steps as graphs, with prep and prep + kernel apart.
  - `evidence/h200-20261115-j/phase_time.py`: the kernel cut after each phase.
- **Timing traps:**
  - With CUDA events around three or more eager launches, Python's launch time leaks into the
    measurement (we saw 160-300 us); time a captured graph instead.
  - ncu needs `--graph-profiling node` for kernels inside a graph, and `--cache-control none` to
    see the warm-cache behaviour.

## Toolchain

`mla_flash_at` needs Vx #3475, and `mla_tc_at` needs #3490 (`lanes::copy_rows_async`). The repo is
tested with Vx main 56f57c3c, which also has #3491 (`lanes::scale_rows`) and #3492 (`as bf16` as
one cvt.rn.bf16.f32). Pods need a toolchain bundle built on the pod (glibc 2.39), with
`scripts/pod_build_bundle.sh` from `~/pod-toolchain/vx-src-56f57c3c.tar.gz`.
