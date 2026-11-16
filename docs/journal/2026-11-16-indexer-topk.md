# Journal: the indexer past index_topk (#32), 2026-11-16

After #42 and #43, the largest gaps left in the attention block were the indexer's, once the context
passes index_topk (2048): its scores, and the top-k of them. Evidence: evidence/h200-20261116-m to
-q (the README in -m covers all five). Same H200 and Vx 4f966fd6 as the absorbs.

## First, what was already there

`topk_radix`, a multi-block radix select written for #32 in October, had passed its GPU tests but was
never timed. Timed now, alone and in the step, it was slower than the default chunk sort and merge
(about 105 against 57 us in the step at 3001 tokens), its last round's one-block sort alone 66 us.

ncu of the default in the step at 3001 tokens: `topk_chunks_at` is two blocks, each sorting 2048
values through 66 bitonic stages with a barrier after each, about 145 instructions a warp a stage, at
1.1% SM throughput. Every path past 2048 tokens ended in such a sort.

## The selection in position order

The attention sums over the selected tokens; their order changes only the rounding of those sums.
Within index_topk the selection is already in position order (#43, as SGLang's). Past it, the set
alone needs no sort: a radix select finds the threshold, and a scan places each selected position.

`topk_select` (+ `_at`), one block of 1024 lanes:
1. Four rounds, one byte of the 32-bit key each, find the threshold key K and how many keys equal to K
   to take (the lowest positions, the stable order's tie rule).
2. Each lane counts the keys above and equal to K over its run of positions; a scan gives each run's
   first slot.
3. Lane r finds slot r (binary search over the scan, then a walk along the run), so the writes are
   linear in r, as the race proof needs.

It is `ResidentGLM(long_topk='position')`. It started opt-in, with the reference model still ordering
by score, so the logits matched it within float32 rounding only (the eager, captured and host-argsort
paths matched each other bit for bit). The next commit made it the default: `GlmMoeDsaModel._index`
and both expanded oracles return the set in position order, as the official traces already record it
(`np.sort` of the causal slots), and `long_topk='merge'` keeps the score order for an A/B.

| top-k in the step | chunk sort + merge | `topk_select_at` |
|---|---|---|
| 3001 tokens | 57.4 us | **13.2 us** |
| 8192 tokens | 68.5 us | **19.4 us** |
| 32768 tokens | 79.7 us | 61.9 us |
| 131072 tokens | 91.2 us | 270.5 us |

One block does not scale to the longest contexts; SGLang's takes 6.0 us at 2048. Tried on the way
(evidence -o, -p): a two-level threshold walk and scan with eight histograms (slower); 32-bit keys
and counts (8% faster, kept); slots through a per-lane cursor in shared memory (the race proof
rejects it: it cannot see the lanes' ranges are disjoint).

## Also: no indexer queries within index_topk

#43 skipped the scores and top-k while the context fits in index_topk, but the indexer's `wq_b` and
`weights_proj` still ran (5.1 + 3.6 us). Only its key row is needed there. The dense layer at context
1000 is now 185.3 us (193.2).

## The scores, not changed

`index_scores_at` is 11.4 us at 3001 tokens and 75 at 32768: a warp a token, every head's query
read again for each token, about 13 instructions a multiply-add. A version with a lane a key row and
the queries in shared memory was faster alone (16.6 against 22.9 us, L2 flushed) but slower in the
step (13.9 against 11.4), where the keys are warm. SGLang's takes 3.6 us with FP8 keys on the tensor
cores; doing the same changes the scores' precision, which is a separate decision.

## Glue fusion (#39), and the A/B

With the top-k down, about 30 us a layer went to the small launches between the big kernels: norms,
the cache rows' copies, RoPE and stores, the residual add. `ResidentGLM(fuse_glue=True)` fuses them,
each fusion with the bits of the launches it replaces, so the flag changes the time and nothing else
(the CPU tests check the logits on and off for equality; the H200 tests check each kernel against the
launches it replaces). Both changes are flags on one build, so one H200 session compared them
(evidence/h200-20261116-r):

| step (us) | `merge` | `position` | + `fuse_glue` |
|---|---|---|---|
| 1 dense layer, context 1000 | 184.9 | 186.1 | 175.4 |
| 1 dense layer, 3001 tokens | 269.6 | 224.3 | 215.1 |
| 1 dense layer, 8192 tokens | 291.1 | 243.1 | 231.4 |
| dense + MoE layer, context 1000 | | 425.9 | 400.1 |

One fusion first gained nothing: the residual add with RMSNorm in registers took 5.1 us where plain
RMSNorm takes 3.3, so as a layer boundary's add (1.8) + RMSNorm (2.7) it broke even. Its weight loads
came after the block sum, a memory round trip in series. Loading them before the sum, as the values
are, took it to 3.1 us, the same bits.

One Vx bug on the way: an `eps <= 0.0` guard with two stores a thread made the prover hand z3 a real
comparison in an integer logic (vx-lang/Vx#3145, fixed by vx-lang/Vx#3517); `index_key_at` has no
guard until the tested Vx has the fix.
