# Where the step grows with context, per rank (#63, docs/TP-SERVING-PLAN.md item 6)

GLM-5.3's first 4 trained layers on one H200 (z16's `real4`): layers 0-2 dense with a full indexer, layer 3 MoE
with a shared indexer (`config-indexer.txt`). `validation/tp_real --ranks 1` with
`moe_down2`, `fuse_router` and `moe_mma`, under `nsys --cuda-graph-trace=node`. `steps-*.txt` give one decode
step's kernels, from `step_breakdown.py`.

Every TP8 rank runs the same full indexer (gpu/tp.py), so its kernels here cost what they cost each rank at TP8.
MLA here runs all 64 heads, against 8 a rank at TP8.

| a full-indexer layer, us | 1000 tokens | 32768 tokens |
|---|---|---|
| `topk_select_ilp_hist_at` (top 2048 of the scores) | not run: every token kept | **38.7** |
| `index_scores_mma_hist_at` | not run | 7.5 |
| `mla_tc_at` (64 heads) | 12.4 | 14.2 |
| `mla_flash_merge_at` | 4.5 | 6.6 |

At TP8 the 21 full-indexer layers cost about 21 x (38.7 + 7.5) = 0.97 ms a token at 32768. That is most of the
1.35 ms our TP8 step grows from 1000 to 32768 (z5); the top-k alone is 0.81 ms. SGLang's step grows 0.25 ms.

For the same k = 2048 at 32768 tokens, SGLang's `fast_topk_v2` took 21.4 us and our ILP top-k about 40 us, timed
alone (`../h200-20261117-y`).

So item 6's first lever is the top-k itself:
- A single-GPU top-k at SGLang's speed saves about 17 us x 21 = 0.36 ms at 32768, with no cross-rank traffic.
- Splitting the indexer over the 8 ranks (scores and a local top-k on an eighth of the keys, an all-gather of the
  candidates, a merge) would also take the scores' 7.5 us down to about 1 us. It adds an all-gather of 2048
  (score, index) pairs a rank and a second top-k over 16384, so it is worth it only once the top-k itself is fast.

The 4 layers' whole step (tp_real, with the logits downloaded each step) is 1667 us at 1000 tokens and 1881 us at
32768.
