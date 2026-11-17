# H200 session z2 (2026-11-17, aditya@204.12.188.116, Vx 6a1d8f47)

`z2.sh` on repo pod-z (topk-cluster 2f0a0dd + moe-group). Log: `z2.log`. The VM went unreachable
(preemptible) during the last router timing below.

- Build, test-spawn-hardware (245 passed, 1 xfailed) and test-resident-graph (27 passed). These include
  router_bf16 against the hand bf16_matvec then the spawn router, bit for bit at 256 x 6144 with ties
  and three launches on one arrivals buffer; fuse_router and fold_shared in the model; and the 3 tests z
  failed on the dtype.
- Top-k A/B, each kernel alone (`topk-hist-*.md`). ilp_hist is the faster one-block top-k at every
  length. The multi-block pair (topk_mb) costs about 30 us flat; it beats one block from 65536 tokens
  (50.7 against 63.9 us with the scores), not at 32768 (43.2 against 40.8). So topk_blocks_from = 65536
  stands. Vx #3588 (f0788ab86) leaves one-block top-k unchanged and makes topk_mb 2-3 us slower;
  `spawn-maxnreg: 64` changes nothing measurable.
- Step at 131072 tokens (`ab-131072.txt`), position against ilp_hist with topk_mb: 431.8 -> 248.0 us
  (topk_select_at 220.7 -> scores_hist 28.0 + topk_mb 22.3 + 5.9); with mma16 scores, 242.5.
- Router at GLM-5.3's 256 x 6144 (`router-ab.md`), each variant one CUDA graph after an L2 flush, in
  us. f32 matvec_rows + hand router 15.3; hand bf16_matvec + router 137.1; bf16_matvec4 + router 20.2
  (today's x4 path); router_bf16 (2f0a0dd) 17.5. The ids are the same in all 8 slots.
- fuse_router in the step (16 experts; `ab-fuse.txt`): bf16_matvec4 8.9 + router 4.6 -> router_bf16 7.5
  us; the step 463.5 -> 455.8.
- Pieces (`router_split.py`, us, the graph floor of 3.2 included; 256 experts / 16):

  | piece | 256 | 16 |
  |---|---|---|
  | empty copy (graph floor) | 3.3 | 3.1 |
  | hand router alone | 8.0 | 6.7 |
  | matvec_rows f32 alone | 7.6 | 5.8 |
  | bf16_matvec4 alone | 12.7 | 12.2 |
  | router_bf16 2f0a0dd, k = 0 (projection only) | 9.7 | 8.4 |
  | router_bf16 2f0a0dd, k = 8 | 17.4 | 10.2 |
  | two-level ranks, 1024 lanes, k = 0 | 9.9 | 8.2 |
  | two-level ranks, 1024 lanes, k = 8 | 16.9 | 14.0 |

  Each loop iteration of the projection's sum and of the ranks waited on its own shared-memory load,
  about 40 cycles. The next version loads 8 before using them; the VM went down before it was timed.
