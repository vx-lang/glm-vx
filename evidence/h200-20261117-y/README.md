# h200-20261117-y: Vx 6a1d8f47, nn_moe_group, top-k against SGLang

Repo pod-y = topk-cluster 899aac6 + moe-group f083ffd; Vx main 6a1d8f47 (#3556, #3569, #3571), its
toolchain bundle built on the pod.

- Toolchain: every kernel builds (build-spawn 425 s); test-spawn-hardware 187 passed, 1 xfailed;
  test_resident_graph_hardware 15 passed (summary.txt). Locally, the PTX of all 87 earlier kernels is
  that of 94e2aa22 but for the compiler's path in assert messages.
- nn_moe_group (Vx #3569, gpu/spawn/moe_group): order, starts and tile_expert equal numpy's stable
  argsort in all 24 cases (moe-group.md). count 7.3 us and place 6.3 at 65536 pairs; offsets 79 (align
  64) to 274 us (align 1, skewed): one lane an expert walks counts' rows in series, and at align 1 the
  hot expert's lane writes every tile. Sent to the Vx side; #3575 changes offsets.
- Top-k, k = 2048, each launch alone (CUDA events after an L2 flush, median of 20), us:

| tokens | SGLang fast_topk_v2 | topk_select_at | topk_select_warp_at | ILP v1 (topk-ilp.md) | quads v2 (topk-ilp2.md) |
|---|---|---|---|---|---|
| 3001 | 10.0 | 16.1 | 16.8 | 13.5 | 21.0 |
| 32768 | 21.4 | 68.2 | 78.1 | 40.4 | 69.6 |
| 131072 | 64.7 | 274.8 | 302.4 | 130.4 | 238.2 |
| 200000 | 95.3 | 344.9 | 394.8 | 192.0 | 362.1 |

  SGLang (sglang-topk.md, sgl_topk_time.py) is one block a row too: the gap is passes and instructions,
  not parallelism. ILP v1 (topk_select_at with four loads in flight a lane and the count and write by
  warp with vote.ballot) is exact; by pass at 131072 tokens, round 0 33 us, rounds 1-3 about 16 each,
  count and write about 24 each. v2 (16-byte loads, an 8-way loop that Vx left rolled, about 45
  instructions a score) is slower: one block is bound by instructions a score. v3 (three passes, the
  threshold bin's keys listed in shared memory) hung (ilp-test.log): a vote.sync of all lanes inside a
  per-lane branch. Fixed after this session and checked against validation/topk_ilp_model.py, not yet
  run on a GPU.
