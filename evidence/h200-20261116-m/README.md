# H200: the indexer past index_topk (#32), Vx 4f966fd6

The same H200 VM and toolchain as h200-20261116-k (`~/glm-h200/vxtc-4f96`). All step profiles are
nsys kernel durations of one warm decode step (graph replay) at GLM-5.3's width, one dense layer,
FP8 Linears through the cvt kernels, `mla_kernel='tc'`, index_topk 2048, after CONTEXT tokens:
`step_profile.py KERNEL CONTEXT LONG_TOPK` (`h200-20261116-k/step-profile/trace_mla.py` lists the
attention block). This README covers five folders, one per pod run:

| folder | what |
|---|---|
| `h200-20261116-m` | top-k kernels alone (`validation/topk_profile.py`); steps at 3001 and 8192 tokens with `long_topk` `'merge'` and `'select'`; ncu of the indexer kernels in the step at 3001 tokens |
| `h200-20261116-n` | first `topk_select`: GPU top-k tests, steps at 1000 to 131072 tokens, ncu of `topk_select_at` |
| `h200-20261116-o`, `-p` | experiments that were not kept (below); their `step_profile.py` took a 4th argument, `index_scores`, an option since removed |
| `h200-20261116-q` | the version committed: GPU tests, `topk_profile`, steps at 1000, 3001 and 8192 tokens |

## What the old path cost (m)

Past index_topk the stable top-k was `topk_sort_at` (it writes nothing there, 1.0 us), then
`topk_chunks_at` and `topk_merge_at` rounds: 1.0 + 50.8 + 5.6 us at 3001 tokens, + 5.4 a round past
4096. `long_topk='select'` (`topk_radix_at`, nine launches) was slower in the step: 6 + 6 + 6 + 5 x 4
+ 66 us, its last round's sort alone 66.

ncu (`ncu-indexer-3001.summary.csv`): `topk_chunks_at` is 2 blocks of 1024 lanes, 611k warp
instructions and 1.1% SM throughput: one block's bitonic sort of 2048 values, 66 stages behind a
barrier each, about 145 instructions a warp a stage. `index_scores_at` is 5.2M warp instructions
(about 13 a multiply-add): a warp a token, every head's query read again for it, 32 shuffle trees.

## topk_select (n, q)

`long_topk='position'`: one launch of `topk_select_at`, the same set in position order, no sort.

| tokens | chunks + merge, alone | `topk_select`, alone | chunks + merge, in the step | `topk_select_at`, in the step |
|---|---|---|---|---|
| 2049 | 59.3 | 15.1 | | |
| 3001 | 59.1 | 15.8 | 57.4 | **13.2** |
| 4096 | 58.9 | 17.2 | | |
| 8192 | 65.9 | 22.6 | 68.5 | **19.4** |
| 32768 | 77.6 | 67.5 | 79.7 (n) | 61.9 (n) |
| 131072 | 207.1 | 272.5 | 91.2 (n) | 270.5 (n) |

Alone: CUDA events, L2 flushed, median of 20 (q/topk-profile.md). In the step: the sum of the top-k
launches, `topk_sort_at` included. The 32768- and 131072-token steps are n's, the first version (about
8% slower alone). One block does not scale: at 32768 tokens it is still ahead in the step, at
131072 the chunk sorts, 64 blocks at once, are three times faster.

The dense layer (q): context 1000 185.3 us (193.2 before; the indexer's `wq_b` and `weights_proj` no
longer run while the context fits in index_topk); 3001 tokens 270.5 -> 226.5 us; 8192 tokens 291.2 ->
240.8 us.

GPU tests (q): `test_spawn_hardware.py -k topk` 28 passed (`topk_select` up to 2^20 tokens, ties, NaN,
-0.0, +-inf, each twice), `test_spawn_vx_launch.py` 83 passed, `test_resident_graph_hardware.py` 9
passed.

## Experiments not kept (o, p)

- `topk_select2`: the threshold by a two-level walk (16 groups, then 16 bytes) instead of eight
  doubling steps a round, a two-level scan, eight histograms by lane: 20.1 against 17.4 us at 3001
  tokens (o/ab-topk.txt; its "warm" columns are Python's launch rate, not the kernel).
- `topk_select4` (32-bit keys and counts): 15.9 against 17.2 us (p/ab-topk.txt). It is the committed
  `topk_select`. A version that wrote each slot through a per-lane shared cursor does not build: the
  race proof cannot see that the lanes' slot ranges are disjoint (E6032).
- `index_scores_rows`, a lane a key row with the queries in shared memory: alone 16.6 against
  `index_scores`' 22.9 us at 3001 tokens, but in the step 13.9 against the hand kernel's 11.4 (p), and
  92.2 against 74.7 at 32768 tokens. Not kept.
