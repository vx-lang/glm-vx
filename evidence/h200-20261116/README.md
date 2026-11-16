# MLA merge rewrite and Vx 56f57c3c, H200 (#33)

This run used the H200 at its new address, on the same VM and disk. The pod's toolchain is
~/glm-h200/vxtc-56f5, built there from Vx main 56f57c3c (with #3491 lanes::scale_rows and #3492
f32 -> bf16 as one cvt.rn.bf16.f32). Every kernel builds with it, and the resident graph, launch and
host GPU tests of the MLA, bf16 and store kernels pass (71 passed).

Timing script: `graph_split.py` (run from the repository root with `PYTHONPATH=.`, a folder of built
kernels, and the tc kernel's name). It times each MLA step as one CUDA graph with CUDA events after an
L2 flush, median of 50. An almost empty graph takes 3.1 us. Shapes are GLM-5.3's: 64 heads, rank 512,
rope 64, and a 4096-token cache.

## The whole step

The first column is mla_flash_at + mla_flash_merge_at on the f32 cache. The second is mla_q_bf16_at +
mla_tc_at + mla_flash_merge_at on the bf16 cache.

| selected tokens | flash, 2026-11-15 | flash, now | tc, 2026-11-15 | tc with #3492 | tc, now | SGLang |
|---|---|---|---|---|---|---|
| 256 | 21.0 us | 15.6 us | 22.8 us | 22.4 us | 19.0 us | |
| 1000 | 27.7 | 24.2 | 28.3 | 27.6 | 22.2 | |
| 2048 | 39.3 | 35.9 | 30.5 | 30.0 | 26.2 | 13.9 |

The max error against float64 attention is unchanged: flash 1e-7, tc 5e-4 to 1e-3 (bf16).

## What changed

**mla_flash_merge_at.** Each block now works out each split's factor exp(m_s - m), and the
denominator, once: lanes::max and lanes::sum over the splits' stats, one lane a split. Then each lane
sums factor_s * numerator_s over the step's splits only (a loop bound from the step buffer). Before,
every lane recomputed the max and the factors for 16 splits (all 64 slots branch-free), which took
4.6 M warp instructions, about 560 a warp. ncu then showed the merge issue- and barrier-bound at about
10.8 us with the cache warm, not bandwidth-bound.

Tried on the way:
- 16 column blocks a head with 8 subgroups of 8 splits: helped at 2048 (30.0 -> 28.6) but hurt at 256
  (22.4 -> 24.8).
- In mla_tc_at, copying O through shared memory so that lanes store coalesced rows, instead of
  lanes::copy from the mma registers: no better (26.1 at 2048, 2 us worse at 256 and 1000).

## Where the tc step's time goes now (2048 tokens, prep and prep + mla_tc_at as their own graphs)

| part | us |
|---|---|
| graph floor | about 3 |
| mla_q_bf16_at | about 1.6 |
| mla_tc_at | about 14.6 (11.3 at 256 tokens) |
| mla_flash_merge_at | about 7 |

## Like-for-like with SGLang (#40): nsys kernel durations inside a decode step

`step-profile/` holds both sides' traces. Both are nsys kernel durations in a real batch-1 decode
step (graph replay, warm cache, no L2 flush) at GLM-5.3's full width.

- **Ours:** `mla_step_profile.py`, one dense layer, FP8 Linears, index_topk 2048, after 1000 or
  2048 tokens; `trace_mla.py` lists the attention block's kernels.
- **SGLang 0.5.22:** `scripts/pod_sglang.sh profile` on the 9-layer FP8 proxy, `--input-len` 1000 or
  2048, `*_cuda_gpu_kern_sum.csv`. Averages over its layers: 9 attention layers and 4 full indexers.

MLA decode attention:

| context | ours `'tc'` (bf16): mla_q_bf16_at + mla_tc_at + merge | ours `'flash'` (f32) | SGLang: prepare_varlen_num_blocks + FlashAttnFwdSm90 + FlashAttnFwdCombine |
|---|---|---|---|
| 1000 | 1.9 + 12.2 + 4.4 = **18.5 us** | 14.3 + 6.5 = 20.8 | 3.3 + 14.4 + 5.8 = **23.5 us** |
| 2048 | 1.8 + 13.7 + 6.7 = **22.2 us** | 24.4 + 6.4 = 30.8 | 3.2 + 16.1 + 5.9 = **25.2 us** |

The rest of the attention block, at context 1000 (2048):

| kernel | ours | SGLang |
|---|---|---|
| absorb matvecs (`matvec_heads`, #42) | 11.5 + 11.7 | 5.9 + 6.5 |
| indexer scores (`index_scores_at` vs `sm90_fp8_paged_mqa_logits`) | 10.5 (10.9) | 3.6 (3.7) |
| indexer top-k (`topk_sort_at`; at 2048 `topk_chunks_at` + `topk_merge_at`, vs `topk_small_batch_cluster_kernel`) | 25.6 (50.7 + 5.7) | 1.7 (6.0) |
| new cache rows (`store_row(_bf16)_at` vs `set_mla_kv_buffer`) | 1.2-1.6 each | 1.2 |

The MLA attention kernels are ahead of SGLang at both contexts. The indexer's top-k is now the
largest gap in the attention block, followed by the absorb matvecs and the indexer's scores.
