# prefill_attention='mla_tc' against 'mha_tc' on one H200, one TP8 rank (#63, TTFT)

This ran on 2026-11-21 on a single Nebius H200 (gpu-h200-sxm, 1gpu-16vcpu-200gb; driver 580, CUDA 13.0). The code was
branch ttft-bf16-gemm, the BF16 Linears over the rows plus 'mla_tc', built with Vx main 10c6f478f. The script is
`z66.sh`.

1. `gpu/test_cublaslt_hardware.py`: 27 passed, 6 skipped. The skips are cuBLASLt heuristics with no algorithm for a
   few small m, as before.
2. `rank_profile.py` once for each option, then `--compare` (`compare.txt`).
   - **Model:** GLM-5.3's geometry cut to 4 layers (0-2 dense with full indexers, 3 MoE with a shared indexer).
     Random weights in the checkpoint's formats. Rank 0 of 8's shapes. The all-reduces are no-ops.
   - **Cases:**
     - W1: 1000 tokens at 0.
     - A 2048-token chunk at 2048 and one at 30720, the chunks before each prefilled first.
   - **Timing:** CUDA events around each layer's attention and FFN. The 78-layer estimate weights the parts by GLM-5.3's
     layer counts (3 dense, 18 MoE with a full indexer, 57 MoE with a shared one).

One run each, so this is a diagnosis, not a result.

| 78-layer GPU estimate | 'mha_tc' | 'mla_tc' |
|---|---|---|
| W1, 1000 tokens | 256 ms | 262 ms |
| chunk of 2048 at 2048 | 3302 ms | 376 ms |
| chunk of 2048 at 30720 | 3709 ms | 430 ms |

Per layer for the chunk at 30720:
- Attention: 47.7-48.9 ms with 'mha_tc' and 5.7-6.8 ms with 'mla_tc' on the full-indexer layers. On the
  shared-indexer layer it is 44.8 ms and 2.8 ms.
- FFN: unchanged, at 0.28 ms (dense) and 1.9 ms (MoE).
- Launches: the same in both, 156 for the 4-layer chunk. No part runs a row at a time.

Logits (`compare.txt`):
- **W1:** identical. Below index_topk both options run 'mha_tc'.
- **Long chunks:** they differ by 25.5% (at 2048) and 17.3% (at 30720) of the largest logit. The argmax agrees at 2048
  and differs at 30720.
- **Cause, not settled:**
  - The likely one is random weights: a small bf16 change in layer 0 moves the keys the later full indexers select.
    The CPU test with one full-indexer layer agrees within 1%.
  - A GPU-only kernel bug would look the same. So the next GPU job runs
    `gpu/test_spawn_hardware.py::test_mla_prefill_sparse_tc_matches_float64_on_bf16_values` first, then the 30K
    passkey and W3 on the real model.

The logits files (`*-logits.npy`, 3.6 MB) are not committed.
