# H200 session z3 (2026-11-18, aditya@204.12.188.116): Vx #3609, and glm-vx against SGLang at batch 1

`z3.sh` on repo pod-z3 (topk-cluster e86dc5a + moe-group; kernels from z2's Vx 6a1d build). Log: `z3.log`.

## A. Vx PR #3609 (the arrivals' count ordered for the device)

One set of glm-vx sources built with Vx main cb0f0428e and with 88098c90d (= cb0f0428e + the fix). Their PTX
differs in the 9 kernels with a last-block count only, and router_bf16's by the two `fence.acq_rel.sys` lines.
Each variant is one CUDA graph after an L2 flush, median of 50, us (`router-3609.log`, `topk-*.md`):

| | cb0f0428e | #3609 |
|---|---|---|
| router_bf16, k = 0, 256 blocks | 9.8 | 8.3 |
| router_bf16, k = 8, 256 blocks | 15.4 | 13.9 |
| router_bf16, k = 0, 16 blocks | 7.8 | 6.8 |
| router_bf16, k = 8, 16 blocks | 11.3 | 10.3 |
| topk_mb_list_at + topk_mb_write_at, 3001 to 200000 tokens | 32.5 - 33.9 | 31.0 - 32.3 |

Kernels without the count are unchanged within noise. Posted by vx-94 on the PR.

## B. glm-vx against SGLang 0.5.22, batch 1, the same GPU and day

One decode step of the 9-layer FP8 proxy (3 dense + 6 MoE, 256 experts top 8; the full indexer on 4 layers, 5
share): GPU kernel time inside the CUDA graph, lm_head included. `compare.py` prints it from the traces:

| context | dense layer | MoE layer | head | glm-vx, 9-layer step | SGLang | glm-vx / SGLang |
|---|---|---|---|---|---|---|
| 1000 | 174.3 | 224.0 | 437.1 | 2288 us | 2848 us | 0.80 |
| 3001 | 207.1 | 258.4 | 436.1 | 2437 | 2878 | 0.85 |
| 32768 | 229.2 | 280.5 | 434.8 | 2527 | 2858 | 0.88 |

- Ours: `step_profile` with production options (graph, cvt, x4, tc MLA, ilp_hist top-k, fold_shared; the router
  is bf16_matvec4 + the hand router). From d1, d1-logits and m2 at each context; the indexer is taken off the 5
  shared layers, and 2 us added a MoE layer for the router at 256 experts (the profile has 16; see compare.py).
  The head is the final norm, the bf16 lm_head (`bf16_matvec4`) and the 16 us logits download (SGLang samples on
  the GPU).
- SGLang: `scripts/pod_sglang.sh profile` (one_batch, `--input-len` 1000 / 3001 / 32768, its last decode step
  under nsys, 278 launches). At 32768 its one-shot prefill ran out of memory at MEM_FRACTION 0.85; rerun at 0.7
  (`sglang-32768.log`).

Where SGLang spends what we don't: act_and_mul about 19 us a layer (fused into our gate_up), moe_sum_reduce about
16 us a MoE layer (our down sums), per-token FP8 activation quantization about 65 us a step (our activations stay
float32), FlashAttention + combine above our MLA. Level: the lm_head (~420 us of kernels each side), the routed
experts (gate_up + down about 112 us a MoE layer each). Ours grows with context (the indexer on full layers: 34
us at 3001, 56 at 32768; MLA), SGLang's step barely does: the lead goes from 20% at 1000 tokens to 12% at 32768.
