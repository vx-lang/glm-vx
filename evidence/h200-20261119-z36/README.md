# W3 (32768 in, 256 out) served with everything + topk_bitmap: TPOT 9.58 ms (SGLang 9.43), TTFT 114 s (#63)

z35's server plus vx-7e's `topk_bitmap` (the 32K top-k in 14.8 us, `../h200-20261119-z34`), branch `prefill` at ed22f2a.
Two prompts (`W3_PROMPTS=2`).

| W3 | median TPOT | median TTFT |
|---|---|---|
| glm-vx z8 c (before fuse_router, device_decode, moe_mma, bitmap) | 11.17 ms | 336 s |
| **glm-vx, everything** (`w3.log`) | **9.58 ms** (p99 ITL 10.09) | **114 s** |
| SGLang 0.5.22 `--tp 8` (z6) | 9.43 ms | 2.6 s |

- The long-context decode gap is down from 18% to 1.6%.
- TTFT: chunks past index_topk still attend with the CUDA-core `mha_prefill_sparse`, item 5b's next step.
- The server log has no errors; the prompt's greedy check passed.
