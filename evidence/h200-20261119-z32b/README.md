# Served W1 with chunked prefill: TTFT 8.54 -> 1.02 s; GSM8K 0.935 (#63, item 5b)

The server of `../h200-20261119-z27c` (device_decode) plus chunked prefill (branch `prefill`; `--prefill-chunk 2048`):
- prefill_rows 2048, cuBLASLt rows, NCCL sums;
- the routed experts grouped (the host's grouping then; on the device since 3ca6fb8);
- prefill_attention 'mha' (the CUDA-core `mha_prefill`).
No moe_mma here.

| W1 | median TTFT | median TPOT |
|---|---|---|
| token at a time (z27c) | 8540 ms | 9.00 ms |
| **chunked prefill** | **1022 ms** | 9.02 ms |

- GSM8K, 5-shot, 200 (`gsm8k-glmvx.json`): **0.935**, no invalid answers, 323 s instead of 1353 s (median TTFT
  0.74 s). Prefill's e4m3 activations (SGLang's block-FP8 recipe) keep the accuracy.
- The prompt's greedy tokens equal tp_real's (`greedy.log`).
- z32a (`../h200-20261119-z32a/compare.txt`) has tp_real's fill rates. With attention and the experts still a token
  at a time outside a graph, prefill is 10-15 tokens/s (Python launches). With everything over the chunk it is 980
  tokens/s at 1000 and 438 at 3001.
