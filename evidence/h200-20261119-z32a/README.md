# GLM-5.3 prefill on 8 GPUs, tp_real: chunked prefill's fill rate, and vx-94's chunk index scores (#63, item 5b)

`hw-index-chunk.log`, `index-chunk.json`: vx-94's index_scores_chunk_at on GPU 0, 7 passed. 2048-row chunks: 458 us at
start 6144 (263 TFLOP/s), 1943 us at start 30720 (274 TFLOP/s).

`validation/tp_real --prefill` (the prompt and the contexts through prefill_chunk), z10's options, branch `prefill`:

| prefill configuration | fill, 1000 tokens | fill, 3001 tokens | first decode step's top-5 vs a token at a time |
|---|---|---|---|
| a token at a time (`tp-real-tokens`) | 117 tok/s | 108 tok/s | |
| cuBLASLt rows, attention and experts a token at a time (eager) | 10.0 | 9.9 | same set, logits within 0.25 |
| + experts grouped (host grouping) | 15.1 | 14.7 | within 0.29 |
| + attention over the chunk (`mha`, CUDA-core) | **980** | **438** | top-3 the same, within 0.15 |

- The eager rows are Python-launch bound: every token's attention and experts are ~20 launches outside a graph.
- The decode step is unchanged in every row (9110 vs 9113 us at 1000).
- The prompt's greedy text differs after 24 tokens: prefill's e4m3 activations. GSM8K in z32b holds at 0.935.
