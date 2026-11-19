# The multi-block top-k at 32768 tokens: no gain (#63, item 6)

The same 4 trained layers as z23, at 32768 tokens, with `--option topk_blocks_from=N`; nsys per-kernel times from
`steps-from*.txt`.

| top-k a full-indexer layer | us |
|---|---|
| `topk_blocks_from=65536` (default): `topk_select_ilp_hist_at` | 37.5-38.6 |
| `16384` or `4096`: `topk_mb_list_at` + `topk_mb_write_at` | 29.3-33.5 + 4.7 = 34-38 |

The tokens are the same and the 4-layer step is 1906 / 1881 / 1881 us. The threshold stays at 65536. A faster
top-k at 32768 needs another algorithm: SGLang's `fast_topk_v2` takes 21.4 us.

`next-8gpu-z27.sh` is the script for the next 8x H200 session (deploy14: af6d4d2 and every kernel).
