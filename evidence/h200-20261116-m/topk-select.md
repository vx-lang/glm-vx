# Stable top-k, k = 2048. CUDA events, L2 flushed, median of 20.

| tokens | topk_stable (radix) us | topk_sort us | topk_chunks + topk_merge us | topk_radix us | topk_select us |
|---|---|---|---|---|---|
| 2049 | 661.6 | - | 59.1 | 652.2 | 16.4 |
| 3001 | 667.1 | - | 59.0 | 621.6 | 17.4 |
| 4096 | 673.4 | - | 59.3 | 692.2 | 19.0 |
| 8192 | 701.0 | - | 65.3 | 654.3 | 26.3 |
| 32768 | 857.6 | - | 78.5 | 699.4 | 79.4 |
| 131072 | 1433.9 | - | 168.2 | 649.3 | 311.7 |
