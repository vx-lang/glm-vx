SGLang 0.5.22 sgl_kernel.fast_topk_v2 (the DSA decode top-k backend 'sgl-kernel'), batch 1, k = 2048,
normal scores, CUDA events after an L2 flush, median of 20 (sgl_topk_time.py). Every row the same set
as torch.topk.

| tokens | us |
|---|---|
| 2049 | 6.1 |
| 3001 | 10.0 |
| 4096 | 6.1 |
| 8192 | 10.6 |
| 16384 | 15.6 |
| 32768 | 21.4 |
| 65536 | 39.0 |
| 131072 | 64.7 |
| 200000 | 95.3 |
