# Fused MLA decode, 64 heads, rank 512, rope 64. CUDA events, L2 flushed, median of 20.

| selected (splits) | mla_decode_at us | mla_chunks_at us | mla_heads_at us | mla_flash_at + merge us (splits) |
|---|---|---|---|---|
| 256 (1) | 406.8 | 134.1 | 149.2 | 21.3 (32) |
| 1024 (4) | 429.6 | 159.9 | 161.6 | 28.7 (64) |
| 2048 (8) | 863.5 | 307.4 | 171.9 | 39.9 (64) |
