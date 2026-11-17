| experts | f32 rows + router | bf16 + router | bf16_matvec4 + router | router_bf16 | ids differ |
|---|---|---|---|---|---|
| 256 | 15.3 | 137.1 | 20.2 | 17.5 | 0 |
| 16 | 10.7 | 148.9 | 17.8 | 10.4 | 0 |

6144 columns, k = 8; each variant one CUDA graph, timed after an L2 flush, median of 50, microseconds; ids differ: slots where router_bf16 and the hand router disagree.
