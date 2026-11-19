| start | queries | keys | mha_prefill | mha_prefill_tc | speedup | tc TFLOP/s |
|---|---|---|---|---|---|---|
| 0 | 512 | 512 | 1277.8 | 164.7 | 7.8x | 7 |
| 0 | 1024 | 1024 | 4725.9 | 312.6 | 15.1x | 14 |
| 0 | 2048 | 2048 | 18435.4 | 618.1 | 29.8x | 28 |
| 1024 | 1024 | 2048 | 13954.7 | 602.2 | 23.2x | 21 |
| 2000 | 48 | 2048 | 947.0 | 654.8 | 1.4x | 1 |

Microseconds, median of 20; mha_prefill_tc checked against float64; PTX from /home/aditya/glm-h200/repo/gpu/build/spawn.
