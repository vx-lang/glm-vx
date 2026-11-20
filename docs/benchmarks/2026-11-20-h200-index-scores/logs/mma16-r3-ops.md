| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 6.8 | 6.3 | 16.1 | 17.4 | 13.7 | 29.7 | 10.9 | 23.0 | 20.0 | 36.0 | 17.2 |
| 8192 | 7.2 | 6.8 | 22.9 | 20.3 | 15.7 | 31.4 | 11.5 | 30.1 | 22.6 | 38.3 | 18.4 |
| 32768 | 9.4 | 11.6 | 68.0 | 36.7 | 28.0 | 31.6 | 14.6 | 77.4 | 39.6 | 43.2 | 26.2 |
| 65536 | 13.0 | 18.3 | 144.0 | 57.3 | 42.3 | 30.8 | 20.6 | 157.0 | 60.6 | 49.1 | 38.9 |
| 131072 | 21.2 | 28.9 | 277.4 | 92.9 | 65.5 | 31.3 | - | 298.6 | 94.4 | 60.2 | - |
| 200000 | 27.9 | 38.4 | 344.3 | 131.7 | 90.4 | 31.9 | - | 372.2 | 128.9 | 70.4 | - |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-nvvm/gpu/build/spawn.
