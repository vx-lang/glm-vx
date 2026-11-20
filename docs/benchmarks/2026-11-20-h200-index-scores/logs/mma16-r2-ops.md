| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 6.9 | 6.4 | 16.2 | 17.4 | 13.9 | 29.9 | 11.3 | 23.2 | 20.3 | 36.3 | 17.7 |
| 8192 | 7.3 | 6.9 | 22.9 | 20.4 | 15.9 | 31.4 | 11.8 | 30.2 | 22.8 | 38.3 | 18.7 |
| 32768 | 9.7 | 11.6 | 68.4 | 36.7 | 27.9 | 32.1 | 15.1 | 78.0 | 39.6 | 43.7 | 26.7 |
| 65536 | 13.1 | 18.6 | 143.7 | 57.5 | 42.6 | 30.8 | 20.6 | 156.8 | 61.2 | 49.4 | 39.2 |
| 131072 | 21.5 | 28.9 | 277.2 | 93.1 | 65.9 | 31.5 | - | 298.7 | 94.8 | 60.4 | - |
| 200000 | 27.9 | 38.5 | 344.5 | 131.9 | 91.3 | 32.1 | - | 372.4 | 129.9 | 70.6 | - |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-nvvm/gpu/build/spawn.
