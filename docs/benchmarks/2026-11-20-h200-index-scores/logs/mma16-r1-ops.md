| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 6.9 | 6.2 | 16.1 | 17.3 | 13.8 | 29.7 | 11.0 | 23.0 | 20.1 | 35.9 | 17.2 |
| 8192 | 7.1 | 6.7 | 22.8 | 20.2 | 15.7 | 31.3 | 11.6 | 29.9 | 22.4 | 38.0 | 18.3 |
| 32768 | 9.4 | 11.4 | 68.1 | 36.6 | 27.9 | 32.0 | 15.0 | 77.5 | 39.3 | 43.4 | 26.4 |
| 65536 | 13.0 | 18.3 | 144.0 | 57.3 | 42.4 | 30.6 | 20.5 | 157.0 | 60.7 | 48.9 | 38.9 |
| 131072 | 21.3 | 28.7 | 277.3 | 93.2 | 65.6 | 31.3 | - | 298.7 | 94.2 | 60.0 | - |
| 200000 | 27.9 | 38.8 | 344.5 | 131.8 | 90.5 | 31.9 | - | 372.4 | 129.3 | 70.7 | - |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-nvvm/gpu/build/spawn.
