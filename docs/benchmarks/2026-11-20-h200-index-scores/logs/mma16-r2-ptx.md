| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.0 | 6.4 | 16.2 | 17.4 | 13.8 | 29.8 | 11.1 | 23.2 | 20.2 | 36.2 | 17.5 |
| 8192 | 7.2 | 6.9 | 22.8 | 20.4 | 15.8 | 31.6 | 11.6 | 30.0 | 22.6 | 38.4 | 18.5 |
| 32768 | 9.6 | 11.6 | 68.3 | 37.1 | 28.5 | 32.1 | 14.7 | 77.9 | 40.1 | 43.7 | 26.4 |
| 65536 | 13.2 | 18.5 | 144.2 | 58.4 | 43.6 | 30.7 | 20.5 | 157.3 | 62.1 | 49.2 | 39.0 |
| 131072 | 21.5 | 28.9 | 277.2 | 95.4 | 68.0 | 31.7 | - | 298.8 | 96.9 | 60.6 | - |
| 200000 | 28.3 | 38.8 | 344.1 | 135.3 | 94.4 | 31.9 | - | 372.4 | 133.2 | 70.7 | - |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-main/gpu/build/spawn.
