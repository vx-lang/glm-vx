| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.0 | 6.3 | 16.2 | 17.4 | 14.0 | 29.8 | 11.1 | 23.2 | 20.2 | 36.1 | 17.3 |
| 8192 | 7.2 | 7.2 | 22.9 | 20.5 | 15.9 | 31.7 | 12.1 | 30.1 | 23.0 | 38.8 | 19.2 |
| 32768 | 9.7 | 11.5 | 68.2 | 37.1 | 28.6 | 32.0 | 15.2 | 78.0 | 40.1 | 43.5 | 26.7 |
| 65536 | 13.1 | 18.7 | 144.2 | 58.4 | 43.6 | 30.9 | 20.7 | 157.3 | 62.3 | 49.6 | 39.5 |
| 131072 | 21.6 | 28.7 | 277.4 | 95.4 | 67.9 | 31.6 | - | 298.9 | 96.7 | 60.3 | - |
| 200000 | 27.9 | 38.7 | 344.2 | 135.3 | 94.8 | 32.1 | - | 372.0 | 133.5 | 70.9 | - |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-main/gpu/build/spawn.
