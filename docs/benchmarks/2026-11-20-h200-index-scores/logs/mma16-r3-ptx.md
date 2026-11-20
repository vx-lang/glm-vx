| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.0 | 6.1 | 16.2 | 17.4 | 13.9 | 29.9 | 11.5 | 23.2 | 20.0 | 36.1 | 17.6 |
| 8192 | 7.2 | 6.8 | 22.9 | 20.5 | 15.9 | 31.5 | 11.8 | 30.1 | 22.7 | 38.4 | 18.6 |
| 32768 | 9.6 | 11.5 | 68.2 | 37.2 | 28.6 | 32.2 | 15.1 | 77.8 | 40.1 | 43.7 | 26.6 |
| 65536 | 13.2 | 18.4 | 143.9 | 58.5 | 43.7 | 30.8 | 20.6 | 157.1 | 62.1 | 49.1 | 39.0 |
| 131072 | 21.7 | 28.9 | 277.6 | 95.4 | 68.0 | 31.6 | - | 299.3 | 96.8 | 60.4 | - |
| 200000 | 27.9 | 38.6 | 344.7 | 135.2 | 94.5 | 32.2 | - | 372.6 | 133.1 | 70.8 | - |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-main/gpu/build/spawn.
