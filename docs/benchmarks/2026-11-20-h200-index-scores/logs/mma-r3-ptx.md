| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 6.9 | 6.5 | 16.1 | 17.3 | 13.9 | 30.2 | 11.1 | 23.1 | 20.4 | 36.7 | 17.6 |
| 8192 | 7.6 | 7.3 | 23.0 | 20.4 | 15.9 | 31.3 | 11.9 | 30.6 | 23.3 | 38.7 | 19.2 |
| 32768 | 11.7 | 12.5 | 68.1 | 37.0 | 28.4 | 32.0 | 15.0 | 79.8 | 40.9 | 44.6 | 27.6 |
| 65536 | 18.4 | 20.8 | 145.1 | 58.2 | 43.1 | 30.8 | 20.7 | 163.5 | 63.9 | 51.6 | 41.5 |
| 131072 | 27.3 | 33.4 | 277.0 | 95.6 | 67.9 | 31.7 | - | 304.3 | 101.2 | 65.1 | - |
| 200000 | 35.9 | 45.5 | 345.1 | 135.3 | 94.6 | 31.8 | - | 381.0 | 140.1 | 77.2 | - |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-main/gpu/build/spawn.
