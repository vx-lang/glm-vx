| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 6.9 | 6.4 | 16.1 | 17.2 | 13.7 | 30.1 | 11.1 | 23.0 | 20.0 | 36.5 | 17.4 |
| 8192 | 7.6 | 7.3 | 22.9 | 20.3 | 15.8 | 31.3 | 11.8 | 30.5 | 23.1 | 38.6 | 19.1 |
| 32768 | 11.5 | 12.5 | 68.0 | 36.9 | 28.2 | 31.8 | 14.9 | 79.6 | 40.7 | 44.4 | 27.4 |
| 65536 | 18.4 | 20.8 | 145.2 | 58.2 | 42.9 | 30.6 | 20.7 | 163.6 | 63.7 | 51.4 | 41.5 |
| 131072 | 27.2 | 33.6 | 277.5 | 95.5 | 67.7 | 31.5 | - | 304.7 | 101.4 | 65.1 | - |
| 200000 | 35.6 | 45.2 | 344.3 | 135.4 | 94.1 | 31.8 | - | 379.9 | 139.4 | 77.0 | - |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-main/gpu/build/spawn.
