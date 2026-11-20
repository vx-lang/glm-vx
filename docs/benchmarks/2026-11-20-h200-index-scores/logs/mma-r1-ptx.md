| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.1 | 6.5 | 16.2 | 17.5 | 14.0 | 30.3 | 11.3 | 23.3 | 20.6 | 36.8 | 17.8 |
| 8192 | 7.8 | 7.6 | 23.1 | 20.5 | 16.3 | 31.2 | 11.9 | 30.9 | 23.9 | 38.8 | 19.5 |
| 32768 | 11.8 | 12.8 | 68.3 | 36.9 | 28.4 | 32.3 | 15.0 | 80.0 | 41.2 | 45.1 | 27.8 |
| 65536 | 18.3 | 20.9 | 145.3 | 58.3 | 43.2 | 31.0 | 20.7 | 163.6 | 64.1 | 51.9 | 41.6 |
| 131072 | 28.0 | 33.5 | 277.4 | 95.6 | 68.0 | 31.9 | - | 305.4 | 101.6 | 65.4 | - |
| 200000 | 35.6 | 45.5 | 344.3 | 135.1 | 94.8 | 32.1 | - | 379.9 | 140.2 | 77.5 | - |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-main/gpu/build/spawn.
