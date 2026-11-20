| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.3 | 6.3 | 16.1 | 17.2 | 13.8 | 30.0 | 10.9 | 23.4 | 20.1 | 36.3 | 17.3 |
| 8192 | 8.2 | 7.3 | 22.9 | 20.4 | 15.8 | 30.9 | 11.7 | 31.1 | 23.2 | 38.2 | 19.0 |
| 32768 | 11.9 | 12.4 | 68.0 | 36.4 | 27.9 | 31.8 | 14.8 | 79.9 | 40.2 | 44.2 | 27.1 |
| 65536 | 19.0 | 20.8 | 144.8 | 57.3 | 41.9 | 30.4 | 20.8 | 163.8 | 62.7 | 51.3 | 41.6 |
| 131072 | 28.4 | 33.4 | 277.6 | 93.6 | 65.5 | 31.6 | - | 306.0 | 98.9 | 65.0 | - |
| 200000 | 36.7 | 45.3 | 344.4 | 131.5 | 90.7 | 31.6 | - | 381.2 | 136.0 | 76.9 | - |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-nvvm/gpu/build/spawn.
