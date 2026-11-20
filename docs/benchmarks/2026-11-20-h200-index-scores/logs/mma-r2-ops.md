| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.4 | 6.5 | 16.2 | 17.3 | 13.9 | 30.4 | 11.1 | 23.7 | 20.4 | 36.9 | 17.6 |
| 8192 | 8.2 | 7.3 | 23.0 | 20.4 | 15.8 | 31.3 | 11.7 | 31.2 | 23.1 | 38.6 | 19.0 |
| 32768 | 12.1 | 12.4 | 68.1 | 36.5 | 27.7 | 32.0 | 14.8 | 80.2 | 40.1 | 44.4 | 27.2 |
| 65536 | 19.0 | 20.7 | 144.9 | 57.3 | 42.5 | 30.7 | 20.7 | 164.0 | 63.1 | 51.4 | 41.3 |
| 131072 | 28.6 | 33.1 | 277.2 | 93.1 | 65.8 | 31.8 | - | 305.9 | 98.8 | 64.9 | - |
| 200000 | 37.0 | 45.6 | 344.9 | 131.9 | 90.8 | 32.2 | - | 381.9 | 136.5 | 77.8 | - |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-nvvm/gpu/build/spawn.
