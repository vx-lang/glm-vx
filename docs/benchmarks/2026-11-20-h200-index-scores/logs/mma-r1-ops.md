| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.2 | 6.4 | 16.1 | 17.2 | 13.7 | 30.2 | 11.2 | 23.4 | 20.2 | 36.6 | 17.7 |
| 8192 | 8.1 | 7.3 | 22.8 | 20.3 | 15.8 | 30.7 | 11.6 | 30.8 | 23.1 | 38.0 | 18.9 |
| 32768 | 12.0 | 12.5 | 68.0 | 36.5 | 27.6 | 31.7 | 14.6 | 80.0 | 40.1 | 44.2 | 27.1 |
| 65536 | 18.8 | 20.5 | 145.2 | 57.5 | 42.1 | 30.4 | 20.4 | 164.0 | 62.6 | 50.9 | 40.9 |
| 131072 | 28.4 | 32.8 | 277.2 | 93.1 | 65.7 | 31.6 | - | 305.5 | 98.5 | 64.4 | - |
| 200000 | 37.2 | 45.1 | 344.5 | 131.4 | 90.7 | 31.7 | - | 381.7 | 135.8 | 76.9 | - |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-nvvm/gpu/build/spawn.
