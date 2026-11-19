| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | bitmap | before | after (one block) | after (blocks) | after (bitmap) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.2 | 6.8 | 16.2 | 17.4 | 13.9 | 30.2 | 11.2 | 23.4 | 20.7 | 37.0 | 18.0 |
| 8192 | 7.7 | 7.4 | 22.9 | 20.5 | 16.0 | 31.2 | 12.0 | 30.7 | 23.4 | 38.6 | 19.4 |
| 32768 | 11.6 | 12.9 | 68.1 | 37.0 | 28.4 | 32.0 | 14.8 | 79.7 | 41.3 | 44.8 | 27.7 |
| 65536 | 18.2 | 20.8 | 145.1 | 58.5 | 43.2 | 30.6 | 20.9 | 163.3 | 64.1 | 51.5 | 41.8 |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /home/aditya/glm-h200/repo-topk/gpu/build/spawn.
