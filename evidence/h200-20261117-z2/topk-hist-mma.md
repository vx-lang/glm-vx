| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.0 | 6.4 | 16.1 | 17.3 | 13.8 | 30.3 | 23.1 | 20.2 | 36.7 |
| 8192 | 7.6 | 7.3 | 22.8 | 20.3 | 15.9 | 30.6 | 30.5 | 23.2 | 37.9 |
| 32768 | 11.5 | 12.7 | 68.3 | 36.9 | 28.1 | 30.5 | 79.9 | 40.8 | 43.2 |
| 65536 | 18.3 | 20.8 | 144.9 | 58.3 | 43.1 | 29.9 | 163.1 | 63.9 | 50.7 |
| 131072 | 27.1 | 33.2 | 277.4 | 95.7 | 67.9 | 31.5 | 304.4 | 101.1 | 64.7 |
| 200000 | 35.5 | 45.3 | 344.8 | 135.0 | 94.3 | 31.8 | 380.4 | 139.6 | 77.1 |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/repo-z2/gpu/build/spawn.
