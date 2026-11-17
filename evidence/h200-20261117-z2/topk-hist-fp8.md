| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.8 | 7.0 | 16.2 | 17.4 | 13.8 | 31.3 | 24.0 | 20.8 | 38.3 |
| 8192 | 7.5 | 7.2 | 22.8 | 20.4 | 16.1 | 31.2 | 30.3 | 23.3 | 38.4 |
| 32768 | 9.6 | 10.9 | 68.1 | 36.8 | 28.1 | 30.5 | 77.7 | 39.1 | 41.5 |
| 65536 | 12.2 | 15.1 | 145.4 | 58.7 | 43.1 | 30.3 | 157.6 | 58.2 | 45.4 |
| 131072 | 19.7 | 23.4 | 276.8 | 95.4 | 67.9 | 31.5 | 296.5 | 91.3 | 54.9 |
| 200000 | 25.2 | 34.1 | 345.6 | 135.0 | 94.6 | 31.8 | 370.8 | 128.7 | 66.0 |

index_scores_fp8; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/repo-z2/gpu/build/spawn.
