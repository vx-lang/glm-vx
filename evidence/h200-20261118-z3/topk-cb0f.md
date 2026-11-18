| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.1 | 6.5 | 16.2 | 17.2 | 14.0 | 32.7 | 23.3 | 20.6 | 39.3 |
| 8192 | 7.7 | 7.4 | 23.0 | 20.5 | 15.9 | 32.9 | 30.7 | 23.3 | 40.4 |
| 32768 | 11.5 | 12.6 | 68.5 | 37.1 | 28.4 | 32.8 | 80.0 | 41.0 | 45.4 |
| 65536 | 18.3 | 20.8 | 145.8 | 58.7 | 43.3 | 32.5 | 164.0 | 64.1 | 53.3 |
| 131072 | 27.3 | 33.4 | 279.0 | 96.3 | 68.1 | 33.5 | 306.3 | 101.6 | 67.0 |
| 200000 | 35.1 | 45.0 | 346.8 | 136.1 | 94.7 | 33.9 | 381.8 | 139.7 | 78.9 |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/scb0f.
