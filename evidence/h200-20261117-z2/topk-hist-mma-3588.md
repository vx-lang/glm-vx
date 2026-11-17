| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.2 | 6.5 | 16.3 | 17.3 | 14.0 | 33.1 | 23.4 | 20.5 | 39.6 |
| 8192 | 7.6 | 7.4 | 22.9 | 20.4 | 16.0 | 33.0 | 30.6 | 23.3 | 40.4 |
| 32768 | 11.8 | 12.5 | 68.1 | 36.9 | 28.5 | 33.1 | 79.9 | 41.0 | 45.6 |
| 65536 | 18.2 | 20.8 | 145.2 | 58.5 | 43.1 | 32.4 | 163.4 | 63.9 | 53.2 |
| 131072 | 27.2 | 33.4 | 277.3 | 95.7 | 68.0 | 33.7 | 304.5 | 101.4 | 67.1 |
| 200000 | 35.9 | 46.0 | 344.7 | 135.3 | 94.3 | 34.0 | 380.6 | 140.3 | 79.9 |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/u3588/spawn-3588.
