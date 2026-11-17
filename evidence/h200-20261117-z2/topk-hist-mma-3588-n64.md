| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.0 | 6.3 | 16.0 | 17.2 | 13.7 | 32.0 | 23.0 | 20.0 | 38.4 |
| 8192 | 7.7 | 7.3 | 22.9 | 20.3 | 16.0 | 32.0 | 30.6 | 23.3 | 39.3 |
| 32768 | 11.4 | 12.7 | 68.1 | 36.9 | 28.1 | 32.3 | 79.6 | 40.8 | 45.0 |
| 65536 | 18.5 | 20.9 | 145.0 | 58.2 | 42.8 | 31.7 | 163.6 | 63.7 | 52.6 |
| 131072 | 27.2 | 33.5 | 277.0 | 95.1 | 67.2 | 32.7 | 304.2 | 100.7 | 66.2 |
| 200000 | 35.8 | 45.2 | 343.7 | 134.3 | 93.3 | 32.7 | 379.5 | 138.4 | 77.9 |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/u3588/spawn-3588-n64.
