| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 7.2 | 6.5 | 16.2 | 17.3 | 13.8 | 30.4 | 23.3 | 20.3 | 36.9 |
| 8192 | 7.7 | 7.5 | 22.9 | 20.4 | 15.9 | 30.5 | 30.7 | 23.4 | 38.0 |
| 32768 | 11.7 | 12.6 | 68.1 | 36.8 | 28.1 | 30.8 | 79.9 | 40.8 | 43.4 |
| 65536 | 18.0 | 20.7 | 145.2 | 58.0 | 42.9 | 29.9 | 163.2 | 63.6 | 50.6 |
| 131072 | 27.5 | 33.3 | 277.5 | 95.2 | 67.2 | 31.3 | 304.9 | 100.5 | 64.6 |
| 200000 | 35.9 | 45.7 | 344.1 | 134.2 | 93.3 | 31.6 | 380.0 | 139.1 | 77.3 |

index_scores_mma; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/u3588/spawn-6a1d-n64.
