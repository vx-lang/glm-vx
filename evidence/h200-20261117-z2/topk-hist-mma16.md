| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) | after (blocks) |
|---|---|---|---|---|---|---|---|---|---|
| 3001 | 6.8 | 6.3 | 16.1 | 17.4 | 13.7 | 30.2 | 23.0 | 20.1 | 36.5 |
| 8192 | 7.2 | 6.9 | 22.8 | 20.4 | 15.8 | 30.7 | 30.0 | 22.6 | 37.6 |
| 32768 | 9.5 | 11.9 | 68.1 | 37.0 | 28.1 | 30.3 | 77.6 | 40.0 | 42.2 |
| 65536 | 13.1 | 18.3 | 143.9 | 58.5 | 43.2 | 30.0 | 157.0 | 61.6 | 48.4 |
| 131072 | 21.5 | 28.7 | 277.2 | 95.3 | 67.9 | 31.2 | 298.7 | 96.6 | 59.9 |
| 200000 | 27.7 | 38.8 | 344.5 | 135.3 | 94.0 | 31.8 | 372.3 | 132.8 | 70.6 |

index_scores_mma16; scores_hist and ilp_hist less a 4096-word copy timed alone; every output checked; median of 20; PTX from /mnt/disk-a4/glm-h200/repo-z2/gpu/build/spawn.
