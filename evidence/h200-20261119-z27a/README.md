# The fused all-reduce + RMSNorm on 8 ranks (#63, item 3): the same bits, slower

8x H200 preemptible, Nebius eu-west1 (`beige-platypus-instance-3`, driver 580.173.02, NVLink NV18 all to all). The
code is branch `tp` with deploy15's kernels (Vx 3609 build). `validation/tp_all_reduce_check --ranks 8` (`z27.sh`
step a), one process a GPU.

| kernel | bits on every rank | us a call |
|---|---|---|
| `tp_all_reduce` (one-shot pull) | pass | 5.07 |
| `tp_sum_rmsnorm`, 8 blocks | pass | 9.96 |
| `tp_sum_rmsnorm`, 16 blocks | pass | 10.38 |
| `tp_sum_rmsnorm`, 32 blocks | pass | 10.61 |

The fused kernel replaces `tp_all_reduce` (5.07 us) and `residual_rmsnorm8` (2.9 us in z5's TP8 profile), about
8.0 us together. At 9.96 us it is about 2 us a call slower, so pushing each value once across NVLink does not
pay for the norm's last-block tail. In the step this is +376 us at 1000 tokens (`../h200-20261119-z27b`), so
`fuse_tp_norm` stays off.
