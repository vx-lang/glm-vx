# GLM-5.3 on 8 GPUs: moe_mma -3.3% a step with the same tokens; the fusions keep the bits but cost +4% (#63)

`validation/tp_real` on the whole FP8 checkpoint, 8 ranks, one-shot sums, z10's options (`moe_down2`,
`fuse_router`), then with `moe_mma`, with `fuse_tp_norm=8` + `fuse_mla_q`, and with all three (`z27.sh` step b, the
same VM as `../h200-20261119-z27a`). Each: 32 greedy tokens after tp_real's prompt with their top-5, and the step at
1000 and 3001 tokens of context (the median of 48 steps, the step's logits downloaded). `tp-real-compare.txt`
has the summary.

| options | tokens vs plain | max top-5 logit diff | step at 1000 | step at 3001 |
|---|---|---|---|---|
| plain | | | 9088 us | 10022 us |
| + `moe_mma` | all 32 equal | 0.159 (some top-5 ids swap) | **8792 us (-3.3%)** | **9712 us (-3.1%)** |
| + `fuse_tp_norm=8`, `fuse_mla_q` | all 32 equal | 0.0 (bit for bit) | 9465 us (+4.1%) | 10409 us |
| + all three | all 32 equal | 0.159 | 9132 us | 10056 us |

- `moe_mma` (the routed experts' down projection on the tensor cores, `gpu/spawn/moe_mma`): about -0.30 ms a
  step. One GPU's ncu predicted -0.6 ms (18.5 -> 10.3 us a layer); in the graph `moe_down2` is faster than its
  cold ncu time. The logits move by up to 0.16 over 78 layers, but the greedy tokens stay the same. Its GSM8K
  is z27d's.
- The two fusions keep forward()'s bits exactly. `fuse_tp_norm` makes the step slower (see z27a), which hides
  `fuse_mla_q`'s ~1 us a layer. The served runs after this keep `fuse_mla_q` and drop `fuse_tp_norm`.
- The plain step on this VM is 9088 us. z5's was 9432 us on another VM, before `moe_down2` and `fuse_router`.
