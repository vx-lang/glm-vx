# `moe_mma`: the expert-rank lookup inside the warp (#63, item 4)

This replaces device.vx's `id_rank!` lane loop, its shared-memory table, the barrier and `slot_of_rank!` with
`slot_ptx()` in `gpu/spawn/moe_mma/gen_moe_mma.py`. Lane j < k loads ids[j] and counts the ids below it through 9
shuffles; a ballot finds the lane of the warp's rank. The values are unchanged (`mma-check.json`: the same
errors as z20, PASS).

ncu, cold, TP8 shapes (`moe-tp8.md`):

| down kernel | k = 8 | k = 9 |
|---|---|---|
| `fp8_moe_cvt_down2` | 16.2 us | 19.0 us |
| `fp8_moe_mma_down_at` (E4M3 act) | 9.8 | 10.4 (z20: 11.2) |
| `fp8_moe_mma16_down_at` (f16 act, `moe_mma=True`) | 9.8 | 10.3 (z20: 10.5) |

The gain is 0.2-0.8 us, not z18's 1.7. That number came from a 127-register build with one block an SM, where
the lookup's latency was not hidden by other blocks.
