# `router_bf16`: the top-k in one warp, the same bits, 10.5 us against 12.2 (#63)

`fuse_router`'s kernel computes the router's matvec in 256 blocks of 1024 lanes, then picks the top 8 of the 256
scores in its last block to finish (the tail).

**Where its time went** (ncu, cold, GLM-5.3's 256 x 6144 BF16 gate, `router_time.py`):
- the whole launch: 12.2 us;
- the same launch with k = 0, which skips the tail: 6.7 us.

So the tail cost 5.5 us. It ran two passes of ranks with shared-memory atomics, lane loops and barriers across the
1024-lane block.

**New tail** (`gpu/spawn/router_bf16.vx`): one warp runs k rounds.
- Lane l holds experts l, l + 32, .. in registers and offers the best still in.
- `lanes::argmax` (the higher score, or the same and the lower id: the router's order) picks round r's winner,
  which is the expert of rank r and goes in slot r.
- The weights are as before: the sigmoid, the `lanes::sum` of the k (the same order in a subgroup as in a
  block), and the scale.

Its registers raised the kernel to 61, one 1024-lane block an SM, which put the 256 rows in two waves (14.7 us).
`// spawn-maxnreg: 32` keeps two blocks an SM with no spill.

| | old tail | new tail |
|---|---|---|
| ncu, whole launch | 12.2 us | **10.5 us** |
| ncu, k = 0 (no tail) | 6.7 us | 6.6 us |
| CUDA events after an L2 flush, `router_ab.py` | 13.8 us | 11.9 us |

**Bits.** `router_ab.py` runs both PTX on 200 random inputs: k = 8, 6 and 1, every fourth case with 40 rows
duplicated (tied scores), and biases. The indices, weights and logits are the same bits in all 200. The CPU host
tests of `router_bf16` (ties included) and the resident `fuse_router` tests pass.

At TP8 every rank runs the router on all 75 MoE layers: −1.7 us x 75 = about −0.13 ms a token, with no change in
results.
