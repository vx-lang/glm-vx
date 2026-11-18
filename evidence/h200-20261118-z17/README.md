# The MoE kernels at TP8's per-rank shapes on one H200: ncu, the FP8 MMA probe and the floor (#63, item 4)

Same machine as z16: one H200 (Nebius), driver 580.173.02. Shapes are one rank of 8 on GLM-5.3:
- hidden 6144;
- each expert's intermediate slice 2048 / 8 = 256;
- 256 experts;
- k = 9 (8 routed experts and the shared one, which `fold_shared` treats as a ninth).

Weights and expert ids are random, which does not change the timing.

## Our kernels under ncu

Run with `validation/moe_profile.py --formats fp8cvt fp8cvt2 --inter 256 --k 8 9 --detail`. Each kernel runs alone
with cold caches; files `moe-tp8.*`.

| kernel (k = 9) | µs | DRAM MB | TB/s | instructions | issue active | warps active |
|---|---|---|---|---|---|---|
| `fp8_moe_cvt_gate_up` | 13.3 | 28.5 | 2.14 | 4.42 M | 39% | 26% |
| `fp8_moe_cvt_down` | 21.7 | 14.2 | 0.66 | 6.02 M | 30% | 18% |
| `fp8_moe_cvt_down2` (served since z9) | 18.1 | 14.2 | 0.78 | 8.62 M | 52% | 35% |

In all three, the stall a warp waits on most is long scoreboard, that is global memory. Occupancy is low. These
match z5's TP8 step profile: 12.7 µs for gate_up and 21.4 µs for down.

## The FP8 MMA probe under the same ncu counters (`probe-ncu.csv`)

The probe is `fp8mma/fp8_moe_mma.cu` (see z16): `mma.sync.m16n8k32.e4m3` with the weights loaded straight into
each lane's registers.

| kernel | µs | DRAM MB | instructions | issue active | warps active | grid × block, regs |
|---|---|---|---|---|---|---|
| gate_up | 13.7 | 29.0 | 1.82 M | 19% | 14% | 144 × 256, 104 |
| down | 9.8 | 14.2 | 1.46 M | 22% | 24% | 384 × 288, 96 |

FP8 MMA cuts instructions by 2.4× for gate_up and by 5.9× for down. Cold and alone:
- down drops from 18.1 to 9.8 µs;
- gate_up does not change (13.7 µs against 13.3). It was waiting on memory, not on instructions: 144 blocks of 8
  warps hold too few loads in flight.

Timed in a graph of 75 layers instead, gate_up takes 11.7 µs and down 8.1 µs.

## The floor (`fp8mma/stream_floor.cu`, `stream_floor.json`)

A kernel that only reads the bytes: 16-byte loads, timed the same way as the probe. The table gives the best of
2 / 4 / 8 blocks an SM and unroll 2 / 4.

| bytes | µs | TB/s |
|---|---|---|
| 14.2 MB (down) | 4.75 | 2.99 |
| 28.3 MB (gate_up) | 7.90 | 3.58 |
| 42.5 MB (both, if one kernel) | 11.06 | 3.84 |

## Variants tried

`probe-variants.txt`, graph-timed:

- **Warps a gate_up block:** 8 warps gives 11.7 µs; 12, 16 and 24 warps give 14.9, 13.4 and 14.3 µs.
- **Non-volatile loads with the next k-block loaded ahead (`fp8_moe_mma_v2.cu`):** 14.6–19.3 µs. It used more
  registers and was slower at every warp count.

The first layout stays the best one found so far.

## What this says

A MoE layer at one rank of 8:

| | µs |
|---|---|
| ours today (`gate_up` + `down2`, ncu) | 31.4 |
| probe (graph) | 20.0 |
| floor (two kernels) | 12.7 |
| floor (one kernel) | 11.1 |

Over 75 MoE layers:
- today to the probe is about −0.8 ms a token;
- today to the floor is about −1.4 ms.

Most of the gap is down's, which FP8 MMA closes. Closing gate_up's needs more memory parallelism: more loads in
flight a warp, or more warps over the same rows.
