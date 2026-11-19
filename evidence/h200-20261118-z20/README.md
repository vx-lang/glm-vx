# `moe_mma` with f16 activations: 10.5 us against `down2`'s 18.5, within 2e-4 of it (#63, item 4)

z19's SASS showed that on sm_90 ptxas turns `mma.sync` e4m3 into f16 unpacks plus `HMMA.16816`. So
`fp8_moe_mma16_down_at` (from `gen_moe_mma.py`) does that conversion itself:
- the FP8 weights are unpacked once with `cvt.rn.f16x2.e4m3x2`, which is exact;
- the activations stay in f16, scaled by a power of two a group of 128 (exact), so each group's max is in
  [2^14, 2^15);
- `mma.sync.m16n8k16.row.col.f32.f16.f16.f32`, f32 sums.

Per 256-column segment the SASS has 64 `F2FP.F16.E4M3.UNPACK_B`, 32 `F2FP.F16.F32.PACK_AB` and 16 `HMMA.16816`. The
E4M3 kernel has 96 unpacks for the same 16 HMMA.

## Correctness (`mma-check.json`, random weights, 257 experts, k = 9)

| cols | mma16 against its NumPy model | mma16 against `down2` (f32 act) | E4M3 `mma` against `down2` |
|---|---|---|---|
| 256 (one rank of 8) | 1.7e-7 | **1.5e-4** | 2.1e-2 |
| 2048 (one GPU) | 2.3e-7 | 2.0e-4 | 2.5e-2 |

All values are relative to max |out|. `gpu/test_moe_mma_hardware.py` passes, with mma16 against `down2` under
2e-3.

**GLM-5.3's first 4 trained layers**, `validation/tp_real --ranks 1`, against `../h200-20261118-z18/tp-real-idle-plain.json`:

| | 32 greedy tokens | top-5 ids, every step | max top-5 logit difference |
|---|---|---|---|
| `moe_mma=True` (f16, `tp-real-mma16.json`) | same | same | 3.4e-4 |
| `moe_mma='e4m3'` (z18) | same | not always (near ties swap) | 2.7e-2 |

## Time (ncu, cold, TP8 shapes, k = 9; `moe-tp8.md`)

| down kernel | us | TB/s |
|---|---|---|
| `fp8_moe_cvt_down2` (served) | 18.5 | 0.77 |
| `fp8_moe_mma_down_at` (E4M3 act) | 11.2 | 1.31 |
| `fp8_moe_mma16_down_at` (f16 act) | **10.5** | 1.41 |

That is −8.0 us a MoE layer, about −0.6 ms a token over 75 layers at TP8. The CUDA probe did 9.8 us and the floor
is 4.75 us.

At one GPU's width (cols 2048), CUDA events give mma16 57.5 us, E4M3 51.4 and `down2` 84.7.

The 4-layer model's step, with one MoE layer, is 1505 / 1620 us at 1000 / 3001 tokens with mma16, against 1533 /
1689 without. These are noisy.

`ResidentGLM(moe_mma=True)` now means the f16 kernel; `moe_mma='e4m3'` keeps the E4M3 one.
