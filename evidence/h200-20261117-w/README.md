# H200: the indexer in FP8, SGLang's recipe, index_kernel='fp8' (#32)

The same VM, toolchain and checkout as h200-20261117-u and -v, with the files of the next commit copied
over (`w.sh`). Two kernels, inline PTX in `mlir!` blocks (`gpu/spawn/index_mma/gen_index_mma.py`),
built for sm_90a with PTX 8.4:

- `index_key_fp8_at`: the key row at step[0] to E4M3 codes and a scale, as SGLang 0.5.22's
  `indexer_k.cuh`: each value rounded to bf16 first, scale = max(1e-4, amax) / 448,
  `cvt.rn.satfinite.e4m3x2.f32` of x * (1 / scale). 22 registers.
- `index_scores_fp8_at`: each head's q quantized in registers, one scale a head (as
  `fused_q_indexer_rope_first_quant`), with q_scale * scale folded into the head's weight. Then
  32 `mma.sync.m16n8k32.e4m3` a warp on 16-byte loads of the codes, and the head sum times the row's
  key scale. 108 registers, no spills.

Tests (`*-rerun.txt`: 15 and 14 passed):
- `test_index_key_fp8_at_quantizes_the_row_at_step_as_sglang`: codes and scales bit for bit against
  the NumPy model. The first run (`test-spawn-hardware.txt`) differed in one code, a -0.0 input: the
  hardware cvt, which SGLang uses too, keeps the sign (0x80), and the model's e4m3_encode did not.
  The model now keeps it.
- `test_spawn_index_scores_fp8_at_matches_float64_on_its_codes`: within 1.2e-8 to 8.5e-8 (of the
  largest sum of |terms|) of float64 on the same codes, 1 to 32768 tokens.
- `test_index_kernel_tc_matches_simt_to_bf16_rounding[fp8]`: under replay against 'simt' (f32), one step
  of 600 (step 283) is outside 2e-2 of the logits' scale: FP8 moved a score across the top-k cut. The
  test now allows 1% of steps for FP8, none for bf16, and the argmax still agrees on more than 95%.

Step profiles against h200-20261117-v's 'mma' (bf16; `ab-*.txt`):

| context | 'mma' `index_scores_mma_at` | 'fp8' `index_scores_fp8_at` + `index_key_fp8_at` | step 'mma' | step 'fp8' |
|---|---|---|---|---|
| 3001 | 3.9 us | 4.2 + 1.4 | 206.8 | 208.1 |
| 8192 | 5.1 | 4.5 + 1.4 | 214.5 | 215.9 |
| 32768 | 6.7 | 5.9 + 1.4 | 242.7 | 244.0 |

At batch 1 these contexts are not bandwidth-bound. FP8 reads a quarter of the f32 cache's bytes, but
each warp quantizes all 32 heads' q, and the key row takes its own launch. So 'fp8' is 1.3 to 1.4 us a
layer slower than 'mma' in the step, at lower precision. SGLang's own FP8 scores take 3.6 us at
1000 and 2048 tokens.
