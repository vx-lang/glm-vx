# One H200 for the TTFT branch: 'mla_tc' against float64, a chunk under nsys, and the host's per-token cost (#63)

This ran on 2026-11-26 on a single Nebius H200 (gpu-h200-sxm, 1gpu-16vcpu-200gb, eu-west1; driver 580.196.07, CUDA
13.0). The code was branch ttft-bf16-gemm 73e5916, with kernels built by Vx main 10c6f478f.
- `rank_profile.py` is mtp-verify's validation/rank_profile.py, which has `--nsys-long`.
- The model is GLM-5.3's geometry, cut to the given number of layers. It has random weights in the checkpoint's
  formats, at rank 0 of 8's shapes. The all-reduces are no-ops.
- These are diagnoses on one GPU, not served results.

**z68** (`z68/z68.sh`, `z68/run.log`):
1. `gpu/test_spawn_hardware.py -k "mla_prefill_sparse or bf16_rows"`: 2 passed (`sparse-tc.log`). The absorbed
   sparse prefill kernel matches float64 attention over the same bf16 values, at GLM-5.3's shape with 2048 selected
   keys, and below index_topk.
2. `gpu/test_cublaslt_hardware.py`: 27 passed, 6 skipped (as on 11-21).
3. **One layer, 'mha_tc' against 'mla_tc'.** The model is layer 0 alone: dense, with a full indexer. The chunk is
   2048 tokens at 30720 (`one-layer-compare.txt`).
   - **Logits:** they differ by 6.3% of the largest logit, and the argmax agrees. W1's logits are identical.
   - **Attention:** 48.9 ms with 'mha_tc', 6.9 ms with 'mla_tc'.
   - **Wall time:** 174 ms against 135 ms, so about 128 ms of the chunk is outside the layer.
4. **nsys of the 4-layer 'mla_tc' chunk at 30720** (`nsys-kern_cuda_gpu_kern_sum.csv`, `nsys-run/summary.txt`). The
   kernels total 23.2 ms:

   | kernel | time | share | launches |
   |---|---|---|---|
   | `mla_prefill_sparse_tc` | 9.3 ms | 40% | 4 |
   | `index_scores_chunk_at` | 5.6 ms | 24% | 3 |
   | `topk_select_rows` | 2.3 ms | 10% | 3 |
   | `fp8_quant_groups` | 1.3 ms | | 31 |
   | grouped FP8 GEMM | 0.8 ms | | 3 |
   | cuBLASLt | under 1 ms | | |

   - The two host-to-device copies took 5.2 ms. The CUDA API calls total about 18 ms.
   - The chunk's wall time was 152 ms, so about 128 ms is Python with no CUDA work.
   - `cprofile-w1-tottime.txt` (W1, 1000 tokens, 4 layers, 91 ms of host time) shows what it is:
     - `memoryview.tobytes`: 26 ms in 4 calls.
     - `ctypes.create_string_buffer`: 14 ms in 2 calls.
     - Loops a token at a time: `_make_rope_coefficients`, `_rope_inputs`, `_embedding` and `_step_buffer`, about
       25 ms per 1000 tokens.
5. The hardware suite on this branch: 509 passed, 7 skipped, 1 xfailed (`suite.log`).

**z71** (`z71/z71.sh`). The host-copy change, A/B, 2 rounds alternating A B, B A. A is 73e5916. B adds:
- `gpu/runtime._host_buffer`: an upload copies from the array's own buffer.
- The chunk's RoPE and step rows as array ops (`kernels/backend._rope_coefficient_rows`, the same bits).
- One embedding gather (`GLMModel._embeddings`).

Both ran 4 layers with prefill_attention='mla_tc'. Results are in `compare.txt` and `rounds.txt`:

| 4 layers | A | B |
|---|---|---|
| W1 wall, median of 3 (r1, r2) | 70.0, 67.9 ms | 14.3, 14.5 ms |
| chunk of 2048 at 30720, wall (r1, r2) | 154.6, 148.5 ms | 43.9, 44.3 ms |
| host enqueue of that chunk | 142.7, 138.6 ms | 31.0, 30.5 ms |
| 78-layer GPU estimate, W1 / chunk | 254 / 431 ms | 255 / 431 ms |

- The logits are identical (max |B - A| = 0), for W1 and for the chunk.
- The saving is per token and does not depend on the layer count:
  - About 55 ms per 1000 tokens, so roughly 365 -> 310 ms for z63's W1 TTFT.
  - About 108 ms per 2048-token chunk, roughly 1.6 s of W3's.
- These are estimates. The served numbers need the 8x H200.
- `host-tests.log`: kernels/test_backend.py's RoPE-rows bit test passes with this box's numpy 2.5.5.

**z76** (`z76/z76.sh`): gpu/test_runtime.py and the hardware suite on B, since every upload takes the new path: 551
passed, 7 skipped, 1 xfailed.

The nsys report (`.nsys-rep`, `.sqlite`) and the logits `.npy` files are not committed.
