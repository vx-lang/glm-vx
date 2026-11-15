# H200 (Nebius): the cvt kernels with `// spawn-maxnreg:` budgets (#38), and the MLA decode kernels (#33)

Without a budget ptxas gives the cvt kernels 64 to 72 registers and moves most of a tile's weight loads
below the first conversions, so a warp has one or two loads in flight. `.maxnreg` 88 to 104 has it issue
them all first; 112 and up lowers occupancy. The sweep (CUDA events, L2 flushed, median of 30; the same
bits in every column; us):

| kernel, shape | ptxas's choice | 72 | 80 | 88 | 96 | 104 | 112 | 128 |
|---|---|---|---|---|---|---|---|---|
| cvt_tiles o_proj 6144 x 16384 | 44.9 | 36.5 | 36.6 | 36.4 | 36.4 | 36.6 | 49.6 | 49.8 |
| cvt_tiles down 6144 x 12288 | 37.3 | 30.1 | 30.2 | 30.0 | 30.0 | 30.1 | 40.0 | 40.2 |
| cvt_tiles q_b 16384 x 2048 | 19.6 | 18.2 | 18.3 | 18.2 | 18.1 | 18.5 | 20.8 | 20.8 |
| cvt_tiles3 q_a + kv_a + wk | 18.6 | 21.4 | 21.6 | 18.9 | 19.2 | 14.3 | 15.2 | 14.3 |
| moe_cvt_gate_up dense 12288 x 6144 | 85.7 | 88.3 | 69.7 | 61.6 | 59.2 | 58.1 | 64.1 | 63.9 |
| moe_cvt_gate_up k = 8, 2 x 2048 x 6144 | 105.8 | 109.0 | 95.7 | 81.7 | 77.9 | 76.9 | 82.9 | 83.0 |
| moe_cvt_down k = 8, 6144 x 2048 | 52.4 | 58.9 | 50.8 | 48.6 | 60.9 | 61.3 | 60.7 | 60.0 |

The kernels take 96 (tiles), 104 (tiles3, gate_up) and 88 (down). Vx has no way to say it: vx-lang/Vx#3463.

* test-spawn-hardware: 153 passed, 1 xfailed, with the budgets.
* resident-fp8cvt-full4: 4 full-width layers, 48 tokens, max |logit diff| 8.8e-6 against float32 (logits
  to 5.6), top-1 equal.
* layer-profile-fp8real (ncu): o_proj 30.7, down 24.5, q_b 13.9, q_a + kv_a + wk 10.8, gate + up + SwiGLU
  51.5 us, against SGLang's 34.7, 28.7, 17.0, 15.0 and 45.0 + 1.8 (h200-20261115-sglang). The dense layer
  is 548 us; mla_partial is 294 of it.
* matvec-profile: matvec_cvt_tiles is under cuBLASLt at every shape (o_proj 36.0 against 40.8 us).
* mla-profile: mla_heads_at takes 153 / 154 / 162 us for 256 / 1024 / 2048 selected tokens at 256 tokens a
  split. With 32 to 128 tokens a split it takes 45 / 82 / 120 us. ncu at 256 tokens and one split: the
  blocks are issue-bound, about 600 warp instructions a head and token against 34 for the fmas, and 56%
  of their stalls are at barriers. A warp scores one token a chunk and reads the queries from shared
  memory for it.
