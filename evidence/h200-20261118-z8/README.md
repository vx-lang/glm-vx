# 8x H200: glm-vx's TP8 server, two changes against z7's serving overhead (#63), 2026-11-18

z7's server added about 0.7 ms a token to tp_real's decode step. Each row below is its own server start
(`glm_vx.server --tp 8 ... --token-budget 34000`), the same client and flags as z6/z7 (concurrency 1).
glm-vx at 2dbfa56.

| server | W1 1000/1000: TPOT median | p99 | TTFT median | vs SGLang (9.18) |
|---|---|---|---|---|
| a. followers block on their pipe between commands (`GLM_VX_TP_SPIN=0`, z7's server) | 10.50 ms | 10.53 | 9386 ms | 1.144 |
| b. followers poll it for 50 ms after each command (the default since 2dbfa56) | 10.13 ms | 10.19 | 9336 ms | 1.104 |
| c. b + the experts' down projection through fp8_moe_cvt_down2 (`--tp-option moe_down2=true`, the same bits) | **9.84 ms** | 9.88 | 9052 ms | **1.072** |

Then, with c:

| workload | TPOT median: glm-vx c | SGLang (z6) | glm-vx / SGLang | TPOT p99: glm-vx | SGLang | TTFT median: glm-vx | SGLang |
|---|---|---|---|---|---|---|---|
| ShareGPT, 50 | 9.60 ms | 9.00 | 1.067 | 10.74 | 9.37 | 2180 ms | 50 |
| W3 32768/256, 2 (SGLang 8) | 11.17 ms | 9.43 | 1.184 | 11.18 | 9.48 | 336344 ms | 2623 |

- a against z7 (10.45): the same server, run again. b: a follower woken from a blocking read started its step
  late, and rank 0's first sum of every step waited for the last follower; polling took 0.37 ms of that off.
- c: fp8_moe_cvt_down2 at GLM-5.3's TP8 slice on one H200 (`down2-time.log`, `down2_time.py`, events after an L2
  flush): 23.0 us against fp8_moe_cvt_down's 27.1 (k = 9 with the folded shared expert), the same bits; in the
  server's step, 0.29 ms over 75 MoE layers. `hw-down2.log`: its sm_90 build gives its _ref build's bits on the
  H200 (gpu/test_spawn_hardware.py).
- Left between the server and tp_real's step (about 0.5 ms at W1's contexts): rank 0's host work in each step's
  critical path (the logits of all 154880 tokens gathered and downloaded, checked and argmaxed in NumPy, then the
  command to the followers), which SGLang avoids by sampling on the GPU and scheduling while the step runs.
- W3: our step grows with context (tp_real: 9.43 ms at 1000, 10.78 at 32768; the DSA indexer scores every key on
  21 layers past index_topk), SGLang's barely does (9.18 / 9.43). TTFT: our prompt runs a token a step under
  tensor parallelism (about 10 ms each, 5.6 min for 32768).
- GSM8K: c has z7's bits (the polling changes no arithmetic; moe_down2 is fp8_moe_cvt_down's bits), so z7's 0.920
  stands for it.
