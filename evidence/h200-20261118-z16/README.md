# One H200: the device decode loop, the fused all-reduce + RMSNorm, and an FP8 MMA probe (#63)

These runs cover what one GPU can check of docs/TP-SERVING-PLAN.md items 2, 3 and 4. GLM-5.3 itself (756 GB)
needs the 8× H200, so none of these numbers is a served TPOT for the whole model.

- **Machine:** one NVIDIA H200 on Nebius, a fresh VM, driver 580.173.02, CUDA 13.0 image.
- **Repo:** branch `tp` at 95f7b7a, plus 447e1d5 (`PeerSum.norm_arguments` passes `arrivals`) and 0852e7b
  (`glm_vx.server --tp-option` at `--tp 1`).
- **Kernels:** gpu/spawn built locally with Vx 3609 (sm_80 PTX, checked by ptxas for sm_90). Every kernel that is
  also in the z5–z10 bundle has the same PTX bytes. gpu/build.sh and cuBLASLt were built on the node.
- **Model:** GLM-5.3's first 4 trained layers (3 dense, 1 MoE), with the embeddings, the final norm and
  `lm_head`. `validation/real_layers.py fetch --layers 4` copied them: 14.88 GB in 233 s.
  - A model cut to 4 layers is not GLM-5.3, so its text means nothing. Its tokens are only compared between
    runs.
- **Options:** gpu/tp_serve.py's `OPTIONS` + `moe_down2` + `fuse_router` (z10's) everywhere.
- **Scripts:** `z16.sh`, `z16b.sh`, `greedy_tokens.py`, `device_loop_time.py`; logs and JSON next to them.

## Item 2: the decode loop on the device

**Server.** `glm_vx.server --tp 1` with the options above. The prompt is tp_real's 28 tokens, then that prompt
repeated to 1000 and 3000 tokens. Each repeated prompt is run 3 times with 128 greedy tokens; the table gives the
median decode time a token (server `decode_seconds` / (tokens − 1)).

| server | 28-token prompt | ~1000 tokens | ~3000 tokens |
|---|---|---|---|
| item 1: greedy on the device, each token read before the next step | 1.661 ms | 1.673 ms | 1.790 ms |
| item 2: `--tp-option device_decode=true` | 1.220 ms | **1.248 ms** | **1.369 ms** |
| change | | −0.425 ms | −0.421 ms |

**Tokens.** Both servers give the same tokens on every prompt, and the 28-token prompt's 32 tokens equal
`validation/tp_real`'s (`greedy-item2.json`: `all_equal: true`; `greedy-item1.json`: `same_as_tp_real: true`).

**Without the server** (`device-loop.json`, 200 steps from position 999, the fastest of 3):

| loop | ms a step |
|---|---|
| chain: `device_step()` back to back, nothing read until the last (the GPU's own time) | 1.249 |
| ahead: step n+1 launched, then token n read (as `ResidentSequenceModel.forward_greedy`) | 1.259 |
| item 1: `forward(token, logits='greedy')`, each token read before the next launch | 1.595 |

The loop one step ahead costs 0.01 ms a step more than the GPU alone. The served item-2 time (1.248 ms) is the
GPU's. Item 1's server pays 0.35 ms a step for waiting on each token, and 0.08 ms more for the server.

For comparison, tp_real's step at one rank, with the logits downloaded each step, is 1515.6 µs at 1000 tokens and
1687.8 µs at 3001 (`tp-real-1.json`).

**Not covered here** (needs 2+ GPUs):
- the NCCL `all_gather` of the greedy pairs captured inside the step graph;
- the followers' `S` / `D` / `W` commands.

On one GPU `ResidentGLM` drops the one-rank TP, so the step takes its own `argmax_pair`.

## Item 3: `tp_sum_rmsnorm`

**Bits** (`validation/tp_all_reduce_check --gpus 1`): the ranks are processes sharing the GPU, so their kernels
take turns and only the bits mean anything here. Each run checks, on every rank:
- `residual_out` = NumPy's rank-order sum + residual;
- `out` = its RMSNorm in `residual_rmsnorm8`'s order;
- 50 eager calls, then a graph of 157 calls replayed 5 times.

| ranks | blocks | result |
|---|---|---|
| 8 | 8 | pass (`norm-8r-8b.json`) |
| 4 | 16 | pass |
| 2 | 32 | pass |
| 8, `tp_all_reduce` alone (control) | | pass |

**Time at one rank**: no peer, only the local work; a graph of 1000 calls, `*-again.json`.

| kernel | µs a call |
|---|---|
| `tp_all_reduce` | 3.09 |
| `tp_sum_rmsnorm`, 8 blocks | 6.06 |
| `tp_sum_rmsnorm`, 16 blocks | 6.27 |
| `tp_sum_rmsnorm`, 32 blocks | 6.32 |

`residual_rmsnorm8` took 2.9 µs in z5's TP8 profile, so the two launches it replaces take about 6.0 µs on one
rank, the same as the fused kernel. The fusion saves no local work.

Whatever item 3 gains at 8 ranks has to come from the push: one NVLink crossing instead of two.
`tp_all_reduce` takes 6.8 µs at 8 ranks against 3.1 µs at one. Only the 8× H200 can measure this.

## Item 4: FP8 tensor cores, a CUDA probe

`fp8mma/fp8_moe_mma.cu` is a measurement, not glm-vx code: Vx's tile matmul takes bf16 and f16 only.

**Shapes:** one MoE layer at batch 1, for one rank of 8:
- 8 routed experts + the shared one;
- per expert, gate_up is 512 × 6144 and down is 6144 × 256, FP8 e4m3 with 128×128 f32 scales.

**Method:**
- `mma.sync.m16n8k32.e4m3` with each lane's A fragment loaded straight from global memory: 16 contiguous bytes,
  with k permuted the same way in A and B.
- Activations quantized to e4m3 in groups of 128 (amax / 448, as SGLang does before its FP8 GEMMs).
- f32 accumulation, with the scales applied per 128 values of k.
- Timed in a graph of 75 layers, each with different random experts out of 256.

**Check:** against a double-precision model of the same quantized inputs, the maximum error is 4.8e-9 on gate_up
values up to 0.048 and 5e-11 on down values up to 6e-4. The fragment layout is right.

Against unquantized activations, gate_up's maximum error is 1.4e-3: the FP8 activation quantization SGLang also
does.

| a MoE layer, one rank of 8 | gate_up | down | both |
|---|---|---|---|
| probe | 11.7 µs (2.4 TB/s) | 8.1 µs (1.75 TB/s) | **20.0 µs** |
| glm-vx today (z5 profile, `cvt` + `down2`) | about 13.2 µs | about 17.3 µs | about 30.5 µs |
| SGLang `--tp 8` (z5 nsys at 1000 tokens) | `fused_moe_kernel` ×2: 24.4 µs | | about 50 µs, with `act_and_mul` 17.7, sum 6.9 and quant |

Over 75 MoE layers, 20 µs against 30.5 µs would be about −0.8 ms a token. The plan estimated −1.4 ms, at 3 TB/s.

The probe is not tuned. One variant, `fp8_moe_mma_v2.cu`, moves the loads out of `asm volatile` and loads the
next k-block ahead. It used 141 registers instead of 96 and made gate_up slower: 15.5 µs. I ran it by hand, so that output is
quoted here and not saved as a file. So the numbers above are a first point, not the limit.

**In Vx:**
- Vx's tile matmul would need FP8 operands, lowered to `mma.sync` e4m3, and an A operand loaded into registers
  without going through shared memory.
- Asked of the Vx peers on 2026-11-18. vx-94 took FP8 and is asking for its priority first; vx-7e took
  NVFP4 / MX for Blackwell.
- glm-vx would use FP8 only. FP4 would be below SGLang's precision.

## What went wrong first (`../h200-20261118-z16-try1`)

- **Item 3:** every `tp_sum_rmsnorm` launch refused its arguments. `PeerSum.norm_arguments` left out the kernel's
  `arrivals` count (its tail runs in the last block). The CPU tests had passed it by hand. Fixed in 447e1d5, with
  a test against the built manifests.
- **tp_real and the servers:** they failed because the kernel tarball lacked `gpu/build/spawn/index_mma/`.
- **Stray server:** the rerun replaced `z16.sh` while the first run was still reading it, so that bash stopped and
  left its item-2 server running for about a minute. z16's one-rank times (`norm-1r-*b.json`, 5.87–6.44 µs) ran
  beside that server. The `*-again` files, run alone, are the ones above.
- **GPU tests:** I stopped `test-spawn-hardware` partway, to free the GPU for z16b. It did not finish, and z16.log's
  "143 failed steps" is the kill's exit status, not a test result. `test-resident-graph` did not run.
