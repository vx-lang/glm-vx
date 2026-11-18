# Closing the gap to SGLang `--tp 8` at batch 1 (#34, #63), 2026-11-18

This is a plan. A number is a measurement only when an evidence directory is named next to it. Everything
marked "expected" is an estimate from z5's profile. Precision rule: FP8 is allowed where SGLang uses FP8, and
nothing goes below SGLang (no FP4).

## Where we stand

GLM-5.3 (block FP8) on the Nebius 8x H200, both engines at concurrency 1, with the same client and flags
(`sglang.benchmark.serving`):

| workload | SGLang `--tp 8` (z6) | glm-vx `--tp 8`, best so far | glm-vx / SGLang |
|---|---|---|---|
| W1 1000/1000: TPOT median | 9.18 ms | 9.53 ms (z10: polling + `moe_down2` + `fuse_router`) | 1.04 |
| ShareGPT 50: TPOT median | 9.00 ms | 9.60 ms (z8 c, before `fuse_router`) | 1.07 |
| W3 32768/256: TPOT median | 9.43 ms | 11.17 ms (z8 c, before `fuse_router`) | 1.18 |
| W1: TTFT median | 88.7 ms | 8681 ms (z10) | 98 |
| W3: TTFT median | 2623 ms | 336344 ms (z8 c) | 128 |

GSM8K, 5-shot, 200 questions: glm-vx 0.920 (z7, which has z8 c's bits), against SGLang's 0.915 (z6) and 0.935
(z9, SGLang's own runner). SGLang's two runs differ on 14 questions, so these are the same accuracy.
`fuse_router` changes the router's rounding, and its GSM8K run (z11) was cut off by a preemption.

## The decode gap is in the server, not the step

The bare decode step, with no HTTP server on either side (z5, `compare.py`):

| context | glm-vx `validation/tp_real.py` | SGLang `sglang.benchmark.one_batch` |
|---|---|---|
| 1000 | 9.43 ms | 9.87 ms |
| 32768 | 10.78 ms | 10.42 ms |

- **Our step was already faster at W1's context.** Since z5, `moe_down2` (−0.29 ms, z8) and `fuse_router`
  (−0.31 ms, z10) took about 0.6 ms more off it. Expected: about 8.8 ms now. It has not been timed on its own.
- **Our server adds time; SGLang's removes it.** Our server takes about 0.7 ms a token longer than that step.
  SGLang's serves 0.7 ms faster than its own step, because it schedules step n+1 while the GPU runs step n.
- **Where our 0.7 ms goes.** It is per-token host work on rank 0 in the critical path:
  - the vocabulary's logits gathered from all 8 ranks, downloaded, checked and argmaxed in NumPy;
  - the scheduler's Python;
  - the command down the followers' pipes.

  tp_real's step itself also has about 0.3 ms of host work between graph replays: the embedding computed on the
  host and uploaded, the eager lm_head, and the gather (z5).

So items 1 and 2 below should put W1 decode below SGLang's on their own. Items 3 and 4 are the GPU-side margin.

## The plan, in order

### 1. Measure greedy decoding on the device (committed, f839322; unmeasured)

- **Change.** Each rank runs `argmax_pair` over its vocabulary rows and the ranks gather one (value, index)
  pair each. The host picks the largest pair, the lower rank on a tie. No logits are gathered or downloaded.
- **Expected.** −0.2 to −0.4 ms a token on W1.
- **Bits.** The same tokens as the host argmax.
- **Run.** `z12.sh`: first z5's prompt (its 32 greedy tokens must equal z11's server, which had the same
  options with the logits sent to the host), then W1, GSM8K 200, ShareGPT 50 and W3 on the final server.
  `z12.sh` and `greedy_check.py` are in the session scratchpad; z12 copies both into
  `evidence/h200-20261118-z12`.

### 2. The decode loop on the device: the host off the critical path

**What changes.** All of a decode step runs from device memory, so rank 0 can launch step n+1 before it reads
token n:

- The token id lives in a device buffer. A gather kernel in the graph reads it and copies the embedding row,
  from the table kept on every rank (bf16 154880 x 6144, about 1.9 GB of each H200's 141 GB).
- cos and sin come from a position-indexed rope table on the device. The step buffer (position, selected
  count) is advanced by a kernel, not uploaded each step as `forward()` does today.
- lm_head, `argmax_pair`, the all-gather of the pairs, and a small pick kernel all go inside the captured graph.
  The pick kernel takes the largest of the 8 pairs, the lower rank on a tie, and writes the token id for the
  next step. Every rank computes the same pick, so no rank needs to be told the token.
- Rank 0 sends the followers a token-less "step" command when it launches each step. A follower therefore reads
  step n+1's command while the GPUs are still busy with step n, and the pipe is no longer on the critical path.
- Rank 0 copies each token to the host asynchronously and handles it one step behind: streaming, detokenizing,
  EOS, stop strings and max_tokens. A request that stops wastes at most one step. Its cache is reset for the
  next request anyway (`ResidentSequenceModel.new_cache`).
- **Scheduler.** A pipelined greedy path for the single running request, in place of one synchronous
  `forward_greedy` per token. Sampled requests (temperature > 0) keep today's path.

**Expected.** The served TPOT comes down to about the step's GPU time: about 8.6-8.8 ms on W1 (SGLang: 9.18).

**Bits.** The same as item 1. The arithmetic is unchanged; only where it runs changes.

**Status: written, tested on the emulator only, not yet run on a GPU.** Enable it with
`--tp-option device_decode=true`.

- Kernels: gpu/spawn/decode_inputs and gpu/spawn/decode_next.
- ResidentGLM: `device_step`, `device_token` and `rewind`.
- Followers: the D, S and W commands.
- `ResidentSequenceModel.forward_greedy` runs a step ahead, and the scheduler passes `last=`.
- Runtime: page-locked `HostBuffer` and `DeviceTensor.download_async`.
- Tests: `gpu/test_tp.py::test_device_steps_give_the_greedy_tokens_one_step_ahead` gives `forward(logits='greedy')`'s
  tokens on one GPU and on two emulated ranks.
- The server's pair gather is NCCL's all-gather inside the step's graph. It has not been captured on GPUs yet.
- Next run: z13 in `z12.sh`.

**Done when:**
- z5's prompt gives z12's tokens;
- W1's served TPOT is within 0.1 ms of tp_real's step at the same context;
- gpu/test_tp's emulated-rank tests cover the pick and the device-side step buffer.

### 3. Fuse the all-reduce with the residual add and RMSNorm

**Why.** Two all-reduces a layer each run as their own kernel and are then followed by a separate
`residual_rmsnorm8`: 156 pairs a token. In z5 the all-reduces took 6.8 us each and the norms 2.9 us each
(1063 us + 449 us a step).

**Change.** One kernel does the one-shot sum, then the residual add and the norm:
- the sum runs in today's rank order;
- the residual add and RMSNorm use `residual_rmsnorm8`'s arithmetic.

SGLang fuses the same pair.

**Expected.** About −0.45 ms a token: the 156 norm launches and the gaps between them.

**Bits.** The same, provided the order of the sum and of the norm's reduction is kept. Verify with gpu/test_tp's
bit-exact tests and `real4-8` on the node.

### 4. FP8 tensor cores for the experts and the replicated projections

**Why.** At TP8 these kernels are limited by instruction issue, not memory bandwidth (ncu: DRAM 12-13%):

| kernel (z5 profile) | per step | data per call | bandwidth |
|---|---|---|---|
| `fp8_moe_cvt_down` (now `down2`, about 1.3 ms) | 1606 us | 12.6 MB | 0.6 TB/s |
| `fp8_moe_cvt_gate_up` | 990 us | 25 MB | 2 TB/s |
| `fp8_matvec_cvt_tiles3` (q_a + kv_a, the same on every rank) | 750 us | | 1.7 TB/s |

Today these kernels convert FP8 to float32 and multiply on the CUDA cores.

**Change.**
- Read the FP8 weights straight into tensor-core MMA (`mma.sync` e4m3, or `wgmma` on sm_90), accumulating in
  float32 and applying the 128-block weight scales.
- Quantize the activations to FP8 in groups of 128, as SGLang does before each FP8 GEMM. This is within the
  precision rule.

**Expected.**
- At about 3 TB/s, gate_up + down would take about 12.5 us a MoE layer, against about 30 us today: about
  −1.4 ms a token over 75 layers.
- tiles3 would save about −0.35 ms more.

**Bits.** Changed.
- Check each layer's error against the float32 path.
- Run GSM8K 200: it must fall inside SGLang's 0.915-0.935 spread.
- Run W1, ShareGPT and W3 again.

These kernels are also what item 5b needs.

### 5. The prompt under tensor parallelism (TTFT)

**Why.** Under TP a prompt runs one token a step, about 9.5 ms each: TTFT is 8.7 s on W1 and 336 s on W3.

**a. `forward_many` under TP.** It runs 1-4 tokens a step and is used for MTP verification. It already works
on one GPU but raises `NotImplementedError` under TP. The scheduler already has a `batched_prefill` /
`prefill_chunk` hook.
- Change: the all-reduces take [count, hidden], and a new Lockstep command carries the tokens.
- Expected: about 1.5-2x on TTFT, a guess. Only the Linears read their weights once for the four tokens; the
  MoE, attention and indexer still run per token.

**b. Chunked prefill.** Chunks of hundreds of tokens, built from:
- item 4's FP8 tensor-core GEMMs;
- MoE GEMMs with the chunk's tokens grouped by expert;
- an MLA prefill attention kernel;
- the indexer over the chunk;
- all-reduces over [chunk, hidden].

This is the only route to a TTFT near SGLang's. The target is within 2x of SGLang on W1 and W3.

### 6. Long context: the indexer on every rank

**Why.** Our step grows 1.35 ms from 1000 to 32768 tokens of context; SGLang's served TPOT grows 0.25 ms.
gpu/tp.py keeps the DSA indexer the same on every rank, so all 8 score every key on its 21 layers.

**First, profile.** nsys a 32K step and split it into indexer scores, top-k (`ilp_hist`) and sparse attention.
For comparison, SGLang's kernel sums at 32768 are in `evidence/h200-20261118-z5/nsys-sglang-32768_cuda_gpu_kern_sum.csv`.

**Then, likely:**
- Each rank scores its eighth of the keys and keeps its own top 2048.
- The ranks all-gather the (score, index) candidates and merge them to 2048.
- A key's score does not depend on the split, so the selection is the same, provided the merge breaks ties as
  the top-k does today.
- Also check whether SGLang's index keys are FP8. If they are, ours may be too.

## Expected totals (estimates)

| after | W1 served TPOT | vs SGLang 9.18 |
|---|---|---|
| today (z10) | 9.53 ms (measured) | 1.04 |
| 1 + 2 | about 8.6-8.8 ms | about 0.95 |
| + 3 | about 8.2-8.4 ms | about 0.90 |
| + 4 | about 6.8-7.2 ms | about 0.76 |

## Where the work happens

- **Local, while the node is down:**
  - item 2's device step: the embedding gather, rope table and step-buffer kernels, the pick kernel, the
    Lockstep "step" command and the pipelined scheduler path;
  - item 3's kernel.

  Both get unit tests against the emulated ranks.
- **On the node, in order:**
  1. z12 (item 1);
  2. item 2's served W1 and tokens check;
  3. item 3's `real4-8` bits and W1;
  4. the 32K profile (item 6);
  5. then items 4 and 5 with a GSM8K run for each change that alters bits.
- **Each result** goes into its own `evidence/h200-YYYYMMDD-zN/README.md` and a comment on #63.
