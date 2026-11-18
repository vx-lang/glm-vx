# 8x H200: GLM-5.3 itself, tensor parallel, against SGLang 0.5.22 --tp 8 (#35, #63), 2026-11-18

Nebius 8x H200 (NVSwitch, NV18 between every pair), driver 580.173.02; the node of z4, restarted after a
preemption with its disk (new IP). glm-vx at 0a42332's changes (aa9827a), Vx 6a1d8f47 toolchain, spawn kernels
built with Vx #3609 with the v2 tp_all_reduce. NCCL_NVLS_ENABLE=0 on both sides (this VM has no multicast
objects). Model: zai-org/GLM-5.3, the published block-FP8 checkpoint (755.7 GB, 141 shards, 78 layers: 3 dense,
75 MoE of 256 experts top 8, the DSA indexer on 21 layers). SGLang 0.5.22 (torch 2.14.1+cu130) with
`--tp 8 --kv-cache-dtype bfloat16 --mem-fraction-static 0.8 --random-seed 0`, no speculative decoding.
Ours: `validation/tp_real.py` (ResidentGLM split over 8 GPUs, the one-shot sum, production kernels: graph, cvt,
x4, tc MLA, ilp_hist top-k, fused glue, mma indexer, folded shared expert).

## Batch-1 decode step (`compare.py`)

| context | glm-vx TP8, us/step (median) | SGLang --tp 8, us/step (median) | glm-vx / SGLang | SGLang, dummy weights |
|---|---|---|---|---|
| 1000 | 9432.5 | 9870.5 | 0.96 | 9785.3 |
| 3001 | 10304.8 | 10349.9 | 1.00 | 10307.9 |
| 32768 | 10778.8 | 10424.5 | 1.03 | 10253.7 |

Wall time of one step, synchronized, at batch 1. Ours: the median of 48 steps after filling the context with
random tokens; each step returns the downloaded logits (sampling on the host). SGLang: `sglang.benchmark.one_batch`
(no HTTP server, CUDA graphs as in serving, sampling on the GPU), the median of its 63 decode steps
(`sglang-one-batch.jsonl`). The dummy-weight column (`sglang-dummy.sh`, run while the checkpoint downloaded) is
within 2% of the real one: dummy weights are good enough for SGLang kernel timing.

## Correctness

- **Bits on real weights.** GLM-5.3's first 4 layers (3 dense + 1 MoE of 256 experts) on 8 GPUs against the same
  8 ranks emulated on GPU 0: every logit of 48 steps bit-exact on all 8 ranks, twice (`real4-8-oneshot.json`,
  `real4-8-oneshot-rerun.json`). The first run's 2280 us/step was measured while SGLang's TP workers (the dummy run
  just before) were still exiting; the rerun on idle GPUs: 864 us/step (the 4-layer proxy of z4c: 840).
- **The whole model.** Prompt "The history of the transistor begins in 1947 at Bell Labs, where John Bardeen, Walter
  Brattain and William Shockley" (28 tokens, no chat template), 32 greedy tokens: " invented the first working
  transistor. This invention revolutionized electronics and paved the way for the development of modern computers,
  smartphones and many other devices. The transistor replaced the" (`real-8-oneshot.json`).
- **Against SGLang, same prompt ids** (`sglang-greedy.json`): the first 3 tokens agree, top-5 in the same order at
  each, log-probabilities within 0.44. At token 4 SGLang has "working" and "point" exactly tied (-1.055 each: its
  log-probabilities come from bf16 logits) and takes "point" ("first point-contact transistor"); ours has "working"
  at -0.858 against -1.371. SGLang also quantizes activations to FP8 before each FP8 GEMM; ours stay float32.
  Agreement on greedy text is therefore not a usable check here; GSM8K accuracy is (z6).

## Where our step goes (`nsys-real-8.gpu0-steps.txt`, context 1000, GPU 0)

1521 kernels a step, 9.15 ms busy in a 9.84 ms span (under nsys, which makes the step 11.2 ms; 9.4 without):

| kernels | per step | each |
|---|---|---|
| fp8_moe_cvt_down (75 MoE layers) | 1606 us | 21.4 us |
| tp_all_reduce (2 a layer) | 1063 us | 6.8 us |
| fp8_moe_cvt_gate_up | 990 us | 12.7 us |
| bf16_matvec4: the router's gate (256 x 6144) + the lm_head | 759 us | 9.3 us + 61 us |
| fp8_matvec_cvt_tiles (q_b, o_proj, shared expert) | 752 us | 4.7 us |
| fp8_matvec_cvt_tiles3 (q_a + kv_a, replicated) | 750 us | 9.6 us |
| mla_tc_at | 649 us | 8.3 us |
| residual_rmsnorm8 | 449 us | 2.9 us |
| router (top-8 of 256) | 443 us | 5.9 us |
| the rest (kv_b, heads, merge, norms, rope, q, kv rows, indexer keys) | ~1.8 ms | |

The leads: the MoE down projection runs at 0.6 TB/s (12.6 MB a rank: under TP8 an expert's down matrix has
256-wide rows, a shape it was not tuned for; gate_up reads 25 MB at 2 TB/s); the router's bf16 gate matvec has 256
rows for 132 SMs; each all-reduce is followed by its own residual+RMSNorm kernel (SGLang fuses the two); the host
work between steps (~0.3 ms: eager lm_head, gather, logits download, embedding) is outside the graph. Together
about 3 ms of the 9.4. SGLang's step at 1000 (`nsys-sglang-1000_cuda_gpu_kern_sum.csv`, GPU 0 only: it profiles
rank 0) is 2280 kernels.

## Files

- `z5.sh` (steps 1-3; stopped during step 3, which ran on), `z5b.sh` (4-6, each once no process holds a GPU),
  `sglang-dummy.sh`, `compare.py`, `step_breakdown.py`.
- `nsys-*_cuda_gpu_kern_sum.csv`: per-kernel sums (ours: all 8 GPUs over the whole run; SGLang: GPU 0, one decode
  step). `nsys-real-8.gpu0-last2-steps.csv`: GPU 0's last two steps, every kernel. The full traces (1.5 GB) are not
  committed.
- Load: the first step uploads every weight; 915 s with half the checkpoint still on disk (read at 205 MB/s), 324 s
  with all of it in the page cache (`nsys-real-8.json`). Each rank copies every full tensor before taking its
  slice; reading the slice only is the next change.
