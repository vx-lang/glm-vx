# 8x H200: SGLang 0.5.22 --tp 8 serving GLM-5.3, the reportable workloads (#63), 2026-11-18

The node and checkpoint of z5 (Nebius 8x H200, zai-org/GLM-5.3 block FP8). SGLang 0.5.22 with
`--tp 8 --kv-cache-dtype bfloat16 --mem-fraction-static 0.8 --random-seed 0`, no speculative decoding (its
default), NCCL_NVLS_ENABLE=0. Every client runs at concurrency 1 (batch 1). glm-vx's side, with the same clients:
`../h200-20261118-z7` (its `compare.py` prints both).

| workload | client | requests | TTFT median | TPOT median | TPOT p99 | ITL median |
|---|---|---|---|---|---|---|
| W1: 1000 in, 1000 out | `sglang.benchmark.serving --backend sglang` (native) | 10 | 88.7 ms | 9.18 ms | 9.18 ms | 9.16 ms |
| W1 | `--backend sglang-oai` (`/v1/completions`, as glm-vx is measured) | 10 | 89.7 ms | 9.18 ms | 9.18 ms | 9.16 ms |
| W3: 32768 in, 256 out | native | 8 | 2623 ms | 9.43 ms | 9.48 ms | 9.43 ms |
| ShareGPT (50 conversations, 23382 tokens in, 11369 out) | `--backend sglang-oai` | 50 | 50.4 ms | 9.00 ms | 9.37 ms | 9.03 ms |

- W1 and W3 are the contract's commands (`benchmarks/SGLANG-SETUP.md`, `scripts/pod_sglang.sh bench`: the
  `random` dataset at fixed lengths, `--random-range-ratio 1.0`, 64 warmup requests, `--flush-cache`),
  in `bench/`.
- GSM8K, 5-shot, the test set's first 200 questions, greedy (`benchmarks/gsm8k.py`, the prompt and scoring of
  `sglang.test.few_shot_gsm8k`): **0.915**, 0 invalid, 186 s (`gsm8k-sglang.json`). SGLang's own runner could not
  download the questions here (a DNS failure, `gsm8k-sglang-official.log`); it reran in z7b from the same file.
- SGLang's served decode step is shorter than its `one_batch` step (z5: 9.87 ms at 1000, 10.42 at 32768): serving
  overlaps the scheduling of a step with the GPU's work on the one before.
- `server/`: the launch command, the checkpoint check and SGLang's own greedy sanity requests; `server.log`: the
  GSM8K and ShareGPT server.
