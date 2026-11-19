# Served with device_decode + moe_mma + fuse_mla_q: W1 8.55 ms, ShareGPT 8.34 ms, GSM8K 0.920 (#63)

`glm_vx.server --tp 8` (z10's options) with `device_decode`, `moe_mma` and `fuse_mla_q` (`fuse_tp_norm` dropped: z27b).
Prompts still a token at a time. Client: SGLang 0.5.22's `sglang.benchmark.serving`, concurrency 1.

| workload | median TPOT | median TTFT | SGLang `--tp 8` (z6, another VM) |
|---|---|---|---|
| W1 1000/1000 (`w1.log`) | **8.55 ms** (p99 ITL 8.84) | 8135 ms | 9.18 ms, 88.7 ms |
| ShareGPT 50 (`sharegpt-glmvx.log`) | **8.34 ms** | 1959 ms | 9.00 ms |

- GSM8K, 5-shot, 200 questions (`gsm8k-glmvx.json`): **0.920**, no invalid answers. That is the same as z7 (0.920,
  without moe_mma) and within SGLang's 0.915 (our client) - 0.935 (its own runner). moe_mma's rounding does not cost
  accuracy.
- z5's prompt gives z11's served tokens (`greedy.log`).
- W3 was stopped by the maintainer (`w3.log` partial); it was measured with chunked prefill in z36.
