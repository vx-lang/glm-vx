# Served W1 with the decode loop on the device: 9.00 ms TPOT, below SGLang's 9.18 (#63, item 2)

`glm_vx.server --backend vx-gpu-resident --tp 8` (z10's options) with `--tp-option device_decode=true`: each greedy
step is launched before the previous token is read (`ResidentSequenceModel.forward_greedy` runs one step
ahead). The client is SGLang 0.5.22's `sglang.benchmark.serving`: W1 is 10 prompts of 1000 tokens, 1000 out,
concurrency 1 (`scripts/pod_sglang.sh client`; `bench/glm-vx-device-decode-W1-vllm.jsonl`, `w1.log`).

| W1 | median TPOT | median TTFT | p99 ITL |
|---|---|---|---|
| glm-vx z10 (each token read before the next step) | 9.53 ms | 8681 ms | |
| **glm-vx, device_decode** | **9.00 ms** | 8540 ms | 9.36 ms |
| SGLang 0.5.22 `--tp 8` (z6, another VM) | 9.18 ms | 88.7 ms | |

- The served TPOT is now the GPU's step. tp_real's step with the same options is 9.09 ms at 1000 tokens
  (`../h200-20261119-z27b`), so the 0.45 ms a token of host work in the critical path is gone, as one GPU
  showed (z16).
- The bits do not change: z5's prompt gives z11's served tokens (`greedy.log`: `same_as_z11: true`).
- TTFT is still a token a step (8.5 s). Chunked prefill is z32's.
- SGLang's W1 on this VM is z27e.
