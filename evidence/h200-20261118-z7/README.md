# 8x H200: glm-vx serving GLM-5.3 tensor parallel (`glm_vx.server --tp 8`), the reportable workloads (#63)

The node and checkpoint of z5 and z6 (Nebius 8x H200, zai-org/GLM-5.3 block FP8, NCCL_NVLS_ENABLE=0).
`python -m glm_vx.server --backend vx-gpu-resident --checkpoint GLM-5.3 --tp 8 --tokenizer GLM-5.3/tokenizer.json
--max-context 33280`: rank 0 serves HTTP, schedules and samples on the host; ranks 1-7 replay its steps
(gpu/tp_serve.py); the decode options of z5's tp_real (graph, cvt, x4, tc MLA, ilp_hist, fused glue, mma indexer,
folded shared expert, the one-shot sum). glm-vx at 93c20fa / 0b6a200. The clients are z6's, at concurrency 1.
`compare.py` prints both sides; z8 has the server's next two changes (followers polling, fp8_moe_cvt_down2).

| workload | requests (glm-vx / SGLang) | TPOT median, ms: glm-vx | SGLang | glm-vx / SGLang | TPOT p99: glm-vx | SGLang | TTFT median, ms: glm-vx | SGLang |
|---|---|---|---|---|---|---|---|---|
| W1 1000/1000 | 10 / 10 | 10.45 | 9.18 | 1.139 | 10.66 | 9.18 | 9380 | 90 |
| ShareGPT | 50 / 50 | 10.15 | 9.00 | 1.127 | 11.28 | 9.37 | 2253 | 50 |

- **GSM8K, 5-shot, 200 questions, greedy, the same client** (`benchmarks/gsm8k.py`): glm-vx **0.920**, SGLang 0.915.
  Both right on 177, only glm-vx on 7, only SGLang on 6; the same final answer on 184 of 200 (`gsm8k-glmvx.json`).
- **The server against tp_real:** for z5's prompt (token ids through /v1/completions) the server's 32 greedy
  tokens are tp_real's (`greedy.log`: same, the same text).
- **Smoke:** the first 4 layers only (`--layers 4`), one request (`smoke.log`, `server-smoke.log`).
- **Load:** every weight preloaded, 94.4 GB a GPU in 83 s (`server.log`; a rank reads its slice of each tensor only).
- **TTFT:** the prompt runs one token a step under tensor parallelism (forward_many is not split yet): about 9.4
  ms a prompt token, 9.4 s for W1's 1000.
- **TPOT against the model's step:** tp_real's step is 9.43 ms at context 1000 and 10.30 at 3001 (z5); the server
  added about 0.7 ms a token. py-spy on rank 0 and a follower during W1: both spend nearly all of the step
  waiting on the GPU, so the time is in the GPUs' step, not in Python: the followers, blocked on their pipe
  between commands, start each step late, and rank 0's first sum waits for them (z8 measures it).
- W3 did not run here: the scheduler admits 8192 tokens a request by default (`--token-budget`), and refused
  the 32768-token prompt (`bench.log`); z8 runs it with `--token-budget 34000`.
- `z7.sh` (smoke, the whole model, greedy, GSM8K, ShareGPT), `z7b.sh` (W1; W3 refused); `bench/`, `client/`: the
  client's results and the server's /health and /v1/models. SGLang's own GSM8K runner did not run here either
  (`--host` given with its scheme; z9).
