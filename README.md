# GLM-Vx

An experimental, from-scratch GLM-MoE-DSA inference engine using **real compiled
Vx numerical kernels**. Python handles checkpoint IO, tensor storage, model
orchestration, request scheduling and HTTP. The serving path does not use vLLM, SGLang, Transformers or
PyTorch. An optional validation environment runs the pinned official model. NumPy supplies storage and an explicit test
oracle; the `vx` backend never silently falls back to NumPy matrix operations.

**Current status: tested CPU prototype plus a GPU-resident decode path tested on A100 and H100.**
Six GPU sessions so far (five on one A100, one on one H100, Oct 15 to 26) ran the kernels against
NumPy, mostly bit for bit, the GPU-resident engine in f32, INT4, FP8 and BF16 against the hybrid
path and, with random weights, against the official PyTorch code; the H100 session passed all 33
steps of `scripts/pod_validate.sh`. The trained model has not run on a GPU, and no GPU throughput
claim is made: the work is per-layer and per-kernel (#29). Evidence is under [`evidence/`](evidence/),
summarized in [#36](https://github.com/vx-lang/glm-vx/issues/36). See
[GPU implementation and run instructions](gpu/README.md).
The full trained GLM-5.3 mixed-bit GGUF checkpoint has executed on CPU through
all 78 base-decoder layers, generating **“Let me”** from a six-token abbreviated
chat prefix. Both tokens match an independent llama.cpp CPU reference. This
original baseline smoke took about 34 minutes and 32.2 GB peak RSS; it is not a complete
answer or a practical-speed result. See [actual model evidence](docs/gguf-evidence/real-chat-summary.json).
The latest [implementation report](docs/OPTIMIZATION-PROGRESS.md) adds packed
Vx kernels, batched prefill, prompt reuse and compiler candidate search. Its full
trained official comparison matches all 306 discrete checks but fails 112 of
308 frozen numerical checks. **Automatic release promotion remains blocked.**
The included tiny model has deterministic random weights and is for testing mechanics only.

The [October 4–5 engine audit](docs/ENGINE-AUDIT-20261004.md) adds corrected DSA
configuration defaults and cache identities, SIMD across independent output rows,
fused packed GGUF batches with fixed-stride activation tiles, and a fused GPU
index-score kernel. The new full trained two-token batched run matches the saved
scalar Vx trace bit-for-bit across 1,898 checks, including both complete vocabulary
outputs. This proves preservation of that baseline, not independent official parity.

[Serving research and prioritized roadmap](docs/SERVING-RESEARCH.md) reviews GLM-specific
provider work, papers, X discussions and their applicability to this engine.

## Run

Prerequisites: Linux x86-64, Python 3.10+, C linker, working Vx v0.0.2 and LLVM 22.
See [Vx installation](https://vxlang.org/docs/getting-started.html).

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
VXC=/path/to/vxc kernels/build.sh
.venv/bin/python -m glm_vx.server --tiny --backend vx
```

On the implementation machine the isolated compiler is already available:

```sh
cd /srv/nanocodex/workspace/glm-vx
VXC=/srv/nanocodex/workspace/vx-toolchain/bin/vxc kernels/build.sh
.venv/bin/python -m glm_vx.server --tiny --backend vx
```

```sh
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[1,2,3],"max_tokens":8,"stream":true}'
```

`/health` reports backend and capacity; `/v1/models` lists the local model.
`/v1/completions` accepts one nonempty token-ID prompt or a string when a local
`--tokenizer tokenizer.json` is provided. Greedy sampling is the default; finite
nonnegative temperature and an optional seed support reproducible sampling.
`ignore_eos`, `stream_options.include_usage`, and `n`/`best_of` equal to 1 are
accepted for OpenAI benchmark clients; other unknown options are rejected. This
is a **completion API subset**, not a claim of complete OpenAI chat/tool-call
compatibility. Each SSE token event carries its exact token ID and a `text`
delta that is emitted only once the decoded text is stable, so split UTF-8/BPE
pieces are held back rather than corrupted. The terminal event depends on the
request. A request that names `model`, as every OpenAI client does, gets OpenAI
framing: the terminal chunk has only the held-back text tail, `finish_reason`
and `usage`, followed by a `choices: []` usage chunk if `include_usage` was set,
and keepalives are blank lines. A request without `model` gets the glm-vx
framing: the final event repeats the complete text and `token_ids` (clients must
not concatenate it again), and keepalives are SSE comments.

The service binds loopback by default. It has no authentication or TLS; place a
suitable gateway in front before exposing it to a network.

## Quantized full-model CPU path

[GGUF setup and limitations](docs/GGUF.md) describes the full GLM-5.3 IQ1_S
checkpoint, memory-mapped loading, bounded expert decoding and CPU generation.
The earlier tiny-model benchmarks remain synthetic; they are not trained-model
performance evidence.

[Correctness gates and Vx performance audit](docs/CORRECTNESS-AND-PERFORMANCE.md)
records the remaining parity work and an isolated compiler optimization experiment.
The [implemented trace gate and current failures](docs/PARITY-PROGRESS.md) now
include full trained-weight layer/logit comparisons and independent codec checks.
[CPU prefill/decode handoff](docs/DISAGGREGATION.md) runs distinct worker processes
with validated cache transfer; it is not yet an HTTP worker pool or GPU speedup.

## GPU implementation

[GPU code](gpu/README.md) adds coalesced F32/FP8 projection, fused residual/RMSNorm,
warp/block reductions, split online-softmax compressed MLA, and a CUDA Driver API
runtime. Use `--backend vx-gpu` for the explicitly hybrid serving adapter, or
`--backend vx-gpu-resident --checkpoint DIR` for device-resident decode (weights, activations
and caches on the GPU; one request at a time, others queue; see [gpu/README.md](gpu/README.md)).
The kernels written in Vx's own GPU model (`spawn` and `across` loops) are in
[gpu/spawn](gpu/spawn/README.md); they need Vx main c1dc2752 or later. Pods restore a prebuilt
toolchain with `scripts/pod_restore.sh` (built on a pod by `scripts/pod_build_bundle.sh`) and run
`scripts/pod_validate.sh`. The [original CPU report](REPORT.md) remains a baseline, not a GPU benchmark.

## Serving performance

[Implemented optimizations and current evidence](docs/OPTIMIZATION-PROGRESS.md)
cover packed GGUF kernels, SIMD batched prefill, exact heap selection, prefix
reuse, official eager validation and an explicit tensor-core GEMM path.

The [HTTP load benchmark and measurements](benchmarks/README.md) track client
TTFT, inter-token gaps, request latency and throughput with mixed prompt lengths.
Prefill skips unused vocabulary heads, GPU MLP intermediates stay on-device,
and decode-first scheduling bounds total prefill work per round.
`--decode-prefill-tokens` (default 4) trades prompt progress for smoother decode.
Measured CPU streaming improvements carry a long-prompt TTFT tradeoff; no GPU
throughput claim is made.

## Implementation

- `kernels/kernels.vx`: float32 matvec, RMSNorm, LayerNorm, interleaved RoPE,
  stable softmax, SwiGLU, sigmoid router with correction bias, top-k and attention.
- `glm_vx/model.py`: dense and sparse blocks, selected routed experts plus shared
  experts, absorbed MLA decode, compressed request-local KV, full/shared DSA
  indexer selection, official expert reduction order and causal sequential decode.
- `glm_vx/prefill.py`: opt-in layer-wise batched prefill with grouped experts.
- `kernels/packed.vx`: fused GGUF unpack-and-dot for all trained checkpoint formats.
- `glm_vx/checkpoint.py`: lazy read-only safetensors mmap with checked headers,
  BF16/F16/F32/FP8 E4M3FN decoding and block scales. No remote code execution.
- `glm_vx/scheduler.py`: decode-first interleaving, bounded optional batched prefill,
  prompt-plus-output token admission, cancellation, failure isolation and cleanup.
  Disconnect detection is cooperative between model forwards (socket polling
  every 250 ms); keep the TCP connection open until the reply. Shutdown raises
  an explicit timeout if a forward is still running, leaving weights open.
  Optional bounded prefix snapshots reuse prompt state; this is CPU serving,
  not a paged GPU KV allocator.
- `glm_vx/server.py`: bounded HTTP handlers, JSON completion and SSE transport.

## Validation and measurement

```sh
.venv/bin/python -m pytest tests kernels/test_backend.py -q
.venv/bin/python kernels/test_kernels.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python scripts/benchmark.py \
  --requests 2 --prompt 8 --tokens 8 --output benchmark.json
```

The benchmark verifies matching greedy outputs first, excludes one-token warmup,
then reports scheduler throughput and per-request time to first token on the
**tiny random model**. It does not measure full-model quality or GPU throughput.
See `REPORT.md` for the measured results and exact validation scope.

## Full checkpoint and remaining work

`--checkpoint DIRECTORY --tokenizer DIRECTORY/tokenizer.json` is an experimental
CPU path for a locally supplied checkpoint. It does not download weights.
Config/header validation and synthetic loader tests are not full-checkpoint
validation. Float32 dequantization is not numerically identical to a production
FP8 activation-quantized GEMM path. Metadata and source revisions are recorded in
`metadata/sources.json` and `docs/model.md`.

The development machine has no NVIDIA GPU, about 93 GiB RAM and roughly 326 GiB
disk free at initial inspection. The original FP8 checkpoint is about 756 GB and
does not fit on this disk. The 216.7 GB mixed-bit GGUF checkpoint was downloaded in
full and all six shard SHA256 hashes verified. GPU work runs on rented single-GPU
pods (see above).

To deliver an optimized production engine still requires: the trained checkpoint on
a GPU (#35) and its logits/generation parity, the per-layer bottlenecks of #29 on
the H100, packed/fused GPU prefill, MTP, FP8 GEMM and quantized activation parity,
paged/distributed KV sharing, tensor/expert parallel collectives on more than one GPU
(#34), GPU topology-specific placement, and throughput/latency benchmarks against
established engines on matching hardware and workloads.
