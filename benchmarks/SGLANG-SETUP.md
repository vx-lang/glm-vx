# SGLang on one H100 for the proxy comparison

Setup notes for Session 1 of #22 (contract #3, proxy #5). `scripts/pod_sglang.sh` implements them.
Researched 2026-10-21. The claims about behaviour come from the SGLang source at the pinned tag, not
only from its docs; links point at that tag. Nothing here has run on a GPU yet. Items marked
**(verify on pod)** are the ones Session 1 must confirm.

`SG` below stands for `https://github.com/sgl-project/sglang/blob/v0.5.22`.

## Version pin: SGLang 0.5.22

- **Pin `sglang==0.5.22`.** Tag `v0.5.22` is commit `baa9befa3815` (2026-10-13). The PyPI wheels were
  uploaded on 2026-10-16 for cp311-cp313, manylinux_2_34 x86_64
  ([PyPI](https://pypi.org/project/sglang/0.5.22/)). The Docker tags `lmsysorg/sglang:v0.5.22` and
  `-runtime` were pushed the same day. This is the newest stable release; 0.5.21 (2026-09-29) is the
  fallback.
- **GLM-5.x support.** `GlmMoeDsaForCausalLM` is a thin subclass of `DeepseekV2ForCausalLM`
  ([SG/python/sglang/srt/models/glm4_moe.py#L1289](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/glm4_moe.py#L1289)).
  It is listed as DSA in `is_deepseek_dsa`
  ([SG/python/sglang/srt/configs/model_config.py#L190](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/configs/model_config.py#L190)).
  The GLM-5.3 cookbook needs `>= v0.5.20` for parser auto-detection
  ([SG/docs/src/snippets/configs/zai-org/glm-5.3.jsx](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/src/snippets/configs/zai-org/glm-5.3.jsx)).
  `--trust-remote-code` is not needed, because transformers 5.19 has `glm_moe_dsa` built in.
- **Pinned dependencies** ([SG/python/pyproject.toml](https://github.com/sgl-project/sglang/blob/v0.5.22/python/pyproject.toml)):
  - `torch==2.14.1`, `sglang-kernel==0.4.9` (FlashMLA, FA3, `fp8_mqa_logits`), `sgl-deep-gemm==0.2.2`
  - `flashinfer_python[cu13]==0.7.1`, `tilelang==0.1.12`, `transformers==5.19.0`, `nvidia-cutlass-dsl[cu13]==4.8.0`
  - The torch 2.14.1 wheel pulls `cuda-toolkit==13.0.3`, cuDNN 9.24 and NCCL 2.30.7
    ([PyPI torch 2.14.1](https://pypi.org/project/torch/2.14.1/)). All of these are sm_90 wheels
    from PyPI. No separate FlashMLA or DeepGEMM build is needed.
- **CUDA 13 only.** SGLang dropped CUDA 12 with torch 2.14. 0.5.19 is the last cu129 release
  ([SG/docs/docs/get-started/install.mdx](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/docs/get-started/install.mdx)).
  So the pod needs **driver R580 or newer**. Our A100 pods had 580.159.04.
- **Install.** The documented pip/uv path is:

  ```sh
  uv venv --python 3.12
  uv pip install --prerelease=allow sglang
  ```

  `--prerelease=allow` matters: without it, uv < 0.12 silently installs 0.5.9. There is no `[all]`
  extra any more; the base package carries the CUDA kernels.

## Pod and image (RunPod H100 SXM)

- **Use our usual image, with SGLang in its own venv** (`/workspace/sglang-venv`). It does not touch
  `.venv` or `/opt/vx-toolchain`. RunPod pods cannot run Docker inside, so the
  `lmsysorg/sglang:v0.5.22` image would have to be the pod's own image, and that costs us our
  toolchain flow.
- **Image requirement: CUDA 13 devel**, for example `nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`.
  This is the base of SGLang's own Dockerfile
  ([SG/docker/Dockerfile](https://github.com/sgl-project/sglang/blob/v0.5.22/docker/Dockerfile)).
  It gives:
  - Python 3.12 and glibc 2.39. Our toolchain bundle needs glibc 2.39.
  - `nvcc`. DeepGEMM JIT-compiles with NVCC by default (`SGL_DG_USE_NVRTC=0`,
    [SG/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py#L56](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py#L56)),
    and FlashInfer JITs its kernels too.
  - `ptxas`, which our `pod_validate.sh` uses.
- When renting, filter for **CUDA 13.0** so the host driver is R580 or newer. The script checks the
  driver and stops below 580.
- `nsys` is not in the CUDA devel image. SGLang's full image installs `nsight-systems-cli` from
  NVIDIA's devtools apt repo. The script tries that, then `cuda-nsight-systems-13-0`, and otherwise
  falls back to the torch profiler.
- Disk: the venv is about 10-15 GB (torch plus the CUDA 13 wheels). Keep it on `/workspace` so a
  restarted pod keeps it.

## What SGLang picks on H100 at `--tp 1`

**DSA attention.** Resolved in `_dsa_kv_cache_dtype_default` and `_dsa_split_backend_resolution`
([SG/python/sglang/srt/arg_groups/overrides.py#L467-L705](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/overrides.py#L467)):

| | KV dtype | prefill | decode | top-k |
|---|---|---|---|---|
| default on sm_90 (`--kv-cache-dtype auto`) | **bfloat16** (FP8 is the default only on sm_100+) | **flashmla_sparse** | **fa3** | `sgl-kernel` |
| `--kv-cache-dtype fp8_e4m3` | fp8_e4m3 | flashmla_kv | flashmla_kv | `sgl-kernel` |

- **Choices** ([SG/python/sglang/srt/arg_groups/fields/exec_.py#L249](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/fields/exec_.py#L248)):
  `flashmla_sparse`, `flashmla_sparse_q8` (prefill only), `flashmla_kv`, `flashmla_auto`,
  `flashinfer_sparse_mla`, `triton_sparse_mla` (prefill, sm_90+), `fa3`, `tilelang`, `triton`
  (ROCm only, rejected on CUDA), `aiter`, `trtllm`.
- **`trtllm` is Blackwell-only.** Auto picks it only for `major >= 10`, and it uses trtllm-gen
  kernels. Do not force it on H100.
- **`tilelang` with an FP8 KV cache** needs both backends set to tilelang.
- **Other kernels:**
  - The indexer always scores in FP8: DeepGEMM `fp8_mqa_logits` / `fp8_paged_mqa_logits`, or the
    sgl-kernel versions ([SG/python/sglang/srt/layers/attention/dsa/dsa_indexer.py](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/layers/attention/dsa/dsa_indexer.py)).
    Its key cache is FP8 even in a "BF16" row.
  - The indexer `wk` and `weights_proj` are fused into one BF16 GEMM. A block-FP8 `wk` is
    dequantized at load.
  - The page size is forced to 64
    ([SG/python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py#L99-L172](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py#L99)).
  - The attention backend is forced to `dsa`.

**Block-FP8 weights** (`quant_method: fp8`, `weight_block_size [128,128]`, dynamic activations):

- These run as W8A8. Activations are quantized per token-group at run time.
- Dense Linears use `--fp8-gemm-backend auto`. That means **DeepGEMM on sm_90** when it imports
  (`ENABLE_JIT_DEEPGEMM` requires sm >= 90,
  [SG/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py#L32](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py#L32)).
- MoE with `--moe-runner-backend auto` and no all-to-all backend is **not** DeepGEMM. DeepGEMM is
  used for MoE only with DeepEP, Mooncake or NIXL
  ([SG/python/sglang/srt/layers/quantization/fp8.py#L1410](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/layers/quantization/fp8.py#L1410)).
  So at TP1 the experts run through the Triton fused-MoE kernel with block-FP8 weights.
  **(verify on pod**: read `moe_runner_backend` and the kernel names from the trace.)
- `kv_b_proj` is re-processed at load into `w_kc`/`w_vc` for absorbed MLA
  ([SG/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py#L547](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py#L547)).

**BF16 needs its own checkpoint.** An FP8 checkpoint cannot be served in BF16:

- The checkpoint's `quantization_config` always wins. `--quantization unquant` only clears the CLI
  value
  ([SG/python/sglang/srt/configs/model_config.py#L1944-L2000](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/configs/model_config.py#L1944)).
- Dropping it with `--json-model-override-args '{"quantization_config": null}'` would cast the FP8
  bytes into BF16 parameters and skip the `weight_scale_inv` tensors. That gives garbage numerics,
  which is fine for kernel timing but not for any output comparison.
- So the BF16 row needs a BF16 copy of the proxy, the same random values dequantized, with no
  `quantization_config`. Upstream does the same with `zai-org/GLM-5.3-BF16`.
- **Size problem:** the 3-dense + 5-MoE proxy is about 54 GB in FP8 but about 105 GB in BF16, which
  does not fit 80 GB. The BF16 proxy has to be shallower, for example 3 dense + 2 MoE (about 46 GB).
  Compare it per layer (ms/layer).
- `write_proxy.py` should emit both copies (`PROXY_BF16`).

**Speculation.** It stays off unless `--speculative-algorithm` is passed. The GLM entries in
`speculative_hook.py` only fill in defaults once EAGLE is requested. The script reads
`/get_server_info` and stops if `speculative_algorithm` is set.

## Truncated proxy: will SGLang load it?

**Yes, with two conditions on the config.** From
[SG/.../deepseek_weight_loader.py#L214-L512](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py#L214):

- **Depth comes from `num_hidden_layers`.** `DeepseekV2ForCausalLM` builds that many layers. The
  only hard-coded count is `num_hidden_layers == 1`, which is special-cased for NextN.
- **Dense or MoE** is `layer_id >= first_k_dense_replace and layer_id % moe_layer_freq == 0`
  ([SG/.../deepseek_v2.py#L2736](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/deepseek_v2.py#L2736)).
  SGLang does not read `mlp_layer_types`.
- **MTP weights are optional with speculation off.** With speculation off (`NextNDisabledConfig`,
  L310), any `model.layers.<i>` tensor with `i >= num_hidden_layers` is skipped. So the checkpoint
  may contain the MTP layer, or not.
  `num_nextn_predict_layers` must exist only when speculation is on: `_initialize_nextn_conf`
  raises if it is missing.
- **Indexer pattern.** `dsa_layer_skips_topk`
  ([SG/.../model_config.py#L410](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/configs/model_config.py#L410))
  uses `indexer_types[layer_id] == "shared"` when `indexer_types` is present. Otherwise it falls back
  to `cli_factor`, then `index_topk_pattern`, then `index_topk_freq`/`index_skip_topk_offset`.
  - Only "full" layers get indexer modules.
  - Checkpoint indexer tensors for "shared" layers are skipped (`indexer_present_prefixes`, L268).
  - An `indexer_types` longer than the depth is harmless to SGLang.
- **Condition 1: per-layer lists must match the depth.** transformers 5.19
  `PreTrainedConfig.validate_layer_type` raises if `mlp_layer_types` or `layer_types` does not have
  exactly `num_hidden_layers` entries
  (`transformers/configuration_utils.py` L567-L582 in the
  [5.19.0 wheel](https://pypi.org/project/transformers/5.19.0/)). Our `model-config.json` has
  78-entry `indexer_types` and `mlp_layer_types`. **The proxy writer must cut both lists to the
  proxy depth.** `scripts/pod_sglang.sh check` tests this.
- **Condition 2: no tensor may be missing.** SGLang never checks that every parameter was loaded:
  - A missing tensor leaves that parameter as uninitialized memory, with no error.
  - An unexpected name only logs `... not found in params_dict` (L493).
  - The script's `check` step therefore lists every tensor SGLang needs, and their
    `weight_scale_inv`, before launch. It also greps the server log for that warning.
- **Expected names** (the official HF layout):
  - Attention: `self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}`.
    `q_a_proj` and `kv_a_proj_with_mqa` are fused at load.
  - Indexer: `self_attn.indexer.{wq_b,wk,weights_proj,k_norm(.weight,.bias)}`.
  - MoE: `mlp.gate.weight`, `mlp.gate.e_score_correction_bias`, `mlp.experts.<e>.{gate,up,down}_proj`,
    `mlp.shared_experts.*` (fused as expert 256).
  - Dense: `mlp.{gate,up,down}_proj`.
  - Each FP8 Linear needs `.weight_scale_inv`.
- **A second way to run truncated: `--load-format dummy`.**
  - `--json-model-override-args '{"num_hidden_layers": 8}'` on the official `config.json` works
    without any weights. The overrides are `setattr`s after parsing, so transformers' list check does
    not rerun
    ([SG/python/sglang/srt/utils/hf_transformers/config.py#L377](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/utils/hf_transformers/config.py#L377)).
  - Dummy init fills each parameter with uniform ±1e-3 and sets every `weight_scale_inv` to 1
    ([SG/python/sglang/srt/model_loader/weight_utils.py#L1753](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/model_loader/weight_utils.py#L1753)).
  - That is good enough for kernel timing. Routing and outputs then differ from glm-vx's, so use it
    only to tell a loader problem from a kernel problem
    (`SGLANG_EXTRA_ARGS="--load-format dummy"`).

## bench_serving commands (W1-W3)

The client lives in `sglang.benchmark.serving`. `python -m sglang.bench_serving` still works and warns
that it is deprecated
([SG/python/sglang/benchmark/serving.py](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/benchmark/serving.py)).

```sh
COMMON="--dataset-name random --random-range-ratio 1.0 --tokenizer $PROXY --model $PROXY --seed 42 \
        --warmup-requests 64 --output-details"
# SGLang server (native /generate). Use --backend sglang-oai to go through /v1/completions like glm-vx.
python -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:30000 $COMMON --flush-cache \
  --random-input-len 1000  --random-output-len 1000 --num-prompts 10   --max-concurrency 1   --output-file W1.jsonl
python -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:30000 $COMMON --flush-cache \
  --random-input-len 1000  --random-output-len 1000 --num-prompts 1000 --max-concurrency 100 --output-file W2.jsonl
python -m sglang.benchmark.serving --backend sglang --base-url http://127.0.0.1:30000 $COMMON --flush-cache \
  --random-input-len 32768 --random-output-len 256  --num-prompts 8    --max-concurrency 1   --output-file W3.jsonl
# glm-vx: same flags, --backend vllm (OpenAI /v1/completions), no --flush-cache, fewer warmups.
```

- **`--random-range-ratio 1.0` is required for fixed lengths.** The default is 0.0, which draws each
  length uniformly from `[max(len*ratio, 1), len]`
  ([SG/python/sglang/benchmark/datasets/common.py#L56](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/benchmark/datasets/common.py#L56)).
  The commands in #3 omit it. The cookbook's reproduce command sets it, together with
  `--warmup-requests 64 --flush-cache`.
- **`random` needs network access.** It samples ShareGPT text and repeats or truncates it to length,
  downloading ShareGPT from the HF Hub on first use
  ([SG/python/sglang/benchmark/datasets/random.py](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/benchmark/datasets/random.py)).
  `random-ids` uses random token IDs and needs no download.
- **The input length is reduced by the tokenizer's special tokens**, so the server sees exactly
  `--random-input-len` tokens.
- **Warmup** requests use the first prompt with at most 32 output tokens, at the run's concurrency.
- **`--flush-cache`** posts `/flush_cache` (SGLang) or `/reset_prefix_cache` (vLLM backends), so it
  must not be passed for glm-vx.
- **The OpenAI-completions client** sends `ignore_eos: true` by default, plus `stream`, `max_tokens`,
  `temperature 0` and `best_of 1`. It needs `/v1/models` unless `--model` is given (#4).
- **`--output-file` appends one JSON line per run.** It holds TTFT, TPOT and ITL as mean, median and
  p99, plus throughput. `--output-details` adds the per-request arrays.

**Cookbook caveat.** The H200 cookbook numbers in its benchmark table are measured with
`SGLANG_SIMULATE_ACC_LEN` pinning the MTP accept length, and are labelled "throughput-mechanism
numbers only"
([SG/docs/src/snippets/configs/zai-org/glm-5.3-benchmarks.jsx](https://github.com/sgl-project/sglang/blob/v0.5.22/docs/src/snippets/configs/zai-org/glm-5.3-benchmarks.jsx)).
They are TP8 with EAGLE, and not comparable to a TP1 proxy without speculation.

## Profiling one decode step at batch 1/8/64

The best tool is `bench_one_batch` under `nsys`. It runs one static batch through the real model
runner, with CUDA graphs as in serving, and no HTTP or scheduler noise
([SG/python/sglang/benchmark/one_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/benchmark/one_batch.py)).

```sh
nsys profile -o decode-bs8 --trace cuda,nvtx --cuda-graph-trace node \
  --capture-range cudaProfilerApi --capture-range-end stop --sample none --cpuctxsw none \
  python -m sglang.benchmark.one_batch --model-path $PROXY --tp 1 --kv-cache-dtype bfloat16 \
    --batch-size 8 --input-len 1000 --output-len 8 \
    --profile --profile-activities CUDA_PROFILER --profile-stage decode --profile-start-step 6
nsys stats --report cuda_gpu_kern_sum --format csv --output decode-bs8 decode-bs8.nsys-rep
```

- **`--cuda-graph-trace node` is required.** Without it, a graph replay shows up as one opaque
  launch.
- **Run one process per (stage, batch size).** `nsys` stops at the first `cudaProfilerStop`.
- **Upstream quirk:** with `CUDA_PROFILER`, one_batch never calls `cudaProfilerStop` after a decode
  step. The stop is gated on a torch-profiler object, which is `None` here (L854). The capture
  therefore runs to process exit. Starting at the last step (`output-len - 2`) keeps the trace to
  that one step. Prefill stops correctly.
- **Without nsys:** `--profile --profile-activities CPU GPU` with `SGLANG_TORCH_PROFILER_DIR=<dir>`
  writes `<prefix>_batch<B>_input<I>_output<O>_<stage>.trace.json.gz`, a torch-profiler trace with
  per-kernel CUDA times.
- **Profiling the live server:** `POST /start_profile` with
  `{"num_steps": 5, "activities": ["GPU"], "profile_by_stage": true}` uses `SGLANG_TORCH_PROFILER_DIR`
  (default `/tmp`). With `"activities": ["CUDA_PROFILER"]` it brackets the steps with
  `cudaProfilerStart/Stop` for an `nsys --capture-range cudaProfilerApi` launch of the server
  ([SG/python/sglang/srt/managers/scheduler_components/profiler_manager.py#L278](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/managers/scheduler_components/profiler_manager.py#L278)).
  This shows scheduler overhead, but the batch composition is whatever is in flight, so use it for
  W-run context, not the per-op table.

## Our side on sm_90

- **The PTX target stays `sm_80`** in `gpu/build.sh` and `gpu/spawn/build.py`. The driver JITs it
  on an H100, and contract #3 asks to record that our sm_80 PTX passes there.
- **ptxas checks the PTX for the real GPU.** `build_ptx.py` and `spawn/build.py` take
  `--ptxas-arch`, which `PTXAS_ARCH` sets. It must not be older than the PTX target.
  - `pod_setup.sh` and `pod_validate.sh` derive `PTXAS_ARCH` from `nvidia-smi` (`sm_90` on an H100).
  - Both use `/usr/local/cuda/bin/ptxas` when present.
  - `pod_setup.sh` used to build sm_90 PTX on Hopper while `pod_validate.sh` built sm_80. Both now
    build sm_80.
  - `pod_validate.sh` names its evidence directory after the GPU (`evidence/h100-<date>`).
- **Checked locally with ptxas 13.1 for `-arch=sm_90`:** all seven PTX modules (28 kernels, no
  spills) and all 25 spawn kernels assemble.
- **The tests' `.target sm_80` strings and `minimum_compute_capability=(8, 0)` checks stay.** They
  are about the build target and the minimum device, and an H100 satisfies both.

## Open questions for Session 1

1. Does the Triton fused-MoE kernel have a tuned config for E=256, N=2048, block-FP8 on H100? If
   not, SGLang logs a warning and uses defaults, which would understate SGLang. Record the warning.
   SGLang can tune with `benchmark/kernels/fused_moe_triton`.
2. How long is the DeepGEMM JIT warmup on first start? `SERVER_TIMEOUT` defaults to 40 min; the
   cache sits under `SGLANG_CACHE_DIR`, so put it on `/workspace` to reuse it.
3. Does the pod's nvcc version matter to DeepGEMM/FlashInfer with the cu13 torch? The script warns
   below 13.
4. Is `flashmla_sparse` prefill + `fa3` decode the fastest valid sm_90 pair for BF16 KV? `tilelang`
   is the other valid choice. A one-flag A/B (`--dsa-decode-backend tilelang`) is cheap.
