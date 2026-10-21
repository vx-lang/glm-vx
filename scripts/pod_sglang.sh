#!/usr/bin/env bash
# SGLang side of the H100 comparison (#22, contract #3): install a pinned SGLang in its own venv,
# serve the single-GPU proxy (#5) with the contract's flags, run W1-W3 with SGLang's own
# bench_serving, capture one per-kernel profile per batch size, and stop the server. The same
# bench_serving commands also run against glm-vx's server (`client`). Background and sources:
# benchmarks/SGLANG-SETUP.md.
#
#   PROXY=/workspace/proxy-fp8 scripts/pod_sglang.sh              # all = install versions bench profile
#   PROXY=... scripts/pod_sglang.sh install|versions|check|bench|profile
#   PROXY=... VARIANT=fp8 scripts/pod_sglang.sh serve             # start one server, wait for Ctrl-C
#   ENGINE_URL=http://127.0.0.1:8000 PROXY=... scripts/pod_sglang.sh client   # glm-vx, same commands
#
# Env:
#   SGLANG_VERSION  pinned release (default 0.5.22); SGLANG_VENV (default /workspace/sglang-venv),
#                   kept apart from the repo's .venv and the Vx toolchain.
#   PROXY           block-FP8 proxy checkpoint dir (validation/write_proxy.py); also the tokenizer
#                   unless TOKENIZER is set.  PROXY_BF16: optional BF16 copy of it (adds the bf16 row;
#                   SGLang cannot run an FP8 checkpoint in BF16, see SGLANG-SETUP.md).
#   VARIANTS        rows to serve (default "fp8", plus "bf16" when PROXY_BF16 is set; "fp8-kvfp8" adds
#                   the FP8-KV row). OUT (default /workspace/sglang-<UTC time>), HOST, PORT (30000).
#   WORKLOADS       default "W1 W2 W3"; CLIENT_WORKLOADS for `client` (default "W1 W3": glm-vx serves
#                   one sequence at a time until #8). WARMUP (bench_serving --warmup-requests, 64 as in
#                   the cookbook), CLIENT_WARMUP (default 2). W3_INPUT (32768), W3_PROMPTS (8).
#   PROFILE_BATCHES ("1 8 64"), PROFILE_INPUTS ("1000"), PROFILER (auto|nsys|torch).
#   MEM_FRACTION (0.85), SERVER_TIMEOUT (s, default 2400: the first start JIT-compiles DeepGEMM),
#   SGLANG_CACHE_DIR (JIT kernel cache, default /workspace/sglang-cache),
#   SGLANG_EXTRA_ARGS (appended to every launch, e.g. "--load-format dummy" to separate loader
#   problems from kernel ones). ENGINE_URL, ENGINE_NAME (glm-vx), ENGINE_MODEL (default: first
#   /v1/models id), CLIENT_BACKEND (bench_serving --backend for glm-vx, default vllm).
#
# Every step logs to OUT/logs/<step>.log and the whole run to OUT/pod_sglang.log; a failed step stops
# the run with the log tail. Benchmark outputs that already exist are kept and skipped, so rerunning
# resumes. Speculative decoding stays off: random MTP weights accept nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

: "${SGLANG_VERSION:=0.5.22}"
: "${SGLANG_VENV:=/workspace/sglang-venv}"
: "${HOST:=127.0.0.1}" "${PORT:=30000}"
: "${OUT:=/workspace/sglang-$(date -u +%Y%m%d-%H%M%S)}"
: "${PROXY:=}" "${PROXY_BF16:=}" "${TOKENIZER:=$PROXY}"
: "${VARIANTS:=fp8${PROXY_BF16:+ bf16}}" "${VARIANT:=fp8}"
: "${WORKLOADS:=W1 W2 W3}" "${CLIENT_WORKLOADS:=W1 W3}"
: "${WARMUP:=64}" "${CLIENT_WARMUP:=2}" "${DATASET:=random}" "${SEED:=42}"
: "${W1_PROMPTS:=10}" "${W2_PROMPTS:=1000}" "${W2_CONCURRENCY:=100}"
: "${W3_INPUT:=32768}" "${W3_OUTPUT:=256}" "${W3_PROMPTS:=8}" "${W3_CONCURRENCY:=1}"
: "${OAI_W1:=1}"
: "${PROFILE_BATCHES:=1 8 64}" "${PROFILE_INPUTS:=1000}" "${PROFILE_OUTPUT:=8}" "${PROFILER:=auto}"
: "${PROFILE_MAX_TOKENS:=262144}"
: "${MEM_FRACTION:=0.85}" "${SERVER_TIMEOUT:=2400}" "${SGLANG_EXTRA_ARGS:=}"
: "${ENGINE_URL:=}" "${ENGINE_NAME:=glm-vx}" "${ENGINE_MODEL:=}" "${CLIENT_BACKEND:=vllm}"
# Compiled DeepGEMM/FlashInfer/Triton kernels land here; on the volume, a restarted pod reuses them.
: "${SGLANG_CACHE_DIR:=/workspace/sglang-cache}"
export SGLANG_CACHE_DIR
PY=$SGLANG_VENV/bin/python
SERVER_PID='' STEP_PID=''

mkdir -p "$OUT"/{logs,bench,server,profile}
exec > >(tee -a "$OUT/pod_sglang.log") 2>&1

log() { echo "[$(date -u +%FT%TZ)] $*"; }
die() { echo "[$(date -u +%FT%TZ)] FATAL: $*" >&2; exit 1; }

# step NAME CMD...: run CMD with its output in OUT/logs/NAME.log; stop the run if it fails.
# CMD runs as a background job so that `set -e` still applies inside shell functions (bash ignores
# -e in anything whose status is tested, such as `if f` or `f || x`).
step() {
  local name=$1 start=$SECONDS code=0
  shift
  log "== $name: $*"
  "$@" > "$OUT/logs/$name.log" 2>&1 < /dev/null &
  STEP_PID=$!
  wait "$STEP_PID" || code=$?
  STEP_PID=
  if [[ $code != 0 ]]; then
    echo "$name FAIL($code) $((SECONDS - start))s" >> "$OUT/steps.txt"
    tail -40 "$OUT/logs/$name.log" >&2
    die "step $name failed (exit $code); full log: $OUT/logs/$name.log"
  fi
  echo "$name ok $((SECONDS - start))s" >> "$OUT/steps.txt"
  log "ok $name ($((SECONDS - start))s)"
}

need_proxy() {
  [[ -n $PROXY ]] || die "set PROXY to the proxy checkpoint directory (validation/write_proxy.py)"
  [[ -f $PROXY/config.json ]] || die "$PROXY/config.json missing"
  [[ -x $PY ]] || die "no SGLang venv at $SGLANG_VENV: run '$0 install' first"
}

# --- environment ---------------------------------------------------------------------------------

check_gpu() {
  command -v nvidia-smi > /dev/null || die "no NVIDIA driver (nvidia-smi missing)"
  local driver cap
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  # SGLang >= 0.5.20 ships only CUDA 13 wheels (torch 2.14 cu130), which need driver R580 or newer.
  [[ ${driver%%.*} -ge 580 ]] || die "driver $driver is too old for CUDA 13 (need >= 580); pick a pod with CUDA 13.0"
  [[ $cap == 9.0 ]] || log "WARNING: compute capability $cap, not 9.0: the contract's SGLang rows are Hopper (sm_90)"
  if [[ -z ${CUDA_HOME:-} && -d /usr/local/cuda ]]; then export CUDA_HOME=/usr/local/cuda; fi
  if [[ -n ${CUDA_HOME:-} && -x $CUDA_HOME/bin/nvcc ]]; then
    export PATH=$CUDA_HOME/bin:$PATH
    local nvcc_version
    nvcc_version=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
    [[ ${nvcc_version%%.*} -ge 13 ]] || log "WARNING: nvcc $nvcc_version; SGLang's torch is CUDA 13 and DeepGEMM/FlashInfer JIT-compile with nvcc"
  elif [[ ${ALLOW_NO_NVCC:-0} != 1 ]]; then
    die "no nvcc (CUDA_HOME=${CUDA_HOME:-unset}): DeepGEMM and FlashInfer JIT-compile with it. Use a CUDA 13 devel image, or ALLOW_NO_NVCC=1 SGL_DG_USE_NVRTC=1 to try NVRTC"
  fi
}

do_install() {
  if [[ -x $PY ]] && "$PY" -I -c "import importlib.metadata as m, sys; sys.exit(m.version('sglang') != '$SGLANG_VERSION')" 2> /dev/null; then
    echo "sglang $SGLANG_VERSION already installed in $SGLANG_VENV"
    return 0
  fi
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "SGLang needs Python >= 3.11; got $(python3 --version)"
  if ! python3 -m venv --help > /dev/null 2>&1 || ! python3 -c 'import ensurepip' 2> /dev/null; then
    command -v apt-get > /dev/null || die "python3-venv missing"
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv
  fi
  [[ -x $PY ]] || python3 -m venv "$SGLANG_VENV"
  "$PY" -m pip install -q --upgrade pip uv
  # --prerelease=allow: some dependencies only publish pre-releases; older uv otherwise falls back
  # to SGLang 0.5.9 without saying so (SGLang install docs).
  "$SGLANG_VENV/bin/uv" pip install --python "$PY" --prerelease=allow "sglang==$SGLANG_VERSION"
  "$PY" -I -c "import importlib.metadata as m, sys; v = m.version('sglang'); print('sglang', v); sys.exit(v != '$SGLANG_VERSION')"
}

do_versions() {
  {
    echo "date=$(date -u +%FT%TZ)"
    echo "glm_vx_git=$(git -C "$REPO" rev-parse HEAD 2> /dev/null || echo "${REPO_REV:-unknown}")"
    echo "vcpus=$(nproc)"
    awk '/MemTotal/{printf "host_ram_gib=%.1f\n",$2/1048576}' /proc/meminfo
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.max.sm --format=csv
    nvcc --version 2> /dev/null | tail -2 || echo "nvcc: none"
    nsys --version 2> /dev/null || echo "nsys: none"
  } | tee "$OUT/host.txt"
  "$PY" -I - "$OUT/versions.json" << 'PYEOF'
import importlib.metadata as m, json, platform, sys
import torch
names = ['sglang', 'sglang-kernel', 'sgl-deep-gemm', 'flashinfer-python', 'torch', 'transformers', 'triton',
         'tilelang', 'flash-attn-4', 'nvidia-cutlass-dsl', 'tokenspeed_mla', 'quack-kernels']
out = {'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
       'cuda_available': torch.cuda.is_available()}
for n in names:
    try:
        out[n] = m.version(n)
    except m.PackageNotFoundError:
        out[n] = None
if torch.cuda.is_available():
    out['device'] = torch.cuda.get_device_name(0)
    out['capability'] = list(torch.cuda.get_device_capability(0))
json.dump(out, open(sys.argv[1], 'w'), indent=2)
print(json.dumps(out, indent=2))
if not out['cuda_available']:
    sys.exit('torch sees no CUDA device')
PYEOF
  "$PY" -m sglang.check_env > "$OUT/check_env.txt" 2>&1 || log "WARNING: sglang.check_env failed (see $OUT/check_env.txt)"
}

# Fail before launching on what SGLang would accept silently: SGLang leaves parameters whose tensors
# are missing uninitialized, and transformers rejects per-layer lists longer than num_hidden_layers.
check_checkpoint() {
  local dir=$1
  "$PY" -I - "$dir" << 'PYEOF'
import json, os, struct, sys
d = sys.argv[1]
c = json.load(open(os.path.join(d, 'config.json')))
problems = []
if c.get('architectures') != ['GlmMoeDsaForCausalLM']:
    problems.append(f"architectures is {c.get('architectures')}, expected ['GlmMoeDsaForCausalLM']")
n = c['num_hidden_layers']
for key in ('indexer_types', 'mlp_layer_types', 'layer_types'):
    if c.get(key) is not None and len(c[key]) != n:
        problems.append(f'{key} has {len(c[key])} entries but num_hidden_layers is {n}')
k = c.get('first_k_dense_replace', 0)
if k > n:
    problems.append(f'first_k_dense_replace {k} > num_hidden_layers {n}')
names = {}
for f in sorted(os.listdir(d)):
    if f.endswith('.safetensors'):
        with open(os.path.join(d, f), 'rb') as h:
            size = struct.unpack('<Q', h.read(8))[0]
            for name, meta in json.loads(h.read(size)).items():
                if name != '__metadata__':
                    names[name] = meta['dtype']
if not names:
    problems.append('no .safetensors files')
fp8 = (c.get('quantization_config') or {}).get('quant_method') == 'fp8'
types = c.get('indexer_types') or ['full'] * n
need = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
quantized = []
for i in range(n):
    p = f'model.layers.{i}.'
    need += [p + 'input_layernorm.weight', p + 'post_attention_layernorm.weight',
             p + 'self_attn.q_a_layernorm.weight', p + 'self_attn.kv_a_layernorm.weight']
    linears = [p + 'self_attn.' + s for s in ('q_a_proj', 'q_b_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj')]
    if types[i] == 'full':
        need += [p + 'self_attn.indexer.k_norm.weight', p + 'self_attn.indexer.weights_proj.weight']
        linears += [p + 'self_attn.indexer.wq_b', p + 'self_attn.indexer.wk']
    if i < k:
        linears += [p + 'mlp.' + s for s in ('gate_proj', 'up_proj', 'down_proj')]
    else:
        need += [p + 'mlp.gate.weight', p + 'mlp.gate.e_score_correction_bias']
        for e in range(c['n_routed_experts']):
            linears += [f'{p}mlp.experts.{e}.{s}' for s in ('gate_proj', 'up_proj', 'down_proj')]
        linears += [p + 'mlp.shared_experts.' + s for s in ('gate_proj', 'up_proj', 'down_proj')]
    need += [l + '.weight' for l in linears]
    quantized += linears
missing = [x for x in need if x not in names]
if fp8:
    missing += [l + '.weight_scale_inv' for l in quantized if names.get(l + '.weight') == 'F8_E4M3'
                and l + '.weight_scale_inv' not in names]
    if not any(v == 'F8_E4M3' for v in names.values()):
        problems.append('quantization_config says fp8 but no F8_E4M3 tensor')
elif any(v == 'F8_E4M3' for v in names.values()):
    problems.append('F8_E4M3 tensors without an fp8 quantization_config')
layers = {int(x.split('.')[2]) for x in names if x.startswith('model.layers.')}
extra = sorted(l for l in layers if l >= n)
if missing:
    problems.append(f'{len(missing)} tensors SGLang needs are missing, e.g. {missing[:5]}')
print(f'{d}: {n} layers ({k} dense), {len(names)} tensors, fp8={fp8}, layers beyond num_hidden_layers '
      f'(MTP, skipped with speculation off): {extra}')
if problems:
    sys.exit('checkpoint problems:\n  ' + '\n  '.join(problems))
PYEOF
}

# --- server ----------------------------------------------------------------------------------------

variant_ckpt() {
  case $1 in
    fp8 | fp8-kvfp8) echo "$PROXY" ;;
    bf16) [[ -n $PROXY_BF16 ]] || die "variant bf16 needs PROXY_BF16"; echo "$PROXY_BF16" ;;
    *) die "unknown variant $1 (fp8, fp8-kvfp8, bf16)" ;;
  esac
}

# Contract flags: one GPU, no speculation (the default), the checkpoint's own precision. The KV dtype
# is explicit so the row label cannot drift: SGLang's auto picks bfloat16 KV on sm_90 for DSA.
variant_args() {
  case $1 in
    fp8 | bf16) VARGS=(--kv-cache-dtype bfloat16) ;;
    fp8-kvfp8) VARGS=(--kv-cache-dtype fp8_e4m3) ;;
  esac
}

launch_args() {
  local variant=$1 ckpt
  ckpt=$(variant_ckpt "$variant")
  variant_args "$variant"
  # shellcheck disable=SC2206  # SGLANG_EXTRA_ARGS is a flag list by design
  LAUNCH=(--model-path "$ckpt" --tokenizer-path "$TOKENIZER" --tp 1 --mem-fraction-static "$MEM_FRACTION"
          --random-seed 0 "${VARGS[@]}" $SGLANG_EXTRA_ARGS)
}

start_server() {
  local variant=$1 logf=$OUT/server/$1.log
  launch_args "$variant"
  [[ -z $SERVER_PID ]] || die "a server is already running (pid $SERVER_PID)"
  if curl -s -m 2 "http://$HOST:$PORT/health" > /dev/null 2>&1; then die "something already listens on $HOST:$PORT"; fi
  check_checkpoint "$(variant_ckpt "$variant")" > "$OUT/server/$variant-checkpoint.txt" 2>&1 \
    || { cat "$OUT/server/$variant-checkpoint.txt" >&2; die "checkpoint check failed for $variant"; }
  cat "$OUT/server/$variant-checkpoint.txt"
  echo "$PY -m sglang.launch_server ${LAUNCH[*]} --host $HOST --port $PORT" | tee "$OUT/server/$variant.cmd"
  # Its own session, so stop_server can signal the whole process group (scheduler, detokenizer).
  setsid "$PY" -m sglang.launch_server "${LAUNCH[@]}" --host "$HOST" --port "$PORT" > "$logf" 2>&1 < /dev/null &
  SERVER_PID=$!
  log "server $variant: pid $SERVER_PID, log $logf"
  local started=$SECONDS deadline=$((SECONDS + SERVER_TIMEOUT))
  until curl -sf -m 10 "http://$HOST:$PORT/health_generate" > /dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2> /dev/null; then
      tail -60 "$logf" >&2; SERVER_PID=; die "SGLang exited during startup ($variant); log: $logf"
    fi
    if ((SECONDS > deadline)); then tail -60 "$logf" >&2; die "SGLang not healthy after ${SERVER_TIMEOUT}s ($variant)"; fi
    sleep 10
  done
  log "server $variant healthy after $((SECONDS - started))s"
  curl -sf "http://$HOST:$PORT/get_server_info" > "$OUT/server/$variant-server_info.json"
  curl -sf "http://$HOST:$PORT/get_model_info" > "$OUT/server/$variant-model_info.json"
  # What SGLang resolved on this GPU: DSA backends, KV dtype, quantization, GEMM/MoE backends.
  grep -E "DSA backends|KV cache dtype|page size|quantiz|deep_gemm|DeepGEMM|moe_runner|fp8_gemm|attention backend|not found in params_dict" \
    "$logf" > "$OUT/server/$variant-resolved.txt" || true
  "$PY" -I - "$OUT/server/$variant-server_info.json" << 'PYEOF'
import json, sys
info = json.load(open(sys.argv[1]))
keys = ['version', 'tp_size', 'quantization', 'kv_cache_dtype', 'attention_backend', 'dsa_prefill_backend',
        'dsa_decode_backend', 'dsa_topk_backend', 'moe_runner_backend', 'fp8_gemm_runner_backend', 'page_size',
        'speculative_algorithm', 'disable_radix_cache', 'cuda_graph_max_bs', 'chunked_prefill_size',
        'max_total_num_tokens', 'context_length']
print({k: info.get(k) for k in keys if k in info})
if info.get('speculative_algorithm'):
    sys.exit(f"speculative decoding is on ({info['speculative_algorithm']}); the contract runs it off")
PYEOF
  # Two greedy requests: the contract's determinism gate on random weights (recorded, not enforced:
  # the second one may hit the radix cache and take another kernel path).
  local body='{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}'
  curl -sf -H 'Content-Type: application/json' -d "$body" "http://$HOST:$PORT/generate" > "$OUT/server/$variant-greedy-1.json"
  curl -sf -H 'Content-Type: application/json' -d "$body" "http://$HOST:$PORT/generate" > "$OUT/server/$variant-greedy-2.json"
  if cmp -s <("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["text"])' "$OUT/server/$variant-greedy-1.json") \
            <("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["text"])' "$OUT/server/$variant-greedy-2.json"); then
    echo "deterministic=yes" > "$OUT/server/$variant-greedy.txt"
  else
    echo "deterministic=no" > "$OUT/server/$variant-greedy.txt"; log "WARNING: two greedy requests differ ($variant)"
  fi
}

stop_server() {
  [[ -n $SERVER_PID ]] || return 0
  log "stopping server pid $SERVER_PID"
  # SIGTERM: launch_server tears down its scheduler/detokenizer processes on it. (SIGINT may be
  # ignored: a background job of a non-interactive shell starts with SIGINT ignored.)
  kill -TERM -- "-$SERVER_PID" 2> /dev/null || true
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2> /dev/null || break; sleep 1; done
  if kill -0 "$SERVER_PID" 2> /dev/null; then
    log "WARNING: server still up 60 s after SIGTERM; killing its process group"
    kill -KILL -- "-$SERVER_PID" 2> /dev/null || true
  fi
  wait "$SERVER_PID" 2> /dev/null || true
  SERVER_PID=
  for _ in $(seq 60); do
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null) ]] && return 0
    sleep 1
  done
  nvidia-smi >&2
  die "GPU memory still held after the server stopped"
}
trap stop_server EXIT
# Background jobs ignore SIGINT in a non-interactive shell, so pass Ctrl-C on to the running step.
on_signal() { [[ -z $STEP_PID ]] || kill -TERM "$STEP_PID" 2> /dev/null || true; die "interrupted"; }
trap on_signal INT TERM

# --- benchmarks ------------------------------------------------------------------------------------

bench_module() {
  if "$PY" -I -c 'import sglang.benchmark.serving' 2> /dev/null; then echo sglang.benchmark.serving; else echo sglang.bench_serving; fi
}

# bench_run ENGINE ROW WORKLOAD BACKEND URL MODEL WARMUP [extra bench_serving flags...]
bench_run() {
  local engine=$1 row=$2 wl=$3 backend=$4 url=$5 model=$6 warmup=$7 in out n c
  shift 7
  case $wl in
    W1) in=1000 out=1000 n=$W1_PROMPTS c=1 ;;
    W2) in=1000 out=1000 n=$W2_PROMPTS c=$W2_CONCURRENCY ;;
    W3) in=$W3_INPUT out=$W3_OUTPUT n=$W3_PROMPTS c=$W3_CONCURRENCY ;;
    *) die "unknown workload $wl" ;;
  esac
  local name=$engine-$row-$wl-$backend
  local file=$OUT/bench/$name.jsonl
  if [[ -s $file ]]; then log "skip $name: $file exists"; return 0; fi
  # --random-range-ratio 1.0 fixes the lengths (the default 0.0 draws them from [1, len]); the
  # cookbook's reproduce command uses the same, with --warmup-requests 64 and --flush-cache.
  step "bench-$name" "$PY" -m "$(bench_module)" --backend "$backend" --base-url "$url" --model "$model" \
    --tokenizer "$TOKENIZER" --dataset-name "$DATASET" --random-input-len "$in" --random-output-len "$out" \
    --random-range-ratio 1.0 --num-prompts "$n" --max-concurrency "$c" --warmup-requests "$warmup" \
    --seed "$SEED" --tag "$name" --output-file "$file" --output-details "$@"
  tail -25 "$OUT/logs/bench-$name.log"
}

do_bench() {
  need_proxy
  check_gpu
  local row wl
  for row in $VARIANTS; do
    start_server "$row"
    local url=http://$HOST:$PORT model
    model=$(variant_ckpt "$row")
    for wl in $WORKLOADS; do
      bench_run sglang "$row" "$wl" sglang "$url" "$model" "$WARMUP" --flush-cache
    done
    # glm-vx is measured through /v1/completions, so W1 also goes through SGLang's OpenAI endpoint.
    if [[ $OAI_W1 == 1 && " $WORKLOADS " == *" W1 "* ]]; then
      bench_run sglang "$row" W1 sglang-oai "$url" "$model" "$WARMUP" --flush-cache
    fi
    stop_server
  done
}

do_client() {
  [[ -n $ENGINE_URL ]] || die "set ENGINE_URL to glm-vx's server, e.g. http://127.0.0.1:8000"
  [[ -n $TOKENIZER ]] || die "set TOKENIZER (or PROXY) to the directory with tokenizer.json"
  [[ -x $PY ]] || die "no SGLang venv at $SGLANG_VENV: run '$0 install' first"
  local dir=$OUT/client
  mkdir -p "$dir"
  curl -sf -m 10 "$ENGINE_URL/health" > "$dir/$ENGINE_NAME-health.txt" || die "$ENGINE_URL/health does not answer"
  curl -sf -m 10 "$ENGINE_URL/v1/models" > "$dir/$ENGINE_NAME-models.json" || die "$ENGINE_URL/v1/models does not answer (bench_serving needs it)"
  local model=$ENGINE_MODEL
  [[ -n $model ]] || model=$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$dir/$ENGINE_NAME-models.json")
  echo "engine=$ENGINE_NAME url=$ENGINE_URL model=$model backend=$CLIENT_BACKEND" | tee "$dir/$ENGINE_NAME.txt"
  local wl
  for wl in $CLIENT_WORKLOADS; do
    # No --flush-cache: it posts to an SGLang/vLLM-only endpoint.
    bench_run "$ENGINE_NAME" "${ENGINE_ROW:-run}" "$wl" "$CLIENT_BACKEND" "$ENGINE_URL" "$model" "$CLIENT_WARMUP"
  done
}

# --- profiles --------------------------------------------------------------------------------------

find_nsys() {
  command -v nsys 2> /dev/null && return 0
  local f
  for f in /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda/nsight-systems-*/bin/nsys; do
    [[ -x $f ]] && { echo "$f"; return 0; }
  done
  if command -v apt-get > /dev/null; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems-cli > /dev/null 2>&1 \
      || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-nsight-systems-$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | cut -d' ' -f2 | tr . -)" > /dev/null 2>&1 || true
    command -v nsys 2> /dev/null && return 0
    for f in /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys; do [[ -x $f ]] && { echo "$f"; return 0; }; done
  fi
  return 1
}

one_batch_module() {
  if "$PY" -I -c 'import sglang.benchmark.one_batch' 2> /dev/null; then echo sglang.benchmark.one_batch; else echo sglang.bench_one_batch; fi
}

# profile_one ROW STAGE BATCH INPUT: one static batch through SGLang's model runner (no HTTP server,
# CUDA graphs as in serving), traced for one prefill or one decode step.
profile_one() {
  local row=$1 stage=$2 bs=$3 in=$4 nsys=$5
  local name=sglang-$row-$stage-bs$bs-in$in
  local base=$OUT/profile/$name
  if [[ -s $base.nsys-rep || -s $base.done ]]; then log "skip $name: exists"; return 0; fi
  launch_args "$row"
  # Decode: trace step PROFILE_OUTPUT-2, the last one. With CUDA_PROFILER, one_batch never calls
  # cudaProfilerStop after decode (its stop is gated on a torch profiler object), so the capture runs
  # to process exit; starting at the last step keeps it to that step.
  local args=(-m "$(one_batch_module)" "${LAUNCH[@]}" --batch-size "$bs" --input-len "$in"
              --output-len "$PROFILE_OUTPUT" --profile --profile-stage "$stage"
              --profile-start-step $((PROFILE_OUTPUT - 2)) --run-name "$name"
              --result-filename "$OUT/profile/one_batch.jsonl")
  if [[ -n $nsys ]]; then
    step "profile-$name" "$nsys" profile --force-overwrite true -o "$base" --trace cuda,nvtx \
      --cuda-graph-trace node --capture-range cudaProfilerApi --capture-range-end stop --sample none \
      --cpuctxsw none "$PY" "${args[@]}" --profile-activities CUDA_PROFILER
    step "stats-$name" "$nsys" stats --force-export true --report cuda_gpu_kern_sum --format csv \
      --output "$base" "$base.nsys-rep"
  else
    mkdir -p "$OUT/profile/torch"
    step "profile-$name" env SGLANG_TORCH_PROFILER_DIR="$OUT/profile/torch" "$PY" "${args[@]}" \
      --profile-activities CPU GPU --profile-prefix "$name"
    touch "$base.done"
  fi
}

do_profile() {
  need_proxy
  check_gpu
  [[ -z $SERVER_PID ]] || die "stop the server first: the profile loads its own copy of the model"
  local nsys=
  case $PROFILER in
    nsys) nsys=$(find_nsys) || die "PROFILER=nsys but no nsys found or installable" ;;
    auto) nsys=$(find_nsys) || log "WARNING: no nsys; falling back to the torch profiler (per-kernel trace.json.gz)" ;;
    torch) ;;
    *) die "PROFILER must be auto, nsys or torch" ;;
  esac
  [[ -z $nsys ]] || "$nsys" --version | tee "$OUT/profile/nsys-version.txt"
  local row in bs
  for row in $VARIANTS; do
    for in in $PROFILE_INPUTS; do
      profile_one "$row" prefill 1 "$in" "$nsys"
      for bs in $PROFILE_BATCHES; do
        if ((bs * in > PROFILE_MAX_TOKENS)); then log "skip decode bs$bs in$in: over PROFILE_MAX_TOKENS"; continue; fi
        profile_one "$row" decode "$bs" "$in" "$nsys"
      done
    done
  done
}

do_serve() {
  need_proxy
  check_gpu
  start_server "$VARIANT"
  log "serving $VARIANT on http://$HOST:$PORT; Ctrl-C stops it"
  wait "$SERVER_PID" || true
}

cmd=${1:-all}
log "pod_sglang $cmd: SGLang $SGLANG_VERSION, venv $SGLANG_VENV, out $OUT"
case $cmd in
  install) check_gpu; step install do_install ;;
  versions) step versions do_versions ;;
  check) need_proxy; for row in $VARIANTS; do step "check-$row" check_checkpoint "$(variant_ckpt "$row")"; cat "$OUT/logs/check-$row.log"; done ;;
  bench) do_bench ;;
  profile) do_profile ;;
  serve) do_serve ;;
  client) do_client ;;
  all)
    check_gpu
    step install do_install
    step versions do_versions
    do_bench
    do_profile
    ;;
  *) die "usage: $0 [all|install|versions|check|bench|profile|serve|client]" ;;
esac
log "done: $OUT"
cat "$OUT/steps.txt" 2> /dev/null || true
