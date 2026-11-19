# z32 (#63, item 5b's first milestone) on the 8x H200, after z27: ResidentGLM.prefill_chunk on all of GLM-5.3.
#   a. validation/tp_real, z10's options, without and with --prefill (prefill_rows 2048, cuBLASLt rows, NCCL sums):
#      the prompt's greedy tokens and top-5, the first decode step's top-5 after 1000 and 3001 random tokens,
#      and the fill rate (tokens/s) of each context.
#      Then the same with prefill_moe "grouped" (the routed experts through vx-94's grouped block-FP8 GEMM, z33), and with
#      prefill_attention "mha" (kv_b_proj expansion + one causal mha_prefill launch while the context fits index_topk),
#      grouped too if that run passed.
#   b. glm_vx.server --tp 8 with device_decode + prefill (grouped if a's grouped run passed): W1 (TTFT) and GSM8K 200
#      (prefill changes bits).
# Resumable as z27: each step that ends with rc 0 leaves $out/.done-NAME.
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T) during ${name:-setup}"; exit 143' TERM
. ./env.sh
[ -e prefill.tar.gz ] && tar xzf prefill.tar.gz -C repo
# (the package carries vx-94's grouped GEMM, its Python and its PTX with bf16 C)
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache GLM_VX_TP_SPIN=0.05
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
SG=$SGV/bin/python URL=http://127.0.0.1:8000
cd repo
run() {
  name=$1; shift
  if [ -e $out/.done-$name ]; then echo "== $name: done before, skipped"; return 0; fi
  echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; local rc=$?
  echo "$name: rc=$rc"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500
  [ $rc = 0 ] && touch $out/.done-$name; return $rc
}
todo() { for n in "$@"; do [ -e $out/.done-$n ] || return 0; done; echo "== $(basename $out): every step done before, skipped"; return 1; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
server=
stop() {
  [ -n "$server" ] || return 0
  kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; server=; idle
}
PREFILL=(prefill_rows=2048 'prefill_gemm="cublaslt"' 'prefill_sum="nccl"')
idle
out=evidence/h200-20261119-z32a; mkdir -p $out; cp ../z32.sh $out/
# vx-94's chunk scores kernel (index_scores_chunk_at) on one GPU first: values, the causal edge, timing.
run hw-index-chunk env CUDA_VISIBLE_DEVICES=0 GLM_VX_INDEX_CHUNK_JSON=$out/index-chunk.json timeout 1800 \
  $PYTHON -m pytest -q -rs gpu/test_index_chunk_hardware.py
cp gpu/build/index_chunk_hardware.json $out/ 2> /dev/null
base="--option moe_down2=true --option fuse_router=true"
run tp-real-tokens timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 1000 3001 $base \
  --output $out/tp-real-tokens.json
idle
opts=$base; for o in "${PREFILL[@]}"; do opts="$opts --option $o"; done
run tp-real-prefill timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 1000 3001 $opts \
  --prefill --output $out/tp-real-prefill.json
idle
run tp-real-grouped timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 1000 3001 $opts \
  --option 'prefill_moe="grouped"' --prefill --output $out/tp-real-grouped.json
idle
mha="--option prefill_attention=\"mha\""; [ -e $out/.done-tp-real-grouped ] && mha="$mha --option prefill_moe=\"grouped\""
run tp-real-mha timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 1000 3001 $opts \
  $mha --prefill --output $out/tp-real-mha.json
idle
python3 - $out <<'P' | tee $out/compare.txt
import json, sys, pathlib
d = {m: json.load(open(f)) for m in ('tokens', 'prefill', 'grouped', 'mha') for f in [pathlib.Path(sys.argv[1]) / f'tp-real-{m}.json'] if f.exists()}
a = d['tokens']
for name, b in d.items():
    if name == 'tokens' or 'timings' not in b:
        continue
    print('==', name)
    same = next((i for i, (x, y) in enumerate(zip(a['generated'], b['generated'])) if x != y), len(a['generated']))
    print('prompt: tokens equal', a['generated'] == b['generated'], 'leading agree', same)
    for n in a['timings']:
        ta, tb = a['timings'][n], b['timings'][n]
        fa, fb = ta.get('first_top'), tb.get('first_top')
        print(n, 'fill tokens/s', ta['fill_tokens_per_s'], '->', tb['fill_tokens_per_s'],
              'step us', ta['us_per_step_median'], tb['us_per_step_median'],
              'first top-5', fa and fa['ids'], fb and fb['ids'],
              'logit diff', fa and fb and round(max(abs(p - q) for p, q in zip(fa['logits'], fb['logits'])), 4))
P
out=evidence/h200-20261119-z32b; mkdir -p $out; cp ../z32.sh $out/ ../greedy_check.py $out/
topts=""; for o in "${PREFILL[@]}"; do topts="$topts --tp-option $o"; done
[ -e $out/../h200-20261119-z32a/.done-tp-real-grouped ] && topts="$topts --tp-option prefill_moe=\"grouped\""
[ -e $out/../h200-20261119-z32a/.done-tp-real-mha ] && topts="$topts --tp-option prefill_attention=\"mha\""
echo "server prefill options: $topts"
if todo greedy w1 gsm8k-glmvx; then
  setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json --port 8000 \
    --max-context 33280 --token-budget 34000 --prefill-chunk 2048 --tp-option moe_down2=true --tp-option fuse_router=true \
    --tp-option device_decode=true $topts > $out/server.log 2>&1 < /dev/null &
  server=$!
  until curl -sf -m 5 $URL/health > /dev/null; do kill -0 $server 2> /dev/null || { echo "server exited"; tail -n 20 $out/server.log; break; }; sleep 10; done
  if kill -0 $server 2> /dev/null; then
    echo "server up: $(date -u +%T)"
    run greedy $PYTHON ../greedy_check.py $out/../h200-20261118-z5/real-8-oneshot.json $URL
    run w1 env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z32b PROXY=$M ENGINE_URL=$URL ENGINE_ROW=device-decode-prefill \
      CLIENT_WORKLOADS=W1 CLIENT_WARMUP=1 timeout 5400 bash scripts/pod_sglang.sh client
    cp -r $HOME/glm-h200/glmvx-z32b/bench $out/ 2> /dev/null
    run gsm8k-glmvx timeout 7200 $SG benchmarks/gsm8k.py --base-url $URL --output $out/gsm8k-glmvx.json
  fi
fi
stop
echo z32-done
