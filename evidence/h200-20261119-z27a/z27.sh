# z27 (#63) on the 8x H200 (e: SGLang W1 on the same VM): what one H200 could not show, with deploy15 (branch tp at f0d42df + every spawn kernel).
#   a. validation/tp_all_reduce_check on 8 GPUs: tp_all_reduce, then gpu/spawn/tp_sum_rmsnorm at 8, 16, 32 blocks (bits
#      on every rank and us a call); B = the fastest that passes. While the page cache fills.
#   b. validation/tp_real on all of GLM-5.3, z10's options (moe_down2, fuse_router) and then + moe_mma, + fuse_tp_norm=B
#      and fuse_mla_q, + all three: tokens, top-5 and the step at 1000 / 3001 tokens. moe_mma changes bits (within 2e-4
#      a layer on one GPU); fuse_tp_norm and fuse_mla_q must not (fused: the same tokens and top-5 as plain).
#   c. glm_vx.server --tp 8 + device_decode (item 2): z5's prompt against z11's tokens (no bits change), W1.
#   d. the server with device_decode + moe_mma + fuse_tp_norm=B + fuse_mla_q: z5's prompt, W1, GSM8K 5-shot 200 (moe_mma changes
#      bits: it must stay in SGLang's 0.915-0.935), ShareGPT 50, W3 (2 prompts).
# Resumable (Nebius preempts: SIGTERM, then 60 s): each step that ends with rc 0 leaves $out/.done-NAME and a rerun
# skips it; a SIGTERM is logged with the step it stopped.
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T) during ${name:-setup}"; exit 143' TERM
. ./env.sh
tar xzf deploy15.tar.gz -C repo
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
serve() {
  setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json --port 8000 \
    --max-context 33280 --token-budget 34000 --tp-option moe_down2=true --tp-option fuse_router=true "$@" \
    > $out/server.log 2>&1 < /dev/null &
  server=$!
  until curl -sf -m 5 $URL/health > /dev/null; do kill -0 $server 2> /dev/null || { echo "server exited"; tail -n 20 $out/server.log; return 1; }; sleep 10; done
  echo "server up: $(date -u +%T)"
}
server=
stop() {
  [ -n "$server" ] || return 0
  kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; server=; idle
}
w() {
  wl=$1 row=$2; shift 2
  run $(echo $wl | tr A-Z a-z) env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-$(basename $out) PROXY=$M ENGINE_URL=$URL \
    ENGINE_ROW=$row CLIENT_WORKLOADS=$wl CLIENT_WARMUP=1 "$@" timeout 5400 bash scripts/pod_sglang.sh client
}
echo "page cache: $(date -u +%T)"; (ls $M/*.safetensors | xargs -P 16 -n 1 cat > /dev/null; echo "cached: $(date -u +%T)") &
cache=$!
idle
out=evidence/h200-20261119-z27a; mkdir -p $out; cp ../z27.sh $out/
run allreduce-tp timeout 900 $PYTHON -m validation.tp_all_reduce_check --ranks 8 --output $out/allreduce-tp.json
for B in 8 16 32; do
  run allreduce-norm-$B timeout 900 $PYTHON -m validation.tp_all_reduce_check --ranks 8 --norm --norm-blocks $B --output $out/allreduce-norm-$B.json
done
python3 - $out <<'P' | tee $out/allreduce-summary.txt
import json, sys, pathlib
rows = []
for f in sorted(pathlib.Path(sys.argv[1]).glob('allreduce-*.json')):
    d = json.loads(f.read_text())
    times = [r.get('us_per_call') for r in d['results'] if r.get('us_per_call') is not None]
    rows.append((f.stem, d['ok'], round(max(times), 2) if times else None))
for row in rows: print(*row)
best = min((r for r in rows if r[0].startswith('allreduce-norm') and r[1] and r[2]), key=lambda r: r[2], default=None)
print('best', best[0].split('-')[-1] if best else 'none')
P
B=$(awk '/^best/ {print $2}' $out/allreduce-summary.txt)
wait $cache
idle
out=evidence/h200-20261119-z27b; mkdir -p $out; cp ../z27.sh $out/
for mode in plain mma fused both; do
  opts="--option moe_down2=true --option fuse_router=true"
  case $mode in mma) opts="$opts --option moe_mma=true";; fused) opts="$opts --option fuse_tp_norm=$B --option fuse_mla_q=true";;
    both) opts="$opts --option moe_mma=true --option fuse_tp_norm=$B --option fuse_mla_q=true";; esac
  [ "$B" = none ] && [ $mode != plain ] && [ $mode != mma ] && continue
  run tp-real-$mode timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 1000 3001 $opts --output $out/tp-real-$mode.json
  idle
done
python3 - $out <<'P' | tee $out/tp-real-compare.txt
import json, sys, pathlib
d = {m: json.load(open(f)) for m in ('plain', 'mma', 'fused', 'both') for f in [pathlib.Path(sys.argv[1]) / f'tp-real-{m}.json'] if f.exists()}
a = d['plain']
for m, b in d.items():
    same = next((i for i, (x, y) in enumerate(zip(a['generated'], b['generated'])) if x != y), len(a['generated']))
    dl = max(abs(p - q) for ta, tb in zip(a['tops'], b['tops']) for p, q in zip(ta['logits'], tb['logits']) if ta['ids'] == tb['ids'])
    print(m, 'tokens equal', a['generated'] == b['generated'], 'leading agree', same, 'max top-5 logit diff', round(dl, 5),
          {n: b['timings'][n]['us_per_step_median'] for n in b['timings']})
P
out=evidence/h200-20261119-z27c; mkdir -p $out; cp ../z27.sh ../greedy_check.py $out/
if todo greedy w1 && serve --tp-option device_decode=true; then
  run greedy $PYTHON ../greedy_check.py $out/../h200-20261118-z5/real-8-oneshot.json $URL
  w W1 device-decode
  cp -r $HOME/glm-h200/glmvx-$(basename $out)/bench $out/ 2> /dev/null
fi
stop
echo z27c-done
out=evidence/h200-20261119-z27d; mkdir -p $out; cp ../z27.sh ../greedy_check.py $out/
# fuse_tp_norm is off: on 8 ranks it keeps the bits but costs ~2 us a call more than the two kernels it replaces
# (z27a: 9.96 us vs 5.07 + ~2.9), and the step +376 us at 1000 tokens (z27b fused vs plain).
fused=""
if todo greedy w1 gsm8k-glmvx sharegpt-glmvx w3 && serve --tp-option device_decode=true --tp-option moe_mma=true --tp-option fuse_mla_q=true $fused; then
  run greedy $PYTHON ../greedy_check.py $out/../h200-20261118-z5/real-8-oneshot.json $URL
  w W1 device-decode-mma-fused
  run gsm8k-glmvx timeout 7200 $SG benchmarks/gsm8k.py --base-url $URL --output $out/gsm8k-glmvx.json
  model=$(curl -s $URL/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  run sharegpt-glmvx timeout 3600 $SG -m sglang.benchmark.serving --backend vllm --base-url $URL --model $model \
    --tokenizer $M --dataset-name sharegpt --num-prompts 50 --max-concurrency 1 --warmup-requests 4 --seed 42 \
    --output-file $out/sharegpt-glmvx.jsonl --output-details
  w W3 device-decode-mma-fused W3_PROMPTS=2
  cp -r $HOME/glm-h200/glmvx-$(basename $out)/bench $out/ 2> /dev/null
fi
stop
echo z27d-done
# e. SGLang --tp 8 W1 on this VM (z6's command), so c and d have a same-machine baseline.
out=evidence/h200-20261119-z27e; mkdir -p $out; cp ../z27.sh $out/
run sglang-w1 env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/sglang-z27e PROXY=$M MEM_FRACTION=0.8 SGLANG_EXTRA_ARGS="--tp 8" \
  SERVER_TIMEOUT=3600 WORKLOADS="W1" timeout 7200 bash scripts/pod_sglang.sh bench
cp -r $HOME/glm-h200/sglang-z27e/bench $HOME/glm-h200/sglang-z27e/steps.txt $HOME/glm-h200/sglang-z27e/server $out/ 2>/dev/null
idle
echo z27-done
