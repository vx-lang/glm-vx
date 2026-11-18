# z8: glm-vx's TP8 server, W1 (1000 / 1000, concurrency 1, 10 prompts) under three settings, one server each:
#   a. followers block on their pipe between commands (GLM_VX_TP_SPIN=0: as z7);
#   b. followers poll it for 50 ms after each command (the default now);
#   c. b, and the experts' down projection through fp8_moe_cvt_down2 (--tp-option moe_down2=true: the same bits).
# Then ShareGPT (50 prompts, concurrency 1) and W3 (32768 / 256, 2 prompts) under the fastest of b and c. Every
# server admits a W3 request (--token-budget 34000; the default 8192 refused z7b's).
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
SG=$SGV/bin/python URL=http://127.0.0.1:8000
cd repo
out=evidence/h200-20261118-z8
mkdir -p $out; cp ../z8.sh $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
serve() { name=$1; shift
  setsid env "$@" $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json --port 8000 \
    --max-context 33280 --token-budget 34000 $OPTS > $out/server-$name.log 2>&1 < /dev/null &
  server=$!
  until curl -sf -m 5 $URL/health > /dev/null; do
    kill -0 $server 2> /dev/null || { echo "server-$name exited: $(date -u +%T)"; return 1; }; sleep 10; done
  echo "server-$name up: $(date -u +%T)"; }
stop() { kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; idle; }
w1() { run w1-$1 env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z8 PROXY=$M ENGINE_URL=$URL ENGINE_ROW=$1 CLIENT_WORKLOADS=W1 \
  CLIENT_WARMUP=1 timeout 3600 bash scripts/pod_sglang.sh client; }
until grep -q z7b-done ../z7b.log 2> /dev/null || ! pgrep -f "^bash z7b.sh" > /dev/null; do sleep 30; done
echo "z7b done: $(date -u +%T)"
idle
# fp8_moe_cvt_down2 on the H200: its sm_90 build gives its _ref build's bits.
run hw-down2 $PYTHON -m pytest -q gpu/test_spawn_hardware.py -k "cvt_sm90 and down2"
OPTS=""; serve blocking GLM_VX_TP_SPIN=0 && { w1 blocking; stop; }
OPTS=""; serve spin GLM_VX_TP_SPIN=0.05 && { w1 spin; stop; }
OPTS="--tp-option moe_down2=true"; serve spin-down2 GLM_VX_TP_SPIN=0.05 && w1 spin-down2
cp -r $HOME/glm-h200/glmvx-z8/bench $out/ 2> /dev/null
best=$(python3 - $out/bench <<'P'
import json, sys
from pathlib import Path
t = {}
for name in ('spin', 'spin-down2'):
    f = Path(sys.argv[1]) / f'glm-vx-{name}-W1-vllm.jsonl'
    if f.exists():
        t[name] = json.loads(f.read_text().splitlines()[-1])['median_tpot_ms']
print(min(t, key=t.get) if t else 'spin')
P
)
echo "fastest: $best"
if [ "$best" = spin ]; then stop; OPTS=""; serve spin2 GLM_VX_TP_SPIN=0.05 || exit 1; fi
model=$(curl -s $URL/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
run sharegpt-$best timeout 3600 $SG -m sglang.benchmark.serving --backend vllm --base-url $URL --model $model \
  --tokenizer $M --dataset-name sharegpt --num-prompts 50 --max-concurrency 1 --warmup-requests 4 --seed 42 \
  --output-file $out/sharegpt-$best.jsonl --output-details
run w3-$best env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z8 PROXY=$M ENGINE_URL=$URL ENGINE_ROW=$best CLIENT_WORKLOADS=W3 \
  W3_PROMPTS=2 CLIENT_WARMUP=1 timeout 5400 bash scripts/pod_sglang.sh client
cp -r $HOME/glm-h200/glmvx-z8/bench $out/ 2> /dev/null
stop
echo z8-done
