# z7b: what z7 and z6 did not finish. 1. glm-vx's W1 and W3 through scripts/pod_sglang.sh client (z7's run
# stopped at once: SGLANG_CACHE_DIR was not set, so the script made /workspace). 2. SGLang's own
# sglang.test.few_shot_gsm8k against SGLang, the check on benchmarks/gsm8k.py (z6's run could not download the
# questions: a DNS failure), from the copy gsm8k.py cached.
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
SG=$SGV/bin/python URL=http://127.0.0.1:8000
cd repo
out=evidence/h200-20261118-z7
mkdir -p $out; cp ../z7b.sh $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
up() { name=$1 url=$2 pid=$3
  until curl -sf -m 5 $url > /dev/null; do
    kill -0 $pid 2> /dev/null || { echo "$name exited: $(date -u +%T)"; return 1; }; sleep 10; done; echo "$name up: $(date -u +%T)"; }
down() { kill -- -$1 2> /dev/null; for i in $(seq 60); do kill -0 $1 2> /dev/null || break; sleep 2; done; kill -9 -- -$1 2> /dev/null
  pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; idle; }
until grep -q z7-done ../z7.log 2> /dev/null || ! pgrep -f "^bash z7.sh" > /dev/null; do sleep 30; done
echo "z7 done: $(date -u +%T)"
idle
# 1. glm-vx: W1 and W3 (2 prompts), one warmup request each.
setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json --port 8000 \
  --max-context 33280 > $out/server-b.log 2>&1 < /dev/null &
server=$!
if up server-b $URL/health $server; then
  run bench env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z7 PROXY=$M ENGINE_URL=$URL CLIENT_WORKLOADS="W1 W3" W3_PROMPTS=2 \
    CLIENT_WARMUP=1 timeout 10800 bash scripts/pod_sglang.sh client
  cp -r $HOME/glm-h200/glmvx-z7/bench $HOME/glm-h200/glmvx-z7/client $HOME/glm-h200/glmvx-z7/steps.txt $out/ 2> /dev/null
fi
down $server
# 2. SGLang: its own GSM8K runner, the questions from gsm8k.py's cache.
PATH=$PATH:$SGV/bin setsid $SG -m sglang.launch_server --model-path $M --tokenizer-path $M --tp 8 --mem-fraction-static 0.8 \
  --random-seed 0 --kv-cache-dtype bfloat16 --host 127.0.0.1 --port 30000 > $out/sglang-server.log 2>&1 < /dev/null &
server=$!
if up sglang http://127.0.0.1:30000/health_generate $server; then
  run gsm8k-sglang-official timeout 3600 $SG -m sglang.test.few_shot_gsm8k --host http://127.0.0.1 --port 30000 \
    --num-questions 200 --parallel 1 --data-path $HOME/.cache/glm-vx/gsm8k-test.jsonl
fi
down $server
echo z7b-done
