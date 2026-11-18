# z7: glm-vx's server, tensor parallel over the 8 H200 (glm_vx.server --tp 8: rank 0 serves, ranks 1-7 replay its
# steps; gpu/tp_serve.py), on GLM-5.3, through the clients and workloads z6 ran against SGLang:
#   1. smoke: the first 4 layers only, one request;
#   2. the whole model: greedy tokens through HTTP against tp_real's (z5, the same kernels), then W1 (1000/1000,
#      concurrency 1, 10 prompts) and W3 (32768/256, concurrency 1; 2 prompts, the first two of z6's 8 with the
#      same seed: our prefill runs one token a step, ~5 min a W3 prompt) through scripts/pod_sglang.sh client
#      (sglang.benchmark.serving, /v1/completions); GSM8K 5-shot 200 with benchmarks/gsm8k.py; ShareGPT 50 at
#      concurrency 1 with the flags of z6. One warmup request a workload (a W3 warmup is a 32768-token prefill).
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
SG=$SGV/bin/python URL=http://127.0.0.1:8000
cd repo
out=evidence/h200-20261118-z7
mkdir -p $out; cp ../z7.sh $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
serve() { name=$1; shift
  setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json --port 8000 "$@" \
    > $out/$name.log 2>&1 < /dev/null &
  server=$!
  until curl -sf -m 5 $URL/health > /dev/null; do
    kill -0 $server 2> /dev/null || { echo "$name exited: $(date -u +%T)"; tail -n 20 $out/$name.log | cut -c1-400; return 1; }; sleep 10; done
  echo "$name up: $(date -u +%T)"; curl -s -m 5 $URL/health; echo; }
stop() { kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; idle; }
until grep -q z6-done ../z6.log 2> /dev/null || ! pgrep -f "^bash z6.sh" > /dev/null; do sleep 30; done
echo "z6 done: $(date -u +%T)"
idle
# 1. Smoke: 4 layers, one request of token ids.
if serve server-smoke --layers 4 --max-context 2048; then
  run smoke curl -s -m 600 $URL/v1/completions -H 'Content-Type: application/json' \
    -d '{"prompt": [791, 3925, 315, 279, 1380, 380, 1563], "max_tokens": 8, "temperature": 0, "ignore_eos": true}'
  stop
fi
# 2. The whole model; the cache holds W3's 32768 + 256.
serve server --max-context 33280 || exit 1
run greedy $PYTHON - $out/../h200-20261118-z5/real-8-oneshot.json $URL <<'P'
import json, sys, urllib.request
ours = json.load(open(sys.argv[1]))
body = dict(prompt=ours['prompt_ids'], max_tokens=len(ours['generated']), temperature=0, ignore_eos=True)
request = urllib.request.Request(sys.argv[2] + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
reply = json.load(urllib.request.urlopen(request, timeout=1800))
got = reply['choices'][0]['token_ids']
print(json.dumps(dict(same=got == ours['generated'], served=got, tp_real=ours['generated'], text=reply['choices'][0]['text'],
                      timings=reply.get('timings'))))
P
run bench env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z7 PROXY=$M ENGINE_URL=$URL CLIENT_WORKLOADS="W1 W3" W3_PROMPTS=2 CLIENT_WARMUP=1 \
  timeout 10800 bash scripts/pod_sglang.sh client
cp -r $HOME/glm-h200/glmvx-z7/bench $HOME/glm-h200/glmvx-z7/client $out/ 2> /dev/null
model=$(curl -s $URL/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
run gsm8k-glmvx timeout 7200 $SG benchmarks/gsm8k.py --base-url $URL --output $out/gsm8k-glmvx.json
run sharegpt-glmvx timeout 7200 $SG -m sglang.benchmark.serving --backend vllm --base-url $URL --model $model \
  --tokenizer $M --dataset-name sharegpt --num-prompts 50 --max-concurrency 1 --warmup-requests 4 --seed 42 \
  --output-file $out/sharegpt-glmvx.jsonl --output-details
stop
echo z7-done
