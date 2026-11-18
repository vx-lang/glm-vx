# z6: SGLang 0.5.22 --tp 8 serving GLM-5.3 on 8x H200, the reportable workloads (glm-vx's side follows once its
# server runs tensor parallel):
#   W1 (1000 in / 1000 out, concurrency 1, 10 prompts) and W3 (32768 / 256, concurrency 1, 8 prompts) through
#   SGLang's own client, sglang.benchmark.serving (scripts/pod_sglang.sh bench: native API, and W1 again through
#   /v1/completions as glm-vx will be measured); ShareGPT at concurrency 1 (50 prompts, /v1/completions);
#   GSM8K 5-shot, first 200 test questions, greedy: benchmarks/gsm8k.py (the client glm-vx will get) and SGLang's
#   own sglang.test.few_shot_gsm8k, whose prompt and scoring it copies, as the check on that client.
# NCCL_NVLS_ENABLE=0: no multicast objects on this VM.
set -u
cd ~/glm-h200
. ./env.sh
export NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
SG=$SGV/bin/python
cd repo
out=evidence/h200-20261118-z6
mkdir -p $out; cp ../z6.sh $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
until grep -q z5-done ../z5b.log 2>/dev/null || ! pgrep -f "^bash z5b.sh" > /dev/null; do sleep 30; done
echo "z5b done: $(date -u +%T)"
idle
# 1. W1, W3: the contract's commands (benchmarks/SGLANG-SETUP.md), --tp 8 appended to its launch flags.
run bench env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/sglang-z6 PROXY=$M MEM_FRACTION=0.8 SGLANG_EXTRA_ARGS="--tp 8" \
  SERVER_TIMEOUT=3600 WORKLOADS="W1 W3" timeout 7200 bash scripts/pod_sglang.sh bench
cp -r $HOME/glm-h200/sglang-z6/bench $HOME/glm-h200/sglang-z6/steps.txt $HOME/glm-h200/sglang-z6/server $out/ 2>/dev/null
idle
# 2. One server for GSM8K and ShareGPT, the launch flags of 1.
PATH=$PATH:$SGV/bin setsid $SG -m sglang.launch_server --model-path $M --tokenizer-path $M --tp 8 --mem-fraction-static 0.8 \
  --random-seed 0 --kv-cache-dtype bfloat16 --host 127.0.0.1 --port 30000 > $out/server.log 2>&1 < /dev/null &
server=$!
until curl -sf -m 10 http://127.0.0.1:30000/health_generate > /dev/null; do
  kill -0 $server 2> /dev/null || { echo "server exited"; tail -n 20 $out/server.log; break; }; sleep 10; done
echo "server up: $(date -u +%T)"
run gsm8k-sglang timeout 3600 $SG benchmarks/gsm8k.py --base-url http://127.0.0.1:30000 --output $out/gsm8k-sglang.json
run gsm8k-sglang-official timeout 3600 $SG -m sglang.test.few_shot_gsm8k --host http://127.0.0.1 --port 30000 --num-questions 200 --parallel 1
run sharegpt-sglang timeout 3600 $SG -m sglang.benchmark.serving --backend sglang-oai --base-url http://127.0.0.1:30000 --model $M \
  --tokenizer $M --dataset-name sharegpt --num-prompts 50 --max-concurrency 1 --warmup-requests 4 --seed 42 \
  --output-file $out/sharegpt-sglang.jsonl --output-details
kill -- -$server 2> /dev/null; sleep 20; kill -9 -- -$server 2> /dev/null
idle
echo z6-done
