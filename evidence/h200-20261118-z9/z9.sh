# z9: SGLang's own sglang.test.few_shot_gsm8k against SGLang --tp 8, the check on benchmarks/gsm8k.py (z6: 0.915).
# z6's and z7b's runs passed --host http://127.0.0.1, which the runner prefixes with http:// again (no such host).
set -u
cd ~/glm-h200
export NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
SG=$SGV/bin/python
cd repo
out=evidence/h200-20261118-z9
mkdir -p $out; cp ../z9.sh $out/
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
until grep -q z8-done ../z8.log 2> /dev/null || ! pgrep -f "^bash z8.sh" > /dev/null; do sleep 30; done
echo "z8 done: $(date -u +%T)"
idle
PATH=$PATH:$SGV/bin setsid $SG -m sglang.launch_server --model-path $M --tokenizer-path $M --tp 8 --mem-fraction-static 0.8 \
  --random-seed 0 --kv-cache-dtype bfloat16 --host 127.0.0.1 --port 30000 > $out/sglang-server.log 2>&1 < /dev/null &
server=$!
until curl -sf -m 10 http://127.0.0.1:30000/health_generate > /dev/null; do
  kill -0 $server 2> /dev/null || { echo "sglang exited"; exit 1; }; sleep 10; done
echo "sglang up: $(date -u +%T)"
timeout 3600 $SG -m sglang.test.few_shot_gsm8k --host 127.0.0.1 --port 30000 --num-questions 200 --parallel 1 \
  --data-path $HOME/.cache/glm-vx/gsm8k-test.jsonl > $out/gsm8k-sglang-official.log 2>&1
echo "gsm8k-sglang-official: rc=$?"; grep -E "Accuracy|Invalid|Latency|Output throughput" $out/gsm8k-sglang-official.log
kill -- -$server 2> /dev/null; sleep 20; kill -9 -- -$server 2> /dev/null; idle
echo z9-done
