# z10: glm-vx's TP8 server, W1 (1000 / 1000, concurrency 1, 10 prompts) with z8's best (followers polling,
# moe_down2) and the router's projection and choice in one launch (--tp-option fuse_router=true:
# gpu/spawn/router_bf16, in place of bf16_matvec4 + router, 9.3 + 5.9 us a MoE layer in z5's profile).
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv
URL=http://127.0.0.1:8000
cd repo
out=evidence/h200-20261118-z10
mkdir -p $out; cp ../z10.sh $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
until grep -q z9-done ../z9.log 2> /dev/null || ! pgrep -f "^bash z9.sh" > /dev/null; do sleep 30; done
echo "z9 done: $(date -u +%T)"
idle
setsid env GLM_VX_TP_SPIN=0.05 $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json \
  --port 8000 --max-context 33280 --token-budget 34000 --tp-option moe_down2=true --tp-option fuse_router=true \
  > $out/server.log 2>&1 < /dev/null &
server=$!
until curl -sf -m 5 $URL/health > /dev/null; do kill -0 $server 2> /dev/null || { echo "server exited"; tail -n 20 $out/server.log; exit 1; }; sleep 10; done
echo "server up: $(date -u +%T)"
run w1-fuse-router env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z10 PROXY=$M ENGINE_URL=$URL ENGINE_ROW=spin-down2-fuse-router \
  CLIENT_WORKLOADS=W1 CLIENT_WARMUP=1 timeout 3600 bash scripts/pod_sglang.sh client
cp -r $HOME/glm-h200/glmvx-z10/bench $out/ 2> /dev/null
kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; idle
echo z10-done
