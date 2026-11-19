# z35 (#63): the maintainer's check after z32's GSM8K, and nothing else: everything we have enabled at once, the served
# W1 again, to confirm z27d's 8.55 ms TPOT holds with chunked prefill. The server: device_decode, moe_mma, fuse_mla_q,
# moe_down2, fuse_router (not fuse_tp_norm: slower, z27b) and chunked prefill with every piece (prefill_rows 2048,
# cuBLASLt rows, NCCL sums, the routed experts grouped on the device, attention on the tensor cores: mha_tc).
# The package is the prefill branch at 3ca6fb8. z5's prompt's greedy tokens as a sanity check, then W1. Resumable.
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T) during ${name:-setup}"; exit 143' TERM
. ./env.sh
tar xzf prefill-all.tar.gz -C repo
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache GLM_VX_TP_SPIN=0.05
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv URL=http://127.0.0.1:8000
cd repo
run() {
  name=$1; shift
  if [ -e $out/.done-$name ]; then echo "== $name: done before, skipped"; return 0; fi
  echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; local rc=$?
  echo "$name: rc=$rc"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500
  [ $rc = 0 ] && touch $out/.done-$name; return $rc
}
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
server=
stop() {
  [ -n "$server" ] || return 0
  kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null; server=; idle
}
idle
out=evidence/h200-20261119-z35; mkdir -p $out; cp ../z35.sh $out/ ../greedy_check.py $out/
if [ ! -e $out/.done-w1 ]; then
  setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json --port 8000 \
    --max-context 33280 --token-budget 34000 --prefill-chunk 2048 \
    --tp-option moe_down2=true --tp-option fuse_router=true --tp-option device_decode=true --tp-option moe_mma=true \
    --tp-option fuse_mla_q=true --tp-option prefill_rows=2048 --tp-option 'prefill_gemm="cublaslt"' \
    --tp-option 'prefill_sum="nccl"' --tp-option 'prefill_moe="grouped"' --tp-option 'prefill_attention="mha_tc"' \
    > $out/server.log 2>&1 < /dev/null &
  server=$!
  until curl -sf -m 5 $URL/health > /dev/null; do kill -0 $server 2> /dev/null || { echo "server exited"; tail -n 20 $out/server.log; break; }; sleep 10; done
  if kill -0 $server 2> /dev/null; then
    echo "server up: $(date -u +%T)"
    run greedy $PYTHON ../greedy_check.py $out/../h200-20261118-z5/real-8-oneshot.json $URL
    run w1 env SGLANG_VENV=$SGV OUT=$HOME/glm-h200/glmvx-z35 PROXY=$M ENGINE_URL=$URL ENGINE_ROW=everything \
      CLIENT_WORKLOADS=W1 CLIENT_WARMUP=1 timeout 5400 bash scripts/pod_sglang.sh client
    cp -r $HOME/glm-h200/glmvx-z35/bench $out/ 2> /dev/null
  fi
  stop
fi
echo z35-done
