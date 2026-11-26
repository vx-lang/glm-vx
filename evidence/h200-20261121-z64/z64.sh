# z64 (#63, TTFT): a long-context check of z63's branch on the real checkpoint, after z63 on the same 8x H200.
# passkey.py (prompts of about 4K-30K tokens, past index_topk, so the DSA indexer selects: the branch computes the
# indexer's weights_proj from bfloat16 rows on cuBLASLt, as SGLang does, where main uses float32 rows) through three
# servers in turn, each fresh: the branch (repo-z63), main 0589bc6 (repo-z61), and SGLang 0.5.22 without speculation
# (z61's launch flags) as the model's own reference. glm-vx with z61's options. Resumable.
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T) during ${name:-setup}"; exit 143' TERM
. ./env.sh
export NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache GLM_VX_TP_SPIN=0.05
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv SG=$HOME/glm-h200/sglang-venv/bin/python
out=$HOME/glm-h200/z64-out
mkdir -p $out; cp z64.sh passkey.py $out/
until grep -q z63-done z63.log 2>/dev/null && ! pgrep -f "^bash z63\.sh" > /dev/null; do sleep 30; done
echo "z63 done: $(date -u +%T)"
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; }
server=
stop() {
  [ -n "$server" ] || return 0
  kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null
  pkill -9 -f "sglang.launch_server --model-path $M" 2> /dev/null; server=; idle; echo "stopped, gpus idle: $(date -u +%T)"
}
up() {  # url health log
  local t=$SECONDS
  until curl -sf -m 10 $1$2 > /dev/null; do
    kill -0 $server 2> /dev/null || { echo "server exited: $(tail -n 5 $3 | tr '\n' ' ' | cut -c1-800)"; return 1; }
    [ $((SECONDS - t)) -gt 3600 ] && { echo "server not up after an hour"; return 1; }; sleep 10
  done
  echo "server up after $((SECONDS - t)) s: $(date -u +%T)"
}
check() {  # engine
  local e=$1 log=$out/$1-server.log
  [ -e $out/.done-$e ] && { echo "== $e: done before"; return 0; }
  idle
  echo "== $e: $(date -u +%T)"
  case $e in
    sglang)
      PATH=$PATH:$SGV/bin setsid $SG -m sglang.launch_server --model-path $M --tokenizer-path $M --tp 8 --mem-fraction-static 0.8 \
        --random-seed 0 --kv-cache-dtype bfloat16 --host 127.0.0.1 --port 30000 > $log 2>&1 < /dev/null &
      server=$! URL=http://127.0.0.1:30000; up $URL /health_generate $log || { stop; return 1; } ;;
    ttft|main)
      local repo=$HOME/glm-h200/repo-z63; [ $e = main ] && repo=$HOME/glm-h200/repo-z61
      (cd $repo && PYTHONPATH=$repo exec setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 \
        --tokenizer $M/tokenizer.json --port 8000 --max-context 33280 --token-budget 34000 --prefill-chunk 2048 \
        --tp-option moe_down2=true --tp-option fuse_router=true --tp-option device_decode=true --tp-option moe_mma=true \
        --tp-option fuse_mla_q=true --tp-option prefill_rows=2048 --tp-option 'prefill_gemm="cublaslt"' \
        --tp-option 'prefill_sum="nccl"' --tp-option 'prefill_moe="grouped"' --tp-option 'prefill_attention="mha_tc"' \
        --tp-option topk_bitmap=true > $log 2>&1 < /dev/null) &
      server=$! URL=http://127.0.0.1:8000; up $URL /health $log || { stop; return 1; } ;;
  esac
  name=$e-passkey
  timeout 7200 $SG passkey.py $URL $M/tokenizer.json $out/$e-passkey.json > $out/$e-passkey.log 2>&1; local rc=$?
  echo "$e-passkey: rc=$rc $(date -u +%T): $(tail -n 1 $out/$e-passkey.log | cut -c1-300)"
  [ $rc = 0 ] && touch $out/.done-$e
  stop
}
for e in ttft main sglang; do check $e; done
echo z64-done
