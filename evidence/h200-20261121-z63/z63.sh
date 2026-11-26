# z63 (#63, TTFT): branch ttft-bf16-gemm end to end, all 78 layers of the real checkpoint, served --tp 8 with z61's glm-vx
# options, on the same 8x H200 right after z61: the router's gate and the indexer's weights_proj over prefill_chunk's
# rows as cuBLASLt's bfloat16 matmul (origin/main 0589bc6 + that change; every kernel built by Vx main 10c6f478f).
# 3 rounds, each a fresh server: z5's greedy check, W1, ShareGPT, W3 (8 prompts) through z61's client; GSM8K in round 1.
# Compared against z61's 3 rounds of main on this VM earlier the same day (not alternated with them). Resumable.
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T) during ${name:-setup}"; exit 143' TERM
. ./env.sh
export NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache GLM_VX_TP_SPIN=0.05
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3 SGV=$HOME/glm-h200/sglang-venv SG=$HOME/glm-h200/sglang-venv/bin/python
R=$HOME/glm-h200/repo-z63 out=$HOME/glm-h200/z63-out REF=$HOME/glm-h200/repo/evidence/h200-20261118-z5/real-8-oneshot.json
mkdir -p $out; cp z63.sh greedy_check.py $out/
if [ ! -e $R/.ready ]; then
  rm -rf $R && mkdir $R && tar xzf z63-ttft.tar.gz -C $R && ln -s $HOME/glm-h200/repo/.venv $R/.venv && touch $R/.ready \
    || { echo "unpack failed"; echo z63-done; exit 1; }
fi
export PYTHONPATH=$R
cd $R  # every server and client runs from the package (z35 ran from repo/)
if [ ! -e $out/env.txt ]; then
  { date -u; hostname; uname -r; nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv
    $SG -c 'import sglang, torch; print("sglang", sglang.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
    $PYTHON -c 'import sys; print("glm-vx python", sys.version)'; echo "glm-vx ttft-bf16-gemm (origin/main 0589bc6 + the BF16 Linears over the rows on cuBLASLt), vxc Vx main 10c6f478f"
    echo "VX_NCCL_LIB=$VX_NCCL_LIB"; nvidia-smi topo -m; } > $out/env.txt 2>&1
fi
run() {  # name command...: once, unless its .done marker exists
  name=$1; shift
  if [ -e $out/.done-$name ]; then echo "== $name: done before, skipped"; return 0; fi
  echo "== $name: $(date -u +%T)"; "$@" > $out/$name.log 2>&1; local rc=$?
  echo "$name: rc=$rc $(date -u +%T): $(tail -n 2 $out/$name.log | cut -c1-600 | tr '\n' ' ')"
  [ $rc = 0 ] && touch $out/.done-$name; return $rc
}
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; }
server=
stop() {
  [ -n "$server" ] || return 0
  kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; pkill -9 -f "gpu.tp_serve --checkpoint $M" 2> /dev/null
  pkill -9 -f "sglang.launch_server --model-path $M" 2> /dev/null; server=; idle; echo "stopped, gpus idle: $(date -u +%T)"
}
up() {  # url health log: wait for the server, at most an hour
  local t=$SECONDS
  until curl -sf -m 10 $1$2 > /dev/null; do
    kill -0 $server 2> /dev/null || { echo "server exited: $(tail -n 5 $3 | tr '\n' ' ' | cut -c1-800)"; return 1; }
    [ $((SECONDS - t)) -gt 3600 ] && { echo "server not up after an hour"; return 1; }; sleep 10
  done
  echo "server up after $((SECONDS - t)) s: $(date -u +%T)"
}
serve() {  # engine tag
  local e=$1 tag=$2 log=$out/$2-server.log
  idle
  case $e in
    sglang|nextn)
      local spec=; [ $e = nextn ] && spec="--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
      PATH=$PATH:$SGV/bin setsid $SG -m sglang.launch_server --model-path $M --tokenizer-path $M --tp 8 --mem-fraction-static 0.8 \
        --random-seed 0 --kv-cache-dtype bfloat16 $spec --host 127.0.0.1 --port 30000 > $log 2>&1 < /dev/null &
      server=$! URL=http://127.0.0.1:30000; up $URL /health_generate $log || { stop; return 1; }
      curl -sf $URL/get_server_info > $out/$tag-server_info.json ;;
    glmvx|vx16)
      local mma=true; [ $e = vx16 ] && mma='"vx16"'
      setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 8 --tokenizer $M/tokenizer.json \
        --port 8000 --max-context 33280 --token-budget 34000 --prefill-chunk 2048 \
        --tp-option moe_down2=true --tp-option fuse_router=true --tp-option device_decode=true --tp-option moe_mma=$mma \
        --tp-option fuse_mla_q=true --tp-option prefill_rows=2048 --tp-option 'prefill_gemm="cublaslt"' \
        --tp-option 'prefill_sum="nccl"' --tp-option 'prefill_moe="grouped"' --tp-option 'prefill_attention="mha_tc"' \
        --tp-option topk_bitmap=true > $log 2>&1 < /dev/null &
      server=$! URL=http://127.0.0.1:8000; up $URL /health $log || { stop; return 1; } ;;
  esac
  model=$(curl -sf -m 10 $URL/v1/models | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])')
  echo "$tag: model id $model"
}
random_wl() {  # engine tag workload
  env SGLANG_VENV=$SGV OUT=$out/$2 PROXY=$M ENGINE_URL=$URL ENGINE_NAME=$1 ENGINE_ROW=$2 CLIENT_WORKLOADS=$3 CLIENT_WARMUP=1 \
    W3_PROMPTS=8 timeout 7200 bash scripts/pod_sglang.sh client
}
sharegpt() {  # tag
  timeout 7200 $SG -m sglang.benchmark.serving --backend vllm --base-url $URL --model "$model" --tokenizer $M \
    --dataset-name sharegpt --num-prompts 50 --max-concurrency 1 --warmup-requests 4 --seed 42 \
    --output-file $out/$1-sharegpt.jsonl --output-details
}
one() {  # round engine [gsm8k]
  local r=$1 e=$2 tag=r$1-$2 steps="w1 sharegpt w3"
  [ $e = vx16 ] && steps="w1"
  case $e in glmvx|vx16) steps="greedy $steps" ;; esac
  [ -n "${3:-}" ] && steps="$steps gsm8k"
  local todo= s; for s in $steps; do [ -e $out/.done-$tag-$s ] || todo="$todo $s"; done
  [ -z "$todo" ] && { echo "== $tag: done before"; return 0; }
  echo "== $tag:$todo, starting $(date -u +%T)"
  serve $e $tag || { echo "$tag: no server"; return 1; }
  for s in $todo; do
    case $s in
      greedy) run $tag-greedy $PYTHON $HOME/glm-h200/greedy_check.py $REF $URL ;;
      w1) run $tag-w1 random_wl $e $tag W1 ;;
      w3) run $tag-w3 random_wl $e $tag W3 ;;
      sharegpt) run $tag-sharegpt sharegpt $tag ;;
      gsm8k) run $tag-gsm8k timeout 7200 $SG benchmarks/gsm8k.py --base-url $URL --output $out/$tag-gsm8k.json ;;
    esac
  done
  stop
}
one 1 glmvx gsm8k
one 2 glmvx
one 3 glmvx
echo "done steps: $(ls -a $out | grep -c '^\.done')"; ls -a $out | grep '^\.done' | tr '\n' ' '; echo
echo z63-done
