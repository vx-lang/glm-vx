# z16 (#63, docs/TP-SERVING-PLAN.md items 2 and 3) on one H200 (Nebius, fresh VM), branch tp at 95f7b7a + the
# server's --tp-option at --tp 1. The full GLM-5.3 (756 GB) does not fit one GPU; this runs what one GPU can:
#   1. item 3, gpu/spawn/tp_sum_rmsnorm, bits across ranks on one GPU (validation/tp_all_reduce_check --gpus 1:
#      the ranks are processes that time-slice the GPU, so bits, not time): 8 ranks at 8 blocks, 4 at 16, 2 at
#      32, tp_all_reduce's 8 ranks as the control; then one rank alone (no peer, the local work only): the time
#      of a call of tp_all_reduce and of tp_sum_rmsnorm at 8, 16, 32 blocks.
#   2. GLM-5.3's first 4 trained layers (3 dense, 1 MoE; validation/real_layers fetch) on one GPU with
#      tp_serve's options + moe_down2 + fuse_router (z10's): validation/tp_real --ranks 1, its 32 greedy tokens
#      and the bare step at 1000 / 3001 tokens.
#   3. glm_vx.server --tp 1 with those options (item 1: greedy on the device, the token to the host each step):
#      tp_real's tokens back, and the server's decode time a token at about 1000 and 3000 tokens of context.
#   4. the same + --tp-option device_decode=true (item 2): every prompt's tokens equal to 3's, the decode time.
#   5. the GPU tests: test-resident-graph (the step graphs), test-spawn-hardware.
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/real4 URL=http://127.0.0.1:8000
cd repo
out=evidence/h200-20261118-z16; mkdir -p $out; cp ../z16.sh ../greedy_tokens.py $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 5; echo "gpu idle: $(date -u +%T)"; }
serve() {  # serve NAME <extra args...>: start the server, wait for /health
  name=$1; shift
  setsid $PYTHON -m glm_vx.server --backend vx-gpu-resident --checkpoint $M --tp 1 --tokenizer $M/tokenizer.json --port 8000 \
    --max-context 8192 --token-budget 34000 --tp-option moe_down2=true --tp-option fuse_router=true "$@" \
    > $out/server-$name.log 2>&1 < /dev/null &
  server=$!
  until curl -sf -m 5 $URL/health > /dev/null; do kill -0 $server 2> /dev/null || { echo "server exited"; tail -n 20 $out/server-$name.log; return 1; }; sleep 5; done
  echo "server up: $(date -u +%T)"
}
stop() {
  kill -- -$server 2> /dev/null; for i in $(seq 60); do kill -0 $server 2> /dev/null || break; sleep 2; done
  kill -9 -- -$server 2> /dev/null; idle
}
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv > $out/gpu.txt
echo "repo 95f7b7a + server --tp-option at --tp 1; spawn kernels: Vx 3609 (local build, sm_80 PTX, ptxas sm_90 checked)" >> $out/gpu.txt
rm -rf gpu/build/spawn && mkdir -p gpu/build/spawn && tar xzf ../spawn-95f7.tar.gz -C gpu/build/spawn
echo "spawn kernels: $(find gpu/build/spawn -name "*.ptx" | wc -l); nccl: $VX_NCCL_LIB"

# 1. item 3's kernel.
for rb in 8:8 4:16 2:32; do
  r=${rb%:*} b=${rb#*:}
  run norm-${r}r-${b}b timeout 1800 $PYTHON -m validation.tp_all_reduce_check --ranks $r --gpus 1 --norm --norm-blocks $b \
    --time-calls 0 --output $out/norm-${r}r-${b}b.json
done
run allreduce-8r timeout 1800 $PYTHON -m validation.tp_all_reduce_check --ranks 8 --gpus 1 --time-calls 0 --output $out/allreduce-8r.json
run allreduce-1r timeout 900 $PYTHON -m validation.tp_all_reduce_check --ranks 1 --output $out/allreduce-1r.json
for b in 8 16 32; do
  run norm-1r-${b}b timeout 900 $PYTHON -m validation.tp_all_reduce_check --ranks 1 --norm --norm-blocks $b --output $out/norm-1r-${b}b.json
done
echo item3-done

# 2. the 4 real layers, tp_real on one rank.
until grep -q "GB of tensors" ../fetch4.log; do pgrep -f "validation.real_layers fetch" > /dev/null || { echo "fetch failed"; tail -5 ../fetch4.log; exit 1; }; sleep 10; done
cp ../fetch4.log $M/fetch.json $out/ 2> /dev/null
idle
run tp-real-1 timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 1 --one-shot --contexts 1000 3001 \
  --option moe_down2=true --option fuse_router=true --output $out/tp-real-1.json
idle

# 3. and 4. the server, item 1 then item 2.
if serve item1; then run greedy-item1 $PYTHON ../greedy_tokens.py $URL $out/tp-real-1.json $out/greedy-item1.json; fi
stop
if serve item2 --tp-option device_decode=true; then
  run greedy-item2 $PYTHON ../greedy_tokens.py $URL $out/tp-real-1.json $out/greedy-item2.json $out/greedy-item1.json
fi
stop
echo item2-done

# 5. the GPU tests.
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^test-(resident-graph|spawn-hardware)$' \
  bash scripts/pod_validate.sh $out-tests > ../z16-tests.log 2>&1; echo "tests: $? failed steps"
grep -hE "FAILED|passed|failed" $out-tests/test-*.txt | tail -8
echo z16-done
