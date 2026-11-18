# z16b, after z16 on the same H200: (1) item 2's step loop timed without the server (device_loop_time.py: the GPU's
# own time a step against one step ahead and item 1's loop); (2) item 3's one-rank times again, alone on the GPU
# (z16's ran beside a stray server); (3) item 4's speed-of-light probe, fp8mma/fp8_moe_mma.cu (CUDA, not glm-vx:
# mma.sync e4m3 straight from the FP8 weights, one MoE layer at TP8's per-rank shapes).
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/real4
cd repo
out=evidence/h200-20261118-z16; cp ../z16b.sh ../device_loop_time.py $out/; mkdir -p $out/fp8mma; cp ../fp8mma/fp8_moe_mma.cu $out/fp8mma/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 5; echo "gpu idle: $(date -u +%T)"; }
idle
run device-loop timeout 1800 $PYTHON ../device_loop_time.py $M $out/tp-real-1.json $out/device-loop.json
idle
run allreduce-1r-again timeout 900 $PYTHON -m validation.tp_all_reduce_check --ranks 1 --output $out/allreduce-1r-again.json
for b in 8 16 32; do
  run norm-1r-${b}b-again timeout 900 $PYTHON -m validation.tp_all_reduce_check --ranks 1 --norm --norm-blocks $b --output $out/norm-1r-${b}b-again.json
done
idle
run fp8mma timeout 900 ../fp8mma/fp8_moe_mma
echo z16b-done
