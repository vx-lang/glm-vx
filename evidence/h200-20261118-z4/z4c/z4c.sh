# z4c: tp_all_reduce v2 (the ranks' flag waits one warp each, all 8 loads before the adds), its bits and
# time; the fixed 8-rank collective check (bf16 values within 256); nsys of an 8-GPU step, NCCL and one-shot.
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0
export VX_NCCL_LIB=$(find ~/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
cd repo
cp ../tpar/tp_all_reduce.ptx ../tpar/tp_all_reduce.json gpu/build/spawn/
out=evidence/h200-20261118-z4c
mkdir -p $out; cp ../z4c.sh $out/
run() { name=$1; shift; echo "== $name: $*"; timeout 2400 "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; tail -n 3 $out/$name.log | cut -c1-500; }
run allreduce-8 $PYTHON -m validation.tp_all_reduce_check --ranks 8 --output $out/allreduce-8.json
run allreduce-2 $PYTHON -m validation.tp_all_reduce_check --ranks 2 --output $out/allreduce-2.json
run allreduce-1gpu $PYTHON -m validation.tp_all_reduce_check --ranks 2 --gpus 1 --time-calls 0 --output $out/allreduce-1gpu.json
run decode-8-oneshot $PYTHON -m validation.tp_decode_check --ranks 8 --one-shot --output $out/decode-8-oneshot.json
run collective-8 $PYTHON -m validation.collective_check --ranks 8 --output $out/collective-8.json
prof() { name=$1; shift
  timeout 2400 nsys profile -f true -o $out/$name --trace=cuda --cuda-graph-trace=node $PYTHON -m validation.tp_decode_check "$@" > $out/$name.log 2>&1
  echo "$name: rc=$?"; tail -n 2 $out/$name.log | cut -c1-400
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name $out/$name.nsys-rep >> $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_kern_sum --format csv -o $out/$name $out/$name.nsys-rep >> $out/$name.log 2>&1; }
prof nsys-8-oneshot --ranks 8 --one-shot --skip-reference --context 4 --time 30
prof nsys-8-nccl --ranks 8 --skip-reference --context 4 --time 30
prof nsys-1 --ranks 1 --context 4 --time 30
echo z4c-done
