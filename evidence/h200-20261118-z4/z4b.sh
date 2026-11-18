# z4b: z4's NCCL steps again. This VM has NVLink but no multicast objects (NVLS: CUDA error 401 at
# ncclCommInitRank), so NCCL_NVLS_ENABLE=0; and ResidentGLM.close / collective_check close their graphs
# before collective.finish (ncclCommDestroy waits for graphs that hold NCCL operations).
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0
export VX_NCCL_LIB=$(find ~/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
cd repo
out=evidence/h200-20261118-z4b
mkdir -p $out; cp ../z4b.sh $out/
run() { name=$1; shift; echo "== $name: $*"; timeout 2400 "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; tail -n 4 $out/$name.log | cut -c1-400; }
run decode-8 $PYTHON -m validation.tp_decode_check --ranks 8 --output $out/decode-8.json
run decode-8-oneshot $PYTHON -m validation.tp_decode_check --ranks 8 --one-shot --output $out/decode-8-oneshot.json
run collective-8 $PYTHON -m validation.collective_check --ranks 8 --output $out/collective-8.json
run decode-2 $PYTHON -m validation.tp_decode_check --ranks 2 --output $out/decode-2.json
run decode-2-oneshot $PYTHON -m validation.tp_decode_check --ranks 2 --one-shot --output $out/decode-2-oneshot.json
for n in 2 8; do
  rm -f /tmp/vx_peer_id
  for r in $(seq 0 $((n - 1))); do LD_LIBRARY_PATH=$HOME/glm-h200/vx-peer:$LD_LIBRARY_PATH VX_NCCL_ID_FILE=/tmp/vx_peer_id timeout 300 ../vx-peer/vx_peer_buffer_check $r $n & done > $out/vx-peer-$n.log 2>&1
  wait; echo "vx-peer $n:"; cat $out/vx-peer-$n.log
done
echo z4b-done
