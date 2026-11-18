# H200 session z4 (glm-vx #62): tensor parallelism on a fresh 8xH200 (Nebius).
# Local first: TC=6a1d8f47 pod_go.sh HOST setup   (branch tp, 46cefa3 -> ~/glm-h200/repo, vxtc, venv, env.sh)
# then copy spawn-tp.tar.gz (every spawn kernel, Vx 3609, sm_80 PTX), vx-peer.tar.gz (Vx PR #3619 @ c6b9b6b3a:
# runtime/, include/, scripts/tools) and z4.sh to ~/glm-h200, and start: setsid -f nohup bash z4.sh > z4.log 2>&1
# Steps in value order (Nebius is preemptible): build, the 8-rank checks, 2-rank, the GPU tests, Vx's peer check.
set -u
cd ~/glm-h200
for f in env.sh repo spawn-tp.tar.gz; do [ -e $f ] || { echo "missing $f"; exit 2; }; done
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo
nvidia-smi -L; nvidia-smi topo -m
G=$(nvidia-smi -L | wc -l)
$PYTHON -m pip install -q nvidia-nccl-cu12
export VX_NCCL_LIB=${VX_NCCL_LIB:-$(find ~/glm-h200/repo/.venv /usr /opt -name 'libnccl.so.2' 2>/dev/null | head -1)}
echo "VX_DISPATCH_LIB=$VX_DISPATCH_LIB VX_NCCL_LIB=$VX_NCCL_LIB GPUs=$G"
cd repo
day=$(date -u +%Y%m%d)
out=evidence/h200-$day-z4
# 0. Hand kernels (gpu/build.sh) and cuBLASLt; then the spawn kernels from spawn-tp (tp_all_reduce needs
#    -fweak-relaxed-atomics, newer than this toolchain's vxc). pod_validate also installs the cu12 runtime wheels.
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^build-(ptx|cublaslt)$' bash scripts/pod_validate.sh $out-build > ../z4-build.log 2>&1; echo "build: $? failed steps"
tail -n 5 ../z4-build.log
rm -rf gpu/build/spawn && mkdir -p gpu/build/spawn && tar xzf ../spawn-tp.tar.gz -C gpu/build/spawn
echo "spawn kernels: $(find gpu/build/spawn -name '*.ptx' | wc -l)"
mkdir -p $out
cp ../z4.sh $out/
nvidia-smi topo -m > $out/topo.txt 2>&1; nvidia-smi -q > $out/nvidia-smi-q.txt 2>&1
run() { name=$1; shift; echo "== $name: $*"; timeout 2400 "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; tail -n 4 $out/$name.log; }
if [ $G -ge 8 ]; then
  # 1. Vx's NCCL collectives on 8, bit for bit, and one all-reduce of 6144 floats in a graph.
  run collective-8 $PYTHON -m validation.collective_check --ranks 8 --output $out/collective-8.json
  # 2. The one-shot sum over CUDA IPC (gpu/spawn/tp_all_reduce), 8 ranks: bits against NumPy's rank-order
  #    sum, eager and in graphs replayed with no reset; the time of a call.
  run allreduce-8 $PYTHON -m validation.tp_all_reduce_check --ranks 8 --output $out/allreduce-8.json
  # 3. Decode split over 8 GPUs against its emulation on one GPU, bit for bit: NCCL gather, then one-shot.
  #    4 layers of GLM-5.3's width (3 dense, 1 MoE), 48 steps compared, 50 timed.
  run decode-8 $PYTHON -m validation.tp_decode_check --ranks 8 --output $out/decode-8.json
  run decode-8-oneshot $PYTHON -m validation.tp_decode_check --ranks 8 --one-shot --output $out/decode-8-oneshot.json
fi
run decode-1 $PYTHON -m validation.tp_decode_check --ranks 1 --output $out/decode-1.json
run collective-2 $PYTHON -m validation.collective_check --ranks 2 --output $out/collective-2.json
run allreduce-2 $PYTHON -m validation.tp_all_reduce_check --ranks 2 --output $out/allreduce-2.json
run allreduce-1gpu $PYTHON -m validation.tp_all_reduce_check --ranks 2 --gpus 1 --time-calls 0 --output $out/allreduce-1gpu.json
run decode-2 $PYTHON -m validation.tp_decode_check --ranks 2 --output $out/decode-2.json
run decode-2-oneshot $PYTHON -m validation.tp_decode_check --ranks 2 --one-shot --output $out/decode-2-oneshot.json
echo z4-tp-done
# 4. GPU tests: the new kernels' (tp_all_reduce one rank, sum_ranks, MLA padded heads) and the graph tests.
STEPS='^test-(spawn-vx-host|resident-graph)$' bash scripts/pod_validate.sh $out-tests > ../z4-tests.log 2>&1; echo "tests: $? failed steps"
grep -hE "FAILED|passed|failed" $out-tests/test-*.txt | tail -8
# 5. Vx PR #3619 (vx-7e): vx_peer_buffer_check on 2 and 8 GPUs; expect "rank R: ok" from every rank.
if [ -f ../vx-peer.tar.gz ]; then
  rm -rf ../vx-peer && tar xzf ../vx-peer.tar.gz -C ..
  (cd ../vx-peer && C=/usr/local/cuda
   sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libffi-dev g++ > /dev/null 2>&1
   g++ -shared -fPIC -O2 runtime/cuda_dispatch.cpp -I$C/include -L$C/lib64 -Wl,-rpath,$C/lib64 -lcudart -lcublas -lffi -o libvx_cuda_dispatch.so &&
   g++ -O1 -std=c++17 scripts/tools/vx_peer_buffer_check.cpp -Iscripts/tools -Iruntime -I$C/include -L$C/lib64 -lcudart \
     -L. -lvx_cuda_dispatch -Wl,-rpath,$PWD -o vx_peer_buffer_check) > $out/vx-peer-build.log 2>&1; echo "vx-peer build: $?"
  for n in 2 $([ $G -ge 8 ] && echo 8); do
    rm -f /tmp/vx_peer_id
    for r in $(seq 0 $((n - 1))); do VX_NCCL_ID_FILE=/tmp/vx_peer_id timeout 300 ../vx-peer/vx_peer_buffer_check $r $n & done > $out/vx-peer-$n.log 2>&1
    wait; echo "vx-peer $n:"; cat $out/vx-peer-$n.log
  done
fi
echo z4-done
