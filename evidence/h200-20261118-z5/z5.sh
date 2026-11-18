# z5 (#35, #63): GLM-5.3 itself (zai-org/GLM-5.3, FP8, 756 GB) on 8x H200, tensor parallel, against SGLang 0.5.22 --tp 8.
# NCCL_NVLS_ENABLE=0: this VM has no multicast objects (NCCL's NVLS fails with CUDA error 401); SGLang runs with it too.
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo NCCL_NVLS_ENABLE=0
export VX_NCCL_LIB=$(find ~/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3
SG=$HOME/glm-h200/sglang-venv/bin/python
# SGLang JIT-builds kernels (DeepGEMM, FlashInfer) with ninja, which only its venv's bin holds on this image;
# its kernel cache stays on the disk, where the dummy-weight warmup (sglang-dummy.sh) filled it.
export SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache PATH=$PATH:$HOME/glm-h200/sglang-venv/bin
# Wait for the shards a step needs (hf download moves a file into place only when it is complete).
have() { $PYTHON - "$M" "$1" <<'P'
import json, os, re, sys
m, layers = sys.argv[1], int(sys.argv[2])
w = json.load(open(f'{m}/model.safetensors.index.json'))['weight_map']
need = {f for k, f in w.items() if not (g := re.match(r'model\.layers\.(\d+)\.', k)) or int(g[1]) < layers}
sys.exit(0 if all(os.path.exists(f'{m}/{f}') for f in need) else 1)
P
}
wait_for() { until [ -f $M/model.safetensors.index.json ] && have $1; do sleep 30; done; echo "shards for $1 layers: $(date -u +%T)"; }
cd repo
out=evidence/h200-20261118-z5
mkdir -p $out; cp ../z5.sh $out/
nvidia-smi --query-gpu=index,name,memory.total --format=csv > $out/gpus.csv
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; timeout ${LIMIT:-3600} "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
wait_for 4
# 1. Bits on real weights: GLM-5.3's first 4 layers (3 dense + 1 MoE of 256 experts) on 8 GPUs against their
#    one-GPU emulation (8 ranks as threads on GPU 0), every logit at 48 steps.
run real4-8-oneshot $PYTHON -m validation.tp_decode_check --checkpoint $M --layers 4 --ranks 8 --one-shot --output $out/real4-8-oneshot.json
# 2. The load path end to end on 4 layers (greedy loop, a context, timing).
run real4-smoke $PYTHON -m validation.tp_real --checkpoint $M --layers 4 --ranks 8 --one-shot --new 4 --contexts 100 --time 10 --output $out/real4-smoke.json
wait_for 1000
# 3. The whole model: greedy text after the prompt, then step time at contexts 1000, 3001, 32768.
LIMIT=7200 run real-8-oneshot $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --new 32 --contexts 1000 3001 32768 --output $out/real-8-oneshot.json
# 4. SGLang on the same prompt ids: greedy tokens and top-5 log-probabilities.
LIMIT=5400 run sglang-greedy $SG validation/sglang_greedy.py --model $M --ours $out/real-8-oneshot.json --tp 8 --output $out/sglang-greedy.json
# 5. SGLang batch-1 decode at the same contexts: its one_batch (no HTTP server, CUDA graphs as in serving), median
#    decode latency. 0.5.22 moved it to sglang.benchmark.one_batch.
LIMIT=5400 run sglang-one-batch $SG -m sglang.benchmark.one_batch --model-path $M --tp 8 --batch-size 1 --input-len 1000 3001 32768 \
  --output-len 64 --kv-cache-dtype bfloat16 --mem-fraction-static 0.8 --random-seed 0 --result-filename $PWD/$out/sglang-one-batch.jsonl
# 6. Where each step's time goes. SGLang: its last decode step under nsys (as scripts/pod_sglang.sh profile, z3).
#    Ours: 20 steps at context 1000, every rank (nsys follows the rank processes).
stats() { for r in cuda_gpu_kern_sum cuda_gpu_trace; do nsys stats -q --force-export true --report $r --format csv -o $out/$1 $out/$1.nsys-rep >> $out/$1.log 2>&1; done; rm -f $out/$1.sqlite; }
for n in 1000 32768; do
  LIMIT=3600 run nsys-sglang-$n nsys profile --force-overwrite true -o $out/nsys-sglang-$n --trace cuda,nvtx --cuda-graph-trace node \
    --capture-range cudaProfilerApi --capture-range-end stop --sample none --cpuctxsw none \
    $SG -m sglang.benchmark.one_batch --model-path $M --tp 8 --batch-size 1 --input-len $n --output-len 8 --kv-cache-dtype bfloat16 \
    --mem-fraction-static 0.8 --random-seed 0 --profile --profile-stage decode --profile-start-step 6 --profile-activities CUDA_PROFILER \
    --run-name nsys-$n --result-filename $PWD/$out/nsys-sglang.jsonl
  stats nsys-sglang-$n
done
LIMIT=5400 run nsys-real-8 nsys profile -f true -o $out/nsys-real-8 --trace=cuda --cuda-graph-trace=node --sample none --cpuctxsw none \
  $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --new 1 --contexts 1000 --time 20 --output $out/nsys-real-8.json
stats nsys-real-8
echo z5-done
