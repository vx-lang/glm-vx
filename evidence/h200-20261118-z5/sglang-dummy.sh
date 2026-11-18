# SGLang 0.5.22 --tp 8 on dummy weights (config + tokenizer only) while GLM-5.3 downloads: JIT-compiles its
# kernels for the TP8 shapes into the cache z5.sh uses, checks that TP8 starts here (NCCL with NVLS off), and
# gives a first batch-1 decode time (random weights route tokens to random experts; z5 reruns on real ones).
set -u
cd ~/glm-h200
export NCCL_NVLS_ENABLE=0 SGLANG_CACHE_DIR=$HOME/glm-h200/sglang-cache PATH=$PATH:$HOME/glm-h200/sglang-venv/bin
out=repo/evidence/h200-20261118-z5; mkdir -p $out; cp sglang-dummy.sh $out/
date -u +%T
timeout 3600 sglang-venv/bin/python -m sglang.benchmark.one_batch --model-path $HOME/glm-h200/GLM-5.3 --load-format dummy --tp 8 \
  --batch-size 1 --input-len 1000 3001 32768 --output-len 64 --kv-cache-dtype bfloat16 --mem-fraction-static 0.8 \
  --random-seed 0 --result-filename $PWD/$out/sglang-one-batch-dummy.jsonl > $out/sglang-one-batch-dummy.log 2>&1
echo "sglang-dummy: rc=$?"; date -u +%T
tail -n 5 $out/sglang-one-batch-dummy.log | cut -c1-600
echo sglang-dummy-done
