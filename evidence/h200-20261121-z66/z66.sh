# z66 (#63, TTFT; the single H200): prefill_attention='mla_tc' (the absorbed sparse attention on the tensor cores,
# gpu/spawn/mla_prefill_sparse_tc + batched bf16 cuBLASLt absorbs) against 'mha_tc' (the served option), branch
# ttft-bf16-gemm built by Vx main 10c6f478f. 1. gpu/test_cublaslt_hardware.py. 2. validation/rank_profile.py once each
# (one TP8 rank, 4 layers; chunks of 2048 at 2048 and 30720, no decode), then --compare. One run each: a diagnosis.
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T)"; exit 143' TERM
. ./env.sh
out=$HOME/glm-h200/z66-out; mkdir -p $out; cp z66.sh rank_profile.py $out/
R=repo-z66
if [ ! -e $R/.ready ]; then
  rm -rf $R && mkdir $R && tar xzf z66-ttft.tar.gz -C $R && ln -s $HOME/glm-h200/repo/.venv $R/.venv && touch $R/.ready \
    || { echo "unpack failed"; echo z66-done; exit 1; }
fi
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
(cd $R && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 timeout 1800 $PYTHON -m pytest -q -rs -p no:cacheprovider gpu/test_cublaslt_hardware.py \
  > $out/test_cublaslt_hardware.log 2>&1); echo "test_cublaslt_hardware: rc=$? $(tail -n 1 $out/test_cublaslt_hardware.log)"
for v in mha_tc mla_tc; do
  [ -s $out/$v/summary.txt ] && continue
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
  (cd $R && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 timeout 3600 $PYTHON ../rank_profile.py --output $out/$v \
    --long 2048,30720 --decode 0 --options "{\"prefill_attention\": \"$v\"}" > $out/$v.log 2>&1); rc=$?
  echo "$v: rc=$rc $(date -u +%T): $(grep -E '^w1 wall|^long start' $out/$v.log | tr '\n' ' ') $(tail -n 2 $out/$v.log | grep -i error | cut -c1-300)"
done
(cd $R && $PYTHON ../rank_profile.py --compare $out/mha_tc $out/mla_tc) > $out/compare.txt 2>&1; echo "compare rc=$?"
grep -E "chunk at|78-layer|logits|attention" $out/compare.txt | head -40
echo z66-done
