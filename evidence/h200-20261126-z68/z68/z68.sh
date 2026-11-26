# z68 (#63, TTFT; one H200, glm-h200-1gpu): branch ttft-bf16-gemm 73e5916 (kernels by Vx main 10c6f478f).
# 1. The absorbed sparse prefill kernel against float64 (test_mla_prefill_sparse_tc_matches_float64_on_bf16_values)
#    and the bf16 rows kernels. 2. gpu/test_cublaslt_hardware.py. 3. Logits, one layer (dense, full indexer):
#    rank_profile mha_tc vs mla_tc, a 2048-token chunk at 30720, as the CPU test's one-layer check. 4. nsys of the
#    4-layer mla_tc chunk at 30720 (kernel table). 5. The rest of the hardware suite on this branch. Resumable.
set -u
cd ~/glm-h200
. ./env.sh
out=$HOME/glm-h200/z68-out; mkdir -p $out; cp z68.sh $out/
R=$HOME/glm-h200/repo-z68 RP=$HOME/glm-h200/z68-rank_profile.py
if [ ! -e $R/.ready ]; then
  rm -rf $R && mkdir $R && tar xzf z68-ttft.tar.gz -C $R && ln -s $HOME/glm-h200/repo/.venv $R/.venv && touch $R/.ready || { echo "unpack failed"; exit 1; }
fi
cd $R; export PYTHONPATH=$R CUDA_VISIBLE_DEVICES=0
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; }
run() {  # name command...: once, unless done
  local name=$1; shift
  [ -e $out/.done-$name ] && { echo "== $name: done before"; return 0; }
  idle; echo "== $name: $(date -u +%T)"; "$@" > $out/$name.log 2>&1; local rc=$?
  echo "$name: rc=$rc $(date -u +%T): $(tail -n 1 $out/$name.log | cut -c1-300)"; touch $out/.done-$name; return $rc
}
[ -e $out/env.txt ] || { date -u; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; $PYTHON -c 'import sys; print(sys.version)'; $PYTHON -c 'from gpu import cublaslt as L; l=L.load_library(); print("cublaslt", l._name, l.cublasLtGetVersion())'; } > $out/env.txt 2>&1
run sparse-tc timeout 1800 $PYTHON -m pytest -q -rs -p no:cacheprovider gpu/test_spawn_hardware.py -k "mla_prefill_sparse or bf16_rows"
run cublaslt timeout 1800 $PYTHON -m pytest -q -rs -p no:cacheprovider gpu/test_cublaslt_hardware.py
for v in mha_tc mla_tc; do
  run one-layer-$v timeout 3600 $PYTHON $RP --output $out/one-layer-$v --layers 1 --long 30720 --decode 0 --verify "" --reps 3 \
    --options "{\"prefill_attention\": \"$v\"}"
done
$PYTHON $RP --compare $out/one-layer-mha_tc $out/one-layer-mla_tc > $out/one-layer-compare.txt 2>&1; echo "one-layer compare rc=$?"
grep -iE "logit|argmax" $out/one-layer-compare.txt | head -8
if [ ! -e $out/.done-nsys ]; then
  idle; echo "== nsys: $(date -u +%T)"
  timeout 3600 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx -o $out/nsys-mla_tc-30720 -f true \
    $PYTHON $RP --output $out/nsys-run --layers 4 --long 30720 --decode 0 --verify "" --reps 1 --nsys-long \
    --options '{"prefill_attention": "mla_tc"}' > $out/nsys.log 2>&1; echo "nsys rc=$?"
  nsys stats --report cuda_gpu_kern_sum --format csv -o $out/nsys-kern $out/nsys-mla_tc-30720.nsys-rep > $out/nsys-stats.log 2>&1; echo "stats rc=$?"
  touch $out/.done-nsys
fi
run suite timeout 3600 $PYTHON -m pytest -q -rs -p no:cacheprovider $(ls gpu/test_*hardware*.py | grep -v stress)
echo z68-done
