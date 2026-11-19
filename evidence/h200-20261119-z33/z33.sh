# z33: vx-94's block-FP8 GEMMs (glm-vx branch vx94-grouped-fp8-gemm 3e27413, built with Vx PR #3656 e37f1c06e)
# on one H200, never run on a GPU before: gpu/test_fp8_gemm_hardware.py (dense m64/m128, split-K) and
# gpu/test_fp8_gemm_grouped_hardware.py (the routed experts' grouped GEMM for prefill, TP8 shapes, timing).
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T)"; exit 143' TERM
. ./env.sh
out=$HOME/glm-h200/repo/evidence/h200-20261119-z33; mkdir -p $out; cp z33.sh $out/
[ -d repo-gemm ] || { mkdir repo-gemm && tar xzf gemm-vx94.tar.gz -C repo-gemm; }
cd repo-gemm
for t in test_fp8_gemm_hardware test_fp8_gemm_grouped_hardware; do
  [ -e $out/.done-$t ] && { echo "$t done before"; continue; }
  echo "== $t $(date -u +%T)"
  PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 GLM_VX_FP8_GROUPED_JSON=$out/grouped.json timeout 3600 \
    $PYTHON -m pytest -q -rs gpu/$t.py > $out/$t.log 2>&1; rc=$?
  echo "$t: rc=$rc $(date -u +%T)"; tail -n 5 $out/$t.log
  [ $rc = 0 ] && touch $out/.done-$t
done
cp gpu/build/fp8_gemm*_hardware.json $out/ 2>/dev/null
echo z33-done
