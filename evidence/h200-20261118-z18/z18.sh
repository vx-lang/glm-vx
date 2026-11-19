# z18 (#63, docs/TP-SERVING-PLAN.md item 4) on the one H200: gpu/spawn/moe_mma/fp8_moe_mma_down_at, the routed experts'
# down projection on the FP8 tensor cores (act quantized to E4M3 in groups of 128, as SGLang), against the NumPy
# model of its arithmetic and against fp8_moe_cvt_down2:
#   1. validation/moe_mma_check at one rank of 8 (cols 256) and one GPU (2048); then budgets of 96/80/72/64 registers;
#   2. gpu/test_moe_mma_hardware.py;
#   3. validation/moe_profile under ncu: fp8cvt2 against fp8mma, TP8 shapes, k = 9;
#   4. the 4 real layers (z16's real4) through validation/tp_real --ranks 1 with --option moe_mma=true: tokens and
#      top-5 against z16's tp-real-1.json (the same options without moe_mma), and the step.
set -u
cd ~/glm-h200
. ./env.sh
export PYTHONPATH=$HOME/glm-h200/repo
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/real4
cd repo
out=evidence/h200-20261118-z18; mkdir -p $out; cp ../z18.sh $out/
run() { name=$1; shift; echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; echo "$name: rc=$?"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500; }
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 5; }
idle
run mma-check timeout 1800 $PYTHON -m validation.moe_mma_check --output $out/mma-check.json
for n in 96 80 72 64; do
  run mma-check-n$n timeout 900 $PYTHON -m validation.moe_mma_check --cols 256 --kernel ../moe_mma_nreg/n$n/fp8_moe_mma_down_at.ptx \
    --output $out/mma-check-n$n.json
done
run test-moe-mma timeout 1200 $PYTHON -m pytest -q -rs gpu/test_moe_mma_hardware.py
idle
run moe-profile timeout 1800 $PYTHON -m validation.moe_profile --formats fp8cvt2 fp8mma --inter 256 --k 9 --detail \
  --output $out/moe-tp8.json
idle
run tp-real-mma timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 1 --one-shot --contexts 1000 3001 \
  --option moe_down2=true --option fuse_router=true --option moe_mma=true --output $out/tp-real-mma.json
echo z18-done
