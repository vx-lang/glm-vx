# z34: vx-7e's topk_select_bitmap_at (glm-vx branch topk-fast 87e9614 (c9f1bf0's kernel + a stats twin), Vx main 8146 vxc; never on a GPU) on the
# 8x H200, after z33: the kernel alone against ilp_hist (validation/topk_hist_ab, every output checked), its GPU
# tests, then GLM-5.3 at 32768 tokens with long_topk ilp_hist without and with topk_bitmap (all 21 full-indexer
# layers on every rank): the step time and the first token's top-5. Resumable (.done-NAME).
set -u
cd ~/glm-h200
trap 'echo "SIGTERM at $(date -u +%T) during ${name:-setup}"; exit 143' TERM
. ./env.sh
export NCCL_NVLS_ENABLE=0 GLM_VX_TP_SPIN=0.05
export VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
M=$HOME/glm-h200/GLM-5.3
out=$HOME/glm-h200/repo/evidence/h200-20261119-z34; mkdir -p $out; cp z34.sh $out/
if [ ! -d repo-topk ]; then
  mkdir repo-topk && tar xzf topk-vx7e.tar.gz -C repo-topk
  cp -a repo/gpu/build repo-topk/gpu/ && rm -rf repo-topk/gpu/build/spawn && cp -a repo-topk/spawn-bitmap repo-topk/gpu/build/spawn
fi
cd repo-topk
export PYTHONPATH=$PWD
run() {
  name=$1; shift
  if [ -e $out/.done-$name ]; then echo "== $name: done before, skipped"; return 0; fi
  echo "== $name: $*"; date -u +%T; "$@" > $out/$name.log 2>&1; local rc=$?
  echo "$name: rc=$rc"; date -u +%T; tail -n 3 $out/$name.log | cut -c1-1500
  [ $rc = 0 ] && touch $out/.done-$name; return $rc
}
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 10; echo "gpus idle: $(date -u +%T)"; }
idle
echo "spawn kernels: $(find gpu/build/spawn -name '*.ptx' | wc -l)"
run topk-ab env CUDA_VISIBLE_DEVICES=0 timeout 1800 $PYTHON -m validation.topk_hist_ab --output $out/topk-bitmap.md \
  --spawn gpu/build/spawn --tokens 3001 8192 32768 65536
run hw-spawn env CUDA_VISIBLE_DEVICES=0 timeout 1800 $PYTHON -m pytest -q -rs gpu/test_spawn_hardware.py -k bitmap
run hw-graph env CUDA_VISIBLE_DEVICES=0 timeout 3600 $PYTHON -m pytest -q -rs gpu/test_resident_graph_hardware.py -k bitmap
idle
base="--option moe_down2=true --option fuse_router=true"
run tp-real-32k-ilp timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 32768 --time 20 \
  $base --output $out/tp-real-32k-ilp.json
idle
run tp-real-32k-bitmap timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 32768 --time 20 \
  $base --option topk_bitmap=true --output $out/tp-real-32k-bitmap.json
idle
# The stats twin (not timed): how often the threshold bin held more than 2048 keys (the rounds read the scores again).
run tp-real-32k-stats timeout 3600 $PYTHON -m validation.tp_real --checkpoint $M --ranks 8 --one-shot --contexts 32768 --time 20 \
  $base --option 'topk_bitmap="stats"' --output $out/tp-real-32k-stats.json
idle
echo z34-done
