# z30b: z30's glm-vx part again after mla_q_rope_bf16_at's fix: every spawn kernel with Vx PR #3649's vxc against the
# 3609 build, then GLM-5.3's first 4 layers on them (tokens and top-5 against z29's plain run).
set -u
cd ~/glm-h200
out=$HOME/glm-h200/repo/evidence/h200-20261119-z30; cp z30b.sh $out/
cp repo/gpu/build/spawn/mla_q_rope_bf16_at.ptx repo/gpu/build/spawn/mla_q_rope_bf16_at.json repo-4d15/gpu/build/spawn-3609/
(
  . ./env.sh
  . vxtc-4d15/env.sh
  cd repo-4d15
  OUT=gpu/build/spawn PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 PYTHON=$PYTHON bash gpu/spawn/build.sh > $out/spawn-build-b.log 2>&1
  echo "spawn build rc=$?: $(find gpu/build/spawn -name '*.ptx' | wc -l) ptx"
  same=0; diff=0; : > $out/spawn-ptx-diff-b.txt
  for f in $(cd gpu/build/spawn-3609 && find . -name '*.ptx'); do
    if [ -f gpu/build/spawn/$f ]; then if cmp -s gpu/build/spawn/$f gpu/build/spawn-3609/$f; then same=$((same+1)); else diff=$((diff+1)); echo "$f" >> $out/spawn-ptx-diff-b.txt; fi; else echo "missing $f" >> $out/spawn-ptx-diff-b.txt; fi
  done
  echo "ptx against the 3609 build: $same identical, $diff differ"; head -20 $out/spawn-ptx-diff-b.txt
)
(
  . ./env.sh
  export PYTHONPATH=$HOME/glm-h200/repo-4d15 VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
  cd repo-4d15
  rm -f $out/tp-real-4d15.json
  timeout 900 $PYTHON -m validation.tp_real --checkpoint $HOME/glm-h200/real4 --ranks 1 --one-shot --contexts 1000 --new 32 --time 20 \
    --option moe_down2=true --option fuse_router=true --output $out/tp-real-4d15.json > $out/tp-real-4d15.log 2>&1; echo "tp_real rc=$?"
  python3 -c "
import json
a = json.load(open('$HOME/glm-h200/repo/evidence/h200-20261119-z29/tp-real-plain.json')); b = json.load(open('$out/tp-real-4d15.json'))
print('against the 3609 kernels (z29 plain): tokens equal', a['generated'] == b['generated'], 'top-5 equal', a['tops'] == b['tops'])"
)
echo z30b-done
