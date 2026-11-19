# z30: Vx PR #3649 (vx-7e, f8e8m0-mxfp4-scales @ 4d15f5315: vx-expand-narrow-floats first in both pipelines) on the
# H200, at their request: the Hopper value tests on the GPU and on the CPU (the same lines), the conversion tests, and
# glm-vx's spawn kernels rebuilt with this vxc against the 3609 build on GLM-5.3's first 4 layers (tokens and top-5).
set -u
cd ~/glm-h200
b=bundle-work/vx-toolchain-4d15f5315.tar.zst
[ -f $b ] || { echo "no $b"; tail -20 bundle-4d15f5315.log; exit 1; }
sudo bash repo/scripts/pod_restore.sh $b $HOME/glm-h200/vxtc-4d15 > restore-4d15.log 2>&1 && sudo chown -R "$USER": vxtc-4d15
rm -rf vx-4d15 && mkdir vx-4d15 && tar xzf vx-src-4d15f5315.tar.gz -C vx-4d15
out=$HOME/glm-h200/repo/evidence/h200-20261119-z30; mkdir -p $out; cp z30-vx.sh $out/
(
  . vxtc-4d15/env.sh
  vxc --version > $out/vxc-version.txt 2>&1
  cd vx-4d15
  for t in tests/optimizations/pass/lane_matmuls_on_tensor_cores.vx tests/optimizations/pass/lane_matmuls_of_fp8_tiles_give_the_sums_of_a_loop.vx; do
    n=$(basename $t .vx)
    timeout 600 vxc $t --host default --machine fleet/h200.vx --run 2>&1 | grep -v "^\[JIT\]" > $out/$n.gpu.txt
    CUDA_VISIBLE_DEVICES= timeout 600 vxc $t --host default --machine fleet/h200.vx --run 2>&1 | grep -v "^\[JIT\]" > $out/$n.cpu.txt
    timeout 600 nsys profile -o $out/$n --force-overwrite true --stats=true vxc $t --host default --machine fleet/h200.vx --run > $out/$n.nsys.txt 2>&1
    echo "== $n"; echo "gpu:"; cat $out/$n.gpu.txt; echo "cpu:"; cat $out/$n.cpu.txt
    cmp -s $out/$n.gpu.txt $out/$n.cpu.txt && echo "gpu == cpu" || echo "GPU AND CPU DIFFER"
    echo "kernels on the GPU: $(grep -A4 cuda_gpu_kern_sum $out/$n.nsys.txt | grep -c vx_npu_kernel)"
    rm -f $out/$n.nsys-rep $out/$n.sqlite
  done
  for t in tests/backend/pass/fp8_and_fp4_conversions.vx tests/backend/pass/f8e8m0_conversions.vx; do
    n=$(basename $t .vx)
    timeout 600 vxc $t --host default --run 2>&1 | grep -v "^\[JIT\]" > $out/$n.txt
    grep -o "EXPECT: .*" $t | sed 's/^EXPECT: //' > $out/$n.expect.txt
    echo "== $n (host)"; cat $out/$n.txt
    diff <(sed 's/ *$//' $out/$n.txt | grep -v '^$') <(sed 's/ *$//' $out/$n.expect.txt) > /dev/null && echo "matches its EXPECT lines" || echo "DIFFERS FROM ITS EXPECT LINES"
  done
)
# glm-vx's spawn kernels with this vxc, in a copy of the repo
rm -rf repo-4d15 && mkdir repo-4d15 && (cd repo && git ls-files 2>/dev/null | head -1 > /dev/null; tar cf - --exclude=./evidence --exclude=./.venv --exclude=./gpu/build/spawn .) | tar xf - -C repo-4d15
ln -s $HOME/glm-h200/repo/.venv repo-4d15/.venv
cp -r repo/gpu/build/spawn repo-4d15/gpu/build/spawn-3609
(
  . ./env.sh
  . vxtc-4d15/env.sh
  cd repo-4d15
  OUT=gpu/build/spawn PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 PYTHON=$PYTHON bash gpu/spawn/build.sh > $out/spawn-build.log 2>&1
  echo "spawn build rc=$?: $(find gpu/build/spawn -name '*.ptx' | wc -l) ptx"
  same=0; diff=0; : > $out/spawn-ptx-diff.txt
  for f in $(cd gpu/build/spawn-3609 && find . -name '*.ptx'); do
    if [ -f gpu/build/spawn/$f ]; then if cmp -s gpu/build/spawn/$f gpu/build/spawn-3609/$f; then same=$((same+1)); else diff=$((diff+1)); echo "$f" >> $out/spawn-ptx-diff.txt; fi; else echo "missing $f" >> $out/spawn-ptx-diff.txt; fi
  done
  echo "ptx against the 3609 build: $same identical, $diff differ"
)
(
  . ./env.sh
  export PYTHONPATH=$HOME/glm-h200/repo-4d15 VX_NCCL_LIB=$(find $HOME/glm-h200/repo/.venv -name libnccl.so.2 | head -1)
  cd repo-4d15
  timeout 900 $PYTHON -m validation.tp_real --checkpoint $HOME/glm-h200/real4 --ranks 1 --one-shot --contexts 1000 --new 32 --time 20 \
    --option moe_down2=true --option fuse_router=true --output $out/tp-real-4d15.json > $out/tp-real-4d15.log 2>&1; echo "tp_real rc=$?"
  python3 -c "
import json
a = json.load(open('$HOME/glm-h200/repo/evidence/h200-20261119-z29/tp-real-plain.json')); b = json.load(open('$out/tp-real-4d15.json'))
print('against the 3609 kernels (z29 plain): tokens equal', a['generated'] == b['generated'], 'top-5 equal', a['tops'] == b['tops'])"
)
echo z30-done
