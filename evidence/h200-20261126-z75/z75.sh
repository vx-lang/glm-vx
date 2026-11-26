# z75 (one H200): glm-vx main 0589bc6 with its kernels built by Vx main 5d313543e, after wrap_mlir_unsafe.py (Vx main's
# E8016: an mlir! block with ensures clauses must be in unsafe { }; 162 helper bodies under gpu/spawn, generators
# included). Each spawn kernel is built alone, so every kernel that fails shows, not only the
# first. Then gpu/build.sh, fp8_gemm, the PTX against 10c6's (repo-z68), and glm-vx's hardware tests. Waits for z70.
set -u
cd ~/glm-h200
. ./env.sh
out=$HOME/glm-h200/z75-out G=$HOME/glm-h200/vxw/glm-main-wrap M=$HOME/glm-h200/vxw/main; mkdir -p $out; cp z75.sh wrap_mlir_unsafe.py wrap_mlir_unsafe2.py $out/
export PATH=$HOME/.cargo/bin:/usr/lib/llvm-22/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0
while pgrep -f "bash z70.sh" > /dev/null; do sleep 20; done
if [ ! -e $out/.done-build ]; then
  rm -rf $G && mkdir -p $G && tar xzf glm-main-0589bc6.tar.gz -C $G && ln -s $HOME/glm-h200/repo/.venv $G/.venv
  (cd $G && python3 $HOME/glm-h200/wrap_mlir_unsafe.py gpu/spawn && python3 $HOME/glm-h200/wrap_mlir_unsafe2.py) > $out/patch.txt 2>&1
  cd $G; export VXC=$M/target/release/vxc PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 PYTHON=$PYTHON
  taskset -c 2-15 nice -n 19 bash gpu/build.sh > $out/build-hand.log 2>&1; echo "gpu/build.sh rc=$?" | tee $out/summary.txt
  fails=
  for src in gpu/spawn/*.vx; do
    k=$(basename $src .vx); grep -q "^pub fn glm_vx_spawn_" $src || continue
    KERNELS=$k taskset -c 2-15 nice -n 19 bash gpu/spawn/build.sh > $out/spawn-$k.log 2>&1 || fails="$fails $k"
  done
  for k in index_mma moe_mma; do
    KERNELS=$k taskset -c 2-15 nice -n 19 bash gpu/spawn/build.sh > $out/spawn-$k.log 2>&1 || fails="$fails $k"
  done
  echo "spawn kernels that failed:${fails:- none}" | tee -a $out/summary.txt
  for k in $fails; do echo "== $k"; grep -m3 -E "Error\[|error:" $out/spawn-$k.log | cut -c1-300; done >> $out/summary.txt
  taskset -c 2-15 nice -n 19 bash gpu/spawn/fp8_gemm/build.sh > $out/build-fp8gemm.log 2>&1; echo "fp8_gemm rc=$? $(grep -m2 -E 'Error\[' $out/build-fp8gemm.log | cut -c1-200)" | tee -a $out/summary.txt
  echo "ptx: $(find gpu/build -name '*.ptx' | wc -l), W1037 $(cat $out/*.log | grep -c W1037)" | tee -a $out/summary.txt
  python3 - $G/gpu/build $HOME/glm-h200/repo-z68/gpu/build > $out/ptx-diff.txt <<'P'
import os, sys
new, old = sys.argv[1], sys.argv[2]; same = asserts = diff = only = 0; changed = []
def body(p): return [l for l in open(p, errors='replace') if not l.startswith('//')]
for root, _, files in os.walk(new):
    for f in files:
        if not f.endswith('.ptx'): continue
        rel = os.path.relpath(os.path.join(root, f), new); o = os.path.join(old, rel)
        if not os.path.exists(o): only += 1; continue
        a, b = body(os.path.join(root, f)), body(o)
        if a == b: same += 1
        elif [l for l in a if 'assert_message' not in l] == [l for l in b if 'assert_message' not in l]: asserts += 1
        else: diff += 1; changed.append(rel)
print(f"same {same}, only assert messages {asserts}, code changed {diff}, new only {only}")
for c in sorted(changed): print('code changed:', c)
P
  echo "ptx-diff: $(head -n 1 $out/ptx-diff.txt)" | tee -a $out/summary.txt; grep "code changed:" $out/ptx-diff.txt >> $out/summary.txt
  touch $out/.done-build
fi
exec 9> $HOME/glm-h200/gpu.lock; flock 9
echo "gpu: $(date -u +%T)" | tee -a $out/summary.txt
(cd $G && PYTHONPATH=$G timeout 3600 $PYTHON -m pytest -q -rs -p no:cacheprovider $(ls gpu/test_*hardware*.py | grep -v stress)) > $out/glm-tests.log 2>&1
echo "glm-tests: rc=$? $(date -u +%T): $(tail -n 1 $out/glm-tests.log)" | tee -a $out/summary.txt
grep -E "^FAILED|^ERROR" $out/glm-tests.log | head -30 >> $out/summary.txt
echo z75-done
