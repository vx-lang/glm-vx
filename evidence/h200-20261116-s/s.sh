# Toolchain move to Vx 64ff9cd4 (branch vx-64ff): restore the bundle, every build and GPU test step of
# pod_validate.sh, then the r session's step profiles again.
set -u
cd ~/glm-h200
sudo bash repo-64ff/scripts/pod_restore.sh bundle-64ff/vx-toolchain-64ff9cd4.tar.zst $HOME/glm-h200/vxtc-64ff > restore-64ff.log 2>&1; echo "restore: $?"
sudo chown -R "$USER": vxtc-64ff
sed -e 's/vxtc-4f96/vxtc-64ff/' -e 's/repo-4f96/repo-64ff/' -e 's/REPO_REV=[0-9a-f]*/REPO_REV=c1ec543/' env-4f96.sh > env-64ff.sh
cd repo-64ff && . ../env-64ff.sh && which vxc && vxc --version
out=evidence/h200-20261116-s
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^(build-|test-)' bash scripts/pod_validate.sh $out > ../s-validate.log 2>&1; echo "validate: $? failed steps"
tail -n 40 ../s-validate.log
cp ../abdiff.py $out/
prof_ab() { name=$1; base=$2; shift 2
  nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1
  $PYTHON $out/abdiff.py ../repo-4f96/evidence/h200-20261116-r/$base.2_cuda_gpu_trace.csv $out/$name.2_cuda_gpu_trace.csv > $out/ab-$name.txt 2>&1; echo "== $name vs 4f96 $base: $(head -n 1 $out/ab-$name.txt)"; }
prof_ab d1-3001-merge d1-3001-merge --context 3001 --long-topk merge
prof_ab d1-3001-position-fuse d1-3001-position-fuse-w8 --context 3001 --long-topk position --fuse-glue
prof_ab moe2-1000-fuse moe2-1000-fuse-w8 --context 1000 --layers 2 --moe --fuse-glue
regs() { for f in $1/gpu/build/spawn/*.ptx; do printf "%s " $(basename $f .ptx); /usr/local/cuda/bin/ptxas -arch=sm_90 -v $f -o /dev/null 2>&1 | grep -o "Used [0-9]* registers" | head -n 1 | grep -o "[0-9]*"; done; }
regs . > $out/registers-sm90.txt; regs ../repo-4f96 > $out/registers-sm90-4f96.txt
join $out/registers-sm90-4f96.txt $out/registers-sm90.txt | awk '$2 != $3' > $out/registers-changed.txt; echo "registers changed:"; cat $out/registers-changed.txt
echo s-done
