# index_kernel='tc' (#32) and the fuse_glue default (#39) on the H200, Vx a4045882 (vxtc-a404).
# Repo 7c40a12 (branch glue-default) in repo-u. Builds, the new GPU tests, then step profiles of the
# indexer's scores, 'tc' against 'simt', at 3001, 8192 and 32768 tokens.
set -u
cd ~/glm-h200
rm -rf repo-u && tar xzf repo-u.tar.gz
sed -e 's/repo-a404/repo-u/' -e 's/REPO_REV=[0-9a-f]*/REPO_REV=7c40a12/' env-a404.sh > env-u.sh
. ./env-u.sh
cd repo-u
out=evidence/h200-20261117-u
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^(build-|test-spawn-hardware|test-resident-graph)' bash scripts/pod_validate.sh $out > ../u-validate.log 2>&1; echo "validate: $? failed steps"
tail -n 12 ../u-validate.log
grep -E "index_scores_tc|index_kernel" $out/test-spawn-hardware.txt $out/test-resident-graph.txt | tail -8
cp ../abdiff.py $out/
prof() { name=$1; shift
  nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1; }
for ctx in 3001 8192 32768; do
  prof d1-$ctx-simt --context $ctx
  prof d1-$ctx-tc --context $ctx --index-kernel tc
  $PYTHON $out/abdiff.py $out/d1-$ctx-simt.2_cuda_gpu_trace.csv $out/d1-$ctx-tc.2_cuda_gpu_trace.csv > $out/ab-$ctx.txt 2>&1
  echo "== $ctx: $(head -n 1 $out/ab-$ctx.txt)"; grep -E "index_scores|topk_select" $out/ab-$ctx.txt
done
/usr/local/cuda/bin/ptxas -arch=sm_90 -v gpu/build/spawn/index_scores_tc_at.ptx -o /dev/null 2>&1 | grep -o "Used [0-9]* registers.*" > $out/index_scores_tc_at-registers.txt
echo u-done
