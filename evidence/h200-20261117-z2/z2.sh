# H200 session z2 (after z): the dtype fix (multi-block top-k, topk_hist_ab), router_bf16 redesigned, Vx 6a1d8f47.
# Copy repo-z2.tar.gz to ~/glm-h200 first; z's moe-group/unroll tarballs and u3588/ are already there.
set -u
cd ~/glm-h200
rm -rf repo-z2 && tar xzf repo-z2.tar.gz
sed -e 's/repo-y/repo-z2/' -e 's/REPO_REV=[0-9a-f]*/REPO_REV=pod-z2/' env-y.sh > env-z2.sh
. ./env-z2.sh
cd repo-z2
out=evidence/h200-$(date -u +%Y%m%d)-z2
# 1. Build, the GPU kernel tests and the graph tests (the 3 that failed on a dtype; fuse_router; router_bf16).
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^(build-|test-spawn-hardware|test-resident-graph)' bash scripts/pod_validate.sh $out > ../z2-validate.log 2>&1; echo "validate: $? failed steps"
tail -n 10 ../z2-validate.log
grep -E "FAILED|passed|failed" $out/test-spawn-hardware.txt $out/test-resident-graph*.txt | tail -12
# 2. The router at GLM-5.3's 256 x 6144 (and 16, step_profile's): launches against router_bf16.
timeout 900 $PYTHON -m validation.router_ab --output $out/router-ab.md > $out/router-ab.log 2>&1; echo "router_ab: $?"; cat $out/router-ab.log
# 3. Top-k A/B (z's steps 2, 5, 6, which failed on the dtype).
for kernel in mma mma16 fp8; do
  timeout 1200 $PYTHON -m validation.topk_hist_ab --kernel $kernel --output $out/topk-hist-$kernel.md > $out/topk-hist-$kernel.log 2>&1; echo "hist $kernel: $?"; cat $out/topk-hist-$kernel.log
done
for v in 3588 6a1d-n64 3588-n64; do
  timeout 1200 $PYTHON -m validation.topk_hist_ab --spawn ../u3588/spawn-$v --output $out/topk-hist-mma-$v.md > $out/topk-hist-mma-$v.log 2>&1; echo "hist mma $v: $?"; cat $out/topk-hist-mma-$v.log
done
# 4. The step at 131072 (z's failed on the dtype), and the router in the step (16 experts).
cp ../abdiff.py $out/
prof() { name=$1; shift
  timeout 900 nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1; }
prof m2-3001-fold --context 3001 --layers 2 --moe --fold-shared
prof m2-3001-fuse --context 3001 --layers 2 --moe --fold-shared --fuse-router
$PYTHON $out/abdiff.py $out/m2-3001-fold.2_cuda_gpu_trace.csv $out/m2-3001-fuse.2_cuda_gpu_trace.csv > $out/ab-fuse.txt 2>&1
echo "== fuse_router: $(head -n 1 $out/ab-fuse.txt)"; grep -E "router|bf16_matvec4" $out/ab-fuse.txt
ctx=131072
prof d1-$ctx-position --context $ctx
prof d1-$ctx-ilp-hist --context $ctx --long-topk ilp_hist
prof d1-$ctx-ilp-hist16 --context $ctx --long-topk ilp_hist --index-kernel mma16
$PYTHON $out/abdiff.py $out/d1-$ctx-position.2_cuda_gpu_trace.csv $out/d1-$ctx-ilp-hist.2_cuda_gpu_trace.csv > $out/ab-$ctx.txt 2>&1
echo "== $ctx: $(head -n 1 $out/ab-$ctx.txt)"; grep -E "topk|index_scores" $out/ab-$ctx.txt
$PYTHON $out/abdiff.py $out/d1-$ctx-position.2_cuda_gpu_trace.csv $out/d1-$ctx-ilp-hist16.2_cuda_gpu_trace.csv > $out/ab16-$ctx.txt 2>&1
echo "== $ctx mma16: $(head -n 1 $out/ab16-$ctx.txt)"; grep -E "topk|index_" $out/ab16-$ctx.txt
echo z2-done
