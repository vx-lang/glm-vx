# H200 session z (after y): long_topk="ilp" / "ilp_hist" (#32), nn_moe_group_offsets (#3575), Vx #3588 unroll,
# fold_shared and fuse_router (MoE launches), Vx 6a1d8f47.
# Copy repo-z.tar.gz, moe-group-3575.tar.gz and unroll-3588.tar.gz to ~/glm-h200 first.
# Repo repo-z = topk-cluster HEAD + moe-group (pod-z). The pod keeps ~/glm-h200/vxtc-6a1d; on a fresh disk,
# restore ~/pod-toolchain/vx-toolchain-6a1d8f47.tar.zst first (scripts/pod_restore.sh) and make env-y.sh.
set -u
cd ~/glm-h200
rm -rf repo-z && tar xzf repo-z.tar.gz
sed -e 's/repo-y/repo-z/' -e 's/REPO_REV=[0-9a-f]*/REPO_REV=pod-z/' env-y.sh > env-z.sh
. ./env-z.sh
cd repo-z
out=evidence/h200-$(date -u +%Y%m%d)-z
# 1. Build everything, the GPU kernel tests and the graph tests (ilp, ilp_hist among them).
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^(build-|test-spawn-hardware|test-resident-graph)' bash scripts/pod_validate.sh $out > ../z-validate.log 2>&1; echo "validate: $? failed steps"
tail -n 10 ../z-validate.log
grep -E "FAILED|passed|failed" $out/test-spawn-hardware.txt $out/test-resident-graph*.txt | tail -12
# 2. Scores and top-k, old against new, each launch alone (bf16 'mma' and FP8 'fp8').
timeout 1200 $PYTHON -m validation.topk_hist_ab --output $out/topk-hist-mma.md > $out/topk-hist-mma.log 2>&1; echo "hist mma: $?"; cat $out/topk-hist-mma.log
timeout 1200 $PYTHON -m validation.topk_hist_ab --kernel mma16 --output $out/topk-hist-mma16.md > $out/topk-hist-mma16.log 2>&1; echo "hist mma16: $?"; cat $out/topk-hist-mma16.log
timeout 1200 $PYTHON -m validation.topk_hist_ab --kernel fp8 --output $out/topk-hist-fp8.md > $out/topk-hist-fp8.log 2>&1; echo "hist fp8: $?"; cat $out/topk-hist-fp8.log
# 3. nn_moe_group with #3575's offsets: PTX built locally with vxc 15d16427 (moe-group-3575.tar.gz).
mkdir -p gpu/build/spawn/moe_group && tar xzf ../moe-group-3575.tar.gz -C gpu/build/spawn/moe_group
timeout 900 $PYTHON -m validation.moe_group_ab --output $out/moe-group-3575.md > $out/moe-group-3575.log 2>&1; echo "moe_group: $?"; cat $out/moe-group-3575.log
# 4. The step with each top-k (dense layer, nsys kernel sums).
cp ../abdiff.py $out/
prof() { name=$1; shift
  timeout 900 nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1; }
for ctx in 3001 32768 131072; do
  prof d1-$ctx-position --context $ctx
  prof d1-$ctx-ilp-hist --context $ctx --long-topk ilp_hist
  prof d1-$ctx-ilp-hist16 --context $ctx --long-topk ilp_hist --index-kernel mma16
  $PYTHON $out/abdiff.py $out/d1-$ctx-position.2_cuda_gpu_trace.csv $out/d1-$ctx-ilp-hist.2_cuda_gpu_trace.csv > $out/ab-$ctx.txt 2>&1
  echo "== $ctx: $(head -n 1 $out/ab-$ctx.txt)"; grep -E "topk|index_scores" $out/ab-$ctx.txt
  $PYTHON $out/abdiff.py $out/d1-$ctx-position.2_cuda_gpu_trace.csv $out/d1-$ctx-ilp-hist16.2_cuda_gpu_trace.csv > $out/ab16-$ctx.txt 2>&1
  echo "== $ctx mma16: $(head -n 1 $out/ab16-$ctx.txt)"; grep -E "topk|index_" $out/ab16-$ctx.txt
done
# 5. Vx PR #3588 (NVPTX unroll of innermost loops of at most 16 trips, f0788ab86): every kernel built
# locally with it (unroll-3588.tar.gz). The same top-k A/B from those PTX files: against step 2's
# table, topk_select_at, ilp, ilp_hist and topk_mb (mb_list spills 24 B there, 4 B on 6a1d; mb_write 32 registers, 29 on 6a1d).
mkdir -p ../u3588 && tar xzf ../unroll-3588.tar.gz -C ../u3588
timeout 1200 $PYTHON -m validation.topk_hist_ab --spawn ../u3588/spawn-3588 --output $out/topk-hist-mma-3588.md > $out/topk-hist-mma-3588.log 2>&1; echo "hist mma 3588: $?"; cat $out/topk-hist-mma-3588.log
# 6. The four top-k kernels with `// spawn-maxnreg: 64` (vx-94: .maxntid 1024 makes ptxas cap them at 32
# registers; one block an SM loses nothing at 64). 6a1d: mb_list 32 -> 60 regs, spill 4 -> 0 B; ilp 32 -> 39.
# 3588: mb_list 32 -> 58, spill 24 -> 0; mb_write 32 -> 39. Same A/B, from unroll-3588.tar.gz.
for v in 6a1d-n64 3588-n64; do
  timeout 1200 $PYTHON -m validation.topk_hist_ab --spawn ../u3588/spawn-$v --output $out/topk-hist-mma-$v.md > $out/topk-hist-mma-$v.log 2>&1; echo "hist mma $v: $?"; cat $out/topk-hist-mma-$v.log
done
# 7. fold_shared: a dense layer then a MoE one (16 routed experts, top 8, GLM-5.3's widths), the shared
# expert in its own three launches against inside the routed two. Its bits are step 1's
# test_fold_shared_gives_the_same_bits_on_the_gpu.
prof m2-3001-plain --context 3001 --layers 2 --moe
prof m2-3001-fold --context 3001 --layers 2 --moe --fold-shared
$PYTHON $out/abdiff.py $out/m2-3001-plain.2_cuda_gpu_trace.csv $out/m2-3001-fold.2_cuda_gpu_trace.csv > $out/ab-fold.txt 2>&1
echo "== fold_shared: $(head -n 1 $out/ab-fold.txt)"; grep -E "moe|matvec_cvt_tiles |add" $out/ab-fold.txt
# 8. fuse_router: the BF16 router's projection and the router in one launch (router_bf16), on top of
# fold_shared. Its bits are step 1's test_spawn_router_bf16_is_bf16_matvec4_then_router (against
# bf16_matvec4 + spawn router) and test_fuse_router_on_the_gpu (near the hand router: ex2.approx).
prof m2-3001-fuse --context 3001 --layers 2 --moe --fold-shared --fuse-router
$PYTHON $out/abdiff.py $out/m2-3001-fold.2_cuda_gpu_trace.csv $out/m2-3001-fuse.2_cuda_gpu_trace.csv > $out/ab-fuse.txt 2>&1
echo "== fuse_router: $(head -n 1 $out/ab-fuse.txt)"; grep -E "router|bf16_matvec4" $out/ab-fuse.txt
grep -E "router_bf16|test_fuse_router|test_fold_shared" $out/test-spawn-hardware.txt $out/test-resident-graph*.txt | tail -12
echo z-done
