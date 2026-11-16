# A/B session: long_topk merge/position, fuse_glue off/on, on branch topk-long-next (one build).
cd ~/glm-h200/repo-4f96  # overlay: git archive topk-long (a8bd6a1) extracted over it first && . ../env-4f96.sh
out=evidence/h200-${DATE}-r; mkdir -p $out
nvidia-smi --query-gpu=name,driver_version --format=csv > $out/host.txt
bash gpu/build.sh > $out/build-hand.txt 2>&1; tail -2 $out/build-hand.txt
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 KERNELS="kv_rows_bf16_at store_row_bf16_at topk_select topk_select_at" bash gpu/spawn/build.sh > $out/build.txt 2>&1; grep -E "rror|ptx:" $out/build.txt
$PYTHON -m pytest -q -rs gpu/test_spawn_hardware.py -k "glue or topk_select or residual_rmsnorm" > $out/test-spawn-hardware.txt 2>&1; echo "spawn-hardware: $(tail -1 $out/test-spawn-hardware.txt)"
$PYTHON -m pytest -q -rs gpu/test_spawn_vx_launch.py > $out/test-vx-launch.txt 2>&1; echo "vx-launch: $(tail -1 $out/test-vx-launch.txt)"
$PYTHON -m pytest -q -rs gpu/test_resident_graph_hardware.py > $out/test-resident-graph-hardware.txt 2>&1; echo "resident-graph: $(tail -1 $out/test-resident-graph-hardware.txt)"
prof() { name=$1; shift
  nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1
  echo "== $name: $(grep -o 'total [0-9.]* us' <($PYTHON evidence/h200-20261116-k/step-profile/trace_mla.py $out/$name.2_cuda_gpu_trace.csv))"; }
for ctx in 1000 3001 8192; do
  prof d1-$ctx-merge --context $ctx --long-topk merge
  prof d1-$ctx-position --context $ctx --long-topk position
  prof d1-$ctx-position-fuse --context $ctx --long-topk position --fuse-glue
done
prof moe2-1000 --context 1000 --layers 2 --moe
prof moe2-1000-fuse --context 1000 --layers 2 --moe --fuse-glue
for ctx in 1000 3001 8192; do $PYTHON ~/glm-h200/abdiff.py $out/d1-$ctx-merge.2_cuda_gpu_trace.csv $out/d1-$ctx-position-fuse.2_cuda_gpu_trace.csv > $out/ab-$ctx.txt; done
$PYTHON ~/glm-h200/abdiff.py $out/moe2-1000.2_cuda_gpu_trace.csv $out/moe2-1000-fuse.2_cuda_gpu_trace.csv > $out/ab-moe2-1000.txt
head -3 $out/ab-*.txt
echo r-done
