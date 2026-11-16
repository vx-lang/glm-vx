cd ~/glm-h200/repo-4f96 && . ../env-4f96.sh
out=evidence/h200-20261116-r
$PYTHON -m pytest -q gpu/test_spawn_hardware.py -k glue > $out/test-glue-w8.txt 2>&1; echo "glue test: $(tail -1 $out/test-glue-w8.txt)"
for a in "moe2-1000-fuse-w8 --context 1000 --layers 2 --moe --fuse-glue" "d1-3001-position-fuse-w8 --context 3001 --long-topk position --fuse-glue"; do set -- $a; name=$1; shift
  nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1
done
$PYTHON ~/glm-h200/abdiff.py $out/moe2-1000-fuse.2_cuda_gpu_trace.csv $out/moe2-1000-fuse-w8.2_cuda_gpu_trace.csv > $out/ab-w8-moe2.txt
$PYTHON ~/glm-h200/abdiff.py $out/d1-3001-position-fuse.2_cuda_gpu_trace.csv $out/d1-3001-position-fuse-w8.2_cuda_gpu_trace.csv > $out/ab-w8-3001.txt
cat $out/ab-w8-*.txt; echo w8-done
