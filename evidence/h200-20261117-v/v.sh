# index_kernel='mma' (#32): the inline-PTX scores kernel on the H200, against u's 'simt' and 'tc'.
set -u
cd ~/glm-h200 && . ./env-u.sh && cd repo-u
tar xzf ../mma-files.tgz
out=evidence/h200-20261117-v; mkdir -p $out
OUT=gpu/build/spawn/index_mma PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 bash gpu/spawn/index_mma/build.sh > $out/build.txt 2>&1; echo "build: $?"
timeout 900 $PYTHON -m pytest -q -rf gpu/test_spawn_hardware.py -k index_scores_tc > $out/test-spawn-hardware.txt 2>&1; tail -1 $out/test-spawn-hardware.txt
timeout 900 $PYTHON -m pytest -q -rf gpu/test_resident_graph_hardware.py > $out/test-resident-graph.txt 2>&1; tail -1 $out/test-resident-graph.txt
cp ../abdiff.py $out/
u=evidence/h200-20261117-u
for ctx in 3001 8192 32768; do
  nsys profile -f true -o $out/d1-$ctx-mma --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile --context $ctx --index-kernel mma > $out/d1-$ctx-mma.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/d1-$ctx-mma.2 $out/d1-$ctx-mma.2.nsys-rep >> $out/d1-$ctx-mma.log 2>&1
  $PYTHON $out/abdiff.py $u/d1-$ctx-simt.2_cuda_gpu_trace.csv $out/d1-$ctx-mma.2_cuda_gpu_trace.csv > $out/ab-$ctx.txt 2>&1
  echo "== $ctx simt -> mma: $(head -n 1 $out/ab-$ctx.txt)"; grep -E "index_scores" $out/ab-$ctx.txt
done
/usr/local/cuda/bin/ptxas -arch=sm_90 -v gpu/build/spawn/index_mma/index_scores_mma_at.ptx -o /dev/null 2>&1 | grep -o "Used [0-9]* registers.*" > $out/registers.txt
echo v-done
