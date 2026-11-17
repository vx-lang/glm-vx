# index_kernel='fp8' (#32): SGLang's FP8 indexer recipe in inline PTX on the H200, against v's 'mma' (bf16).
set -u
cd ~/glm-h200 && . ./env-u.sh && cd repo-u
tar xzf ../fp8-files.tgz
out=evidence/h200-20261117-w; mkdir -p $out
OUT=gpu/build/spawn/index_mma PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 bash gpu/spawn/index_mma/build.sh > $out/build.txt 2>&1; echo "build: $?"; tail -3 $out/build.txt
timeout 900 $PYTHON -m pytest -q -rf -s gpu/test_spawn_hardware.py -k "fp8_at or index_scores_tc" > $out/test-spawn-hardware.txt 2>&1; grep -E "err|passed|failed" $out/test-spawn-hardware.txt | tail -8
timeout 1200 $PYTHON -m pytest -q -rf gpu/test_resident_graph_hardware.py > $out/test-resident-graph.txt 2>&1; tail -3 $out/test-resident-graph.txt
cp ../abdiff.py $out/
v=evidence/h200-20261117-v
for ctx in 3001 8192 32768; do
  nsys profile -f true -o $out/d1-$ctx-fp8 --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile --context $ctx --index-kernel fp8 > $out/d1-$ctx-fp8.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/d1-$ctx-fp8.2 $out/d1-$ctx-fp8.2.nsys-rep >> $out/d1-$ctx-fp8.log 2>&1
  $PYTHON $out/abdiff.py $v/d1-$ctx-mma.2_cuda_gpu_trace.csv $out/d1-$ctx-fp8.2_cuda_gpu_trace.csv > $out/ab-$ctx.txt 2>&1
  echo "== $ctx mma -> fp8: $(head -n 1 $out/ab-$ctx.txt)"; grep -E "index_scores|index_key" $out/ab-$ctx.txt
done
for k in index_key_fp8_at index_scores_fp8_at; do echo "$k $(/usr/local/cuda/bin/ptxas -arch=sm_90a -v gpu/build/spawn/index_mma/$k.ptx -o /dev/null 2>&1 | grep -o 'Used [0-9]* registers')"; done > $out/registers.txt
echo w-done
