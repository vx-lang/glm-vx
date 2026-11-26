# z71 (#63, TTFT; one H200, glm-h200-1gpu): the host side of a prefill chunk. A = ttft-bf16-gemm 73e5916 (z68's
# package, repo-z68); B = the same plus: uploads straight from the numpy buffer (gpu/runtime.py: no tobytes and
# create_string_buffer copies, which were three host copies of the chunk's embeddings), the chunk's RoPE and step
# rows as array ops, and one embedding gather. rank_profile, 4 layers, prefill_attention='mla_tc', W1 and a
# 2048-token chunk at 30720, alternating A B, B A for 2 rounds; then --compare (logits must be identical: the change
# is host-side only). Also kernels/test_backend.py's RoPE rows test on this box's numpy. Waits for gpu.lock.
set -u
cd ~/glm-h200
. ./env.sh
out=$HOME/glm-h200/z71-out; mkdir -p $out; cp z71.sh $out/
A=$HOME/glm-h200/repo-z68 B=$HOME/glm-h200/repo-z71 RP=$HOME/glm-h200/z68-rank_profile.py
if [ ! -e $B/.ready ]; then
  rm -rf $B && mkdir $B && tar xzf z71-hostcopy.tar.gz -C $B && ln -s $HOME/glm-h200/repo/.venv $B/.venv && touch $B/.ready || { echo "unpack failed"; exit 1; }
fi
(cd $B && PYTHONPATH=$B $PYTHON -m pytest -q -p no:cacheprovider kernels/test_backend.py -k rope_rows gpu/test_runtime.py) > $out/host-tests.log 2>&1
echo "host tests: rc=$? $(tail -n 1 $out/host-tests.log)"
while pgrep -f "bash z68.sh" > /dev/null; do sleep 20; done
exec 9> $HOME/glm-h200/gpu.lock; flock 9
echo "gpu: $(date -u +%T)"
export CUDA_VISIBLE_DEVICES=0
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; }
for r in 1 2; do
  order="A B"; [ $r = 2 ] && order="B A"
  for v in $order; do
    run=$out/$v-r$r; [ -s $run/summary.txt ] && continue
    repo=$A; [ $v = B ] && repo=$B
    idle; (cd $repo && PYTHONPATH=$repo timeout 3600 $PYTHON $RP --output $run --layers 4 --long 30720 --decode 0 --verify "" --reps 3 \
      --options '{"prefill_attention": "mla_tc"}' > $run.log 2>&1)
    echo "$v-r$r: rc=$? $(date -u +%T): $(grep -E '^w1 wall|^long start' $run.log | tr '\n' ' ' | cut -c1-300)"
  done
done
$PYTHON $RP --compare $out/A-r1 $out/B-r1 > $out/compare.txt 2>&1; echo "compare rc=$?"
grep -E "wall|logits" $out/compare.txt | head -12
for v in A B; do for r in 1 2; do
  [ -s $out/$v-r$r/summary.txt ] && echo "$v-r$r: $(grep -E 'wall median|^  wall' $out/$v-r$r/summary.txt | tr '\n' ' ' | cut -c1-300)"
done; done | tee $out/rounds.txt
echo z71-done
