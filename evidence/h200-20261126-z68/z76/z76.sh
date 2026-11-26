# z76 (one H200): glm-vx's hardware tests on z71's B (ttft-bf16-gemm 73e5916 + the host-copy change: uploads from the
# numpy buffer, the chunk's RoPE and step rows and embeddings at once), since every upload now takes the new path.
set -u
cd ~/glm-h200
. ./env.sh
out=$HOME/glm-h200/z76-out R=$HOME/glm-h200/repo-z71; mkdir -p $out; cp z76.sh $out/
while pgrep -f "bash z7[01].sh" > /dev/null; do sleep 20; done
exec 9> $HOME/glm-h200/gpu.lock; flock 9
echo "gpu: $(date -u +%T)"
cd $R; export PYTHONPATH=$R CUDA_VISIBLE_DEVICES=0
timeout 3600 $PYTHON -m pytest -q -rs -p no:cacheprovider gpu/test_runtime.py $(ls gpu/test_*hardware*.py | grep -v stress) > $out/suite.log 2>&1
echo "suite: rc=$? $(date -u +%T): $(tail -n 1 $out/suite.log)"; grep -E "^FAILED|^ERROR" $out/suite.log | head -20
echo z76-done
