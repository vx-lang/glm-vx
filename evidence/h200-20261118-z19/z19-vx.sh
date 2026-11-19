# z19: Vx PR #3646 (vx-94, fp8-tensor-core-matmul @ 66ff5d17e), FP8 tile matmul on the tensor cores, on the one H200.
# After `TC=66ff5d17e pod_go.sh HOST bundle` built bundle-work/vx-toolchain-66ff5d17e.tar.zst: restore it beside the
# glm-vx toolchain (vxtc stays 6a1d), then the PR's value test on the GPU, under nsys to show the kernel ran there.
set -u
cd ~/glm-h200
b=bundle-work/vx-toolchain-66ff5d17e.tar.zst
[ -f $b ] || { echo "no $b"; tail -20 bundle-66ff5d17e.log; exit 1; }
sudo bash repo/scripts/pod_restore.sh $b $HOME/glm-h200/vxtc-66ff > restore-66ff.log 2>&1 && sudo chown -R "$USER": vxtc-66ff
rm -rf vx-66ff && mkdir vx-66ff && tar xzf vx-src-66ff5d17e.tar.gz -C vx-66ff
out=$HOME/glm-h200/repo/evidence/h200-20261118-z19; mkdir -p $out; cp z19-vx.sh $out/
(
  . vxtc-66ff/env.sh
  cd vx-66ff
  t=tests/optimizations/pass/lane_matmuls_of_fp8_tiles_give_the_sums_of_a_loop.vx
  vxc --version > $out/vxc-version.txt 2>&1
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $out/vxc-version.txt
  vxc $t --host default --machine fleet/h200.vx --emit-ptx -o $out/fp8-tiles.ptx > $out/emit-ptx.log 2>&1; echo "emit-ptx rc=$?"
  timeout 600 vxc $t --host default --machine fleet/h200.vx --run > $out/run.log 2>&1; echo "run rc=$?"
  cat $out/run.log
  timeout 900 nsys profile -o $out/run-nsys --force-overwrite true --stats=true vxc $t --host default --machine fleet/h200.vx --run \
    > $out/run-nsys.log 2>&1; echo "nsys rc=$?"
  grep -A8 "cuda_gpu_kern_sum" $out/run-nsys.log | head -12
  grep -c "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32\|mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32" $out/fp8-tiles.ptx
)
echo z19-done
