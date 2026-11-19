# z25: Vx PR #3648 (vx-94, mixed-f16-by-fp8-tiles @ 7933f6af1, on #3646): f16 x f8 tile matmul, on the one H200.
# After `TC=7933f6af1 pod_go.sh HOST bundle` built bundle-work/vx-toolchain-7933f6af1.tar.zst: restore it beside the
# glm-vx toolchain (vxtc stays 6a1d), then the PR's value test on the GPU, under nsys to show the kernel ran there.
set -u
cd ~/glm-h200
b=bundle-work/vx-toolchain-7933f6af1.tar.zst
[ -f $b ] || { echo "no $b"; tail -20 bundle-7933f6af1.log; exit 1; }
sudo bash repo/scripts/pod_restore.sh $b $HOME/glm-h200/vxtc-7933 > restore-66ff.log 2>&1 && sudo chown -R "$USER": vxtc-7933
rm -rf vx-7933 && mkdir vx-7933 && tar xzf vx-src-7933f6af1.tar.gz -C vx-7933
out=$HOME/glm-h200/repo/evidence/h200-20261119-z25; mkdir -p $out; cp z25-vx.sh $out/
(
  . vxtc-7933/env.sh
  cd vx-7933
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
echo z25-done
(
  . vxtc-7933/env.sh
  cd vx-7933
  out=$HOME/glm-h200/repo/evidence/h200-20261119-z25
  /usr/local/cuda/bin/ncu --clock-control none --page source --print-source sass --csv vxc tests/optimizations/pass/lane_matmuls_of_fp8_tiles_give_the_sums_of_a_loop.vx --host default --machine fleet/h200.vx --run > $out/ncu-sass.csv 2> $out/ncu-sass.err
  echo "kernels in the SASS: $(grep -c '^"[0-9]' $out/ncu-sass.csv)"
  grep -oE "[A-Z]*MMA[A-Z0-9_.]*|F2FP[A-Z0-9_.]*|LDSM[A-Z0-9_.]*" $out/ncu-sass.csv | sort | uniq -c
)
