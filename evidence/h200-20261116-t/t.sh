# Vx main a4045882 (after #3517: #3523-#3538, with #3531's lanes::copy): restore the bundle, run
# #3531's test on the H200 for vx-94, then every build and GPU test step of pod_validate.sh, the
# step profiles and the registers against 64ff9cd4 (h200-20261116-s). The first run stopped after the
# lane tests (it made the evidence dir first, and pod_validate.sh keeps an existing one); the rest
# then ran again as written here.
set -u
cd ~/glm-h200
tar xzf repo-a404.tar.gz
sudo bash repo-64ff/scripts/pod_restore.sh bundle-a404/vx-toolchain-a4045882.tar.zst $HOME/glm-h200/vxtc-a404 > restore-a404.log 2>&1; echo "restore: $?"
sudo chown -R "$USER": vxtc-a404
sed -e 's/vxtc-64ff/vxtc-a404/' -e 's/repo-64ff/repo-a404/' -e 's/REPO_REV=[0-9a-f]*/REPO_REV=4501cfb/' env-64ff.sh > env-a404.sh
. ./env-a404.sh && which vxc && vxc --version
# vx-lang/Vx#3531's test: the RUN lines (the CPU copy), then the same on the GPU.
FC=/usr/lib/llvm-22/bin/FileCheck; T=tests/optimizations/pass/lane_copy_into_and_out_of_a_tile.vx
mkdir -p lane3531; ( cd bundle-a404/src
  lane() { name=$1; shift; "$@" > ../../lane3531/$name.txt 2>&1; rc=$?
    $FC $T --implicit-check-not=Error --input-file=../../lane3531/$name.txt > ../../lane3531/$name.filecheck 2>&1; echo "lane $name: vxc $rc filecheck $?"; }
  nvidia-smi -L > ../../lane3531/gpu.txt
  lane cpu-h100 env CUDA_VISIBLE_DEVICES= vxc $T --host default --machine fleet/h100-sxm.vx --run
  lane cpu-h100-j2 env CUDA_VISIBLE_DEVICES= vxc $T -j 2 --host default --machine fleet/h100-sxm.vx --run
  lane cpu-h100-legacy env CUDA_VISIBLE_DEVICES= vxc $T --legacy-codegen --host default --machine fleet/h100-sxm.vx --run
  lane gpu-h200 vxc $T --host default --machine fleet/h200.vx --run
  lane gpu-h200-j2 vxc $T -j 2 --host default --machine fleet/h200.vx --run
  lane gpu-h200-legacy vxc $T --legacy-codegen --host default --machine fleet/h200.vx --run
  lane gpu-h100 vxc $T --host default --machine fleet/h100-sxm.vx --run )
echo lane-done
cd repo-a404
out=evidence/h200-20261116-t
PTXAS=/usr/local/cuda/bin/ptxas PTXAS_ARCH=sm_90 STEPS='^(build-|test-)' bash scripts/pod_validate.sh $out > ../t-validate.log 2>&1; echo "validate: $? failed steps"
tail -n 40 ../t-validate.log
cp -r ../lane3531 $out/
cp ../abdiff.py $out/
prof_ab() { name=$1; shift
  nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1
  $PYTHON $out/abdiff.py ../repo-64ff/evidence/h200-20261116-s/$name.2_cuda_gpu_trace.csv $out/$name.2_cuda_gpu_trace.csv > $out/ab-$name.txt 2>&1; echo "== $name vs 64ff: $(head -n 1 $out/ab-$name.txt)"; }
prof_ab d1-3001-merge --context 3001 --long-topk merge
prof_ab d1-3001-position-fuse --context 3001 --long-topk position --fuse-glue
prof_ab moe2-1000-fuse --context 1000 --layers 2 --moe --fuse-glue
regs() { for f in $1/gpu/build/spawn/*.ptx; do printf "%s " $(basename $f .ptx); /usr/local/cuda/bin/ptxas -arch=sm_90 -v $f -o /dev/null 2>&1 | grep -o "Used [0-9]* registers" | head -n 1 | grep -o "[0-9]*"; done; }
regs . > $out/registers-sm90.txt; regs ../repo-64ff > $out/registers-sm90-64ff.txt
join $out/registers-sm90-64ff.txt $out/registers-sm90.txt | awk '$2 != $3' > $out/registers-changed.txt; echo "registers changed:"; cat $out/registers-changed.txt
echo t-done
