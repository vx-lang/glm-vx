#!/usr/bin/env bash
# The measurement of this folder (z60 step 1), on the single H200. repo-main and repo-nvvm are glm-vx at 7970a7b and
# 8eae0c0, each with gpu/build/ from:
#   VXC=<vxc of Vx main 10c6f478f> PTXAS=<CUDA 13.1 ptxas> PTXAS_ARCH=sm_90 gpu/build.sh && gpu/spawn/build.sh && gpu/spawn/fp8_gemm/build.sh
# Three rounds a kernel option, the order of the two builds alternating (ops-ptx, ptx-ops, ops-ptx), the GPU idle
# before each run.
set -u
out=$PWD/logs; mkdir -p $out
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 2; }
for kernel in mma mma16; do
  for r in 1 2 3; do
    order="ops ptx"; [ $r = 2 ] && order="ptx ops"
    for b in $order; do
      dir=repo-nvvm; [ $b = ptx ] && dir=repo-main
      idle
      (cd $dir && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 python -m validation.topk_hist_ab --kernel $kernel \
        --output $out/$kernel-r$r-$b.md > $out/$kernel-r$r-$b.log 2>&1)
    done
  done
done
