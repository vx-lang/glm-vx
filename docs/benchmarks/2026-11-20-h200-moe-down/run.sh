#!/usr/bin/env bash
# The measurement of this folder (z60 step 2 and the reruns after it), on the single H200. repo-vxmm16b is glm-vx
# branch moe-vxmm16 with gpu/build/ from (Vx main 10c6f478f's vxc, CUDA 13.1 ptxas):
#   VXC=... PTXAS=... PTXAS_ARCH=sm_90 VX_DISPATCH_LIB=... gpu/build.sh && gpu/spawn/build.sh && gpu/spawn/fp8_gemm/build.sh
# A variant of variants/ replaces gpu/spawn/moe_mma/fp8_moe_vxmm16_down_at.vx and is built with
#   python gpu/spawn/build.py --source <variant>.vx --output gpu/build/spawn/moe_mma/fp8_moe_vxmm16_down_at.ptx \
#     --machine gpu/spawn/index_mma/machine-sm90a.vx --ptx-version 8.4 --ptxas $PTXAS --ptxas-arch sm_90a
# Each run: gpu/test_moe_mma_hardware.py, then 5 rounds of the three kernels, the order reversed every other round.
set -u
idle() { until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done; sleep 2; }
name=${1:?run name}
mkdir -p logs
cd repo-vxmm16b
idle
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs gpu/test_moe_mma_hardware.py > ../logs/test_moe_mma_hardware-$name.log 2>&1
idle
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 python ../moe_down_bench.py --ptx $PWD/gpu/build/spawn/moe_mma --rounds 5 --repeats 50 \
  --output ../logs/$name.json > ../logs/$name.log 2>&1
