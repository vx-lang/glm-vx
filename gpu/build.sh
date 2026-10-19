#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3}"
: "${VX_GPU_ARCH:=sm_80}"
extra=()
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS"); fi
mkdir -p gpu/build
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/kernels.vx --output gpu/build/kernels.ptx --arch "$VX_GPU_ARCH" "${extra[@]}"
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/fp8.vx --output gpu/build/fp8.ptx --arch "$VX_GPU_ARCH" "${extra[@]}"
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/gemm.vx --output gpu/build/gemm.ptx --shared-floats 320 --arch "$VX_GPU_ARCH" "${extra[@]}"
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/int4.vx --output gpu/build/int4.ptx --arch "$VX_GPU_ARCH" "${extra[@]}"
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/fp8_batched.vx --output gpu/build/fp8_batched.ptx --arch "$VX_GPU_ARCH" "${extra[@]}"
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/bf16.vx --output gpu/build/bf16.ptx --arch "$VX_GPU_ARCH" "${extra[@]}"
"$PYTHON" gpu/toolchain/build_ptx.py --source gpu/topk.vx --output gpu/build/topk.ptx --arch "$VX_GPU_ARCH" "${extra[@]}"
