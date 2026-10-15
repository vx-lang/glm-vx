#!/usr/bin/env bash
# M0: bring up a RunPod (or any Linux+NVIDIA) box for GLM-Vx GPU work.
# Idempotent. Checks the driver, locates/installs the toolchain, builds PTX,
# records host RAM/vCPUs (offload results depend on them), runs hardware tests.
#
# Env: VOLUME (weights dir, default /workspace), VXC, VX_GPU_LLVM_BIN, PTXAS,
#      VX_GPU_ARCH (default sm_80; use sm_89 for L4/4090, sm_86 for A10).
set -euo pipefail
cd "$(dirname "$0")/.."
VOLUME=${VOLUME:-/workspace}
OUT=${OUT:-$VOLUME/glm-vx-pod}
mkdir -p "$OUT"

command -v nvidia-smi >/dev/null || { echo "no NVIDIA driver (nvidia-smi missing)" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv | tee "$OUT/gpu.csv"
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
[[ ${cap%%.*} -ge 8 ]] || { echo "need compute capability >= 8.0, got $cap" >&2; exit 2; }
: "${VX_GPU_ARCH:=sm_${cap/./}}"; export VX_GPU_ARCH

# Host resources: record for every benchmark.
{ echo "vcpus=$(nproc)"; awk '/MemTotal/{printf "host_ram_gib=%.1f\n",$2/1048576}' /proc/meminfo
  echo "arch=$VX_GPU_ARCH"; echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"; } | tee "$OUT/host.env"

: "${VXC:=$(command -v vxc || true)}"
[[ -n $VXC ]] || { echo "vxc not found: install Vx v0.0.2 + LLVM 22 (docs/toolchain.md), set VXC" >&2; exit 3; }
export VXC
command -v mlir-translate opt llc >/dev/null || [[ -n ${VX_GPU_LLVM_BIN:-} ]] \
  || { echo "mlir-translate/opt/llc not on PATH; set VX_GPU_LLVM_BIN" >&2; exit 3; }

[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/pip install -q -e '.[test]'
gpu/build.sh
# Real-device tests only: the CPU SIMT/.so tests need symbols that Vx main (#1745) now hides.
.venv/bin/python -m pytest gpu/test_hardware.py gpu/test_gemm_hardware.py gpu/test_index_scores_hardware.py -q -rs | tee "$OUT/gpu-tests.txt"
.venv/bin/python -m gpu.benchmark --kernel all --repeats 20 --output "$OUT/kernels.json"
echo "ok: results in $OUT"
