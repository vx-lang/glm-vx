#!/usr/bin/env bash
# Build every gpu/spawn kernel into gpu/build/spawn/. Needs VXC (a vxc with --machine) on PATH
# or set; PTXAS optionally assembles each kernel and fails the build if it does not assemble.
# The PTX target stays sm_80 (build.py --arch); PTXAS_ARCH is the GPU ptxas checks it for, so set it
# to the real device (sm_90 on an H100).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PYTHON:=python3}"
: "${PTXAS_ARCH:=sm_80}"
extra=()
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS" --ptxas-arch "$PTXAS_ARCH"); fi
# VX_DISPATCH_LIB (libvx_cuda_dispatch.so from the vxc build) also builds lib<kernel>.so, which
# Vx's own runtime launches; vxc then needs llvm-config on PATH.
library=${VX_DISPATCH_LIB:+yes}
for source in gpu/spawn/*.vx; do
  name=$(basename "$source" .vx)
  [[ $name == machine || $name == device ]] && continue
  lib=()
  if [[ -n $library ]]; then lib=(--library "gpu/build/spawn/lib$name.so"); fi
  "$PYTHON" gpu/spawn/build.py --source "$source" --output "gpu/build/spawn/$name.ptx" "${extra[@]}" "${lib[@]}"
done
