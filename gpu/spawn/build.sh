#!/usr/bin/env bash
# Build every gpu/spawn kernel into gpu/build/spawn/. Needs VXC (a vxc with --machine) on PATH
# or set; PTXAS optionally assembles each kernel and fails the build if it does not assemble.
# MACHINE picks the machine file (default gpu/spawn/machine.vx, an A100: sm_80 PTX); the PTX target
# is its `chip:` (Vx's fleet/h100-sxm.vx says sm_90a). PTXAS_ARCH is the GPU ptxas checks it for,
# default the PTX target; set sm_90 to check sm_80 PTX for an H100. OUT is the output directory.
# KERNELS (names, space-separated) builds only those.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PYTHON:=python3}"
: "${OUT:=gpu/build/spawn}"
extra=()
if [[ -n "${MACHINE:-}" ]]; then extra+=(--machine "$MACHINE"); fi
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS" ${PTXAS_ARCH:+--ptxas-arch "$PTXAS_ARCH"}); fi
# VX_DISPATCH_LIB (libvx_cuda_dispatch.so from the vxc build) also builds lib<kernel>.so, which
# Vx's own runtime launches; vxc then needs llvm-config on PATH.
library=${VX_DISPATCH_LIB:+yes}
for source in gpu/spawn/*.vx; do
  name=$(basename "$source" .vx)
  [[ $name == machine || $name == device ]] && continue
  [[ -n ${KERNELS:-} && " $KERNELS " != *" $name "* ]] && continue
  lib=()
  if [[ -n $library ]]; then lib=(--library "$OUT/lib$name.so"); fi
  "$PYTHON" gpu/spawn/build.py --source "$source" --output "$OUT/$name.ptx" "${extra[@]}" "${lib[@]}"
done
