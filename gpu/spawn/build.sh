#!/usr/bin/env bash
# Build every gpu/spawn kernel into gpu/build/spawn/. Needs VXC (a vxc with --machine) on PATH
# or set; PTXAS optionally assembles each kernel and fails the build if it does not assemble.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${PYTHON:=python3}"
extra=()
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS"); fi
for source in gpu/spawn/*.vx; do
  name=$(basename "$source" .vx)
  [[ $name == machine || $name == device ]] && continue
  "$PYTHON" gpu/spawn/build.py --source "$source" --output "gpu/build/spawn/$name.ptx" "${extra[@]}"
done
