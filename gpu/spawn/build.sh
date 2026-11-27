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
"$PYTHON" gpu/spawn/gen_topk_bitmap_stats.py --check
extra=()
if [[ -n "${MACHINE:-}" ]]; then extra+=(--machine "$MACHINE"); fi
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS" ${PTXAS_ARCH:+--ptxas-arch "$PTXAS_ARCH"}); fi
# VX_DISPATCH_LIB (libvx_cuda_dispatch.so from the vxc build) also builds lib<kernel>.so, which
# Vx's own runtime launches; vxc then needs llvm-config on PATH.
library=${VX_DISPATCH_LIB:+yes}
for source in gpu/spawn/*.vx; do
  name=$(basename "$source" .vx)
  # Only files with a kernel entry: not machine*.vx, device.vx or the variant includes (fp8_cvt_*.vx).
  grep -qE "^pub (extern )?fn glm_vx_spawn_" "$source" || continue
  [[ -n ${KERNELS:-} && " $KERNELS " != *" $name "* ]] && continue
  # `// spawn-variant: SUFFIX INCLUDE [MACHINE]` lines (build.py): one build each, <name><SUFFIX>.
  variants=$(sed -n 's|^//[[:space:]]*spawn-variant:[[:space:]]*\([^[:space:]]*\).*|\1|p' "$source")
  for suffix in ${variants:-none}; do
    out=$name
    args=()
    if [[ $suffix != none ]]; then
      args=(--variant "$suffix")
      [[ $suffix != - ]] && out=$name$suffix
    fi
    lib=()
    if [[ -n $library ]]; then lib=(--library "$OUT/lib$out.so"); fi
    "$PYTHON" gpu/spawn/build.py --source "$source" --output "$OUT/$out.ptx" "${extra[@]}" "${lib[@]}" "${args[@]}"
  done
done
# The kernels in subdirectories with their own build (inline PTX: no host library); KERNELS=index_mma or
# KERNELS=moe_mma builds only those.
if [[ -z ${KERNELS:-} || " $KERNELS " == *" index_mma "* ]]; then
  OUT="$OUT/index_mma" bash gpu/spawn/index_mma/build.sh
fi
if [[ -z ${KERNELS:-} || " $KERNELS " == *" moe_mma "* ]]; then
  OUT="$OUT/moe_mma" bash gpu/spawn/moe_mma/build.sh
fi
