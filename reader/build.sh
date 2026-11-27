#!/usr/bin/env bash
# Build glm-vx-reader (#68): VXC=/path/to/vxc reader/build.sh. Needs Vx main 6322eb757 or later, as
# front/build.sh does, whose JSON parser it uses.
set -euo pipefail
cd "$(dirname "$0")"
: "${VXC:=vxc}"
: "${VX_RUNTIME_DIR:=$(dirname "$(command -v "$VXC")")}"
: "${VX_SHIMS_DIR:=$(dirname "$(find "$VX_RUNTIME_DIR/build" -name libvx_mlir_shims.so -newer "$VXC" 2>/dev/null | head -1 || true)")}"
: "${CC:=clang}"
LLVM_LIB=$(llvm-config --libdir)
FLAGS=(--package reader="$PWD/src" --package front="$PWD/../front/src" --no-trusted-code reader --no-trusted-code front)
mkdir -p build
"$VXC" main.vx "${FLAGS[@]}" --action emit-obj -o build/glm-vx-reader.o
"$CC" build/glm-vx-reader.o -o build/glm-vx-reader -Wl,-rpath,"$LLVM_LIB" -Wl,-rpath,"$VX_RUNTIME_DIR" -Wl,-rpath,"$VX_SHIMS_DIR" \
  "$LLVM_LIB/libmlir_c_runner_utils.so" "$LLVM_LIB/libmlir_runner_utils.so" "$VX_RUNTIME_DIR/libvx_std_core.so" \
  "$VX_SHIMS_DIR/libvx_mlir_shims.so" -Wl,--allow-shlib-undefined -lm
"$VXC" main.vx "${FLAGS[@]}" --audit > build/audit.txt 2>&1
echo "built: build/glm-vx-reader"
