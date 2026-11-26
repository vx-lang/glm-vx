#!/bin/bash
# Build glm-vx-front and its test programs (#68, step 2) into front/build/.
# The package `front` (front/src) is compiled with `--no-trusted-code front`: no `unsafe`, `extern`,
# `mlir!`, `extern impl` or `unsafe impl` in it. Needs Vx main from 2026-11-26 on (Vx #4122, #4130,
# #4134, #4136, #4137). VXC names vxc; VX_RUNTIME_DIR the folder of libvx_std_core.so (vxc's target
# folder); VX_SHIMS_DIR the folder of libvx_mlir_shims.so (vxc's build script output).
set -euo pipefail
cd "$(dirname "$0")"
: "${VXC:=vxc}"
: "${VX_RUNTIME_DIR:=$(dirname "$(command -v "$VXC")")}"
: "${VX_SHIMS_DIR:=$(dirname "$(find "$VX_RUNTIME_DIR/build" -name libvx_mlir_shims.so -newer "$VXC" 2>/dev/null | head -1 || true)")}"
: "${CC:=clang}"
LLVM_LIB=$(llvm-config --libdir)
mkdir -p build
link() {  # MAIN.vx OUT
  "$VXC" "$1" --package front="$PWD/src" --no-trusted-code front --action emit-obj -o "build/$2.o"
  "$CC" "build/$2.o" -o "build/$2" -Wl,-rpath,"$LLVM_LIB" -Wl,-rpath,"$VX_RUNTIME_DIR" -Wl,-rpath,"$VX_SHIMS_DIR" \
    "$LLVM_LIB/libmlir_c_runner_utils.so" "$LLVM_LIB/libmlir_runner_utils.so" "$VX_RUNTIME_DIR/libvx_std_core.so" \
    "$VX_SHIMS_DIR/libvx_mlir_shims.so" -Wl,--allow-shlib-undefined -lm
}
link main.vx glm-vx-front
link tests/json_echo.vx json_echo
link tests/request_check.vx request_check
"$VXC" main.vx --package front="$PWD/src" --no-trusted-code front --audit > build/audit.txt 2>&1
echo "built: $(ls build | grep -v '\.o$' | tr '\n' ' ')"
