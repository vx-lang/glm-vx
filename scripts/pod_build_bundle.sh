#!/usr/bin/env bash
# Build a Vx toolchain bundle for scripts/pod_restore.sh, on a pod: vxc must link against the pods'
# glibc (2.39 on the Ubuntu 24.04 images), not the dev box's (2.43), so it cannot be built there.
#
#   scripts/pod_build_bundle.sh VX_SRC.tar.gz OLD_BUNDLE.tar.zst [WORK]   # WORK default /root/vx-bundle
#
# VX_SRC is `git archive` of a Vx commit, named vx-src-<commit>.tar.gz (on the dev box:
# git -C ~/go/Vx archive --format=tar.gz -o vx-src-<commit>.tar.gz <commit>); OLD_BUNDLE is any
# earlier vx-toolchain-*.tar.zst, whose llvm-22/ the new bundle reuses. It installs apt.llvm.org's
# LLVM 22 and MLIR development packages and Rust (when missing), builds Vx, and writes
# WORK/vx-toolchain-<commit>.tar.zst: bin/vxc, lib/ (libvx_std_core, libvx_cuda_dispatch,
# libvx_mlir_shims), stdlib/ and llvm-22/. APT=0 skips the installs (they need root).
set -euo pipefail
src=$(realpath "${1:?usage: pod_build_bundle.sh VX_SRC.tar.gz OLD_BUNDLE.tar.zst [WORK]}")
old=$(realpath "${2:?usage: pod_build_bundle.sh VX_SRC.tar.gz OLD_BUNDLE.tar.zst [WORK]}")
work=${3:-/root/vx-bundle}
commit=$(basename "$src" .tar.gz); commit=${commit#vx-src-}
name=vx-toolchain-$commit
mkdir -p "$work"
work=$(realpath "$work")

if [[ ${APT:-1} != 0 ]] && command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq wget gnupg lsb-release software-properties-common build-essential cmake pkg-config \
    zstd libzstd-dev zlib1g-dev libz3-dev libffi-dev libxml2-dev libedit-dev curl
  if [[ ! -x /usr/lib/llvm-22/bin/mlir-translate ]]; then
    wget -qO /tmp/llvm.sh https://apt.llvm.org/llvm.sh
    bash /tmp/llvm.sh 22
    apt-get install -y -qq libmlir-22-dev mlir-22-tools clang-22 lld-22 llvm-22-dev
  fi
fi
if ! command -v cargo >/dev/null && [[ ! -x $HOME/.cargo/bin/cargo ]]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH=$HOME/.cargo/bin:/usr/lib/llvm-22/bin:$PATH

rm -rf "$work/src" "$work/$name"
mkdir -p "$work/src"
tar -xzf "$src" -C "$work/src"
# A full build: it also makes libvx_std_core.so, which programs with a GPU spawn link against.
(cd "$work/src" && CARGO_TARGET_DIR="$work/target" cargo build --release)
out=$(ls -d "$work"/target/release/build/vxc-*/out | while read -r d; do
  [[ -f $d/libvx_cuda_dispatch.so ]] && echo "$d"; done | head -1)
[[ -n $out ]] || { echo "no libvx_cuda_dispatch.so in the build (is the CUDA toolkit installed?)" >&2; exit 3; }

mkdir -p "$work/$name/bin" "$work/$name/lib"
cp "$work/target/release/vxc" "$work/$name/bin/"
cp "$work/target/release/libvx_std_core.so" "$out/libvx_cuda_dispatch.so" "$out/libvx_mlir_shims.so" "$work/$name/lib/"
cp -r "$work/src/stdlib" "$work/$name/stdlib"
tar --use-compress-program=unzstd -xf "$old" -C "$work/$name" --strip-components=1 --wildcards '*/llvm-22'

nm -D "$work/$name/lib/libvx_cuda_dispatch.so" | grep -q ' vx_cuda_set_stream$' \
  || { echo "libvx_cuda_dispatch.so does not export vx_cuda_set_stream" >&2; exit 3; }
echo "glibc needed by vxc: $(objdump -T "$work/$name/bin/vxc" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
tar -C "$work" --use-compress-program='zstd -T0 -10' -cf "$work/$name.tar.zst" "$name"
ls -la "$work/$name.tar.zst"
echo "ok: check it with scripts/pod_restore.sh $work/$name.tar.zst, then gpu/spawn/build.sh"
