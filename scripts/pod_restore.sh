#!/usr/bin/env bash
# Restore a saved Vx toolchain bundle on a fresh Linux box (instead of building Vx and
# LLVM there). Takes seconds; the build it replaces took several minutes of rented time.
#
#   scripts/pod_restore.sh vx-toolchain-<vx-commit>.tar.zst [PREFIX]   # PREFIX default /opt/vx-toolchain
#
# Then:  source PREFIX/env.sh   (sets VXC and puts LLVM 22 first on PATH)
#
# The bundle holds: bin/vxc (built from Vx at the commit in the file name), stdlib/,
# lib/, and llvm-22/ (the apt.llvm.org tools without static libraries). It must have been
# built on a glibc no newer than this box's; build it on the oldest box you rent (Ubuntu
# 24.04, glibc 2.39). Only distro runtime libraries are installed here, no compilers or Rust.
set -euo pipefail
bundle=${1:?usage: pod_restore.sh BUNDLE.tar.zst [PREFIX]}
prefix=${2:-/opt/vx-toolchain}
mkdir -p "$prefix"
case $bundle in
  *.zst) tar --use-compress-program=unzstd -xf "$bundle" -C "$prefix" --strip-components=1 ;;
  *.gz)  tar -xzf "$bundle" -C "$prefix" --strip-components=1 ;;
  *) echo "unknown bundle type: $bundle" >&2; exit 2 ;;
esac
if command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq libz3-4 libzstd1 zlib1g libstdc++6 libxml2 libedit2 libtinfo6 build-essential
fi
cat > "$prefix/env.sh" <<E
export VX_TOOLCHAIN=$prefix
export PATH=$prefix/llvm-22/bin:$prefix/bin:\$PATH
export VXC=$prefix/bin/vxc
export VX_GPU_LLVM_BIN=$prefix/llvm-22/bin
E
. "$prefix/env.sh"
missing=$(for f in "$prefix/bin/vxc" "$prefix"/llvm-22/bin/{mlir-translate,opt,llc,clang}; do ldd "$f" | grep 'not found' || true; done)
[[ -z $missing ]] || { echo "missing shared libraries:" >&2; echo "$missing" >&2; exit 3; }
"$VXC" --version
mlir-translate --version | head -2
echo "ok: source $prefix/env.sh"
