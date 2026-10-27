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
# Bundles packed from apt's LLVM keep libLLVM.so.22.1 and libclang-cpp.so.22.1 as symlinks into
# /usr/lib/x86_64-linux-gnu (371c2bae's does). They resolve only where apt.llvm.org's LLVM 22 is
# installed, so install its two runtime libraries and point the links there.
lib=$prefix/llvm-22/lib
if [[ ! -e $lib/libLLVM.so.22.1 || ! -e $lib/libclang-cpp.so.22.1 ]]; then
  if [[ ! -e /usr/lib/x86_64-linux-gnu/libLLVM.so.22.1 ]] && command -v apt-get >/dev/null; then
    apt-get install -y -qq wget gnupg ca-certificates
    wget -qO /etc/apt/trusted.gpg.d/apt.llvm.org.asc https://apt.llvm.org/llvm-snapshot.gpg.key
    echo "deb http://apt.llvm.org/$(. /etc/os-release; echo "$VERSION_CODENAME")/ llvm-toolchain-$(. /etc/os-release; echo "$VERSION_CODENAME")-22 main" \
      > /etc/apt/sources.list.d/llvm22.list
    apt-get update -qq
    apt-get install -y -qq libllvm22 libclang-cpp22
  fi
  ln -sf /usr/lib/x86_64-linux-gnu/libLLVM.so.22.1 "$lib/libLLVM.so.22.1"
  ln -sf /usr/lib/x86_64-linux-gnu/libclang-cpp.so.22.1 "$lib/libclang-cpp.so.22.1"
fi
cat > "$prefix/env.sh" <<E
export VX_TOOLCHAIN=$prefix
export PATH=$prefix/llvm-22/bin:$prefix/bin:\$PATH
export VXC=$prefix/bin/vxc
export VX_GPU_LLVM_BIN=$prefix/llvm-22/bin
export LD_LIBRARY_PATH=$prefix/llvm-22/lib:$prefix/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
E
# vxc links a program with a GPU spawn against VX_DISPATCH_LIB. Its built-in default is a path on
# the machine that built vxc, so without this the link fails on vx_plugin_* symbols.
if [[ -f $prefix/lib/libvx_cuda_dispatch.so ]]; then
  echo "export VX_DISPATCH_LIB=$prefix/lib/libvx_cuda_dispatch.so" >> "$prefix/env.sh"
fi
# Vx's collectives (gpu/collective.py) dlopen libnccl.so.2. When only the pip wheel
# (nvidia-nccl-cu12) has it, the loader cannot find it, so point VX_NCCL_LIB at the wheel's copy.
if ! ldconfig -p 2>/dev/null | grep -q 'libnccl\.so\.2 '; then
  nccl=$(python3 -c 'import nvidia.nccl, pathlib; print(pathlib.Path(nvidia.nccl.__path__[0]) / "lib" / "libnccl.so.2")' 2>/dev/null || true)
  if [[ -n $nccl && -f $nccl ]]; then
    echo "export VX_NCCL_LIB=$nccl" >> "$prefix/env.sh"
  else
    echo "note: no libnccl.so.2 found; Vx's collectives need it (pip install nvidia-nccl-cu12)" >&2
  fi
fi
. "$prefix/env.sh"
missing=$(for f in "$prefix/bin/vxc" "$prefix"/llvm-22/bin/{mlir-translate,opt,llc,clang}; do ldd "$f" | grep 'not found' || true; done)
[[ -z $missing ]] || { echo "missing shared libraries:" >&2; echo "$missing" >&2; exit 3; }
"$VXC" --version
mlir-translate --version | head -2
echo "ok: source $prefix/env.sh"
