#!/usr/bin/env bash
# Build Vx main 8146e0d05 (after #3646-#3652) and the H200-tested #3649 head 4d15f5315, then build
# glm-vx HEAD's spawn kernels with each (and with the 3609 vxc of deploy15) and compare the PTX.
set -uo pipefail
D=/mnt/scratch/glm-vx-claude-d134c2f5
S=/tmp/claude-1000/-home-ubuntu-go-glm-vx/d134c2f5-ebfa-4ac3-a987-f90b0db413ca/scratchpad
REPO=/home/ubuntu/go/glm-vx
export PATH=$HOME/.cargo/bin:/usr/lib/llvm-22/bin:$PATH
export PTXAS=/mnt/scratch/nvhpc/sdk/Linux_x86_64/26.9/cuda/13.3/bin/ptxas PTXAS_ARCH=sm_90

build_vx() {  # tag srcdir
  local tag=$1 src=$2
  cp -a $D/vxtarget-3609 $D/vxtarget-$tag
  find $src -type f -exec touch {} +
  ( cd $src && CARGO_TARGET_DIR=$D/vxtarget-$tag cargo build --release ) > $D/vx-$tag-build.log 2>&1
  echo "vx $tag cargo rc=$?: $(tail -1 $D/vx-$tag-build.log)"
  $D/vxtarget-$tag/release/vxc --version 2>&1 | head -1
}

mkdir $D/vx-4d15 && tar -xzf ~/pod-toolchain/vx-src-4d15f5315.tar.gz -C $D/vx-4d15
build_vx 8146 $D/vx-8146
build_vx 4d15 $D/vx-4d15

# The kernels from a clean copy of HEAD, so a build never writes into the work tree.
mkdir $D/head-c3ae && git -C $REPO archive HEAD | tar -x -C $D/head-c3ae
for tag in 3609 4d15 8146; do
  ( cd $D/head-c3ae && VXC=$D/vxtarget-$tag/release/vxc OUT=$D/kern-$tag gpu/spawn/build.sh ) > $D/kern-$tag.log 2>&1
  echo "kernels $tag rc=$?: $(cd $D/kern-$tag && find . -name '*.ptx' | wc -l) ptx; errors: $(grep -ci 'error' $D/kern-$tag.log)"
done

cmp_dirs() {  # a b
  local same=0 diff=0 list=()
  for f in $(cd $D/kern-$1 && find . -name '*.ptx' | sort); do
    if cmp -s $D/kern-$1/$f $D/kern-$2/$f; then same=$((same+1)); else diff=$((diff+1)); list+=($f); fi
  done
  echo "$1 vs $2: $same identical, $diff differ ${list[*]}"
}
cmp_dirs 4d15 8146
cmp_dirs 3609 8146
cmp_dirs 3609 4d15
echo vxmain-done
