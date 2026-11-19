#!/usr/bin/env bash
# Compare two spawn build directories with the vxc source directory normalized: the PTX embeds the path of
# Vx's stdlib (core/sync/atomic.vx) in its assert messages as bytes, here .../vx-4d15/, .../vx-8146/ or
# .../vx-3609/ (the same length), which is the only difference between those builds but mla_tc_at's.
#   ptx_same.sh DIR_A DIR_B
a=$1 b=$2
norm() { sed -E 's/118, 120, 45, (52, 100, 49, 53|56, 49, 52, 54|51, 54, 48, 57)/118, 120, 45, TAG/g' "$1"; }
same=0 differ=()
for f in $(cd "$a" && find . -name '*.ptx' | sort); do
  if cmp -s <(norm "$a/$f") <(norm "$b/$f"); then same=$((same+1)); else differ+=("$f"); fi
done
echo "$same identical, ${#differ[@]} differ: ${differ[*]}"
