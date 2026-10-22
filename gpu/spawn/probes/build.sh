#!/usr/bin/env bash
# Build the tensor-core probes into gpu/build/spawn/probes/ (not part of gpu/spawn/build.sh). A
# probe named *_sm80.vx builds with machine-sm80.vx, *_sm90a.vx with machine-sm90a.vx. Needs a
# vxc with register tensors (Vx PR #2058, which replaced #1991) and, for sm_90a, the chip from the
# machine file (#1989).
# With PTXAS set, each probe is assembled for its chip and the script prints ptxas -v, whether the
# PTX has a __local_depot, and the size of the cubin's .text section (the SASS).
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${PYTHON:=python3}"
: "${OUT:=gpu/build/spawn/probes}"
mkdir -p "$OUT"
for source in gpu/spawn/probes/*_sm[0-9]*.vx; do
  name=$(basename "$source" .vx)
  chip=${name##*_}
  "$PYTHON" gpu/spawn/build.py --source "$source" --output "$OUT/$name.ptx" \
    --machine "gpu/spawn/probes/machine-$chip.vx"
  if grep -q __local_depot "$OUT/$name.ptx"; then echo "$name: PTX has a __local_depot"; fi
  if [[ -n "${PTXAS:-}" ]]; then
    "$PTXAS" -arch="sm_${chip#sm}" -v "$OUT/$name.ptx" -o "$OUT/$name.cubin" 2>&1 | sed "s/^/$name: /"
    "$PYTHON" - "$OUT/$name.cubin" <<'EOF'
import struct, sys
data = open(sys.argv[1], 'rb').read()
shoff, = struct.unpack_from('<Q', data, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 0x3A)
def section(i):
    return struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize)
names = section(shstrndx)
for i in range(shnum):
    s = section(i)
    name = data[names[4] + s[0]:].split(b'\0', 1)[0].decode()
    if name.startswith('.text.'):
        print(f'{sys.argv[1]}: {name} {s[5]} bytes ({s[5] // 16} SASS instructions)')
EOF
  fi
done
