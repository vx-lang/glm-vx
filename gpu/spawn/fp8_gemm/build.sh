#!/usr/bin/env bash
# Build the block-FP8 GEMM (sm_90a) into gpu/build/spawn/fp8_gemm/: the kernel variants and the
# split-K reduction, with machine-sm90a.vx. Not part of gpu/spawn/build.sh (whose kernels are
# sm_80). Needs a vxc from Vx main (which has #1989, sm_90a; #1995, dynamic shared memory; #1996,
# aligned(N); and #2058, REG tensors, which replaced #1991) with #2065 (grid/block). With
# PTXAS set, each kernel is assembled for sm_90a and the script prints its registers, stack, spills
# and shared memory.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${PYTHON:=python3}"
: "${OUT:=gpu/build/spawn/fp8_gemm}"
mkdir -p "$OUT"
"$PYTHON" gpu/spawn/fp8_gemm/gen_fp8_gemm.py --check
extra=()
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS"); fi
for source in gpu/spawn/fp8_gemm/fp8_gemm_*.vx; do
  name=$(basename "$source" .vx)
  "$PYTHON" gpu/spawn/build.py --source "$source" --output "$OUT/$name.ptx" --ptx-version 8.3 \
    --machine gpu/spawn/fp8_gemm/machine-sm90a.vx "${extra[@]}"
  if grep -q __local_depot "$OUT/$name.ptx"; then echo "$name: PTX has a __local_depot" >&2; exit 1; fi
  if [[ -n "${PTXAS:-}" ]]; then
    "$PYTHON" - "$OUT/$name.json" <<'EOF'
import json, re, sys
j = json.load(open(sys.argv[1]))
log = ' '.join(j['ptxas_log'])
regs = re.search(r'Used (\d+) registers', log)[1]
stack = re.search(r'(\d+) bytes stack frame', log)[1]
spills = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', log)
print(f"{j['entry']}: {regs} registers, {stack} B stack, {spills[1]}/{spills[2]} B spill stores/loads, "
      f"{j.get('dynamic_shared_bytes', 0)} B dynamic shared, block {j['launch']['block']}")
if int(stack) or int(spills[1]) or int(spills[2]):
    sys.exit(f"{j['entry']}: stack or spills")
EOF
  fi
done
