#!/usr/bin/env bash
# Build the MoE kernels whose warp's work is inline PTX in an mlir! block (#63, docs/TP-SERVING-PLAN.md item 4)
# into gpu/build/spawn/moe_mma/: fp8_moe_mma_down_at (FP8 E4M3 mma.sync) and fp8_moe_mma16_down_at (act in f16), machine-sm90a.vx, PTX 8.4. PTX only: no
# host library (lib<kernel>.so), and the CPU emulator cannot run them; their checks are the GPU's
# (validation/moe_mma_check.py). gpu/spawn/build.sh runs this script.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${PYTHON:=python3}"
: "${OUT:=gpu/build/spawn/moe_mma}"
mkdir -p "$OUT"
"$PYTHON" gpu/spawn/moe_mma/gen_moe_mma.py --check
ptxas=()
if [[ -n "${PTXAS:-}" ]]; then ptxas=(--ptxas "$PTXAS"); fi
for name in fp8_moe_mma_down_at fp8_moe_mma16_down_at fp8_moe_mma16_gate_up_at; do
  "$PYTHON" gpu/spawn/build.py --source "gpu/spawn/moe_mma/$name.vx" --output "$OUT/$name.ptx" \
    --machine gpu/spawn/index_mma/machine-sm90a.vx --ptx-version 8.4 "${ptxas[@]}" ${PTXAS:+--ptxas-arch sm_90a}
done
