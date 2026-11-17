#!/usr/bin/env bash
# Build index_scores_mma_at (#32, ResidentGLM(index_kernel='mma')) into gpu/build/spawn/index_mma/. Its
# warp's work is inline PTX in an mlir! block, so it has PTX only: no host library (lib<kernel>.so),
# and the CPU emulator cannot run it; its tests are the GPU's. gpu/spawn/build.sh runs this script.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${PYTHON:=python3}"
: "${OUT:=gpu/build/spawn/index_mma}"
mkdir -p "$OUT"
"$PYTHON" gpu/spawn/index_mma/gen_index_mma.py --check
extra=()
if [[ -n "${MACHINE:-}" ]]; then extra+=(--machine "$MACHINE"); fi
if [[ -n "${PTXAS:-}" ]]; then extra+=(--ptxas "$PTXAS" ${PTXAS_ARCH:+--ptxas-arch "$PTXAS_ARCH"}); fi
"$PYTHON" gpu/spawn/build.py --source gpu/spawn/index_mma/index_scores_mma_at.vx --output "$OUT/index_scores_mma_at.ptx" "${extra[@]}"
