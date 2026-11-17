#!/usr/bin/env bash
# Build the indexer kernels whose warp's work is inline PTX in an mlir! block (#32) into
# gpu/build/spawn/index_mma/: index_scores_mma_at (bf16, ResidentGLM(index_kernel='mma'), sm_80 PTX),
# and index_key_fp8_at and index_scores_fp8_at (FP8 E4M3, index_kernel='fp8': machine-sm90a.vx, PTX
# 8.4 for the e4m3 mma.sync), and topk_select_warp_at (long_topk='warp', sm_80 PTX). PTX only: no host library (lib<kernel>.so), and the CPU emulator cannot
# run them; their tests are the GPU's. gpu/spawn/build.sh runs this script.
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${PYTHON:=python3}"
: "${OUT:=gpu/build/spawn/index_mma}"
mkdir -p "$OUT"
"$PYTHON" gpu/spawn/index_mma/gen_index_mma.py --check
ptxas=()
if [[ -n "${PTXAS:-}" ]]; then ptxas=(--ptxas "$PTXAS"); fi
machine=()
if [[ -n "${MACHINE:-}" ]]; then machine=(--machine "$MACHINE"); fi
"$PYTHON" gpu/spawn/build.py --source gpu/spawn/index_mma/index_scores_mma_at.vx --output "$OUT/index_scores_mma_at.ptx" \
  "${machine[@]}" "${ptxas[@]}" ${PTXAS:+${PTXAS_ARCH:+--ptxas-arch "$PTXAS_ARCH"}}
"$PYTHON" gpu/spawn/build.py --source gpu/spawn/index_mma/topk_select_warp_at.vx --output "$OUT/topk_select_warp_at.ptx" \
  "${machine[@]}" "${ptxas[@]}" ${PTXAS:+${PTXAS_ARCH:+--ptxas-arch "$PTXAS_ARCH"}}
"$PYTHON" gpu/spawn/build.py --source gpu/spawn/index_mma/topk_select_ilp_at.vx --output "$OUT/topk_select_ilp_at.ptx" \
  "${machine[@]}" "${ptxas[@]}" ${PTXAS:+${PTXAS_ARCH:+--ptxas-arch "$PTXAS_ARCH"}}
for name in index_key_fp8_at index_scores_fp8_at; do
  "$PYTHON" gpu/spawn/build.py --source "gpu/spawn/index_mma/$name.vx" --output "$OUT/$name.ptx" \
    --machine gpu/spawn/index_mma/machine-sm90a.vx --ptx-version 8.4 "${ptxas[@]}" ${PTXAS:+--ptxas-arch sm_90a}
done
