#!/usr/bin/env bash
# Every pending GPU check in one run, for a pod with the toolchain restored (pod_restore.sh).
# Rebuilds the PTX and the CPU library from this checkout, runs the hardware tests and the
# resident comparisons (float32, INT4, FP8, BF16; small and full width), a concurrent-sharing
# stress pass, and writes everything to an evidence directory. It keeps going after a failure
# and ends with a summary table; the exit code is the number of failed steps.
#
#   source /opt/vx-toolchain/env.sh
#   scripts/pod_validate.sh [EVIDENCE_DIR]      # default evidence/a100-<UTC date>
#
# Optional: GLM_VX_INT4_CHECKPOINT=<dir from validation/fetch_tensors.py> adds the real
# compressed-tensors Linears to the INT4 device test. PYTHON (default python3), LAYERS (full-width
# layers, default 4), TOKENS (default 48), STRESS (concurrent stress processes, default 8).
# Use OMP_NUM_THREADS well below the host's vCPU count (NumPy thrashes on 128-vCPU pods).
set -uo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3}" "${LAYERS:=4}" "${TOKENS:=48}" "${STRESS:=8}" "${OMP_NUM_THREADS:=16}"
export OMP_NUM_THREADS
out=${1:-evidence/a100-$(date -u +%Y%m%d)}
if [[ -e $out ]]; then echo "$out exists; preserve evidence" >&2; exit 2; fi
mkdir -p "$out"
failed=0
summary=()

step() {  # step NAME CMD...: run, log to $out/NAME.txt, record the result
  local name=$1; shift
  local start=$SECONDS
  echo "== $name: $*"
  "$@" > "$out/$name.txt" 2>&1
  local code=$?
  summary+=("$(printf '%-28s %-4s %5ss' "$name" "$([[ $code == 0 ]] && echo ok || echo FAIL)" $((SECONDS - start)))")
  if [[ $code != 0 ]]; then failed=$((failed + 1)); tail -5 "$out/$name.txt"; fi
}

{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "repo $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  echo "vxc $("${VXC:-vxc}" --version 2>&1 | head -1)"
  echo "llvm $(llc --version 2>&1 | grep -m1 -o 'LLVM version [0-9.]*')"
  $PYTHON --version
} > "$out/host.txt" 2>&1

# PTXAS (optional, e.g. /usr/local/cuda/bin/ptxas) also assembles each PTX to an sm_80 cubin; this
# needs no GPU, so it is worth running before renting one too: PTXAS=... gpu/build.sh
step build-ptx bash gpu/build.sh
step build-cpu-lib bash kernels/build.sh
sha256sum gpu/build/*.ptx > "$out/ptx.sha256"

# Hardware tests skip without a usable device; a run where nothing passed is a failure, not "ok".
hw() { $PYTHON -m pytest -q -rs "$@" | tee /dev/stderr | grep -qE '[0-9]+ passed'; }
step test-hardware hw gpu/test_hardware.py gpu/test_gemm_hardware.py gpu/test_index_scores_hardware.py gpu/test_topk_hardware.py
step test-quant-hardware hw gpu/test_quant_hardware.py
step test-hardware-stress hw gpu/test_hardware_stress.py

for mode in f32 int4 fp8 bf16; do
  flag=$([[ $mode == f32 ]] && echo "" || echo "--$mode")
  step resident-$mode-small $PYTHON -m validation.resident_compare $flag --tokens "$TOKENS" --output "$out/resident-$mode-small.json"
  step resident-$mode-full$LAYERS $PYTHON -m validation.resident_compare $flag --geometry full --layers "$LAYERS" \
    --tokens "$TOKENS" --output "$out/resident-$mode-full$LAYERS.json"
done

# Sharing: the upload race (docs/GPU-BRINGUP.md) only showed with several processes on one GPU.
stress() {
  local pids=() code=0
  for s in $(seq 1 "$STRESS"); do
    $PYTHON -m gpu.stress_ops --seed "$s" --iters 300 > "$out/stress-$s.txt" 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || code=1; done
  ! grep -l MISMATCH "$out"/stress-*.txt && return $code
}
step stress stress

{
  echo "pod_validate $(date -u +%FT%TZ), $(head -2 "$out/host.txt" | tail -1 | cut -d, -f1)"
  printf '%s\n' "${summary[@]}"
  echo "failed steps: $failed"
} | tee "$out/summary.txt"
exit "$failed"
