#!/usr/bin/env bash
# Every pending GPU check in one run, for a pod with the toolchain restored (pod_restore.sh).
# Rebuilds the PTX and the CPU library from this checkout, runs the hardware tests and the
# resident comparisons (float32, INT4, FP8, BF16; small and full width), a concurrent-sharing
# stress pass, and writes everything to an evidence directory. It keeps going after a failure
# and ends with a summary table; the exit code is the number of failed steps.
#
#   source /opt/vx-toolchain/env.sh
#   scripts/pod_validate.sh [EVIDENCE_DIR]      # default evidence/<gpu>-<UTC date>, e.g. h100-20261022
#
# Optional: GLM_VX_INT4_CHECKPOINT=<dir from validation/fetch_tensors.py> adds the real
# compressed-tensors Linears to the INT4 device test. PYTHON (default python3), LAYERS (full-width
# layers, default 4), TOKENS (default 48), REFERENCE_TOKENS (tokens of the full-width hybrid
# reference, default 20), STRESS (concurrent stress processes, default 8).
# Use OMP_NUM_THREADS well below the host's vCPU count (NumPy thrashes on 128-vCPU pods).
#
# Toolchain: one vxc bundle from Vx main, 371c2bae or later (#2240: mla_decode_at reads its step
# buffer; #2247: its 1024-thread block is declared to ptxas, or the launch is refused). It also has
# #2134 (one library per kernel in a process), the `lanes::` reductions of #260 (#2117, #2124,
# #2128), the `across` loops of #2095 to #2097, `if` values of #2112, the code of a group of blocks
# and its arrival counts that mla_decode uses (#2185 to #2188), a library's pub fn that gives
# mla_decode a device arrival count (#2232) and loads without vx_stderr_write (#2230), and
# vx_cuda_set_stream of #2100, which the graph step needs (host.txt says whether the dispatch
# library exports it). VXC, and VX_DISPATCH_LIB or <bundle>/lib/libvx_cuda_dispatch.so.
# Do not export VX_CUDA_SYNC or VX_TIME_KERNEL for the whole run: Vx stops a graph capture on either.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON:=python3}" "${LAYERS:=4}" "${TOKENS:=48}" "${REFERENCE_TOKENS:=20}" "${STRESS:=8}" "${OMP_NUM_THREADS:=16}"
export OMP_NUM_THREADS
# The PTX target stays sm_80 (gpu/build.sh), which the driver JITs on newer GPUs; ptxas checks it for
# the device's own arch (PTXAS_ARCH, e.g. sm_90 on an H100) and runs whenever a CUDA ptxas is found.
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]' | grep -oE '\b[abhlt][0-9]{2,3}\b' | head -1)
if [[ -n $cap && -z ${PTXAS_ARCH:-} ]]; then export PTXAS_ARCH=sm_${cap/./}; fi
if [[ -z ${PTXAS:-} && -x /usr/local/cuda/bin/ptxas ]]; then export PTXAS=/usr/local/cuda/bin/ptxas; fi
out=${1:-evidence/${gpu:-gpu}-$(date -u +%Y%m%d)}
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
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
  echo "ptx target ${VX_GPU_ARCH:-sm_80}, ptxas ${PTXAS:-none} for ${PTXAS_ARCH:-${VX_GPU_ARCH:-sm_80}}"
  if git rev-parse -q HEAD >/dev/null 2>&1; then
    echo "repo $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
  else
    echo "repo ${REPO_REV:-unknown} (no .git; set REPO_REV)"
  fi
  echo "vxc $("${VXC:-vxc}" --version 2>&1 | head -1)"
  echo "llvm $(llc --version 2>&1 | grep -m1 -o 'LLVM version [0-9.]*')"
  $PYTHON --version
} > "$out/host.txt" 2>&1

# PTXAS (default /usr/local/cuda/bin/ptxas when present) also assembles each PTX to a cubin for
# PTXAS_ARCH; this needs no GPU, so it is worth running before renting one too:
#   PTXAS=... PTXAS_ARCH=sm_90 gpu/build.sh
step build-ptx bash gpu/build.sh
# The spawn kernels also become lib<kernel>.so, launched by Vx's own runtime, when the bundle has
# the CUDA dispatch library vxc was built with (lib/libvx_cuda_dispatch.so).
if [[ -z ${VX_DISPATCH_LIB:-} ]]; then
  candidate=$(dirname "$(command -v "${VXC:-vxc}")")/../lib/libvx_cuda_dispatch.so
  [[ -f $candidate ]] && export VX_DISPATCH_LIB=$candidate
fi
{
  echo "dispatch lib ${VX_DISPATCH_LIB:-none}"
  echo "vx_cuda_set_stream (Vx #2100): $(nm -D --defined-only "${VX_DISPATCH_LIB:-/dev/null}" 2>/dev/null | grep -qw vx_cuda_set_stream && echo exported || echo MISSING)"
} >> "$out/host.txt"
step build-spawn bash gpu/spawn/build.sh
step build-cpu-lib bash kernels/build.sh
sha256sum gpu/build/*.ptx gpu/build/spawn/*.ptx gpu/build/spawn/*.so > "$out/ptx.sha256" 2>/dev/null

# Hardware tests skip without a usable device; a run where nothing passed is a failure, not "ok".
hw() { $PYTHON -m pytest -q -rs "$@" | tee /dev/stderr | grep -qE '[0-9]+ passed'; }
step test-hardware hw gpu/test_hardware.py gpu/test_gemm_hardware.py gpu/test_index_scores_hardware.py gpu/test_topk_hardware.py
step test-quant-hardware hw gpu/test_quant_hardware.py
step test-spawn-hardware hw gpu/test_spawn_hardware.py
# The same libraries on the CPU (Vx runs the regions in order with no device visible), against NumPy.
step test-spawn-vx-host hw gpu/test_spawn_vx_host.py
step test-spawn-vx-launch hw gpu/test_spawn_vx_launch.py
hw_sync() { VX_CUDA_SYNC=1 hw "$@"; }  # Vx waits after each launch, and names the kernel on a fault
step test-spawn-vx-launch-sync hw_sync gpu/test_spawn_vx_launch.py
# Vx's launches on our stream and captured in CUDA graphs (glm-vx #9). Each scenario runs in a child
# process, since Vx stops the program on what a capture cannot hold; -s keeps every child's stderr
# (Vx's messages) and the eager-vs-replay timing lines in the log, on success too.
step test-spawn-vx-graph hw -s gpu/test_spawn_vx_graph.py
# The block-FP8 GEMM (glm-vx #10, gpu/FP8-GEMM.md) is not built: its wgmma, TMA and mbarrier
# `mlir!` blocks cannot be in an `across` loop until Vx #2076 (declared writes) lands, and Vx main
# no longer takes `grid([..]) block(n)`. Its step comes back with that.
step test-hardware-stress hw gpu/test_hardware_stress.py
# Resident decode with the routed experts chosen on the device (no host read of the router ids)
# against the host loop, bit for bit, for each weight format.
step test-resident-device-experts hw gpu/test_resident_device_experts_hardware.py
# Resident decode with each layer's attention in one launch (mla_decode) against mla_partial then
# mla_merge, over 700 steps: up to 3 splits per head, one arrival-count buffer throughout.
step test-resident-fused-attention hw gpu/test_resident_fused_attention_hardware.py
# The decode step captured once as a CUDA graph and replayed every token (#25, #26) against the
# per-launch path, bit for bit, 5 weight layouts and 700 steps through 1 to 3 MLA splits.
step test-resident-graph hw gpu/test_resident_graph_hardware.py

# Full-width float32 checks resident against the hybrid path, which reads every result back (about
# 33 s per token): REFERENCE_TOKENS of them (default 20, above the full config's index_topk of 16).
for mode in f32 int4 fp8 bf16; do
  flag=$([[ $mode == f32 ]] && echo "" || echo "--$mode")
  step resident-$mode-small $PYTHON -m validation.resident_compare $flag --tokens "$TOKENS" --output "$out/resident-$mode-small.json"
  reference=$([[ $mode == f32 ]] && echo "--reference-tokens $REFERENCE_TOKENS" || echo "")
  step resident-$mode-full$LAYERS $PYTHON -m validation.resident_compare $flag $reference --geometry full --layers "$LAYERS" \
    --tokens "$TOKENS" --output "$out/resident-$mode-full$LAYERS.json"
done
# Graph replay against the per-launch path (#26), bit for bit, every routed expert resident.
for mode in f32 int4 fp8 bf16; do
  step graph-$mode-small $PYTHON -m validation.resident_graph --format $mode --tokens "$TOKENS" --output "$out/graph-$mode-small.json"
done
# Where one full-width layer spends its GPU time, kernel by kernel, under Nsight Compute: layer 0
# (dense) and layer 3 (MoE) of one decode step, with DRAM bytes and % of peak bandwidth and compute.
# Then the fused MoE expert kernels alone at GLM-5.3's shapes, k = 1 to 8 routed experts, against
# the bytes they must read. Without ncu or its counters (ERR_NVGPUCTRPERM in a container), both time
# each launch with CUDA events instead, and their reports say so.
for mode in f32 int4 fp8 bf16; do
  step layer-profile-$mode $PYTHON -m validation.layer_profile --format $mode --output "$out/layer-profile-$mode.json"
done
step moe-profile $PYTHON -m validation.moe_profile --output "$out/moe-profile.json"
# For Vx: mla_decode_at as built (ptxas caps it at 32 registers on sm_90, an 8 B spill) against
# its PTX with `.minnctapersm 1` (64 registers, 1 block per SM), bit for bit and timed.
step mla-hint $PYTHON -m validation.mla_hint --output "$out/mla-hint.json"

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
