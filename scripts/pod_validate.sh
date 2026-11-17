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
# reference, default 20), STRESS (concurrent stress processes, default 8), TIMER (ncu, or events to
# skip the ncu pass where a container blocks its counters; it falls back to events anyway),
# BASELINE (the evidence directory the layer gate compares with; default the newest earlier one of
# this GPU that has layer reports, e.g. evidence/h100-20261026). STEPS (a regex: only the steps whose
# names match run), e.g. the bottleneck profiles alone:
#   STEPS='^(build-|layer-|moe-|matvec-|topk-|mla-)' scripts/pod_validate.sh
# Use OMP_NUM_THREADS well below the host's vCPU count (NumPy thrashes on 128-vCPU pods).
#
# Toolchain: one vxc bundle from Vx main, 950a6f71 or later (tested: a4045882; mla_flash_at needs Vx #3475 to keep its tiles in registers, mla_tc_at #3490's
# lanes::copy_rows_async). Before 950a6f71
# (Vx #3326) step build-cpu-lib takes hours: packed.vx's raw-pointer writes asked z3 a question per
# pair. The GPU kernels alone need 12fea615 (#2979: topk_radix's last-block
# scan and sort; #2949, #2943, #2969: its atomics; #2938: mla_heads_at's lane rows in registers; a9d832fd: its lane-0 writes, and
# #2648's shfl.bfly sums; #2375: read-only and 16-byte loads;
# #2363: bf16_decode!'s f32::from_bits calls in one across loop; #2240: mla_decode_at reads its step
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
: "${PYTHON:=python3}" "${LAYERS:=4}" "${TOKENS:=48}" "${REFERENCE_TOKENS:=20}" "${STRESS:=8}" "${OMP_NUM_THREADS:=16}" "${TIMER:=ncu}"
export OMP_NUM_THREADS
# The PTX target stays sm_80 (gpu/build.sh), which the driver JITs on newer GPUs; ptxas checks it for
# the device's own arch (PTXAS_ARCH, e.g. sm_90 on an H100) and runs whenever a CUDA ptxas is found.
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]' | grep -oE '\b[abhlt][0-9]{2,3}\b' | head -1)
if [[ -n $cap && -z ${PTXAS_ARCH:-} ]]; then export PTXAS_ARCH=sm_${cap/./}; fi
if [[ -z ${PTXAS:-} && -x /usr/local/cuda/bin/ptxas ]]; then export PTXAS=/usr/local/cuda/bin/ptxas; fi
out=${1:-evidence/${gpu:-gpu}-$(date -u +%Y%m%d)}
if [[ -e $out ]]; then echo "$out exists; preserve evidence" >&2; exit 2; fi
: "${BASELINE:=$(for d in evidence/"${gpu:-none}"-*; do compgen -G "$d/layer-*-*.json" >/dev/null && echo "$d"; done | sort | tail -1)}"
mkdir -p "$out"
failed=0
summary=()

step() {  # step NAME CMD...: run, log to $out/NAME.txt, record the result
  local name=$1; shift
  if [[ -n ${STEPS:-} && ! $name =~ $STEPS ]]; then return; fi
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
# The dispatch library links libcudart.so.12 and libcublas.so.12. A CUDA 13 image (Nebius H200,
# driver 580) has neither, and then every spawn build fails on "Symbols not found: vx_plugin_*".
# The CUDA 12 wheels run on that driver: install them and put their lib directories on the path.
# Not a step, so a STEPS filter that skips the builds still gets the path.
if [[ -n ${VX_DISPATCH_LIB:-} ]] && ldd "$VX_DISPATCH_LIB" 2>/dev/null | grep -qE 'lib(cudart|cublas)\.so\.12 => not found'; then
  $PYTHON -m pip install -q nvidia-cuda-runtime-cu12 'nvidia-cublas-cu12>=12.9' >> "$out/host.txt" 2>&1
  wheels=$($PYTHON -c 'import nvidia, pathlib; print(":".join(str(pathlib.Path(p) / d / "lib") for p in nvidia.__path__ for d in ("cuda_runtime", "cublas")))' 2>/dev/null)
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$wheels
  echo "dispatch lib's CUDA 12 runtime from the wheels: $(ldd "$VX_DISPATCH_LIB" | grep -cE 'lib(cudart|cublas)\.so\.12 => /') of 2 found" >> "$out/host.txt"
fi
step build-spawn bash gpu/spawn/build.sh
step build-cpu-lib bash kernels/build.sh
sha256sum gpu/build/*.ptx gpu/build/spawn/*.ptx gpu/build/spawn/*.so > "$out/ptx.sha256" 2>/dev/null
# cuBLASLt 12.9+ for gpu/cublaslt.py (Hopper's block-scaled FP8); CUDA 12.8 images ship 12.8. When none
# loads, pip installs the wheel: nvidia-cublas (CUDA 13) on a 580+ driver, nvidia-cublas-cu12 before.
cublaslt_check() { $PYTHON -c 'from gpu import cublaslt as L; lib = L.load_library(); print("cublaslt", lib._name, lib.cublasLtGetVersion())'; }
cublaslt_setup() {
  cublaslt_check && return
  local driver; driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  $PYTHON -m pip install -q "$([[ ${driver%%.*} -ge 580 ]] && echo nvidia-cublas || echo 'nvidia-cublas-cu12>=12.9')" && cublaslt_check
}
step build-cublaslt cublaslt_setup
grep -h '^cublaslt' "$out/build-cublaslt.txt" >> "$out/host.txt" 2>/dev/null

# Hardware tests skip without a usable device; a run where nothing passed is a failure, not "ok".
hw() { $PYTHON -m pytest -q -rs "$@" | tee /dev/stderr | grep -qE '[0-9]+ passed'; }
step test-hardware hw gpu/test_hardware.py gpu/test_gemm_hardware.py gpu/test_index_scores_hardware.py gpu/test_topk_hardware.py \
    gpu/test_small_kernels_hardware.py gpu/test_cublaslt_hardware.py
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
# forward_many (MTP verification, #16): 1 to 4 tokens as one step, bit for bit the one-at-a-time run,
# with fp8_matvec4_rows reading each FP8 weight once for all of them; under graph=True, forward_many's
# graphs against forward's.
step test-resident-many hw gpu/test_resident_many_hardware.py
# MTP speculative decoding (#16): the MTP layer drafts on the device, forward_many verifies; the tokens
# and cache rows equal plain greedy decoding's, also with all three replaying graphs.
step test-speculative-resident hw gpu/test_speculative_resident_hardware.py

# Full-width float32 checks resident against the hybrid path, which reads every result back (about
# 33 s per token): REFERENCE_TOKENS of them (default 20, above the full config's index_topk of 16).
for mode in f32 int4 fp8 bf16; do
  flag=$([[ $mode == f32 ]] && echo "" || echo "--$mode")
  step resident-$mode-small $PYTHON -m validation.resident_compare $flag --tokens "$TOKENS" --output "$out/resident-$mode-small.json"
  reference=$([[ $mode == f32 ]] && echo "--reference-tokens $REFERENCE_TOKENS" || echo "")
  step resident-$mode-full$LAYERS $PYTHON -m validation.resident_compare $flag $reference --geometry full --layers "$LAYERS" \
    --tokens "$TOKENS" --output "$out/resident-$mode-full$LAYERS.json"
done
# FP8 through fp8_matvec4 and fp8_moe4_* (#30, #31): another summation order, so the gate is the
# FP8 one (top-1 equal, |diff| <= 1e-4 * max |logit| against float32 on the dequantized weights).
step resident-fp8x4-small $PYTHON -m validation.resident_compare --fp8 --fp8-kernels x4 --tokens "$TOKENS" \
  --output "$out/resident-fp8x4-small.json"
step resident-fp8x4-full$LAYERS $PYTHON -m validation.resident_compare --fp8 --fp8-kernels x4 --geometry full \
  --layers "$LAYERS" --tokens "$TOKENS" --output "$out/resident-fp8x4-full$LAYERS.json"
# fp8_matvec_fma for the dense Linears (#30, one fma a code): x4's gate.
step resident-fp8fma-full$LAYERS $PYTHON -m validation.resident_compare --fp8 --fp8-kernels fma --geometry full \
  --layers "$LAYERS" --tokens "$TOKENS" --output "$out/resident-fp8fma-full$LAYERS.json"
# fp8_kernels cvt (glm-vx #38): the sm_90 builds on Hopper, x4's gate.
step resident-fp8cvt-full$LAYERS $PYTHON -m validation.resident_compare --fp8 --fp8-kernels cvt --geometry full \
  --layers "$LAYERS" --tokens "$TOKENS" --output "$out/resident-fp8cvt-full$LAYERS.json"
# cuBLASLt on e4m3 activations (#29): near the float32-activation logits, not equal (SGLang's w8a8).
step resident-fp8lt-full$LAYERS $PYTHON -m validation.resident_compare --fp8 --fp8-kernels cublaslt --geometry full \
  --layers "$LAYERS" --tokens "$TOKENS" --output "$out/resident-fp8lt-full$LAYERS.json"
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
  step layer-profile-$mode $PYTHON -m validation.layer_profile --format $mode --timer "$TIMER" \
    --output "$out/layer-profile-$mode.json"
done
step layer-profile-fp8x4 $PYTHON -m validation.layer_profile --format fp8 --fp8-kernels x4 --timer "$TIMER" \
  --output "$out/layer-profile-fp8x4.json"
# The dense FP8 Linears through fp8_matvec_fma (#30), the rest as fp8x4.
step layer-profile-fp8fma $PYTHON -m validation.layer_profile --format fp8 --fp8-kernels fma --timer "$TIMER" \
  --output "$out/layer-profile-fp8fma.json"
# What an FP8 checkpoint runs: its lm_head is BF16 (bf16_matvec4 with matvec_kernels x4), every FP8 Linear
# and routed expert at the codes' true values (fp8_kernels cvt: Hopper's conversion, glm-vx #38; q_a/kv_a/wk
# and gate/up fused), RMSNorm with 1024 threads. Context
# 1000 and GLM-5.3's index_topk 2048, as SGLang's batch-1 profile (pod_sglang.sh PROFILE_INPUTS), so the
# indexer and MLA attend to what SGLang's do (every token, below the top-k).
step layer-profile-fp8real $PYTHON -m validation.layer_profile --format fp8 --fp8-kernels cvt --matvec-kernels x4 \
  --bf16-head --norm-block 1024 --context 1000 --index-topk 2048 --timer "$TIMER" --output "$out/layer-profile-fp8real.json"
# The dense FP8 Linears through cuBLASLt on e4m3 activations (#29), the rest as fp8x4.
step layer-profile-fp8lt $PYTHON -m validation.layer_profile --format fp8 --fp8-kernels cublaslt --timer "$TIMER" \
  --output "$out/layer-profile-fp8lt.json"
for mode in int4 bf16; do
  for kernels in spawn x4; do
    step layer-profile-$mode-$kernels $PYTHON -m validation.layer_profile --format $mode --matvec-kernels $kernels \
      --timer "$TIMER" --output "$out/layer-profile-$mode-$kernels.json"
  done
done
# The per-layer timing gate (#29): each report above against BASELINE's of the same format, and
# each other kernel choice against this run's hand kernels. It fails on a slowdown, not on a missed
# target (#30, #31, #32), which it reports.
step layer-gate $PYTHON -m validation.layer_gate ${BASELINE:+--baseline "$BASELINE"} --output "$out/layer-gate.md" \
  "$out"/layer-profile-*.json
step moe-profile $PYTHON -m validation.moe_profile --timer "$TIMER" --output "$out/moe-profile.json"
# The dense matvec kernels (hand, spawn, and the *_matvec4 of #30) at GLM-5.3's projection shapes.
step matvec-profile $PYTHON -m validation.matvec_profile --output "$out/matvec-profile.json"
# cuBLASLt with the weights as operand B, as in its block-scaled FP8 sample (the step above: operand A).
step matvec-profile-cublaslt-b $PYTHON -m validation.matvec_profile --cublaslt-orientation weights_b \
    --output "$out/matvec-profile-cublaslt-b.json"
for mode in bf16 int4; do
  step matvec-profile-$mode $PYTHON -m validation.matvec_profile --format $mode --output "$out/matvec-profile-$mode.json"
done
# The indexer's top-k (#32): the radix select against topk_sort, 64 to 32k tokens.
step topk-profile $PYTHON -m validation.topk_profile --output "$out/topk-profile.json"
# The fused MLA decode (#33): mla_decode_at against mla_chunks_at at a full-width layer's shape.
step mla-profile $PYTHON -m validation.mla_profile --output "$out/mla-profile.json"
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
