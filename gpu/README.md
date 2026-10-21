# Explicit NVIDIA GPU path

GPU kernels are written in Vx and compile to **real PTX** with hardware thread
indices, warp shuffle reductions, shared memory and block barriers. A small LLVM
adapter supplies hardware intrinsics; there is no CUDA C numerical replacement,
PyTorch/CuPy runtime, or implicit CPU fallback. Target: NVIDIA SM80+ by default.

**Not yet GPU-benchmarked or proven optimal.** This workspace has no CUDA driver
or GPU. Offline compilation, assembly (where recorded), and CPU SIMT arithmetic
validation are distinct from real-device execution. See GPU-REPORT.md for receipts.

## Build and run

Install Vx v0.0.2 and LLVM 22. Ensure `vxc`, `mlir-translate`, `opt`, `llc` are on
PATH, or set `VXC` and `VX_GPU_LLVM_BIN`. Then:

```sh
gpu/build.sh
# Optional: PTXAS=/path/to/ptxas gpu/build.sh validates native assembly too.
# On the original build machine, first: source ../vx-toolchain/env.sh
# All three modules emit PTX plus source/hash/toolchain manifests in gpu/build/.
```

On a machine with an SM80+ NVIDIA GPU and compatible CUDA driver:

```sh
.venv/bin/python -m gpu.benchmark --kernel all --repeats 20 --output gpu-result.json
# Use the actual 6144 projection / 512-latent dimensions:
.venv/bin/python -m gpu.benchmark --model-shapes --output gpu-model-shapes.json
.venv/bin/python -m glm_vx.server --tiny --backend vx-gpu
```

The benchmark first checks numerical outputs, then uses CUDA events over resident
inputs. Timings exclude upload/download, startup and serving; they are not tokens
per second. Missing CUDA exits with status 2 and an explicit unavailable result.
Any numerical/JIT/launch failure is an error, not a CPU timing result.

`--backend vx-gpu` is an explicitly **hybrid compatibility adapter**: the existing
model still orchestrates on Python/NumPy host arrays. Every primitive uses PTX,
but intermediate activations transfer between calls. Read-only weights are held
in a bounded device LRU (default 512 MiB; `--gpu-weight-cache-mib` changes it).
Mutable arrays are re-uploaded; temporary buffers are released after synchronized
completion. The compatibility model uses fused compressed MLA across heads, but still gathers
and uploads the selected KV rows on each attention call. Direct FP8 projection
is currently a resident API, not integrated into checkpoint serving.

`--backend vx-gpu-resident` serves through `gpu/resident.py` (`ResidentGLM`) instead: weights,
activations and the MLA / indexer caches stay on the device, and block-FP8 / INT4 / BF16 Linears
stay packed. It holds one sequence's caches, so it serves one request at a time (`--max-sequences`
is fixed at 1 until batched decode, #8); further requests wait in a FIFO (`--max-waiting`, default
32) rather than getting 429. `--max-context` (default 8192) fixes the cache length at startup;
each request resets the position and reuses the buffers. At startup `ResidentGLM.preload()` uploads
every weight decode can use (every layer and routed expert, final norm, lm_head) and logs the bytes
and seconds; if they would not fit in free device memory (cuMemGetInfo, less a 256 MiB reserve) the
server exits with the bytes needed against the bytes free. `--no-preload` uploads on first use instead,
so steady state takes many requests. The prefix cache, `--batched-prefill` and `--gpu-weight-cache-mib`
(nothing is evicted) are refused; sampling stays on the host (#17). Completion `timings` include
`queued_seconds`.

```sh
.venv/bin/python -m glm_vx.server --backend vx-gpu-resident --checkpoint DIR --tokenizer DIR/tokenizer.json --max-context 4096
```

## Measured layout tuning

```sh
.venv/bin/python -m gpu.tune --rows 6144 --cols 6144 --output tuning.json
.venv/bin/python -m glm_vx.server --tiny --backend vx-gpu --gpu-tuning tuning.json
# Quantized projection candidate measurements (resident API only):
.venv/bin/python -m gpu.tune --fp8 --rows 6144 --cols 6144 --output fp8-tuning.json
```

The tuner checks outputs and compares 32/64/128/256-thread layouts using actual
CUDA events. A plan records the GPU, driver, PTX hash and exact matrix shape.
The compatibility backend rejects mismatched plans and uses a measured layout
only for its matching F32 shape. This selects the best *tested* layout; it does
not prove global optimality. No plan is emitted when CUDA is unavailable.

## Resident kernels

- [Tiled GEMM](GEMM.md): explicit TF32 tensor-core MMA with shared staging and
  arbitrary residual shapes, alongside a separate full F32 reference mode.

- [Fused DSA index scores](INDEX-SCORES.md): all signed head contributions in
  one launch, with one upload per input and one final score readback in the
  compatibility backend; a separate resident API supports output reuse.
- Coalesced warp-per-row float32 matvec; tails need no padded weight allocation.
- Direct **FP8 E4M3FN + 128×128 block scales** matvec; no expanded float32 weight
  matrix or per-element scale tensor in GPU memory.
- Fused residual-add + RMSNorm with warp/block reductions; residual output is
  retained for the next residual connection.
- Grid-strided SwiGLU and GLM interleaved-input, half-split-output RoPE.
- Split compressed MLA decode: each CTA streams selected latent KV entries with
  online softmax; a second kernel stably merges partitions. This avoids expanded
  per-head K/V and a full attention score matrix. Scratch is O(heads×splits×rank).
- Standalone RMSNorm, stable softmax and corrected sigmoid routing for the
  compatibility backend.

Use `CUDAContext.tensor()` and `GLMKernels` from `gpu.runtime` to keep inputs and
outputs resident across launches. Launch partial MLA and merge on the same stream
(or provide an explicit event dependency). `gpu/kernels.h` defines the exact ABI,
shape bounds, alias rules and launch geometry. Scalar storage is float32 except
quantized FP8 weights, int32 selected indices and routing indices.

## Validation layers

```sh
# Mock CUDA API lifecycle/argument tests; no hardware claim:
.venv/bin/python -m pytest gpu/test_runtime.py -q
# Compile identical Vx source into a test-only CPU lane simulator:
source ../vx-toolchain/env.sh   # or supply your LLVM/Vx environment
gpu/testing/build_simt.sh
.venv/bin/python -m pytest gpu/test_simt.py gpu/test_backend.py -q
# Compiler IR/PTX contracts and exhaustive FP8 tests:
.venv/bin/python -m pytest gpu/toolchain/test_build_ptx.py gpu/test_fp8.py -q -rs
```

CPU simulation executes Vx kernels with host threads, shuffle emulation and CTA
barriers; it tests lane ownership, reductions, tails, mask behavior and merge
math. It does not model GPU scheduling, compiler-induced races, register pressure
or the GPU approximate/FTZ exponential. Real GPU cases explicitly skip when no
CUDA device exists and fail on missing PTX when a device does exist. The production
runtime cannot import or select the simulator as a fallback.

## Single-GPU proxy checkpoint (SGLang comparison, #5)

```sh
.venv/bin/python -m validation.write_proxy --output /data/glm53-proxy --tokenizer-dir GLM53_SNAPSHOT
python -m sglang.launch_server --model-path /data/glm53-proxy --tp 1 --mem-fraction-static 0.85
.venv/bin/python -m glm_vx.server --checkpoint /data/glm53-proxy --backend vx-gpu-resident \
  --tokenizer /data/glm53-proxy/tokenizer.json
```

`validation/write_proxy.py` writes the real GLM-5.3 config cut to 9 layers (3 dense + 6 MoE, the
real indexer pattern, every width unchanged) with seeded random weights in the official block-FP8
layout and tensor names: 64.25 GB, which leaves ~8 GB of KV cache in SGLang's 72.7 GB static pool
on an 80 GB H100. It streams ~5 GB shards at about 1 GB/s and records every file's sha256 and every
tensor in `proxy.json`. `--format bf16` (default 3 dense + 2 MoE, 45.7 GB) writes the same model
unquantized for SGLang's BF16 row; `--layers N` and `--mtp` change the depth; the docstring has the
byte arithmetic.

## Remaining performance work

The code removes several obvious data-movement/serialization costs inside the
kernels, but no optimality claim follows. High-throughput full GLM serving still
needs tensor-core FP8/BF16 batched GEMM, fully resident model orchestration,
quantized checkpoint placement/sharding, tensor/expert collectives, paged KV,
prefill fusion, and GPU tuning. Single-node GPU choice, interconnect, latency/SLO
and workload determine the next optimization. No GPU was rented or purchased.
