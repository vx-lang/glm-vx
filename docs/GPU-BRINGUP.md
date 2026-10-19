# GPU bring-up plan (issue #1)

Maps the milestones in [vx-lang/glm-vx#1](https://github.com/vx-lang/glm-vx/issues/1)
to what this repo has. The prototype started CPU-only; it now has a PTX path (`gpu/`) that has
run on an A100 and matches the official Transformers implementation on random weights.
Trained-weight parity, BF16 and performance are not done.

## Status

| Milestone | Repo state | Next step |
|---|---|---|
| M0 pod setup | `scripts/pod_setup.sh`: driver check, PTX build, hardware tests, records vCPUs/RAM | done |
| M0 HF reference | `validation/hf_reference.py` + `layer_compare.py`, `gpu_vs_official.py`, `prefill_compare.py` | done |
| M0 bring-up model | decided: random-weight parity until M4 (see below) | revisit if a small trained checkpoint appears |
| M1 GPU ops | PTX for RMSNorm, RoPE, SwiGLU, matvec, FP8 matvec, TF32 GEMM, MLA decode, index scores; parity vs. HF passes on A100 (F32/TF32) | BF16 activations/compute (BF16 weight storage exists: `gpu/BF16.md`); device-resident activations/KV |
| M1 KV cache | contiguous host buffers (`glm_vx/kvcache.py`), 4x faster row gather | device-resident cache |
| M2 library | layout recommended in `docs/LIBRARY-LAYOUT.md`; paged KV CPU reference (`glm_vx/paged_kv.py`); loader tested on BF16/FP8/sharded checkpoints (`validation/write_checkpoint.py`) | GQA attention, grouped GEMM, sampling, device paged cache; extraction |
| M3 second family | not started; gap list in `docs/LIBRARY-LAYOUT.md` | after M2 ops |
| M4 hybrid MoE | admission verdict (`glm_vx/admission.py`, validated within 0.011% of the official checkpoint size); per-layer expert placement on a host backend (`glm_vx/placement.py`), bit-identical logits on CPU backends | GPU attention + device KV with host experts; port the verdict into `vxc` |
| M5 quantization / benchmarks | INT4 and block-FP8 reference math (`glm_vx/quant.py`); cross-engine harness (`benchmarks/compare.py`, tested on fake servers) | INT4 (incl. compressed-tensors bf16-scale checkpoints, `gpu/INT4.md`) block-FP8 and BF16 (`gpu/BF16.md`) Linears run as stored in resident decode (SIMT-checked, not yet on hardware); batched FP8/INT4 matmul not in the model path; run against llama.cpp and vLLM on the pod |

## First real-device result (2026-10-15)

A100-SXM4-80GB (sm_80, driver 580.159.04, 128 vCPUs, ~2 TB host RAM), Vx main
`a7a3aa65` built on the pod, LLVM 22 from apt.llvm.org. `gpu/test_hardware.py`,
`test_gemm_hardware.py`, `test_index_scores_hardware.py`: **16 passed**. Resident
kernel benchmark: all match the oracle (max abs error ≤ 8e-7). Raw output in
`evidence/a100-20261015/`. Timings are for tiny shapes (not model shapes) and are
not performance claims.

Toolchain notes from the bring-up:
- Newer `vxc` emits `define hidden`, `@abort` and `@puts` (overflow/div checks);
  `gpu/toolchain/build_ptx.py` now accepts them (abort -> `llvm.trap`, puts -> no-op).
- Vx #1745 stopped exporting program function symbols, which emptied the CPU `.so`
  files built by `kernels/build.sh` and `gpu/testing/build_simt.sh`. Vx #1839 (issue
  #1827) added `pub fn`: a top-level `pub fn` keeps its name and is exported. The 58
  kernels we call from outside are `pub fn`; the helpers (`add`, `axpy`, `attention`,
  `sparse_attention`, `rope`, `rope_offset`, `sigmoid_scalar`, `topk_*`) stay hidden.
  Needs a `vxc` at or after 8d28aa86; older ones reject `pub`. `pub` does not change
  the PTX output (checked byte for byte).
- Per the Vx maintainers, the mlir-translate/opt/llc PTX pipeline is not the
  compiler's own path (`spawn on(Topology::X)` + `deviceImageOf()` is). Moving
  there is a candidate for M2.
- Prebuilt `vxc` from another host is unusable (glibc 2.43); build on the pod.
- Saved toolchain, to skip the Vx/LLVM build on a new pod: build once on the pod
  (`scripts/provision/setup_linux.sh`, `./setup.sh`, `cargo build --release` in the Vx
  checkout; about 5 minutes on a 256-vCPU host), pack `bin/vxc`, `stdlib/`,
  `lib/libvx_std_core.so` and `/usr/lib/llvm-22` without `*.a` into
  `vx-toolchain-<vx-commit>.tar.zst` (115 MB for Vx 8d28aa86), and keep it off the pod.
  On the next pod: `scripts/pod_restore.sh vx-toolchain-<commit>.tar.zst`, then
  `source /opt/vx-toolchain/env.sh`. It must be packed on the oldest glibc you will
  rent (Ubuntu 24.04 / 2.39); the bundle made on the pod does run on this newer box.
  Rebuilding PTX from the restored toolchain gave byte-identical files to a local build.

## Device-resident decode (`gpu/resident.py`)

`ResidentGLM` runs a decode step with weights, activations and the MLA/indexer caches on the
GPU, through the same Vx kernels as `GPUBackend` (new kernels: add, copy, scale, scaled_add,
layernorm, rope_strided, matvec_heads; later topk_stable). Host traffic per token: the embedding
row, RoPE cos/sin, the k routed expert ids, and the logits. The indexer top-k runs on the device
(`gpu/topk.vx`, `gpu/TOPK.md`: single-block radix select, the model's exact
`argsort(-scores, kind='stable')` rule, SIMT-tested bit-identical to the host argsort including
exact ties; not yet on hardware, and the A100 evidence below predates it). Weights upload lazily on first use and stay resident.

On the A100 (random weights, F32, `validation/resident_compare.py`, evidence in
`evidence/a100-20261018/`): logits are **bitwise equal** to the hybrid backend's on the 6-layer
256-wide model (48 tokens, six concurrent runs, seeds 1-6) and on the full-width slice (6144
hidden, 256 experts, vocab 154880, 4 layers, 12 tokens); `gpu_vs_official --backend vx-gpu-resident`
keeps the max logit difference to Transformers at <= 8.8e-6 with 16 greedy tokens matching.

Single-run timings, steady state after a warmup pass, not a benchmark: 6-layer model 11.6 ms/token
resident vs 78.9 ms hybrid; full-width 4-layer slice 10.5 ms/token vs 34.0 s/token (the hybrid
path re-uploads weights from the host each token). A per-kernel profile at full width (157 launches
per token, synchronized after each launch, so inflated) puts about 9 of 16 ms in `matvec`.
INT4 (compressed-tensors W4A16, `gpu/INT4.md`): a Linear the weight source reports as INT4
(`SafetensorsCheckpoint`, or `glm_vx.quant.CompressedInt4Weights`) stays packed on the device and
runs through `glm_vx_gpu_int4_matvec` with its bf16 group scales; `kv_b_proj`, which absorbed MLA
reads transposed per head, is dequantized exactly to float32 once. Checked in the CPU SIMT emulator
only (`gpu/test_resident.py`: INT4 resident vs the NumPy model on the exact dequantization, and vs
float32 resident on the same weights). `resident_compare --int4` is the A100 check; it has not run.
Block FP8 (the official GLM checkpoint format, `gpu/FP8.md`) works the same way: a Linear whose
`.weight` is `F8_E4M3` stays as bytes on the device with its 128x128 `weight_scale_inv` grid and
runs through `glm_vx_gpu_fp8_matvec`; `kv_b_proj` is dequantized to float32. It is weight-only FP8
(float32 activations, like the NumPy model), not the official per-token FP8 activation
quantization. Checked in the SIMT emulator only, including an FP8 safetensors directory read by
`SafetensorsCheckpoint`; `resident_compare --fp8` is the A100 check; it has not run.
BF16 (`gpu/BF16.md`): every other Linear stored as a bfloat16 matrix, including the router and
lm_head (the unquantized parts of the FP8 and INT4 checkpoints), stays 16-bit on the device and runs
through `glm_vx_gpu_bf16_matvec`. Its exact decode and float32 matvec summation order make the
logits bit-identical to float32 resident on the same values (SIMT-tested, also from a BF16
safetensors directory); `resident_compare --bf16` is the A100 check; it has not run.

Not done: fewer launches (fusion/graphs), INT4/FP8/BF16 on hardware, and a full-depth model (only 4 layers ran at full width).

## M1 correctness vs. the official model (random weights, A100)

`python -m validation.gpu_vs_official --backend vx-gpu` runs a 6-layer, 256-wide
random-weight GLM-MoE-DSA (dense + MoE + shared expert, full and shared indexers,
top-k 8 over 24-token prompts, 16 cached greedy decode steps) against unmodified
Transformers (commit 469230357a, eager F32 CPU). Seeds 1–3 on the A100: **max
per-position logit abs diff ≤ 8.8e-6, top-1 matches at every prompt position, all
16 greedy tokens match** (numpy-reference backend: ≤ 9.9e-6). Evidence:
`evidence/a100-20261015/vs-official/`. This is F32/TF32-class agreement on a
*random* model; it is not a BF16 result and not a trained-model result.

Harness lesson: with few indexer heads, ReLU can zero all heads for a key, giving
exact score ties that `torch.topk` and Vx break differently (Vx: lowest index).
That flips the selection and the logits (a 4.0 jump) without any numerical error,
so the model uses 16 indexer heads. Real checkpoints can hit such ties too; if
bit-for-bit selection parity matters, the tie rule should be specified.
Do not run the NumPy backend with default BLAS threads on a many-core host
(`OMP_NUM_THREADS=8`).

## Bug found: kernels read stale data after uploads when the GPU was shared (fixed)

Found by the per-layer harness. A 2300-token run failed once (position 1428, layer 2
MLP) and passed on rerun; three identical concurrent runs gave three different,
wrong answers (`evidence/a100-20261015/concurrency/rep-*`). `gpu/stress_ops.py`
(16 processes on one A100, every `GPUBackend` op vs. an independent oracle) showed
wrong results of order 1e3 in up to ~20% of `mlp`/`compressed_attention` calls and,
at lower rates, in single-kernel ops with small uploads (matvec, softmax). Not
ECC/hardware (no errors logged).

Root cause (confirmed by the fix, not by a driver trace): `DeviceBuffer.upload()`
synchronized the context *before* `cuMemcpyHtoD_v2` but not after. A pageable HtoD
copy may return before its DMA lands, and it is ordered on the legacy default
stream, not against kernels on our non-blocking streams, so a kernel launched right
after could read stale device memory. It needed GPU contention to show. Fix: sync
after the copy in `upload()` (so "blocking" is true). Evidence it is the cause and
not a mask: a first workaround, a host sync per kernel launch, also removed the
failures but was unnecessary; with only the upload fix, 16-process stress shows 0
mismatches for all ops (and `mlp`/`compressed_attention` 16x500), the resident
pytest loop passes 64/64 (it failed 1-2 per 36-64 before), and three concurrent
2300-token comparisons are bit-identical and pass (`final-*`).
Implications: results obtained while the GPU was shared could be wrong, so the
earlier idle-GPU parity numbers stand but anything else should be rerun; a device
async upload on the kernel stream would also avoid the context sync later.

## Open questions answered from the repo

- *Which GLM checkpoints does the prototype support?* Only the GLM-MoE-DSA
  architecture (`GlmMoeDsaForCausalLM`): the official GLM-5.3 (FP8 safetensors,
  ~756 GB) and its mixed-bit GGUF, plus tiny random-weight test models. There is
  no smaller trained GLM-MoE-DSA checkpoint known to this repo.
- *Bring-up model (decision, 2026-10-15).* A web search found no official or community
  trained GLM-MoE-DSA checkpoint below the ~744-754 B-parameter family (GLM-5 / 5.1 / 5.2 /
  5.3). The only small one is a third-party random-weight fixture
  ([`aday777/glm_moe_dsa_tiny_fixture`](https://huggingface.co/aday777/glm_moe_dsa_tiny_fixture):
  274 k parameters, 4 layers, 64 wide, 8 experts, explicitly untrained, unverified under any
  Transformers version; not used here). Our own random-weight models are larger and sharper
  tests (real per-head geometry up to the real 6144-wide config). So: **correctness is
  gated on random-weight parity against the pinned Transformers class
  (`validation/layer_compare.py`, `gpu_vs_official.py`, `prefill_compare.py`); trained-weight
  parity is deferred to M4**, when the hybrid CPU+GPU path can run the real checkpoint. If a
  small trained GLM-MoE-DSA ever appears, point `validation/hf_reference.py` at it.

## M1 pass criteria (proposed, grounded in measured numbers; the issue left these blank)

Random-weight parity gate, F32/TF32 path (what exists today). Measured on the A100: worst
per-stage relative error 7e-6 to 2e-5 up to 78 layers; 2.9e-6 at real width.
- Every stage of every layer at every position: `max|a-b| / max(1, max|a|) <= 1e-4`
  (`layer_compare --tol 1e-4`; currently 2e-4, 5x-10x above what is observed).
- DSA top-k selections identical, except exact-score ties, which are reported separately
  (a tie is not an error; see the harness note).
- Greedy decode matches Transformers for 16 tokens on every reference prompt.
- Repeatable under GPU sharing: identical concurrent runs give identical logits hashes.

BF16 compute path (BF16 activations; not built yet, BF16 *weights* with float32 compute are bit-identical
to float32 and need no tolerance): tolerance is not guessed. Measure the noise of the
official model itself, official-BF16 vs. official-F32 on the same inputs, and require our
BF16 stages to be within 4x of that, with greedy decode matching on prompts where the
official top-1 margin exceeds that noise.

Trained weights (M4): perplexity on a fixed eval set within 1% of Transformers, and greedy
decode matching for the first 16 tokens on a fixed prompt set wherever the official top-1
margin is above the BF16 noise.
