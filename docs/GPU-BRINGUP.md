# GPU bring-up plan (issue #1)

Maps the milestones in [vx-lang/glm-vx#1](https://github.com/vx-lang/glm-vx/issues/1)
to what this repo already has. The issue calls the prototype CPU-only; it now has
a PTX path (`gpu/`), but **nothing has run on a real device yet** (`gpu/GPU-REPORT.md`).

## Status

| Milestone | Repo state | Next step |
|---|---|---|
| M0 pod setup | `scripts/pod_setup.sh` (new): driver check, build PTX, hardware tests, records vCPUs/RAM | run on the pod |
| M0 HF reference | `validation/hf_reference.py` (new): dump logits + greedy, compare with tolerances | run on pod with chosen checkpoint |
| M0 bring-up model | **undecided** (see below) | decide |
| M1 GPU ops (random-weight parity vs. HF passes on A100) | RMSNorm, RoPE, SwiGLU, matvec, FP8 matvec, TF32 GEMM, MLA decode, index scores exist as PTX; hybrid `vx-gpu` adapter round-trips activations through host | run `gpu/test_hardware.py`; fix first real-device failures; then make activations/KV device-resident; BF16 is not yet implemented (F32/TF32/FP8 only) |
| M2 library | not started; ops live in `gpu/`, `kernels/` | decide location (open question) |
| M3–M5 | not started | after M1 passes |

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
- Vx main (#1745) no longer exports program function symbols, so the CPU `.so`
  paths (`kernels/build.sh`, `gpu/testing/build_simt.sh`) fail to expose
  `glm_vx_*`; they need `_vx_`-compatible entry points. PTX path is unaffected.
- Per the Vx maintainers, the mlir-translate/opt/llc PTX pipeline is not the
  compiler's own path (`spawn on(Topology::X)` + `deviceImageOf()` is). Moving
  there is a candidate for M2.
- Prebuilt `vxc` from another host is unusable (glibc 2.43); build on the pod.

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

## Bug found: chained GPU ops returned wrong results under GPU sharing (fixed)

Found by the per-layer harness. A 2300-token run failed once (position 1428, layer 2
MLP) and passed on rerun; three identical concurrent runs gave three different,
wrong answers (`evidence/a100-20261015/concurrency/rep-*`). `gpu/stress_ops.py`
(16 processes on one A100, every `GPUBackend` op vs. an independent oracle) isolated
it: single-kernel ops (matvec, rmsnorm, ...) never failed, but `mlp` (4 chained
launches) and `compressed_attention` (2) were wrong in up to ~20% of calls, with
errors of order 1e3. Not ECC/hardware (no errors logged). A host sync either before
an op's first launch or between launches removes all failures (0 in 8000+ calls
each); the root cause is **not yet understood** (a question is out to the Vx
runtime peers). `GPUBackend._launch` now syncs before the first launch of an op and
between launches; with it, 16-process stress shows 0 mismatches and the three
concurrent 2300-token runs are bit-identical (`fix-*`, all pass).
Implications: (1) any result obtained while the GPU was shared could be wrong, so
parity runs should be repeated alone and under load; (2) the resident APIs
(`GLMKernels`, GEMM, tuning) use the same upload-then-launch pattern and have not
been stress-tested under sharing; (3) this costs a sync per launch, so performance
numbers must wait until the cause is known.

## Open questions answered from the repo

- *Which GLM checkpoints does the prototype support?* Only the GLM-MoE-DSA
  architecture (`GlmMoeDsaForCausalLM`): the official GLM-5.3 (FP8 safetensors,
  ~756 GB) and its mixed-bit GGUF, plus tiny random-weight test models. There is
  no smaller trained GLM-MoE-DSA checkpoint known to this repo.
- *Bring-up model.* GLM-5.3 cannot fit an L4/A10/4090 or one A100-80GB without the
  M4 hybrid offload. Until a small trained checkpoint exists, M1 correctness uses
  (a) tiny random-weight models vs. the pinned Transformers class
  (`validation/official_eager_fixture.py` already does this on CPU) and
  (b) scaled-down random configs on the pod via `hf_reference.py`. Trained-model
  parity waits for M4.

## M1 pass criteria (proposed; the issue left these blank)

- Greedy decode equals the HF reference for the first 16 tokens on every prompt in
  the reference set.
- Per-position logit max-abs-diff ≤ 1e-2 for BF16 reference vs. TF32/FP8 Vx
  (tighten once real-device numbers exist).
- Perplexity on a fixed eval set within 1% of HF.
