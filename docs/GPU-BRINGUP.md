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
| M1 GPU ops | RMSNorm, RoPE, SwiGLU, matvec, FP8 matvec, TF32 GEMM, MLA decode, index scores exist as PTX; hybrid `vx-gpu` adapter round-trips activations through host | run `gpu/test_hardware.py`; fix first real-device failures; then make activations/KV device-resident; BF16 is not yet implemented (F32/TF32/FP8 only) |
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
