# Where the shared LLM operator library should live (issue #1, M2) — recommendation

This is a recommendation for the maintainers and the Vx peers to accept or change. Nothing here
moves code.

## What exists today vs. the M2 operator list

| M2 operator | State in this repo | Gap for a library |
|---|---|---|
| RMSNorm | PTX kernels (standalone + fused residual add), CPU Vx kernel; parity-tested on A100 | none for GLM; eps/offset variants are parameters |
| RoPE | GLM's interleaved-input RoPE (PTX + CPU); half-split output | Llama/Qwen half-split *input* convention is a different kernel or a flag |
| GQA attention | **absent**. GLM uses MLA (compressed latent KV) with DSA top-k selection | a GQA/MHA decode + prefill kernel (flash-style), and a dense-selection path |
| Gated MLP | SwiGLU + matvec (PTX), resident fused MLP in the backend | batched (GEMM) form for prefill; TF32 GEMM exists in `gpu/gemm.vx` |
| MoE routing | sigmoid + correction-bias top-k, single group (PTX + CPU) | softmax top-k (Mixtral/Qwen-MoE style), grouped routing (`n_group > 1`) |
| Grouped GEMM | **absent**: experts run as per-expert matvec/MLP loops | one launch over (expert, token-block) tiles |
| Sampling | greedy argmax on host | top-k/top-p/temperature kernels |
| Paged KV cache | CPU reference with copy-on-write forking (`glm_vx/paged_kv.py`) | device-side block pool and gather kernels |
| HF loader | `glm_vx/checkpoint.py`: safetensors (F32/BF16/FP8 block-scaled), sharded index, manifest validation | model-definition mapping is GLM-specific; needs a per-family tensor-name table |
| Per-op tests vs PyTorch | layer/stage-level parity vs Transformers exists (`validation/layer_compare.py`); per-kernel tests use NumPy oracles | op-level PyTorch references, mostly mechanical |
| Quantized matmul (M5) | CPU reference math for INT4 and block-FP8 (`glm_vx/quant.py`); standalone FP8 matvec kernel | INT4 kernels; FP8 inside the model path |

Roughly half of M2 is a refactor of what exists; GQA attention, grouped GEMM, sampling and the
device paged cache are new kernels.

## What a second model family needs (M3)

Llama or Qwen are dense/MoE GQA models. Relative to GLM they need: GQA attention with a plain KV
cache (no MLA absorption, no DSA indexer), half-split RoPE input convention, softmax routing for the
MoE variants, optional QKV bias (Qwen), tied embeddings, and different config key names. They do not
need: MLA, the DSA indexer, sigmoid routing with correction bias, shared experts (some do), MTP.
"Mostly config changes" is therefore only true for the ops that already take parameters (norms, MLP,
embedding, lm_head); attention and routing are new code.

## Options

1. **Separate repository.** Decouples release cadence from the compiler. But the Vx package manager
   is still in progress (Vx issue #487), so there is no supported way for the library to be consumed
   across repos today; the Vx packages README says packages are imported via `-I` include paths. And
   the kernels co-evolve with the compiler: Vx #1745 (program symbols no longer exported) silently
   broke our CPU `.so` build, and the GPU pipeline here is a hand-built MLIR→PTX path that the
   maintainers say is not the compiler's own route.
2. **In the Vx repo, `packages/vx_llm`** (or inside `vx_nn`, which is a documented placeholder for
   "Transformers"). Follows the existing first-party package rules: tests live inside the package,
   minimal sibling dependencies, no cycles, and the dependency graph recorded in
   `docs/implementation_plans/libraries_design_deps.md`. Compiler changes and kernel changes land
   together and CI sees both. Cost: coupling to the compiler's churn and review process.
3. **Keep it in `glm-vx`** and publish later. Cheapest now; the library never gets a public API
   boundary, and "GLM rewritten on top of the library" (M2's last item) has no meaning.

## Recommendation

Stage it:

1. **Until M3 starts, keep operators in this repo**, but give them the library's shape: one module per
   operator, a config-driven entry point, no GLM names in kernels, and per-op tests with PyTorch
   references. This is where the missing GQA/grouped-GEMM kernels get written and proven on the A100.
2. **Then move to `packages/vx_llm` in the Vx repo (option 2)**, following the package guidelines
   above, with `glm-vx` as the first consumer. Reasons: no cross-repo package mechanism exists yet;
   the kernels depend on compiler behavior that already broke us once; and the Vx peers are
   already the people who fix compiler-side issues. Revisit if the package manager lands and the
   compiler churn becomes a drag.
3. Open question for the peers: whether the library should emit PTX through the compiler's own
   `spawn on(Topology::X)` path or the hand-built pipeline used here. That decides the kernel
   source form, so it should be settled before the move.
