# Trained GLM-5.3 weights, first 3 layers, against the official implementation (#35)

The first run of glm-vx on the trained checkpoint rather than random weights. This is a CPU-only
run, on the dev box.

- **Weights:** zai-org/GLM-5.3 at `aca966e4` (metadata/sources.json). The run uses decoder layers
  0–2, the 3 dense layers, each with the full indexer, plus `embed_tokens`, the final norm and
  `lm_head`. `validation/real_layers.py fetch` copied them byte for byte out of 3 of the 141 shards
  by HTTP range requests: 5.01 GB in 110 s. `fetch.json` has each file's sha256; `config.json` is
  the real config cut to 3 layers, with no MTP layer.
- **Reference:** the official `GlmMoeDsaForCausalLM` from Transformers at the pinned revision
  (5.19.0.dev0, torch 2.15.0+cpu), eager, float32, on CPU. Its FP8 weights are decoded by torch
  (`float8_e4m3fn` × the F32 128×128 block scale), not by glm-vx's decoder.
- **Candidates:**
  - `numpy-reference`: glm-vx's model with NumPy math.
  - `vx`: the same model with every op in the Vx CPU library (`kernels/libglm_vx.so`), built by a
    vxc from Vx PR #3327 (e43d9fcb, on main 22a0a84f).
- **Prompts:** two short English and code prompts, 12 and 17 tokens. The logits are compared at
  every prompt position, then 8 greedy tokens follow, run with no cache on the official side.
- **Criteria:** the same as `validation/gpu_vs_official.py` uses on random weights: all logits within
  atol = rtol = 2e-3, the same top-1 at every position, and the same greedy tokens.

| Candidate | Prompt tokens | Max \|Δ logit\| | Mean \|Δ logit\| | Max \|logit\| | Top-1 all positions | Greedy 8/8 | Time |
|---|---|---|---|---|---|---|---|
| numpy-reference | 12 | 9.5e-6 | 6.8e-7 | 11.4 | yes | yes | 800 s for both prompts |
| numpy-reference | 17 | 1.0e-5 | 7.1e-7 | 20.0 | yes | yes | |
| vx | 12 | 3.5e-5 | 2.6e-6 | 11.4 | yes | yes | 608 s for both prompts |
| vx | 17 | 4.4e-5 | 2.7e-6 | 20.0 | yes | yes | |

Both pass, at about 1e-5 to 4e-5 of logits up to 20: the size of a float32 summation-order
difference.

## What this does and does not cover

- **Covered, on trained values:**
  - the FP8 block-scale decode of every attention and MLP Linear;
  - the BF16 embeddings, norms, `lm_head` and indexer `weights_proj`;
  - RMSNorm, the interleaved RoPE (θ = 8e6), MLA attention, the full indexer's scores, and the
    dense SwiGLU MLP.
- **Not covered:**
  - **Sparse selection.** `index_topk` is 2048 and the prompts are under 20 tokens, so the
    indexer keeps every token.
  - **The MoE layers: the router, shared and routed experts.** Layer 3 is 9.7 GB of FP8, and the
    official model needs about 39 GB to hold its 256 experts in float32, more than this box has
    free. That check needs a machine with 64 GB of RAM or more.
  - **The MTP layer** (layer 78, for #16).
  - **The GPU path.**
- A model cut to 3 layers is not GLM-5.3: its tokens mean nothing. This shows that glm-vx computes
  what the official code computes on the trained weights, not anything about output quality.

## Reproduce

```sh
python -m validation.real_layers fetch --output DATA/glm53-l3 --layers 3
python -m validation.real_layers compare --checkpoint DATA/glm53-l3 --output RUN --backend numpy-reference
GLM_VX_LIBRARY=kernels/build/libglm_vx.so python -m validation.real_layers compare --checkpoint DATA/glm53-l3 \
    --output RUN-vx --backend vx
```

`compare` needs torch, Transformers at the pinned revision (`transformers_revision` in
metadata/sources.json) and `tokenizers`. About 25 GB of RAM at 3 layers.
