"""M1 gate on real hardware: Vx GPU backend vs. unmodified Transformers GLM-MoE-DSA.

Scaled-down *random-weight* model (not a trained checkpoint): exercises every GLM
code path (dense + MoE layers, shared experts, full + shared DSA indexers with
real top-k selection, MLA, RoPE) at dimensions large enough for the GPU kernels'
multi-warp/multi-block code. Needs torch + the pinned transformers (optional
validation dependency; never imported by serving). Writes a JSON report and an
.npz of logits and exits nonzero if any criterion fails.

  python -m validation.gpu_vs_official --output run/ [--backend vx-gpu|vx|numpy-reference]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel, tiny_weights

CONFIG = dict(
    hidden_size=256, vocab_size=512, num_hidden_layers=6,
    num_attention_heads=8, num_key_value_heads=8, q_lora_rank=96,
    kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=16,
    v_head_dim=32, index_n_heads=16, index_head_dim=64, index_topk=8,
    intermediate_size=384, moe_intermediate_size=96,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    routed_scaling_factor=2.5, rms_norm_eps=1e-5,
    indexer_types=['full', 'shared', 'full', 'shared', 'full', 'shared'],
    mlp_layer_types=['dense', 'sparse', 'sparse', 'sparse', 'sparse', 'sparse'],
    rope_parameters={'rope_type': 'default', 'rope_theta': 10000},
    norm_topk_prob=True, n_group=1, topk_group=1, max_position_embeddings=256)


# Real GLM-5.3 per-head geometry (64 heads, kv rank 512, nope 192, rope 64, value
# 256, indexer 32x128, theta 8e6) with a small hidden width and few layers/experts,
# so kernel launch sizes and rank/dim edge cases match the real model.
REAL_GEOMETRY = dict(CONFIG, hidden_size=1024, num_hidden_layers=6, num_attention_heads=64,
    num_key_value_heads=64, q_lora_rank=512, kv_lora_rank=512, qk_nope_head_dim=192,
    qk_rope_head_dim=64, v_head_dim=256, index_n_heads=32, index_head_dim=128, index_topk=16,
    intermediate_size=1536, moe_intermediate_size=256, n_routed_experts=16, num_experts_per_tok=4,
    rope_parameters={'rope_type': 'default', 'rope_theta': 8000000}, max_position_embeddings=1024)

# 16 indexer heads: with few heads, ReLU can zero every head for a key, and the
# resulting exact score ties are broken differently by torch.topk and Vx
# (lowest index), which is not a numerical bug but flips the selection.

def with_layers(config, n):
    """Deepen a config using GLM-5.3's real layer pattern (verified against model-config.json):
    first 3 layers dense, indexer 'full' at layers 0-2 and every 4th from layer 6, else 'shared'."""
    return dict(config, num_hidden_layers=n,
                indexer_types=['full' if i < 3 or (i >= 6 and (i - 6) % 4 == 0) else 'shared' for i in range(n)],
                mlp_layer_types=['dense' if i < 3 else 'sparse' for i in range(n)])


def random_weights(config, seed):
    """Same names/shapes as tiny_weights but with non-degenerate values."""
    shapes = tiny_weights(config, seed=0, parameter_limit=10_000_000_000)
    rng = np.random.default_rng(seed)
    out = {}
    for name, ref in shapes.items():
        if name.endswith('layernorm.weight') or name.endswith('norm.weight'):
            out[name] = rng.uniform(.7, 1.3, ref.shape).astype(np.float32)
        elif name.endswith('.bias') or name.endswith('e_score_correction_bias'):
            out[name] = rng.normal(0, .1, ref.shape).astype(np.float32)
        else:
            fan_in = ref.shape[-1]
            out[name] = rng.normal(0, 1 / np.sqrt(fan_in), ref.shape).astype(np.float32)
    return out


def official_model(config, weights):
    import torch
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa as configuration
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as official
    cfg = configuration.GlmMoeDsaConfig(**config, attention_dropout=0.0)
    cfg._attn_implementation = 'eager'
    model = official.GlmMoeDsaForCausalLM(cfg).float().eval()
    state = {}
    for name in model.state_dict():
        if name.endswith('.experts.gate_up_proj'):
            p = name.removesuffix('.experts.gate_up_proj')
            value = np.stack([np.concatenate([weights[f'{p}.experts.{e}.{k}_proj.weight'] for k in ('gate', 'up')])
                              for e in range(config['n_routed_experts'])])
        elif name.endswith('.experts.down_proj'):
            p = name.removesuffix('.experts.down_proj')
            value = np.stack([weights[f'{p}.experts.{e}.down_proj.weight'] for e in range(config['n_routed_experts'])])
        else:
            value = weights[name]
        state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state, strict=True)
    return model


def official_run(model, tokens, new):
    import torch
    with torch.no_grad():
        logits = model(torch.tensor([tokens]), use_cache=False).logits[0].numpy().copy()
        ids = list(tokens)
        for _ in range(new):  # no cache: simplest unambiguous reference
            ids.append(int(model(torch.tensor([ids]), use_cache=False).logits[0, -1].argmax()))
    return logits, np.array(ids[len(tokens):])


def candidate_run(model, tokens, new):
    cache = model.new_cache()
    logits = np.stack([model.forward(t, cache) for t in tokens])
    out = []
    last = logits[-1]
    for _ in range(new):
        out.append(int(np.argmax(last)))
        last = model.forward(out[-1], cache)
    return logits, np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['vx-gpu', 'vx', 'numpy-reference'], default='vx-gpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    p.add_argument('--prompt-len', type=int, default=24)
    p.add_argument('--new', type=int, default=16)
    p.add_argument('--atol', type=float, default=2e-3)
    p.add_argument('--rtol', type=float, default=2e-3)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    import torch
    import transformers
    if args.backend == 'vx-gpu':
        from gpu.backend import GPUBackend
        backend = GPUBackend()
    elif args.backend == 'vx':
        from kernels.backend import VxBackend
        backend = VxBackend()
    else:
        from glm_vx.backend import NumpyBackend
        backend = NumpyBackend()
    reports, arrays = [], {}
    for seed in args.seeds:
        weights = random_weights(CONFIG, seed)
        for w in weights.values():
            w.flags.writeable = False
        tokens = [int(t) for t in np.random.default_rng(100 + seed).integers(0, CONFIG['vocab_size'], args.prompt_len)]
        ref_logits, ref_ids = official_run(official_model(CONFIG, weights), tokens, args.new)
        got_logits, got_ids = candidate_run(GlmMoeDsaModel(CONFIG, weights, backend), tokens, args.new)
        diff = np.abs(ref_logits - got_logits)
        # argmax-stability: HF's top-1 margin must exceed the error for a mismatch to matter
        top1_match = bool(np.array_equal(ref_logits.argmax(-1), got_logits.argmax(-1)))
        first_mismatch = next((i for i, (a, b) in enumerate(zip(ref_ids, got_ids)) if a != b), len(ref_ids))
        ok = (bool(np.allclose(got_logits, ref_logits, rtol=args.rtol, atol=args.atol))
              and top1_match and first_mismatch == len(ref_ids))
        reports.append(dict(seed=seed, tokens=tokens, max_abs_diff=float(diff.max()),
                            mean_abs_diff=float(diff.mean()), top1_match_all_positions=top1_match,
                            greedy_tokens=len(ref_ids), greedy_first_mismatch=first_mismatch, ok=ok))
        arrays.update({f'seed{seed}.ref_logits': ref_logits, f'seed{seed}.cand_logits': got_logits,
                       f'seed{seed}.ref_ids': ref_ids, f'seed{seed}.cand_ids': got_ids})
        print(json.dumps({k: v for k, v in reports[-1].items() if k != 'tokens'}))
    args.output.mkdir(parents=True)
    np.savez(args.output / 'arrays.npz', **arrays)
    failed = not all(r['ok'] for r in reports)
    report = dict(status='fail' if failed else 'pass', backend=args.backend, config=CONFIG,
                  atol=args.atol, rtol=args.rtol, torch=torch.__version__,
                  transformers=transformers.__version__,
                  scope='random-weight scaled model vs official eager F32 CPU; not a trained-model result',
                  reports=reports)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
