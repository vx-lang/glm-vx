"""Per-layer, per-stage comparison of a Vx backend against unmodified Transformers.

Runs one teacher-forced token sequence through the official eager F32 model
(batched, with forward hooks) and through Vx (token by token with a trace), then
compares every layer's stages at every position, in execution order:

  input_norm -> attention.output -> post_attention_norm -> mlp.output -> output

plus the DSA top-k selection of every layer. The report names the *first*
(position, layer, stage) outside tolerance, so a bug is localized instead of
showing up only as a wrong token. Selection mismatches are classified: a tie
(all differing keys score within `--tie-eps` of the k-th score in Vx's own
indexer scores) is reported separately from a real selection error, because
torch.topk and Vx break exact ties differently and that is not a numerical bug.

Random-weight scaled model (see gpu_vs_official.py); never imported by serving.

  python -m validation.layer_compare --backend vx-gpu --output run/layers
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel
from validation.gpu_vs_official import CONFIG, REAL_GEOMETRY, official_model, random_weights

STAGES = ('input_norm', 'attention.output', 'post_attention_norm', 'mlp.output', 'output')


def official_stages(model, tokens):
    """{(pos, layer, stage): vector} and {(pos, layer): sorted causal selection}."""
    import torch
    got, sel = {}, {}

    def put(layer, stage, tensor):
        for t in range(tensor.shape[1]):
            got[(t, layer, stage)] = tensor[0, t].detach().numpy().astype(np.float64)

    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(lambda m, a, o, i=i: put(i, 'input_norm', o))
        layer.self_attn.register_forward_hook(lambda m, a, o, i=i: put(i, 'attention.output', o[0]))
        layer.post_attention_layernorm.register_forward_hook(lambda m, a, o, i=i: put(i, 'post_attention_norm', o))
        layer.mlp.register_forward_hook(lambda m, a, o, i=i: put(i, 'mlp.output', o))

        def layer_hook(m, a, o, i=i):
            put(i, 'output', o[0])
            ids = o[1][0].numpy()
            for t in range(ids.shape[0]):
                sel[(t, i)] = np.sort(ids[t][ids[t] <= t]).astype(np.int64)
        layer.register_forward_hook(layer_hook)
    with torch.no_grad():
        result = model(torch.tensor([tokens]), use_cache=False)
    for t in range(len(tokens)):
        got[(t, -1, 'logits')] = result.logits[0, t].numpy().astype(np.float64)
    return got, sel


def candidate_stages(model, tokens):
    raw = {}
    model_ = model
    model_.trace = lambda p, n, v: raw.__setitem__((p, n), np.array(v, dtype=np.float64))
    cache = model_.new_cache()
    for t in tokens:
        last = model_.forward(t, cache)
    got = {}
    for (p, name), v in raw.items():
        if name == 'logits':
            got[(p, -1, 'logits')] = v
            continue
        parts = name.split('.', 2)
        if parts[0] == 'layer' and parts[2] in STAGES:
            got[(p, int(parts[1]), parts[2])] = v
    return got, raw


def classify_selection(raw, layers, mlp_full_layers, pos, layer, official, mine, topk, eps):
    """'match' | 'tie' | 'mismatch' for one (pos, layer)."""
    if np.array_equal(official, mine):
        return 'match'
    # Shared layers inherit from the previous full layer; its tie decides.
    full = layer
    while full not in mlp_full_layers:
        full -= 1
    scores = raw.get((pos, f'layer.{full}.indexer.scores'))
    if scores is None:
        return 'mismatch'
    scores = np.asarray(scores, dtype=np.float64)
    k = min(topk, len(scores))
    boundary = np.sort(scores)[::-1][k - 1]
    differing = np.setxor1d(official, mine)
    return 'tie' if all(abs(scores[d] - boundary) <= eps for d in differing) else 'mismatch'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['vx-gpu', 'vx', 'numpy-reference'], default='vx-gpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--prompt-len', type=int, default=48)
    p.add_argument('--geometry', choices=['small', 'real'], default='small',
                   help="'real' uses GLM-5.3's per-head dimensions (64 heads, rank 512, 192+64 qk, 256 v, 32x128 indexer)")
    p.add_argument('--topk', type=int, default=None)
    p.add_argument('--tol', type=float, default=2e-4,
                   help='max abs diff / max(1, max abs reference) per stage tensor')
    p.add_argument('--perturb', metavar='WEIGHT:REL',
                   help='harness self-check: scale this Vx-side weight by 1+REL (e.g. '
                        'model.layers.3.mlp.shared_experts.down_proj.weight:1e-3); the report must blame that layer')
    p.add_argument('--tie-eps', type=float, default=1e-5)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    if args.backend == 'vx-gpu':
        from gpu.backend import GPUBackend
        backend = GPUBackend()
    elif args.backend == 'vx':
        from kernels.backend import VxBackend
        backend = VxBackend()
    else:
        from glm_vx.backend import NumpyBackend
        backend = NumpyBackend()
    base = REAL_GEOMETRY if args.geometry == 'real' else CONFIG
    args.topk = args.topk or base['index_topk']
    config = dict(base, index_topk=args.topk,
                  max_position_embeddings=max(base['max_position_embeddings'], args.prompt_len + 16))
    full_layers = {i for i, t in enumerate(config['indexer_types']) if t == 'full'}
    runs, failures = [], {}
    for seed in args.seeds:
        weights = random_weights(config, seed)
        for w in weights.values():
            w.flags.writeable = False
        tokens = [int(t) for t in np.random.default_rng(500 + seed).integers(0, config['vocab_size'], args.prompt_len)]
        ref, ref_sel = official_stages(official_model(config, weights), tokens)
        mine_w = weights
        if args.perturb:
            name, rel = args.perturb.rsplit(':', 1)
            mine_w = dict(weights)
            mine_w[name] = (weights[name] * (1 + float(rel))).astype(np.float32)
            mine_w[name].flags.writeable = False
        got, raw = candidate_stages(GlmMoeDsaModel(config, mine_w, backend), tokens)
        rows, first, ties, bad_sel = [], None, 0, []
        order = [(t, l, s) for t in range(len(tokens)) for l in range(config['num_hidden_layers']) for s in STAGES]
        order += [(t, -1, 'logits') for t in range(len(tokens))]
        worst = {}
        for key in order:
            a, b = ref[key], got[key]
            err = float(np.abs(a - b).max() / max(1.0, np.abs(a).max()))
            worst[key[1:]] = max(worst.get(key[1:], 0.0), err)
            if err > args.tol and first is None:
                first = dict(position=key[0], layer=key[1], stage=key[2], error=err)
                failing = {f'seed{seed}.official': a, f'seed{seed}.vx': b}
        for t in range(len(tokens)):
            for l in range(config['num_hidden_layers']):
                mine = np.sort(np.asarray(raw[(t, f'layer.{l}.selected')], dtype=np.int64))
                verdict = classify_selection(raw, config['num_hidden_layers'], full_layers, t, l,
                                             ref_sel[(t, l)], mine, args.topk, args.tie_eps)
                if verdict == 'tie':
                    ties += 1
                elif verdict == 'mismatch':
                    bad_sel.append(dict(position=t, layer=l, official=ref_sel[(t, l)].tolist(), vx=mine.tolist()))
        table = {f'{l}.{s}': e for (l, s), e in sorted(worst.items())}
        # A tie flips later stages legitimately; only trust "first" when no tie came earlier.
        if first is not None:
            failures.update(failing)
        digest = hashlib.sha256(b''.join(got[(t, -1, 'logits')].astype('<f8').tobytes()
                                         for t in range(len(tokens)))).hexdigest()
        runs.append(dict(seed=seed, tokens=len(tokens), vx_logits_sha256=digest, first_outside_tolerance=first,
                         selection_ties=ties, selection_mismatches=bad_sel[:10],
                         selection_mismatch_count=len(bad_sel), worst_error_by_layer_stage=table))
        status = 'ok' if first is None and not bad_sel else ('tie-affected' if ties and not bad_sel else 'FAIL')
        print(f'seed {seed}: {status}  ties={ties} bad_selections={len(bad_sel)} first={first}')
    ok = all(r['first_outside_tolerance'] is None and not r['selection_mismatch_count'] for r in runs)
    args.output.mkdir(parents=True)
    report = dict(status='pass' if ok else 'fail', backend=args.backend, topk=args.topk, tol=args.tol,
                  stages=list(STAGES), geometry=args.geometry, config=config,
                  scope='random-weight scaled model vs official eager F32 CPU, teacher-forced; not trained-model parity',
                  runs=runs)
    if failures:
        np.savez(args.output / 'first_failure_vectors.npz', **failures)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
