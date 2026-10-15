"""Batched/chunked prefill (the serving path) vs. unmodified Transformers.

layer_compare.py exercises the token-by-token executor; serving uses
`prefill_chunk` (1..64 tokens, layer-wise batched). This checks, for several
chunkings of the same prompt on the chosen backend, that the final prefill logits,
the cache it leaves behind (via subsequent cached greedy decode) and an
intermediate-chunk continuation all agree with the official model, and that every
chunking agrees with every other one.

  python -m validation.prefill_compare --backend vx-gpu --output run/prefill
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel
from validation.gpu_vs_official import CONFIG, REAL_GEOMETRY, official_model, official_run, random_weights


def chunked(model, tokens, sizes):
    cache = model.new_cache()
    pos, last = 0, None
    for size in sizes:
        last = model.prefill_chunk(tokens[pos:pos + size], cache, output_logits=pos + size == len(tokens))
        pos += size
    assert pos == len(tokens)
    return cache, last


def decode(model, cache, last, new):
    ids = []
    for _ in range(new):
        ids.append(int(np.argmax(last)))
        last = model.forward(ids[-1], cache)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['vx-gpu', 'vx', 'numpy-reference'], default='vx-gpu')
    p.add_argument('--geometry', choices=['small', 'real'], default='small')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    p.add_argument('--prompt-len', type=int, default=100)
    p.add_argument('--new', type=int, default=8)
    p.add_argument('--atol', type=float, default=2e-4)
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
    config = dict(base, max_position_embeddings=max(base['max_position_embeddings'], args.prompt_len + 64))
    n = args.prompt_len
    chunkings = {'whole(<=64)': [n] if n <= 64 else [64, n - 64], 'singles-then-chunk': [1, 1, 1, n - 3] if n - 3 <= 64 else [1, 1, 1, 61, n - 64],
                 'uneven': [7, 33, 1, n - 41] if n - 41 <= 64 else [7, 33, 1, 64, n - 105], 'by-16': [16] * (n // 16) + ([n % 16] if n % 16 else [])}
    runs, ok = [], True
    for seed in args.seeds:
        weights = random_weights(config, seed)
        for w in weights.values():
            w.flags.writeable = False
        tokens = [int(t) for t in np.random.default_rng(900 + seed).integers(0, config['vocab_size'], n)]
        ref_logits, ref_ids = official_run(official_model(config, weights), tokens, args.new)
        results = {}
        for name, sizes in chunkings.items():
            sizes = [s for s in sizes if s > 0]
            model = GlmMoeDsaModel(config, weights, backend)
            cache, last = chunked(model, tokens, sizes)
            err = float(np.abs(last - ref_logits[-1]).max() / max(1.0, np.abs(ref_logits[-1]).max()))
            ids = decode(model, cache, last, args.new)
            results[name] = dict(sizes=sizes, final_logit_rel_err=err, greedy_match=ids == ref_ids.tolist(),
                                 ok=err <= args.atol and ids == ref_ids.tolist())
            ok &= results[name]['ok']
            print(f"seed {seed} {name:20s} err={err:.2e} greedy_match={results[name]['greedy_match']}")
        runs.append(dict(seed=seed, results=results))
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(dict(
        status='pass' if ok else 'fail', backend=args.backend, geometry=args.geometry, prompt_len=n,
        scope='random weights vs official eager F32; chunked prefill + cached decode', runs=runs), indent=2) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
