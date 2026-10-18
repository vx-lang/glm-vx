"""ResidentGLM vs the hybrid GPUBackend model on the same random weights, on a real GPU.

Reports (a) whether the logits are bitwise equal (both run the same kernels in the same order, so
equality is the expectation, and any difference localizes a resident-path orchestration bug),
(b) max abs / rel difference, and (c) wall-clock seconds per token for each path with the
transfers each one does. Timing is a measurement of this script on this machine, not a
benchmark: single run, F32, random weights, includes host work, no warmup beyond one token.

  python -m validation.resident_compare --output run/resident.json [--geometry small|real] [--tokens 64]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel
from validation.gpu_vs_official import CONFIG, random_weights, full_geometry, with_layers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry', choices=['small', 'full'], default='small')
    p.add_argument('--layers', type=int, default=4, help='layers for --geometry full')
    p.add_argument('--tokens', type=int, default=48)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    from gpu.backend import GPUBackend
    from gpu.resident import ResidentGLM
    from glm_vx.backend import NumpyBackend
    config = CONFIG if args.geometry == 'small' else full_geometry(args.layers)
    weights = random_weights(config, args.seed)
    for w in weights.values():
        w.flags.writeable = False
    tokens = [int(t) for t in np.random.default_rng(100 + args.seed).integers(0, config['vocab_size'], args.tokens)]

    hybrid_backend = GPUBackend(weight_cache_bytes=8 * 1024 ** 3)
    hybrid = GlmMoeDsaModel(config, weights, hybrid_backend)
    cache = hybrid.new_cache()
    hybrid.forward(tokens[0], cache)  # warm the weight cache and kernels
    cache = hybrid.new_cache()
    start = time.perf_counter()
    expected = np.stack([hybrid.forward(t, cache) for t in tokens])
    hybrid_s = time.perf_counter() - start
    hybrid_backend.close()

    resident = ResidentGLM(GlmMoeDsaModel(config, weights, NumpyBackend()), max_tokens=len(tokens) + 1)
    try:
        t0 = time.perf_counter()
        resident.forward(tokens[0])  # uploads every weight the first time
        upload_s = time.perf_counter() - t0
        resident.reset()
        start = time.perf_counter()
        got = np.stack([resident.forward(t) for t in tokens])
        resident_s = time.perf_counter() - start
        weight_gib = resident.weight_bytes / 1024 ** 3
    finally:
        resident.close()

    diff = np.abs(got - expected)
    report = dict(
        geometry=args.geometry, layers=config['num_hidden_layers'], hidden=config['hidden_size'], tokens=len(tokens),
        bitwise_equal=bool(np.array_equal(got, expected)), max_abs_diff=float(diff.max()),
        max_rel_diff=float((diff / np.maximum(np.abs(expected), 1e-6)).max()),
        top1_equal=bool(np.array_equal(got.argmax(-1), expected.argmax(-1))),
        hybrid_seconds_per_token=hybrid_s / len(tokens), resident_seconds_per_token=resident_s / len(tokens),
        resident_first_token_with_weight_upload_s=upload_s, resident_weights_gib=weight_gib,
        note='single run, F32, random weights; timings include host work and are not a benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['top1_equal'] and report['max_abs_diff'] < 1e-4 else 1


if __name__ == '__main__':
    raise SystemExit(main())
