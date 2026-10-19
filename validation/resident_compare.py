"""ResidentGLM vs the hybrid GPUBackend model on the same random weights, on a real GPU.

Reports (a) whether the logits are bitwise equal (both run the same kernels in the same order, so
equality is the expectation, and any difference localizes a resident-path orchestration bug),
(b) max abs / rel difference, and (c) wall-clock seconds per token for each path with the
transfers each one does. Timing is a measurement of this script on this machine, not a
benchmark: single run, F32, random weights, includes host work, no warmup beyond one token.

  python -m validation.resident_compare --output run/resident.json [--geometry small|real] [--tokens 64]

With --int4 the Linears the published GLM W4A16 checkpoint quantizes are encoded as
compressed-tensors INT4 (bf16 group-128 scales) and the comparison is ResidentGLM on the packed
weights (INT4 kernel) vs ResidentGLM float32 on their exact dequantization. Only the summation
order differs, so the gate is top-1 equality and max |diff| <= 1e-4 * max |logit|.

With --fp8 the Linears the official GLM FP8 checkpoint quantizes are encoded as block-scaled
E4M3FN (128x128 float32 scales) and compared the same way (FP8 kernel vs float32 on the exact
dequantization, same gate). Weight-only: activations stay float32 in both.

With --bf16 every Linear except the embedding is rounded to bfloat16 and stored as bits; the BF16
kernel decodes exactly and sums in the float32 matvec's order, so bitwise_equal is the expectation
(the gate is the same as above; a non-bitwise result is reported, not hidden).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from glm_vx.model import GlmMoeDsaModel
from validation.gpu_vs_official import CONFIG, random_weights, full_geometry, with_layers

GLM_INT4 = ('.q_a_proj.', '.q_b_proj.', '.kv_a_proj_with_mqa.', '.kv_b_proj.', '.o_proj.',
            '.gate_proj.', '.up_proj.', '.down_proj.')


def int4_source(weights):
    from glm_vx.quant import CompressedInt4Weights, compressed_int4_encode
    linears = {}
    for name, w in weights.items():
        if w.ndim == 2 and name.endswith('.weight') and any(p in name for p in GLM_INT4):
            words, bits = compressed_int4_encode(w)
            linears[name[:-len('.weight')]] = (words, bits, w.shape)
    rest = {n: w for n, w in weights.items() if n[:-len('.weight')] not in linears}
    return CompressedInt4Weights(rest, linears)


def fp8_source(weights):
    from glm_vx.quant import BlockFP8Weights
    from validation.write_checkpoint import UNQUANTIZED, fp8_quantize
    linears = {name[:-len('.weight')]: fp8_quantize(w) for name, w in weights.items()
               if w.ndim == 2 and name.endswith('.weight') and not any(u in name for u in UNQUANTIZED)}
    rest = {n: w for n, w in weights.items() if n[:-len('.weight')] not in linears}
    return BlockFP8Weights(rest, linears)


def bf16_source(weights):
    from glm_vx.quant import BF16Weights
    from validation.write_checkpoint import to_bf16_bits
    linears = {name[:-len('.weight')]: to_bf16_bits(w) for name, w in weights.items()
               if w.ndim == 2 and name.endswith('.weight') and 'embed_tokens' not in name}
    rest = {n: w for n, w in weights.items() if n[:-len('.weight')] not in linears}
    return BF16Weights(rest, linears)


def timed_resident(model, tokens):
    from gpu.resident import ResidentGLM
    resident = ResidentGLM(model, max_tokens=len(tokens) + 1)
    try:
        # Weights upload lazily on first use (only the experts a token routes to), so run the whole
        # sequence once to make the timed pass steady-state, and report that warmup separately.
        t0 = time.perf_counter()
        for t in tokens:
            resident.forward(t)  # logits on: also uploads the final norm and lm_head
        warmup_s = time.perf_counter() - t0
        resident.reset()
        start = time.perf_counter()
        got = np.stack([resident.forward(t) for t in tokens])
        return got, (time.perf_counter() - start) / len(tokens), warmup_s, resident.weight_bytes / 1024 ** 3, dict(resident.launches)
    finally:
        resident.close()


def quantized_main(args, config, weights, tokens, fmt):
    from glm_vx.backend import NumpyBackend
    source = {'int4': int4_source, 'fp8': fp8_source, 'bf16': bf16_source}[fmt](weights)
    candidate = GlmMoeDsaModel(config, source, NumpyBackend())
    baseline = GlmMoeDsaModel(config, source, NumpyBackend())
    baseline_source = baseline.weights
    baseline.weights = lambda name: baseline_source(name)  # hides the packed formats: float32 path
    got, got_s, got_warm, got_gib, launches = timed_resident(candidate, tokens)
    expected, exp_s, exp_warm, exp_gib, _ = timed_resident(baseline, tokens)
    diff = np.abs(got - expected)
    scale = float(np.abs(expected).max())
    report = dict(
        mode=f'{fmt}-vs-f32-dequantized', geometry=args.geometry, layers=config['num_hidden_layers'],
        hidden=config['hidden_size'], tokens=len(tokens), **{f'{fmt}_linears': len(source.linears)},
        bitwise_equal=bool(np.array_equal(got, expected)), max_abs_diff=float(diff.max()), max_abs_logit=scale,
        top1_equal=bool(np.array_equal(got.argmax(-1), expected.argmax(-1))),
        **{f'{fmt}_seconds_per_token': got_s}, f32_seconds_per_token=exp_s,
        **{f'{fmt}_weights_gib_uploaded': got_gib}, f32_weights_gib_uploaded=exp_gib,
        **{f'{fmt}_matvec_launches': launches.get(f'{fmt}_matvec', 0)},
        note=f'single run, synthetic {fmt.upper()} encoding of random weights; kv_b_proj is dequantized to float32 in both; lazy uploads; not a benchmark')
    return report, report['top1_equal'] and report['max_abs_diff'] <= 1e-4 * max(scale, 1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry', choices=['small', 'full'], default='small')
    p.add_argument('--layers', type=int, default=4, help='layers for --geometry full')
    p.add_argument('--tokens', type=int, default=48)
    p.add_argument('--seed', type=int, default=1)
    quant = p.add_mutually_exclusive_group()
    quant.add_argument('--int4', action='store_true', help='INT4 resident vs float32 resident on the dequantized weights')
    quant.add_argument('--fp8', action='store_true', help='block-FP8 resident vs float32 resident on the dequantized weights')
    quant.add_argument('--bf16', action='store_true', help='BF16 resident vs float32 resident on the same bf16 values')
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
    if args.int4 or args.fp8 or args.bf16:
        fmt = 'int4' if args.int4 else 'fp8' if args.fp8 else 'bf16'
        report, ok = quantized_main(args, config, weights, tokens, fmt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return 0 if ok else 1

    hybrid_backend = GPUBackend(weight_cache_bytes=8 * 1024 ** 3)
    hybrid = GlmMoeDsaModel(config, weights, hybrid_backend)
    cache = hybrid.new_cache()
    hybrid.forward(tokens[0], cache)  # warm the weight cache and kernels
    cache = hybrid.new_cache()
    start = time.perf_counter()
    expected = np.stack([hybrid.forward(t, cache) for t in tokens])
    hybrid_s = time.perf_counter() - start
    hybrid_backend.close()

    got, resident_per_token, warmup_s, weight_gib, _ = timed_resident(GlmMoeDsaModel(config, weights, NumpyBackend()), tokens)
    resident_s = resident_per_token * len(tokens)

    diff = np.abs(got - expected)
    report = dict(
        geometry=args.geometry, layers=config['num_hidden_layers'], hidden=config['hidden_size'], tokens=len(tokens),
        bitwise_equal=bool(np.array_equal(got, expected)), max_abs_diff=float(diff.max()),
        max_rel_diff=float((diff / np.maximum(np.abs(expected), 1e-6)).max()),
        top1_equal=bool(np.array_equal(got.argmax(-1), expected.argmax(-1))),
        hybrid_seconds_per_token=hybrid_s / len(tokens), resident_seconds_per_token=resident_s / len(tokens),
        resident_warmup_pass_with_lazy_weight_upload_s=warmup_s, resident_weights_gib_uploaded=weight_gib,
        note='single run, F32, random weights; weights upload lazily so only experts routed in this sequence are resident; timings include host work and are not a benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['top1_equal'] and report['max_abs_diff'] < 1e-4 else 1


if __name__ == '__main__':
    raise SystemExit(main())
