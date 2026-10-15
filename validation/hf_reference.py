"""M0 reference harness: dump Hugging Face logits for a fixed prompt set.

Runs the pinned Transformers GLM-MoE-DSA model (optional dependency, never
imported by serving) and writes an .npz of per-position logits plus a greedy
continuation per prompt, for comparison with the Vx GPU path (see
compare_to_reference below and docs/GPU-BRINGUP.md for pass criteria).

  python -m validation.hf_reference dump --model DIR --prompts prompts.json \
      --max-new 16 --output ref.npz
  python -m validation.hf_reference compare --reference ref.npz --candidate cand.npz \
      --atol 1e-2 --min-match 16

prompts.json: list of token-id lists. The same file drives the Vx run; the
candidate npz uses identical keys: logits_<i> [T,V] float32 and greedy_<i> [N].
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def dump(args):
    import torch
    from transformers import AutoModelForCausalLM
    prompts = json.loads(Path(args.prompts).read_text())
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    out = {}
    with torch.no_grad():
        for i, ids in enumerate(prompts):
            x = torch.tensor([ids])
            out[f'logits_{i}'] = model(x).logits[0].float().numpy()
            gen = model.generate(x, max_new_tokens=args.max_new, do_sample=False)
            out[f'greedy_{i}'] = gen[0, len(ids):].numpy()
    np.savez(args.output, **out)


def compare(args):
    ref, cand = np.load(args.reference), np.load(args.candidate)
    failures = []
    for key in sorted(k for k in ref.files if k.startswith('greedy_')):
        i = key.split('_')[1]
        a, b = ref[key], cand[key]
        n = min(len(a), len(b))
        match = int(np.argmin(np.append(a[:n] == b[:n], False)))  # first mismatch
        if match < min(args.min_match, len(a)):
            failures.append(f'prompt {i}: greedy diverges at token {match}')
        diff = float(np.abs(ref[f'logits_{i}'] - cand[f'logits_{i}']).max())
        print(f'prompt {i}: greedy_match={match}/{len(a)} logit_max_abs_diff={diff:.4g}')
        if diff > args.atol:
            failures.append(f'prompt {i}: logit diff {diff:.4g} > {args.atol}')
    for f in failures:
        print('FAIL', f, file=sys.stderr)
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--model', required=True)
    d.add_argument('--prompts', required=True)
    d.add_argument('--max-new', type=int, default=16)
    d.add_argument('--dtype', default='bfloat16')
    d.add_argument('--output', required=True)
    c = sub.add_parser('compare')
    c.add_argument('--reference', required=True)
    c.add_argument('--candidate', required=True)
    c.add_argument('--atol', type=float, required=True)
    c.add_argument('--min-match', type=int, required=True)
    args = p.parse_args()
    if args.cmd == 'dump':
        if Path(args.output).exists():
            p.error('output exists; preserve evidence')
        dump(args)
        return 0
    return compare(args)


if __name__ == '__main__':
    sys.exit(main())
