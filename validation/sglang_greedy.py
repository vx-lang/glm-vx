"""SGLang's greedy tokens and top-5 log-probabilities for validation/tp_real.py's prompt (#63).

Run with SGLang's own venv (no glm-vx imports): the same prompt token ids as tp_real's output JSON,
temperature 0, --new tokens, SGLang's offline Engine with --tp. Writes its tokens, text and per-step
top-5 (ids, log-probabilities) next to ours, and how many leading tokens agree.

  NCCL_NVLS_ENABLE=0 sglang-venv/bin/python validation/sglang_greedy.py --model ~/GLM-5.3 \\
      --ours run/real.json --tp 8 --output run/sglang-greedy.json
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ours', type=Path, required=True, help="validation/tp_real.py's --output")
    p.add_argument('--tp', type=int, default=8)
    p.add_argument('--mem-fraction', type=float, default=0.8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    ours = json.loads(args.ours.read_text())
    prompt, new = ours['prompt_ids'], len(ours['generated'])
    import sglang as sgl
    engine = sgl.Engine(model_path=str(args.model), tp_size=args.tp, mem_fraction_static=args.mem_fraction,
                        kv_cache_dtype='bfloat16', random_seed=0, log_level='warning')
    try:
        out = engine.generate(input_ids=prompt, sampling_params=dict(temperature=0, max_new_tokens=new),
                              return_logprob=True, top_logprobs_num=5)
    finally:
        engine.shutdown()
    meta = out['meta_info']
    generated = [int(t[1]) for t in meta['output_token_logprobs']]
    tops = [dict(ids=[int(e[1]) for e in step], logprobs=[float(e[0]) for e in step]) for step in meta['output_top_logprobs']]
    agree = next((i for i, (a, b) in enumerate(zip(generated, ours['generated'])) if a != b), min(len(generated), new))
    steps = []
    for i, (theirs, mine) in enumerate(zip(tops, ours['tops'])):
        # Log-probability differences on the tokens both top-5s hold, while the two sequences agree.
        if i >= agree:
            break
        shared = {t: lp for t, lp in zip(theirs['ids'], theirs['logprobs'])}
        diffs = [abs(shared[t] - lp) for t, lp in zip(mine['ids'], mine['logprobs']) if t in shared]
        steps.append(dict(step=i, top1_same=theirs['ids'][0] == mine['ids'][0], shared=len(diffs),
                          max_logprob_diff=max(diffs) if diffs else None))
    summary = dict(model=str(args.model), tp=args.tp, prompt_ids=prompt, generated=generated, text=out['text'],
                   ours_generated=ours['generated'], ours_text=ours.get('text'), leading_tokens_agree=agree, of=new,
                   max_logprob_diff=max((s['max_logprob_diff'] or 0) for s in steps) if steps else None,
                   steps=steps, tops=tops)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('tops', 'steps', 'prompt_ids')}))


if __name__ == '__main__':
    main()
