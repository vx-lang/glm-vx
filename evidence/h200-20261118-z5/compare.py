"""z5: glm-vx TP8 against SGLang --tp 8 on GLM-5.3, batch 1. Wall time of one decode step at each context:
ours is validation/tp_real (median of the timed steps; each returns the downloaded logits), SGLang's is its
one_batch median decode latency (each step synchronized; it samples on the GPU). Greedy agreement from
sglang_greedy.py.   python3 compare.py EVIDENCE_DIR"""
import json
import sys
from pathlib import Path


def sglang(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[int(r['input_len'])] = r
    return rows


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    ours = json.loads((d / 'real-8-oneshot.json').read_text()) if (d / 'real-8-oneshot.json').exists() else {}
    theirs, dummy = sglang(d / 'sglang-one-batch.jsonl'), sglang(d / 'sglang-one-batch-dummy.jsonl')
    timings = ours.get('timings', {})
    contexts = sorted({int(n) for n in timings} | set(theirs) | set(dummy))
    print('| context | glm-vx TP8, us/step (median) | SGLang --tp 8, us/step (median) | glm-vx / SGLang | SGLang, dummy weights |')
    print('|---|---|---|---|---|')
    for n in contexts:
        mine = timings.get(str(n), {}).get('us_per_step_median')
        sg = theirs.get(n, {}).get('median_decode_latency')
        sg = sg * 1e6 if sg is not None else None
        dm = dummy.get(n, {}).get('median_decode_latency')
        ratio = f'{mine / sg:.2f}' if mine and sg else ''
        print(f"| {n} | {mine if mine else ''} | {f'{sg:.1f}' if sg else ''} | {ratio} | {f'{dm * 1e6:.1f}' if dm else ''} |")
    if ours:
        print(f"\nglm-vx: ok={ours.get('ok')}, load {ours.get('setup_s')} s + first step {ours.get('first_step_s')} s; "
              f"text after the prompt: {ours.get('text')!r}")
        for n, t in timings.items():
            print(f"  context {n}: fill {t['fill_tokens_per_s']} tokens/s; step mean {t['us_per_step_mean']} min {t['us_per_step_min']} us")
    g = d / 'sglang-greedy.json'
    if g.exists():
        g = json.loads(g.read_text())
        print(f"greedy: {g.get('leading_tokens_agree')} of {g.get('of')} leading tokens agree; "
              f"max |logprob difference| on shared top-5 entries {g.get('max_logprob_diff')}; SGLang text {g.get('text')!r}")


if __name__ == '__main__':
    main()
