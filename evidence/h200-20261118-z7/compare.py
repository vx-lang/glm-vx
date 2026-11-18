"""glm-vx (z7, glm_vx.server --tp 8) against SGLang 0.5.22 --tp 8 (z6) on GLM-5.3, 8x H200, concurrency 1:
the same client for both (sglang.benchmark.serving, benchmarks/gsm8k.py).   python3 compare.py [EVIDENCE_ROOT]"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
z6, z7 = root / 'h200-20261118-z6', root / 'h200-20261118-z7'


def bench(path):
    if not path.exists():
        return None
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return lines[-1] if lines else None


def cell(r, key, scale=1.0, digits=1):
    return '' if r is None or key not in r else f'{r[key] * scale:.{digits}f}'


rows = [('W1 1000/1000', bench(z7 / 'bench/glm-vx-run-W1-vllm.jsonl'), bench(z6 / 'bench/sglang-fp8-W1-sglang-oai.jsonl')),
        ('W3 32768/256', bench(z7 / 'bench/glm-vx-run-W3-vllm.jsonl'), bench(z6 / 'bench/sglang-fp8-W3-sglang.jsonl')),
        ('ShareGPT', bench(z7 / 'sharegpt-glmvx.jsonl'), bench(z6 / 'sharegpt-sglang.jsonl'))]
print('| workload | requests (glm-vx / SGLang) | TPOT median, ms: glm-vx | SGLang | glm-vx / SGLang | TPOT p99: glm-vx | SGLang '
      '| TTFT median, ms: glm-vx | SGLang |')
print('|---|---|---|---|---|---|---|---|---|')
for name, ours, theirs in rows:
    ratio = f"{ours['median_tpot_ms'] / theirs['median_tpot_ms']:.3f}" if ours and theirs else ''
    print(f"| {name} | {cell(ours, 'completed', digits=0)} / {cell(theirs, 'completed', digits=0)} | {cell(ours, 'median_tpot_ms', digits=2)} "
          f"| {cell(theirs, 'median_tpot_ms', digits=2)} | {ratio} | {cell(ours, 'p99_tpot_ms', digits=2)} | {cell(theirs, 'p99_tpot_ms', digits=2)} "
          f"| {cell(ours, 'median_ttft_ms', digits=0)} | {cell(theirs, 'median_ttft_ms', digits=0)} |")
print()
for name, path in (('glm-vx', z7 / 'gsm8k-glmvx.json'), ('SGLang', z6 / 'gsm8k-sglang.json')):
    if path.exists():
        g = json.loads(path.read_text())
        print(f"GSM8K 5-shot, {g['num_questions']} questions, greedy, benchmarks/gsm8k.py: {name} {g['accuracy']:.3f} "
              f"(invalid {g['invalid']:.3f}, {g['seconds']:.0f} s, median TTFT {g['median_ttft_s']:.2f} s)")
official = z7 / 'gsm8k-sglang-official.log'
if official.exists():
    tail = [l for l in official.read_text().splitlines() if l.startswith(('Accuracy', 'Invalid', 'Latency'))]
    print('SGLang, its own sglang.test.few_shot_gsm8k: ' + '; '.join(tail))
a, b = (json.loads(p.read_text()) for p in (z7 / 'gsm8k-glmvx.json', z6 / 'gsm8k-sglang.json')) if (z7 / 'gsm8k-glmvx.json').exists() else (None, None)
if a and b:
    both = sum(x['correct'] and y['correct'] for x, y in zip(a['results'], b['results']))
    only_a = sum(x['correct'] and not y['correct'] for x, y in zip(a['results'], b['results']))
    only_b = sum(y['correct'] and not x['correct'] for x, y in zip(a['results'], b['results']))
    same = sum(x['pred'] == y['pred'] for x, y in zip(a['results'], b['results']))
    print(f'per question: both right {both}, only glm-vx {only_a}, only SGLang {only_b}; the same answer on {same} of {len(a["results"])}')
