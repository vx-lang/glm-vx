"""Greedy tokens and decode time from glm_vx.server's /v1/completions on one GPU (z16).

  greedy_tokens.py URL TP_REAL.json OUT.json [OTHER.json]

Prompts: tp_real's prompt ids (its 32 greedy tokens must come back) and that prompt repeated to about 1000
and 3000 tokens (128 tokens each, 3 times: the median server decode time a token, decode_seconds over
completion_tokens - 1). With OTHER.json (another server's OUT), every prompt's tokens must equal its."""
import json
import statistics
import sys
import time
import urllib.request

url, ref_path, out_path = sys.argv[1:4]
other = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
ref = json.load(open(ref_path))
base = ref['prompt_ids']


def complete(prompt, new):
    body = dict(prompt=prompt, max_tokens=new, temperature=0, ignore_eos=True)
    request = urllib.request.Request(url + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    start = time.perf_counter()
    reply = json.load(urllib.request.urlopen(request, timeout=3600))
    wall = time.perf_counter() - start
    choice, timings = reply['choices'][0], reply.get('timings') or {}
    done = reply['usage']['completion_tokens']
    return dict(tokens=choice['token_ids'], completion_tokens=done, wall_seconds=round(wall, 4),
                decode_ms_per_token=round(1e3 * timings['decode_seconds'] / max(1, done - 1), 4)
                if timings.get('decode_seconds') is not None else None, timings=timings)


runs = {}
first = complete(base, len(ref['generated']))
runs['tp_real_prompt'] = dict(prompt_tokens=len(base), **first, same_as_tp_real=first['tokens'] == ref['generated'])
for n in (1000, 3000):
    prompt = (base * (n // len(base) + 1))[:n]
    tries = [complete(prompt, 128) for _ in range(3)]
    runs[f'context_{n}'] = dict(prompt_tokens=n, tokens=tries[0]['tokens'],
                                repeat_tokens_equal=all(t['tokens'] == tries[0]['tokens'] for t in tries),
                                decode_ms_per_token=[t['decode_ms_per_token'] for t in tries],
                                decode_ms_per_token_median=statistics.median(t['decode_ms_per_token'] for t in tries),
                                wall_seconds=[t['wall_seconds'] for t in tries])
result = dict(url=url, tp_real=ref_path, runs=runs)
if other is not None:
    result['same_tokens_as'] = sys.argv[4]
    result['tokens_equal'] = {k: v['tokens'] == other['runs'][k]['tokens'] for k, v in runs.items()}
    result['all_equal'] = all(result['tokens_equal'].values())
json.dump(result, open(out_path, 'w'), indent=1)
print(json.dumps({k: (v if k != 'runs' else {r: {f: x for f, x in d.items() if f not in ('tokens', 'timings')}
                                              for r, d in v.items()}) for k, v in result.items()}))
