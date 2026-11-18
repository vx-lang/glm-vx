"""z16: how much of the served decode time is the GPU's (#63, docs/TP-SERVING-PLAN.md item 2), on one GPU.

  device_loop_time.py CHECKPOINT TP_REAL.json OUT.json

The model as glm_vx.server --tp 1 --tp-option moe_down2=true fuse_router=true device_decode=true builds it
(gpu/tp_serve.start, one rank), a 1000-token prompt (tp_real's repeated), then from position 1000, 200 steps
each way (rewound between):
  chain:  device_step() back to back, nothing read until the last: the GPU's own time a step;
  ahead:  as ResidentSequenceModel.forward_greedy: step n+1 launched, then token n read;
  item1:  forward(token, logits='greedy'), the token read before the next step is launched.
"""
import json
import sys
import time

from gpu import tp_serve

checkpoint, ref_path, out_path = sys.argv[1:4]
ref = json.load(open(ref_path))
prompt = (ref['prompt_ids'] * 40)[:1000]
N = 200
model, lock = tp_serve.start(checkpoint, 1, 8192, one_shot=True,
                             options=dict(moe_down2=True, fuse_router=True, device_decode=True))
r = lock.resident
r.preload()
for token in prompt[:-1]:
    r.forward(token, logits=False)
base = r.position  # 999: the last prompt token goes in at this position
results = {}


def chain():
    r.rewind(base)
    first = r.device_step(prompt[-1])
    r.device_token(first)
    start = time.perf_counter()
    for _ in range(N - 1):
        r.device_step(readback=False)
    last = r.device_step()
    token = r.device_token(last)
    return (time.perf_counter() - start) / N, token


def ahead():
    r.rewind(base)
    pending = r.device_step(prompt[-1])
    tokens = []
    start = time.perf_counter()
    for _ in range(N):
        following = r.device_step()
        tokens.append(r.device_token(pending))
        pending = following
    elapsed = time.perf_counter() - start
    r.device_token(pending)
    return elapsed / N, tokens


def item1():
    r.rewind(base)
    token = r.forward(prompt[-1], logits='greedy')
    tokens = []
    start = time.perf_counter()
    for _ in range(N):
        tokens.append(token)
        token = r.forward(token, logits='greedy')
    return (time.perf_counter() - start) / N, tokens


for name, fn in (('chain', chain), ('ahead', ahead), ('item1', item1)):
    fn()  # warm: the graphs of each short/long shape captured
    runs = [fn() for _ in range(3)]
    results[name] = dict(ms_per_step=[round(1e3 * t, 4) for t, _ in runs])
    results[name]['ms_per_step_min'] = min(results[name]['ms_per_step'])
    if name != 'chain':
        results[name]['tokens'] = runs[0][1]
results['ahead_tokens_equal_item1'] = results['ahead']['tokens'] == results['item1']['tokens']
results['ahead_ms_over_chain'] = round(results['ahead']['ms_per_step_min'] - results['chain']['ms_per_step_min'], 4)
results['item1_ms_over_chain'] = round(results['item1']['ms_per_step_min'] - results['chain']['ms_per_step_min'], 4)
lock.close()
json.dump(dict(checkpoint=checkpoint, prompt_tokens=len(prompt), steps=N, **results), open(out_path, 'w'), indent=1)
print(json.dumps({k: v for k, v in results.items() if k not in ('ahead', 'item1')} |
                 {k: {f: x for f, x in results[k].items() if f != 'tokens'} for k in ('ahead', 'item1')}))
