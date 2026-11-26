"""z64 (#63, TTFT): passkey retrieval through a server's /v1/completions, greedy, as a long-context check of the DSA
indexer's selection (prompts past index_topk = 2048 tokens). A five-digit key is hidden in repeated filler at depths
0.1 .. 0.9 of prompts of about 4K, 8K, 16K and 30K tokens (token ids, the tokenizer's own prefix first); the answer is
right when the key is in the first 8 tokens generated. The same prompts for every engine (seed 64).

    python passkey.py URL TOKENIZER_JSON OUT_JSON
"""
import json
import random
import sys
import time
import urllib.request

from tokenizers import Tokenizer

url, tokenizer_path, out_path = sys.argv[1:4]
tok = Tokenizer.from_file(tokenizer_path)
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
prefix = tok.encode('', add_special_tokens=True).ids  # the template's special tokens, if any
head = enc('There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. '
           'I will quiz you about the important information there.\n\n')
tail = enc('\n\nWhat is the pass key? The pass key is')
filler = enc('The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ')
models = json.load(urllib.request.urlopen(url + '/v1/models', timeout=60))
model = models['data'][0]['id']
rng = random.Random(64)
results = []
for length in (4000, 8000, 16000, 30000):
    for depth in (0.1, 0.3, 0.5, 0.7, 0.9):
        key = str(rng.randint(10000, 99999))
        needle = enc(f' The pass key is {key}. Remember it. {key} is the pass key. ')
        room = length - len(prefix) - len(head) - len(needle) - len(tail)
        body = (filler * (room // len(filler) + 1))[:room]
        at = int(len(body) * depth)
        ids = prefix + head + body[:at] + needle + body[at:] + tail
        request = urllib.request.Request(url + '/v1/completions', json.dumps(dict(
            model=model, prompt=ids, max_tokens=8, temperature=0)).encode(), {'Content-Type': 'application/json'})
        t = time.perf_counter()
        reply = json.load(urllib.request.urlopen(request, timeout=3600))
        seconds = time.perf_counter() - t
        text = reply['choices'][0]['text']
        results.append(dict(tokens=len(ids), depth=depth, key=key, text=text, ok=key in text, seconds=round(seconds, 2)))
        print(json.dumps(results[-1]), flush=True)
summary = {}
for r in results:
    s = summary.setdefault(r['tokens'] // 1000 * 1000, [0, 0])
    s[0] += r['ok']
    s[1] += 1
report = dict(url=url, model=model, prefix=prefix, results=results,
              correct=sum(r['ok'] for r in results), total=len(results),
              by_length={f'{k}': f'{v[0]}/{v[1]}' for k, v in sorted(summary.items())})
json.dump(report, open(out_path, 'w'), indent=1)
print(f"passkey: {report['correct']}/{report['total']} right; by length {report['by_length']}", flush=True)
