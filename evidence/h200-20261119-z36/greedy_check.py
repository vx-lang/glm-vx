"""z5's prompt through the server's /v1/completions, greedy: its 32 tokens against tp_real's (z5) and against
z11's server (the same options as z12, the logits to the host)."""
import json
import sys
import urllib.request

ours = json.load(open(sys.argv[1]))
body = dict(prompt=ours['prompt_ids'], max_tokens=len(ours['generated']), temperature=0, ignore_eos=True)
request = urllib.request.Request(sys.argv[2] + '/v1/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
reply = json.load(urllib.request.urlopen(request, timeout=1800))
got = reply['choices'][0]['token_ids']
z11 = [35316, 279, 1156, 3238, 97113, 13, 1096, 27041, 13782, 1506, 30373, 323, 62135, 279, 1616, 369, 279, 4401, 315, 6481,
       18464, 11, 34848, 323, 1008, 14336, 7611, 13, 576, 97113, 12569, 77198]
agree = next((i for i, (a, b) in enumerate(zip(got, ours['generated'])) if a != b), len(got))
print(json.dumps(dict(same_as_z11=got == z11, same_as_tp_real=got == ours['generated'], leading_agree_tp_real=agree,
                      served=got, text=reply['choices'][0]['text'], timings=reply.get('timings'))))
