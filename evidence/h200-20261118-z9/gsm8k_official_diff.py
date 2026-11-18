"""Per question: SGLang's own few_shot_gsm8k (z9, tmp_output_gsm8k.txt) against benchmarks/gsm8k.py on SGLang (z6)
and on glm-vx (z7): the answer each extracted and whether it was right."""
import ast, json, re, sys
ev, data = sys.argv[1], sys.argv[2]
text = open(f'{ev}/h200-20261118-z9/tmp_output_gsm8k.txt').read()
lines = [json.loads(l) for l in open(data)]
mine = json.load(open(f'{ev}/h200-20261118-z6/gsm8k-sglang.json'))['results']
ours = json.load(open(f'{ev}/h200-20261118-z7/gsm8k-glmvx.json'))['results']
def val(s):
    n = re.findall(r'\d+', s.replace(',', ''))
    try:
        return ast.literal_eval(n[-1]) if n else -9999999
    except SyntaxError:
        return -9999999
parts = re.split(r'^=+ (\d+) =+$', text, flags=re.M)
states = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
print('states', len(states))
states = [states[k] for k in sorted(states)]
off = {'same_text': 0, 'official_right': 0}
rows = []
for i, st in enumerate(states[:200]):
    q = 'Question: ' + lines[i]['question'] + '\nAnswer:'
    k = st.rfind(q)
    ans = st[k + len(q):] if k >= 0 else ''
    ans = re.split(r'\n=+\n', ans)[0]
    label = val(lines[i]['answer'])
    o = val(ans)
    rows.append((i, label, o, mine[i]['pred'], ours[i]['pred']))
    off['official_right'] += o == label
    off['same_text'] += ans.strip()[:200] == mine[i]['text'].strip()[:200]
print(off)
both = sum(1 for i, l, o, m, u in rows if o == l and m == l)
print('official right', sum(o == l for i, l, o, m, u in rows), 'client-on-SGLang right', sum(m == l for i, l, o, m, u in rows),
      'glm-vx right', sum(u == l for i, l, o, m, u in rows))
print('official vs client on SGLang: same answer', sum(o == m for i, l, o, m, u in rows), 'of', len(rows))
diff = [(i, l, o, m, u) for i, l, o, m, u in rows if o != m]
print('questions where the two SGLang runs answer differently:', len(diff))
for i, l, o, m, u in diff[:30]:
    print(f'  q{i}: label {l} official {o} client-on-SGLang {m} glm-vx {u}')
