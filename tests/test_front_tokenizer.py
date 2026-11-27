"""The front end's tokenizer (front/src/tokenizer.vx) against HF `tokenizers` (#68, step 3).

front/build/tokenize_check loads a tokenizer.json and writes the ids of each text. They must be
HF's `Tokenizer.encode(text).ids`, for:
- the cases below: contractions, white space and line ends, numbers, scripts, added tokens;
- every code point, in a few contexts;
- random text over all of those (FRONT_TOKENIZER_FUZZ texts, 3000 by default);
- GSM8K's questions, when GSM8K_JSONL names its test.jsonl.

GLM_TOKENIZER names zai-org/GLM-5.3's tokenizer.json (20 MB, not in the repository). FRONT_PRELOAD
runs the program under AddressSanitizer (see tests/front_server.py).
"""
import json
import os
import random
import struct
import subprocess
from pathlib import Path

import pytest

CHECK = Path(__file__).resolve().parents[1] / 'front' / 'build' / 'tokenize_check'
TOKENIZER = os.environ.get('GLM_TOKENIZER')
ENV = {**os.environ, **({'LD_PRELOAD': os.environ['FRONT_PRELOAD'], 'ASAN_OPTIONS': 'detect_leaks=0'}
                        if os.environ.get('FRONT_PRELOAD') else {})}
pytestmark = pytest.mark.skipif(not CHECK.exists() or not TOKENIZER, reason='needs front/build.sh and GLM_TOKENIZER')


@pytest.fixture(scope='module')
def hf():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER)


def ours(texts, tokenizer=None):
    data = b''.join(struct.pack('<I', len(t.encode())) + t.encode() for t in texts)
    out = subprocess.run([str(CHECK), tokenizer or TOKENIZER], input=data, capture_output=True, timeout=1800, env=ENV)
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
    got, at = [], 0
    for _ in texts:
        n, = struct.unpack_from('<I', out.stdout, at)
        got.append(list(struct.unpack_from(f'<{n}I', out.stdout, at + 4)))
        at += 4 + 4 * n
    assert at == len(out.stdout)
    return got


def agree(hf, texts):
    theirs = [e.ids for e in hf.encode_batch(texts)]
    bad = [(t, o, h) for t, o, h in zip(texts, ours(texts), theirs) if o != h]
    assert not bad, [(t[:80], o[:20], h[:20]) for t, o, h in bad[:5]] + [f'{len(bad)} of {len(texts)} differ']


CASES = [
    '', ' ', '\n', '\r\n', "'", 'Hello, world!', 'a  b   c\t\td\n\ne \r\n\r\nf', '  \n  x', 'x  ', 'x \n',
    "I'm, you're, he's, we've, they'll, I'd, it'S, IT'LL, We'Ve, 'ſ, 'ſt, 'Ss, ''s, 'x, '1, ' 's",
    # After a letter a contraction is its own piece: "x'Some" is "x" "'S" "ome", whose ids differ from "'Some"'s.
    # (After a space, "'" goes with the space.)
    "x'some x'Some x'tome x'Tome x'mome x'Mome x'dome x'Dome x'reome x'REome x'veome x'VEome x'llome x'LLome x'Llome",
    'ﬅ ß İ ı K Ω Å', '12 123 1234 12345 1,234.5 ١٢٣ ௧௨ ⅠⅡⅢ ½ ² 𝟙𝟚𝟛', '数字和汉字 日本語のテキスト 한국어',
    'naïve café résumé — “quotes” ‘single’ …', '😀👍🏽 👨‍👩‍👧 🇩🇪', 'é ä ‍ ​ ﻿',
    'a b c　d e f\u0085g\u000bh\u000ci j k',
    '<|user|>hi<|assistant|>', '<|user|<|user|>', '<|endoftext|><|endoftext|>', '<think>why</think>\n/nothink',
    '[gMASK]<sop>x<eop>', '< |user|>', '<|user|> <|system|>\n', '<tool_call>{"a": 1}</tool_call>',
    'x' * 5000, 'ab' * 3000, ' ' * 1000 + 'x', '\n' * 50, '!' * 2000, '1' * 100, 'é' * 2000,
    'def f(x):\n    return x ** 2  # square\n', '<div class="a">&nbsp;</div>', 'http://example.com/a?b=c&d=e',
    'Ünïcödé ŞĞÇÖ ĀĒĪŌŪ', 'Привет, мир! Ελληνικά עברית العربية हिन्दी ไทย',
]


def test_the_cases_agree(hf):
    agree(hf, CASES)


def test_every_code_point_agrees(hf):
    chars = [chr(c) for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    texts = [f"a{c}b {c}{c}'{c} 1{c} {c}\n{c} x'{c}ome" for c in chars]
    agree(hf, texts)


def test_random_text_agrees(hf):
    rng = random.Random(68)
    pools = [
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789', " \t\n\r\x0b\x0c", "'sStTmMdDrReEvVlLſ",
        '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', 'éüñçßøåœ', 'ΑΩαωЖжא', '中文日本語한국', '١٢٣௧ⅠⅡ½²𝟙', '😀👍🏽🇩🇪‍́',
        '  　  \u0085 ', '\x00\x01\x7f',
    ]
    added = ['<|user|>', '<|assistant|>', '<think>', '</think>', '/nothink', '[gMASK]', '<|endoftext|>', '<|user', '|>']
    texts = []
    for _ in range(int(os.environ.get('FRONT_TOKENIZER_FUZZ', '3000'))):
        parts = []
        for _ in range(rng.randint(1, 30)):
            if rng.random() < 0.08:
                parts.append(rng.choice(added))
            else:
                pool = rng.choice(pools)
                parts.append(''.join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
        texts.append(''.join(parts))
    agree(hf, texts)


def test_gsm8k_questions_agree(hf):
    path = os.environ.get('GSM8K_JSONL')
    if not path:
        pytest.skip('GSM8K_JSONL names GSM8K test.jsonl')
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    agree(hf, [r['question'] for r in rows] + [r['answer'] for r in rows])


def test_another_tokenizer_is_refused(tmp_path):
    t = json.loads(Path(TOKENIZER).read_text())
    for change in ({'normalizer': {'type': 'NFC'}}, {'model': {**t['model'], 'ignore_merges': False}},
                   {'pre_tokenizer': {**t['pre_tokenizer'], 'pretokenizers': t['pre_tokenizer']['pretokenizers'][:1]}}):
        path = tmp_path / 'tokenizer.json'
        path.write_text(json.dumps({**t, **change}))
        out = subprocess.run([str(CHECK), str(path)], input=b'', capture_output=True, timeout=300, env=ENV)
        assert out.returncode == 4 and out.stderr.startswith(b'tokenizer.json: '), (change.keys(), out)


class Glm:
    """GLM's vocabulary; each step's logits peak at a token that depends on the context."""
    config = {'vocab_size': 154856, 'max_position_embeddings': 1 << 20, 'eos_token_id': []}

    def new_cache(self):
        return []

    def forward(self, t, c):
        import numpy as np
        c.append(t)
        out = np.zeros(self.config['vocab_size'], dtype=np.float32)
        out[(sum(c) + 1) % 5000] = 10
        return out


def test_the_engine_gets_token_ids_and_no_text(hf, monkeypatch):
    import http.client
    from front_server import FRONT, FrontServer
    from glm_vx import link
    from glm_vx.scheduler import Scheduler
    if not FRONT.exists():
        pytest.skip('needs front/build.sh')
    kinds, prompts = [], []
    parse = link.parse_request
    monkeypatch.setattr(link, 'parse_request', lambda payload: (kinds.append(payload[0]), parse(payload))[1])
    sched = Scheduler(Glm())
    submit = sched.submit
    sched.submit = lambda prompt, *a, **k: (prompts.append(list(prompt)), submit(prompt, *a, **k))[1]
    server = FrontServer(('127.0.0.1', 0), sched, hf, tokenizer_path=TOKENIZER)
    try:
        texts = ["Hello, world! It's <|user|> naïve 😀", 'x' * 3000, '数字 12345\n\n']
        for text in texts:
            c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=120)
            c.request('POST', '/v1/completions', json.dumps({'prompt': text, 'max_tokens': 2}))
            r = c.getresponse()
            body = json.loads(r.read())
            c.close()
            assert r.status == 200, body
            assert body['usage']['prompt_tokens'] == len(hf.encode(text).ids)
        assert prompts == [hf.encode(t).ids for t in texts]
        assert kinds.count(link.SUBMIT) == len(texts) and link.SUBMIT_TEXT not in kinds
    finally:
        server.server_close(); sched.close()
