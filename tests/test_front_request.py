"""The front end's /v1/completions checks (front/src/request.vx) against glm_vx/server.py's (#68, step 2).

front/build/request_check reads a body and writes the submission as JSON, or `400 message`. The
reference is the HTTP server's own code: `Handler._options`, then `Scheduler.submit`'s checks (on a
closed scheduler, so that a request that passes them gets BusyError and runs nothing). The engine
checks the vocabulary and the context length, so the reference has room for any u32 id. The front
end rejects more, on purpose: a max_tokens past 2**32 - 1 and a seed past 2**64 - 1, which do not fit
the engine link's fields (any real context is shorter, and no client sends such a seed).
"""
import json
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_vx.scheduler import BusyError, Scheduler
from glm_vx.server import Handler

CHECK = Path(__file__).resolve().parents[1] / 'front' / 'build' / 'request_check'
# FRONT_PRELOAD=/path/to/libclang_rt.asan-x86_64.so runs the binary under AddressSanitizer, which stops it
# (a nonzero exit) at a double free, a use after free or a read past a buffer.
ENV = {**os.environ, **({'LD_PRELOAD': os.environ['FRONT_PRELOAD'], 'ASAN_OPTIONS': 'detect_leaks=0'}
                        if os.environ.get('FRONT_PRELOAD') else {})}
pytestmark = pytest.mark.skipif(not CHECK.exists(), reason='needs front/build.sh')


class Tok:
    def encode(self, text):
        return SimpleNamespace(ids=[1])


class Wide:
    config = {'vocab_size': 2**32, 'max_position_embeddings': 2**40, 'eos_token_id': []}


CLOSED = Scheduler(Wide())
CLOSED.close()


def front(data):
    out = subprocess.run([str(CHECK)], input=data, capture_output=True, timeout=30, env=ENV)
    assert out.returncode == 0, out
    if out.stdout.startswith(b'400 '):
        return ('400', out.stdout[4:].decode())
    got = json.loads(out.stdout)
    got['temperature'] = float(got['temperature'])  # Vx writes 1e308 as its 309 digits
    return got


def reference(data):
    try:
        body = json.loads(data)
        if not isinstance(body, dict):
            raise ValueError('JSON object required')
        fake = SimpleNamespace(server=SimpleNamespace(model_name='glm-test', tokenizer=Tok()))
        prompt, submit, opts = Handler._options(fake, body)
        raw_prompt = body.get('prompt')
        try:
            CLOSED.submit(prompt, **submit)
        except BusyError:
            pass
    except (ValueError, TypeError, OverflowError) as exc:
        return ('400', str(exc))
    is_text = isinstance(raw_prompt, str)
    return dict(ids=[] if is_text else prompt, text=raw_prompt if is_text else '', is_text=is_text,
                max_tokens=submit['max_tokens'], temperature=float(submit['temperature']), seed=submit['seed'],
                ignore_eos=submit['ignore_eos'], stream=opts['stream'], openai=opts['openai'],
                include_usage=opts['include_usage'])


def stricter(data):
    body = json.loads(data)
    return isinstance(body, dict) and (
        isinstance(body.get('max_tokens'), int) and body['max_tokens'] > 2**32 - 1
        or isinstance(body.get('seed'), int) and body['seed'] > 2**64 - 1)


def agree(data):
    ours, theirs = front(data), reference(data)
    if isinstance(theirs, tuple):
        return isinstance(ours, tuple)
    return ours == theirs or isinstance(ours, tuple) and stricter(data)


CASES = [
    {'prompt': [1, 2], 'max_tokens': 5},
    {'prompt': [1], 'temperature': 0.7, 'seed': 3, 'ignore_eos': True},
    {'prompt': 'héllo', 'model': 'glm-test', 'stream': True, 'stream_options': {'include_usage': True}},
    {'prompt': [1], 'n': 1, 'best_of': None, 'stream_options': None},
    {'prompt': [1], 'temperature': 0, 'seed': -0, 'max_tokens': 1},
    {'prompt': [4294967295], 'seed': 2**64 - 1, 'max_tokens': 2**32 - 1},
    {'prompt': [1], 'temperature': -0.0}, {'prompt': [1], 'temperature': 1e308},
]
BAD = [
    [], 'x', 1, None, {}, {'prompt': []}, {'prompt': [1.0]}, {'prompt': [True]}, {'prompt': [-1]},
    {'prompt': [4294967296]}, {'prompt': {'a': 1}}, {'prompt': [1], 'stream': 'yes'}, {'prompt': [1], 'stop': 'x'},
    {'prompt': [1], 'max_tokens': True}, {'prompt': [1], 'max_tokens': 0}, {'prompt': [1], 'max_tokens': 1.0},
    {'prompt': [1], 'max_tokens': None}, {'prompt': [1], 'temperature': -1}, {'prompt': [1], 'temperature': True},
    {'prompt': [1], 'temperature': None}, {'prompt': [1], 'seed': -1}, {'prompt': [1], 'seed': 1.5},
    {'prompt': [1], 'n': 2}, {'prompt': [1], 'best_of': 1.0}, {'prompt': [1], 'model': 'other'},
    {'prompt': [1], 'model': 5}, {'prompt': [1], 'stream_options': {'include_usage': True}},
    {'prompt': [1], 'stream': True, 'stream_options': {'x': 1}}, {'prompt': [1], 'stream': True, 'stream_options': 1},
    {'prompt': [1], 'ignore_eos': 1}, {'max_tokens': 3}, {'prompt': [1], 'zz': 1, 'aa': 2},
]


@pytest.mark.parametrize('body', CASES)
def test_accepted_bodies_give_the_same_submission(body):
    data = json.dumps(body).encode()
    assert front(data) == reference(data)


@pytest.mark.parametrize('body', BAD)
def test_rejected_bodies_are_400_with_the_same_message(body):
    data = json.dumps(body).encode()
    ours, theirs = front(data), reference(data)
    assert isinstance(theirs, tuple) and isinstance(ours, tuple), (ours, theirs)
    if body != [] and not isinstance(body, (str, int)) and body is not None:
        assert ours[1] == theirs[1]


def test_big_numbers_and_inf_text():
    for text in [b'{"prompt":[1],"temperature":1e400}', b'{"prompt":[1],"temperature":' + b'9' * 400 + b'}',
                 b'{"prompt":[1],"max_tokens":' + b'9' * 30 + b'}', b'{"prompt":[1],"seed":' + b'9' * 30 + b'}']:
        assert agree(text), text


def test_fuzzed_bodies_agree():
    rng = random.Random(4134)
    keys = ['prompt', 'max_tokens', 'temperature', 'seed', 'stream', 'ignore_eos', 'stream_options', 'n', 'best_of',
            'model', 'stop', 'top_p']
    values = [None, True, False, 0, 1, -1, 2, 16, 2**31, 2**33, 2**65, 0.0, -0.0, 0.5, 1.0, 2.5e2, -3.5, 'glm-test',
              'x', '', [], [0], [1, 2, 3], [1.5], [-1], [True], {}, {'include_usage': True}, {'include_usage': 1}]
    bad = []
    for _ in range(400):
        body = {k: rng.choice(values) for k in rng.sample(keys, rng.randint(0, 6))}
        if rng.random() < 0.7:
            body['prompt'] = rng.choice([[1, 2], [5], 'text', [0, 4294967295]])
        data = json.dumps(body).encode()
        if not agree(data):
            bad.append((body, front(data), reference(data)))
    assert not bad, bad[:5]
