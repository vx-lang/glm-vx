"""The front end's JSON parser (front/src/json.vx) against Python's json.loads (#68, step 2).

front/build/json_echo parses its standard input and writes the tree back as JSON, or `N!` with the
offset where the input stopped being JSON. Where both accept, the values must agree. Where Python
rejects, the front end must too. The front end rejects more, on purpose:
- NaN, Infinity and -Infinity, which are not JSON;
- a lone surrogate escape such as "\\ud800", since its strings are UTF-8;
- arrays and objects nested past 32.
"""
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

ECHO = Path(__file__).resolve().parents[1] / 'front' / 'build' / 'json_echo'
# FRONT_PRELOAD=/path/to/libclang_rt.asan-x86_64.so runs the binary under AddressSanitizer, which stops it
# (a nonzero exit) at a double free, a use after free or a read past a buffer.
ENV = {**os.environ, **({'LD_PRELOAD': os.environ['FRONT_PRELOAD'], 'ASAN_OPTIONS': 'detect_leaks=0'}
                        if os.environ.get('FRONT_PRELOAD') else {})}
pytestmark = pytest.mark.skipif(not ECHO.exists(), reason='needs front/build.sh')


def echo(data):
    out = subprocess.run([str(ECHO)], input=data, capture_output=True, timeout=30, env=ENV)
    assert out.returncode == 0, out
    if out.stdout.endswith(b'!'):
        return None
    return out.stdout


REJECTED = object()


def python(data):
    try:
        return json.loads(data)
    except (ValueError, RecursionError):
        return REJECTED


def stricter(data):
    """True when the front end rejects `data` on purpose although Python accepts it."""
    text = data.decode('utf-8', errors='replace')
    if any(w in text for w in ('NaN', 'Infinity')):
        return True
    def deep(v, d=0):
        if isinstance(v, (list, dict)):
            if d + 1 > 32:
                return True
            return any(deep(x, d + 1) for x in (v.values() if isinstance(v, dict) else v))
        return False
    def surrogate(v):
        if isinstance(v, str):
            return any(0xD800 <= ord(c) <= 0xDFFF for c in v)
        if isinstance(v, dict):
            return any(surrogate(k) or surrogate(x) for k, x in v.items())
        if isinstance(v, list):
            return any(surrogate(x) for x in v)
        return False
    value = python(data)
    return deep(value) or surrogate(value)


VALID = [
    b'{}', b'[]', b'0', b'-0', b'1.5e-3', b'1E+2', b'"x"', b'true', b'false', b'null', b' \t\n\r[1 , 2]\n',
    b'{"prompt":[1,2,3],"max_tokens":16,"temperature":0.7,"stream":true,"stream_options":{"include_usage":true}}',
    b'{"a":{"b":{"c":[[],[{}]]}}}', b'"\\"\\\\\\/\\b\\f\\n\\r\\t"', b'"\\u0000\\u001f\\u00e9\\u20ac\\ud83d\\ude00"',
    '"naïve ☃ 😀"'.encode(), b'{"k":1,"k":2}', b'[' * 32 + b']' * 32, b'123456789012345678901234567890',
    b'{"prompt":"hello world","model":"glm"}',
]
INVALID = [
    b'', b' ', b'[1,]', b'{"a":1,}', b'01', b'1.', b'.5', b'+1', b'-', b'1e', b'1e+', b'[1 2]', b'{"a" 1}',
    b'{1:2}', b"'x'", b'"\x01"', b'"abc', b'"\\x"', b'"\\u12"', b'tru', b'nul', b'[1]x', b'{"a":1}}',
    b'\xff', b'"\xc3"', b'"\xed\xa0\x80"', b'[' * 40 + b']' * 40, b'NaN', b'Infinity', b'"\\ud800"', b'"\\udc00x"',
]


@pytest.mark.parametrize('data', VALID)
def test_valid_json_agrees(data):
    got = echo(data)
    assert got is not None, data
    assert json.loads(got) == json.loads(data)


@pytest.mark.parametrize('data', INVALID)
def test_invalid_json_is_rejected(data):
    assert echo(data) is None, data


def test_fuzzed_bodies_agree_with_python():
    rng = random.Random(68)
    seeds = [d for d in VALID if len(d) > 2]
    alphabet = b'{}[]",:0123456789-+.eE \\untrfalsx\x00\xc3\xa9'
    disagree = []
    for case in range(int(os.environ.get('FRONT_JSON_FUZZ', '600'))):
        data = bytearray(rng.choice(seeds))
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            at = rng.randrange(len(data) + 1)
            if op < 0.4 and data:
                del data[min(at, len(data) - 1)]
            elif op < 0.8:
                data.insert(at, rng.choice(alphabet))
            else:
                data[at:at] = rng.choice(seeds)
        data = bytes(data)
        ours, theirs = echo(data), python(data)
        if ours is not None:
            if theirs is REJECTED or json.loads(ours) != theirs:
                disagree.append((data, ours, theirs))
        elif theirs is not REJECTED and not stricter(data):
            disagree.append((data, ours, theirs))
    assert not disagree, disagree[:5]
