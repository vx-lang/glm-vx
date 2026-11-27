"""Each `lit("...", n)` in the front end gives the literal's exact length (#68, step 2).

`String::from_bytes(literal, n)` is safe code: the prover checks that the literal has room for `n`
bytes, so a length too long is an error (E8006; since Vx #4152 also n + 1, the literal's NUL). A
length too short builds and cuts the text. That is not a memory bug, so this test checks the lengths.
"""
import re
from pathlib import Path

FRONT = Path(__file__).resolve().parents[1] / 'front'
LIT = re.compile(r'lit\(\s*"((?:[^"\\]|\\.)*)"\s+as\s+\*const\s+u8\s*,\s*(\d+)\s*\)', re.S)
ESCAPES = {'n': b'\n', 'r': b'\r', 't': b'\t', '\\': b'\\', '"': b'"', "'": b"'", '0': b'\0'}


def literal_bytes(text):
    out, i = bytearray(), 0
    while i < len(text):
        if text[i] == '\\':
            out += ESCAPES[text[i + 1]]
            i += 2
        else:
            out += text[i].encode()
            i += 1
    return bytes(out)


def test_every_literal_length_is_exact():
    wrong, count = [], 0
    for path in sorted(FRONT.rglob('*.vx')):
        source = path.read_text()
        for m in LIT.finditer(source):
            count += 1
            n = len(literal_bytes(m.group(1)))
            if n != int(m.group(2)):
                line = source.count('\n', 0, m.start()) + 1
                wrong.append(f'{path.relative_to(FRONT)}:{line}: "{m.group(1)}" has {n} bytes, not {m.group(2)}')
    assert count > 50
    assert not wrong, '\n'.join(wrong)


def test_no_literal_reaches_from_bytes_but_through_lit():
    for path in sorted(FRONT.rglob('*.vx')):
        if path.name == 'bytes.vx':
            continue
        assert 'from_bytes(' not in path.read_text(), path
