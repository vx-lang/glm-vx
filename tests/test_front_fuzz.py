"""Random HTTP bytes at the front end (#68, step 5).

Each case is a mutated request, or several in one write, sent on its own connection; eight run
at a time. When the front end is quiet for FRONT_FUZZ_QUIET seconds, which is how it waits for
the rest of a request or the next one, the client half closes. Both servers take a half close
before the reply for a client that has gone, so the wait comes first. The front end must:
- answer with well-formed HTTP/1.1 replies, or close without a reply, within FRONT_FUZZ_SECONDS;
- never stop. A panic ends the process, the engine restarts it, and FrontServer.server_close fails
  on any restart. FRONT_PRELOAD runs it under AddressSanitizer (see tests/front_server.py).

The number of cases is FRONT_HTTP_FUZZ (default 300).
"""
import json
import os
import random
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from front_server import FRONT, FrontServer
from glm_vx.scheduler import Scheduler
from test_scheduler import Model

pytestmark = pytest.mark.skipif(not FRONT.exists(), reason='needs front/build.sh')

SECONDS = float(os.environ.get('FRONT_FUZZ_SECONDS', '20'))
QUIET = float(os.environ.get('FRONT_FUZZ_QUIET', '0.25'))


def post(path, body, extra=b''):
    body = json.dumps(body).encode() if not isinstance(body, bytes) else body
    return (b'POST ' + path + b' HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: '
            + str(len(body)).encode() + b'\r\n' + extra + b'\r\n' + body)


SEEDS = [
    post(b'/v1/completions', {'prompt': [1, 2, 3], 'max_tokens': 4}),
    post(b'/v1/completions', {'prompt': [5], 'max_tokens': 3, 'stream': True}),
    post(b'/v1/completions', {'prompt': [1, 2], 'max_tokens': 2, 'stream': True,
                              'stream_options': {'include_usage': True}, 'temperature': 0.5, 'seed': 7}),
    post(b'/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 2}),
    post(b'/generate', {'input_ids': [1, 2], 'sampling_params': {'max_new_tokens': 2}}),
    post(b'/v1/completions', {'prompt': [1], 'max_tokens': 2}, b'Connection: close\r\n'),
    b'GET /health HTTP/1.1\r\nHost: x\r\n\r\n',
    b'GET /v1/models HTTP/1.1\r\n\r\n',
    b'GET /health HTTP/1.0\r\nConnection: keep-alive\r\n\r\n',
]
ALPHABET = [b'\r\n', b'\r', b'\n', b':', b' ', b'\t', b'0', b'9', b'-', b'+', b'"', b'{', b'}', b'[', b']', b',',
            b'\x00', b'\xff', b'\xc3', b'a', b'Content-Length: 5\r\n', b'Transfer-Encoding: chunked\r\n',
            b'Connection: close\r\n', b'99999999999', b'HTTP/1.1', b'/', b'\r\n\r\n']


def mutate(rng):
    data = bytearray(b''.join(rng.choice(SEEDS) for _ in range(rng.choice([1, 1, 1, 2, 3]))))
    for _ in range(rng.randint(0, 6)):
        op, at = rng.random(), rng.randrange(len(data) + 1)
        if op < 0.35 and data:
            del data[at:at + rng.randint(1, 8)]
        elif op < 0.75:
            data[at:at] = rng.choice(ALPHABET)
        elif op < 0.85:
            data[at:at] = bytes(rng.randrange(256) for _ in range(rng.randint(1, 16)))
        elif op < 0.95:
            del data[at:]  # cut short
        else:
            data[at:at] = rng.choice(SEEDS)
    return bytes(data)


STATUS = re.compile(rb'HTTP/1\.1 [1-5][0-9][0-9] [ -~]*\r\n')


def check_replies(got):
    """Each reply: a status line, header lines, and a body by Content-Length or to the close."""
    while got:
        m = STATUS.match(got)
        assert m, got[:200]
        head_end = got.find(b'\r\n\r\n', m.end() - 2)
        assert head_end >= 0, got[:400]
        headers = got[m.end():head_end].split(b'\r\n') if head_end > m.end() - 2 else []
        length = None
        for line in headers:
            if not line:
                continue
            name, sep, value = line.partition(b':')
            assert sep and re.fullmatch(rb'[A-Za-z0-9-]+', name), line
            if name.lower() == b'content-length':
                length = int(value)
        body_at = head_end + 4
        if length is None:  # a stream: SSE to the close
            return
        got = got[body_at + length:]


def exchange(port, data):
    """The bytes the front end sent before it closed."""
    s = socket.create_connection(('127.0.0.1', port), 5)
    deadline, half, got = time.monotonic() + SECONDS, False, b''
    try:
        try:
            s.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            return got
        while True:
            left = deadline - time.monotonic()
            assert left > 0, ('no close', data, got[:300])
            s.settimeout(left if half else min(QUIET, left))
            try:
                part = s.recv(65536)
            except socket.timeout:
                assert not half, ('a hang', data, got[:300])
                s.shutdown(socket.SHUT_WR)
                half = True
                continue
            except ConnectionResetError:
                return got
            if not part:
                return got
            got += part
    finally:
        s.close()


def test_random_requests_get_well_formed_replies_and_the_front_end_stays_up():
    sched = Scheduler(Model())
    server = FrontServer(('127.0.0.1', 0), sched)
    rng = random.Random(int(os.environ.get('FRONT_FUZZ_SEED', '68')))
    cases = [mutate(rng) for _ in range(int(os.environ.get('FRONT_HTTP_FUZZ', '300')))]
    try:
        with ThreadPoolExecutor(8) as pool:
            for start in range(0, len(cases), 8):
                batch = cases[start:start + 8]
                for data, got in zip(batch, pool.map(lambda d: exchange(server.server_port, d), batch)):
                    try:
                        check_replies(got)
                    except AssertionError as exc:
                        raise AssertionError((data, got[:500])) from exc
                assert not server.front.restarts, batch
        health = exchange(server.server_port, b'GET /health HTTP/1.1\r\nConnection: close\r\n\r\n')
        assert health.startswith(b'HTTP/1.1 200 ')
    finally:
        server.server_close(); sched.close()
