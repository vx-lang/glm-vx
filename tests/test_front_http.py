"""The front end's HTTP limits, keep-alive and lifecycle (#68, step 2).

The cases that both servers share are in test_server.py and test_openai_compat.py. These are the front
end's own: its limits are stricter than http.server's, and it is a separate process.
"""
import json
import os
import socket
import subprocess
import threading
import time

import pytest

from front_server import FRONT, FrontServer, free_port
from glm_vx.scheduler import Scheduler
from test_scheduler import Model

pytestmark = pytest.mark.skipif(not FRONT.exists(), reason='needs front/build.sh')


@pytest.fixture
def front():
    sched = Scheduler(Model())
    server = FrontServer(('127.0.0.1', 0), sched)
    yield server
    server.server_close(); sched.close()


def raw(port, data, timeout=5):
    """Send `data`, read until the server closes or `timeout`; the bytes it sent."""
    s = socket.create_connection(('127.0.0.1', port), timeout)
    try:
        s.sendall(data)
    except (BrokenPipeError, ConnectionResetError):
        pass
    got = b''
    try:
        while True:
            part = s.recv(65536)
            if not part:
                break
            got += part
    except (socket.timeout, ConnectionResetError):  # a reset: the server closed with our bytes unread
        pass
    s.close()
    return got


def post_bytes(body, extra=b'', version=b'HTTP/1.1'):
    return (b'POST /v1/completions ' + version + b'\r\nHost: x\r\nContent-Length: ' + str(len(body)).encode()
            + b'\r\n' + extra + b'\r\n' + body)


BODY = json.dumps({'prompt': [1, 2], 'max_tokens': 3}).encode()


def status(reply):
    return int(reply.split(b' ', 2)[1]) if reply.startswith(b'HTTP/1.1 ') else None


def test_header_limits(front):
    p = front.server_port
    assert status(raw(p, post_bytes(BODY, b'X-Pad: ' + b'a' * 17000 + b'\r\n'))) == 431
    assert status(raw(p, post_bytes(BODY, b''.join(b'X-%d: 1\r\n' % i for i in range(70))))) == 431
    assert status(raw(p, post_bytes(BODY, b''.join(b'X-%d: 1\r\n' % i for i in range(60))))) == 200


def test_bodies_are_read_one_way_only(front):
    p = front.server_port
    assert status(raw(p, post_bytes(BODY, b'Transfer-Encoding: chunked\r\n'))) == 400
    assert status(raw(p, post_bytes(BODY, b'Content-Length: 5\r\n'))) == 400  # two lengths that differ
    assert status(raw(p, post_bytes(BODY, b'Content-Length: ' + str(len(BODY)).encode() + b'\r\n'))) == 200
    assert status(raw(p, post_bytes(BODY).replace(b'Content-Length:', b'Content-Length :'))) == 400
    assert status(raw(p, post_bytes(BODY).replace(b'Content-Length: ', b'Content-Length: +'))) == 400
    big = b'POST /v1/completions HTTP/1.1\r\nContent-Length: 1048577\r\n\r\n'
    assert status(raw(p, big)) == 400
    assert status(raw(p, b'POST /v1/completions HTTP/1.1\r\nContent-Length: 0\r\n\r\n')) == 400


def test_request_lines(front):
    p = front.server_port
    assert status(raw(p, post_bytes(BODY, version=b'HTTP/2.0'))) == 505
    assert status(raw(p, b'GET  /health HTTP/1.1\r\n\r\n')) == 400
    assert status(raw(p, b'GET /health HTTP/1.1 x\r\n\r\n')) == 400
    assert status(raw(p, b'DELETE /health HTTP/1.1\r\n\r\n')) == 501
    assert status(raw(p, b'GET /nope HTTP/1.1\r\nConnection: close\r\n\r\n')) == 404
    assert raw(p, b'') == b''


def test_keep_alive_and_pipelining(front):
    p = front.server_port
    one = post_bytes(BODY)
    health = b'GET /health HTTP/1.1\r\n\r\n'
    # Three requests in one write; the third closes the connection.
    got = raw(p, one + health + post_bytes(BODY, b'Connection: close\r\n'))
    assert got.count(b'HTTP/1.1 200 OK') == 3, got
    # HTTP/1.0 closes after one reply unless it asks to keep the connection.
    assert raw(p, post_bytes(BODY, version=b'HTTP/1.0') + one).count(b'HTTP/1.1 200') == 1
    assert raw(p, post_bytes(BODY, b'Connection: keep-alive\r\n', b'HTTP/1.0') + post_bytes(BODY, b'Connection: close\r\n')
               ).count(b'HTTP/1.1 200') == 2


def test_at_most_32_connections(front):
    p = front.server_port
    idle = [socket.create_connection(('127.0.0.1', p), 5) for _ in range(32)]
    try:
        time.sleep(0.3)
        # The front end accepts in arrival order, so it reaches the 33rd after the 32 and closes it.
        assert raw(p, post_bytes(BODY), timeout=2) == b''
        for s in idle[:4]:
            s.close()
        # A slot is free once the front end has seen a close; on a busy host that can take a while.
        deadline = time.monotonic() + 30
        while (got := raw(p, post_bytes(BODY, b'Connection: close\r\n'))) == b'' and time.monotonic() < deadline:
            time.sleep(0.1)
        assert status(got) == 200
    finally:
        for s in idle:
            s.close()


def test_the_front_end_exits_when_the_engine_closes_the_link():
    port = free_port('127.0.0.1')
    proc = subprocess.Popen([str(FRONT), f'127.0.0.1:{port}', 'm'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(('127.0.0.1', port), 0.2).close()
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.02)
        # A request waits for the engine, which reads its frame and then goes away.
        got = {}
        t = threading.Thread(target=lambda: got.update(reply=raw(port, post_bytes(BODY), timeout=10)))
        t.start()
        head = proc.stdout.read(4)
        assert len(head) == 4
        proc.stdin.close()
        t.join(15)
        assert status(got['reply']) == 500 and b'the engine is gone' in got['reply']
        assert proc.wait(10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_front_end_stops_at_a_malformed_engine_frame():
    port = free_port('127.0.0.1')
    proc = subprocess.Popen([str(FRONT), f'127.0.0.1:{port}', 'm'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        proc.stdin.write(b'\x05\x00\x00\x00\x99abcd')  # an unknown kind
        proc.stdin.flush()
        proc.wait(10)  # the reader stops; the router sees it end and so does the front end
    finally:
        if proc.poll() is None:
            proc.kill()
