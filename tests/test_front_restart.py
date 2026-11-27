"""The engine restarts a front end that stops (#68, step 2).

A panic in the front end ends its process (Vx aborts, or exits nonzero), and so does a signal. The
engine then cancels the requests that front end carried and starts a new one on the same address.
A front end that keeps stopping is given up after `max_restarts` restarts within `window` seconds.
"""
import json
import signal
import socket
import threading
import time

import pytest

from front_server import FRONT, FrontServer, free_port
from glm_vx.link import FrontEnd, ProtocolError
from glm_vx.scheduler import Scheduler
from test_front_http import BODY, post_bytes, raw, status
from test_scheduler import Model


class Gated(Model):
    """The toy model, whose forward() waits for `gate`."""

    def __init__(self):
        self.gate = threading.Event()

    def forward(self, t, c):
        assert self.gate.wait(30)
        return super().forward(t, c)


def until(condition, seconds=10):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, condition
        time.sleep(0.02)


def listening(port):
    try:
        socket.create_connection(('127.0.0.1', port), 0.2).close()
        return True
    except OSError:
        return False


def script(tmp_path, body):
    """An executable that stands in for the front end."""
    path = tmp_path / 'front'
    path.write_text('#!/bin/sh\n' + body + '\n')
    path.chmod(0o755)
    return path


@pytest.mark.skipif(not FRONT.exists(), reason='needs front/build.sh')
@pytest.mark.parametrize('sig', [signal.SIGABRT, signal.SIGKILL])
def test_a_front_end_that_stops_is_restarted(sig):
    model = Gated()
    sched = Scheduler(model)
    server = FrontServer(('127.0.0.1', 0), sched)
    front, port = server.front, server.server_port
    try:
        got = {}
        client = threading.Thread(target=lambda: got.update(reply=raw(port, post_bytes(BODY), timeout=20)))
        client.start()
        until(lambda: front.link.requests)
        (req,) = front.link.requests.values()
        old = front.process
        old.send_signal(sig)  # what a panic does: Vx aborts
        client.join(20)
        assert got['reply'] == b''  # its connection ended with the process
        assert req.cancelled.wait(10)
        until(lambda: front.restarts and front.process is not old and listening(port))
        assert front.restarts[0][1] == -sig and front.error is None
        model.gate.set()
        until(lambda: sched.status()['reserved_tokens'] == 0)
        assert status(raw(port, post_bytes(BODY, b'Connection: close\r\n'))) == 200
        health = raw(port, b'GET /health HTTP/1.1\r\nConnection: close\r\n\r\n')
        assert json.loads(health.split(b'\r\n\r\n', 1)[1])['front_end_restarts'] == 1
    finally:
        model.gate.set()
        front.close(); sched.close()
    assert front.process.poll() is not None


def test_a_front_end_that_keeps_stopping_is_given_up(tmp_path):
    sched = Scheduler(Model())
    front = FrontEnd(script(tmp_path, 'exit 3'), '127.0.0.1', free_port('127.0.0.1'), sched,
                     max_restarts=3, window=60)
    try:
        assert front.done.wait(20)
        assert front.wait() == 3 and len(front.restarts) == 3
        assert 'stopped 4 times' in str(front.error)
    finally:
        front.close(); sched.close()


def test_a_malformed_frame_stops_the_front_end(tmp_path):
    # Length 1, an unknown kind; then it would wait forever, so the engine must kill it.
    front_end = script(tmp_path, "printf '\\001\\000\\000\\000\\167'; exec sleep 60")
    sched = Scheduler(Model())
    front = FrontEnd(front_end, '127.0.0.1', free_port('127.0.0.1'), sched, max_restarts=1, window=60)
    try:
        assert front.done.wait(20)
        assert front.wait() == -signal.SIGKILL
        assert len(front.restarts) == 1 and isinstance(front.restarts[0][2], ProtocolError)
        assert isinstance(front.error, ProtocolError)
    finally:
        front.close(); sched.close()


@pytest.mark.parametrize('delay', [0, 0.03, 0.1, 0.3])
def test_close_stops_the_restarts(tmp_path, delay):
    sched = Scheduler(Model())
    front = FrontEnd(script(tmp_path, 'exec sleep 0.05'), '127.0.0.1', free_port('127.0.0.1'), sched,
                     max_restarts=1000, window=60)
    try:
        time.sleep(delay)
        front.close()
        assert front.done.is_set() and front.process.poll() is not None
        process = front.process
        time.sleep(0.3)
        assert front.process is process  # nothing started after close()
    finally:
        front.close(); sched.close()
