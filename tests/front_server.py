"""The Vx front end (front/build/glm-vx-front) behind glm_vx.server.Server's interface, so that the
HTTP tests run against both servers (#68, step 2)."""
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

FRONT = Path(__file__).resolve().parents[1] / 'front' / 'build' / 'glm-vx-front'


def free_port(host):
    s = socket.socket()
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FrontServer:
    """server_port, server_address, model_name, serve_forever(), shutdown() and server_close()."""

    def __init__(self, address, scheduler, tokenizer=None, model_name='glm-vx-tiny-random'):
        from glm_vx.link import FrontEnd
        from glm_vx.server import Handler
        host, port = address
        port = port or free_port(host)
        # FRONT_PRELOAD=/path/to/libclang_rt.asan-x86_64.so: the front end under AddressSanitizer.
        env = ({**os.environ, 'LD_PRELOAD': os.environ['FRONT_PRELOAD'], 'ASAN_OPTIONS': 'detect_leaks=0'}
               if os.environ.get('FRONT_PRELOAD') else None)
        self.front = FrontEnd(FRONT, host, port, scheduler, tokenizer, model_name,
                              heartbeat_seconds=Handler.heartbeat_seconds, env=env)
        self.server_port, self.server_address, self.model_name = port, (host, port), model_name
        self._stop = threading.Event()
        deadline = time.monotonic() + 10
        while True:
            try:
                socket.create_connection(self.server_address, 0.2).close()
                return
            except OSError:
                if self.front.process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'the front end did not start: {self.front.process.poll()}')
                time.sleep(0.02)

    def serve_forever(self):
        self._stop.wait()

    def shutdown(self):
        self._stop.set()

    def server_close(self):
        alive = self.front.process.poll() is None
        code = self.front.process.poll()
        self.front.close()
        assert self.front.error is None, self.front.error
        assert alive, f'the front end died during the test with {code}'


def use(request, monkeypatch, module_name):
    """For the parameter 'front', the test module's `Server` is the front end."""
    if request.param == 'front':
        if not FRONT.exists():
            pytest.skip('needs front/build.sh')
        monkeypatch.setattr(sys.modules[module_name], 'Server', FrontServer)
    return request.param
