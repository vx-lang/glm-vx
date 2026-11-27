"""The engine's side of the engine link (#68, step 1; docs/ENGINE-LINK.md).

The front end sends fixed binary frames over its standard output, and reads the engine's frames
on its standard input. Each frame is parsed with `struct`, after a check of its length.
"""
import json
import queue
import struct
import sys
import threading
import time

from .detok import IncrementalText
from .scheduler import BusyError

MAX_PAYLOAD = 1 << 22

SUBMIT, SUBMIT_TEXT, CANCEL, STATUS = 0x01, 0x02, 0x03, 0x04
TOKEN, DONE, ERROR, STATUS_REPLY, ACCEPTED = 0x81, 0x82, 0x83, 0x84, 0x85

REASONS = {'length': 1, 'stop': 2, 'cancelled': 3}
BUSY, INVALID, FAILED = 1, 2, 3
IGNORE_EOS = 1
TIMINGS = ('prefill_seconds', 'decode_seconds', 'prefix_seconds', 'queued_seconds')

_LENGTH = struct.Struct('<I')
_SUBMIT = struct.Struct('<BQIdQBI')     # kind, id, max_tokens, temperature, seed, flags, count
_ID = struct.Struct('<BQ')              # kind, id: CANCEL and STATUS
_TOKEN = struct.Struct('<BQII')         # kind, id, token, text length
_DONE = struct.Struct('<BQBIII4dII')    # kind, id, reason, prompt, completion, cached, timings, tail, text
_ERROR = struct.Struct('<BQBI')         # kind, id, code, message length
_STATUS_REPLY = struct.Struct('<BQI')   # kind, id, JSON length
_ACCEPTED = struct.Struct('<BQQI')      # kind, id, created (Unix seconds), name length


class ProtocolError(Exception):
    """A frame that breaks docs/ENGINE-LINK.md's rules. It ends the link."""


def read_frame(stream):
    """The next payload, or None when the stream ends between frames."""
    head = _read_exact(stream, _LENGTH.size, at_start=True)
    if head is None:
        return None
    (n,) = _LENGTH.unpack(head)
    if not 1 <= n <= MAX_PAYLOAD:
        raise ProtocolError(f'payload length {n} is not in 1..{MAX_PAYLOAD}')
    return _read_exact(stream, n)


def _read_exact(stream, n, at_start=False):
    parts, have = [], 0
    while have < n:
        part = stream.read(n - have)
        if not part:
            if at_start and have == 0:
                return None
            raise ProtocolError(f'the stream ended inside a frame ({have} of {n} bytes)')
        parts.append(part)
        have += len(part)
    return b''.join(parts)


def frame(payload):
    if not 1 <= len(payload) <= MAX_PAYLOAD:
        raise ValueError(f'payload length {len(payload)} is not in 1..{MAX_PAYLOAD}')
    return _LENGTH.pack(len(payload)) + payload


def parse_request(payload):
    """A front end payload as (kind, id, fields). Raises ProtocolError for a malformed one."""
    kind = payload[0]
    if kind in (SUBMIT, SUBMIT_TEXT):
        if len(payload) < _SUBMIT.size:
            raise ProtocolError(f'a submit payload of {len(payload)} bytes is shorter than its fields')
        _, rid, max_tokens, temperature, seed, flags, count = _SUBMIT.unpack_from(payload)
        if flags & ~IGNORE_EOS:
            raise ProtocolError(f'unknown flag bits {flags:#x}')
        size = count * 4 if kind == SUBMIT else count
        if len(payload) != _SUBMIT.size + size:
            raise ProtocolError(f'a submit of {count} needs {_SUBMIT.size + size} bytes, not {len(payload)}')
        body = payload[_SUBMIT.size:]
        prompt = list(struct.unpack(f'<{count}I', body)) if kind == SUBMIT else body
        return kind, rid, dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature, seed=seed,
                               ignore_eos=bool(flags & IGNORE_EOS))
    if kind in (CANCEL, STATUS):
        if len(payload) != _ID.size:
            raise ProtocolError(f'a {"cancel" if kind == CANCEL else "status"} payload is {_ID.size} bytes, '
                                f'not {len(payload)}')
        return kind, _ID.unpack(payload)[1], {}
    raise ProtocolError(f'unknown kind {kind:#x}')


def _utf8(text):
    return text.encode('utf-8', errors='replace')


def token_frame(rid, token, text=''):
    data = _utf8(text)
    return frame(_TOKEN.pack(TOKEN, rid, token, len(data)) + data)


def done_frame(rid, event, tail='', text=''):
    tail, text = _utf8(tail), _utf8(text)
    timings = [float(event['timings'][k]) for k in TIMINGS]
    return frame(_DONE.pack(DONE, rid, REASONS[event['reason']], event['prompt_tokens'], event['completion_tokens'],
                            event['cached_prompt_tokens'], *timings, len(tail), len(text)) + tail + text)


def error_frame(rid, code, message):
    data = _utf8(message)[:MAX_PAYLOAD - _ERROR.size]
    return frame(_ERROR.pack(ERROR, rid, code, len(data)) + data)


def status_frame(rid, data):
    return frame(_STATUS_REPLY.pack(STATUS_REPLY, rid, len(data)) + data)


def accepted_frame(rid, created, name):
    data = _utf8(name)
    return frame(_ACCEPTED.pack(ACCEPTED, rid, created, len(data)) + data)


def submit_frame(rid, prompt, max_tokens, temperature=0.0, seed=0, ignore_eos=False):
    """A front end's SUBMIT (token ids) or SUBMIT_TEXT (str), for tests and tools."""
    if isinstance(prompt, str):
        body, kind = prompt.encode('utf-8'), SUBMIT_TEXT
        count = len(body)
    else:
        body, kind, count = struct.pack(f'<{len(prompt)}I', *prompt), SUBMIT, len(prompt)
    return frame(_SUBMIT.pack(kind, rid, max_tokens, temperature, seed, IGNORE_EOS if ignore_eos else 0, count) + body)


def cancel_frame(rid):
    return frame(_ID.pack(CANCEL, rid))


def status_request_frame(rid):
    return frame(_ID.pack(STATUS, rid))


def parse_reply(payload):
    """An engine payload as a dict, for tests and tools (the front end's side is Vx)."""
    kind = payload[0]
    if kind == TOKEN:
        _, rid, token, n = _TOKEN.unpack_from(payload)
        return dict(kind='token', id=rid, token_id=token, text=payload[_TOKEN.size:_TOKEN.size + n].decode())
    if kind == DONE:
        fields = _DONE.unpack_from(payload)
        _, rid, reason, prompt, completion, cached = fields[:6]
        tail_n, text_n = fields[10:]
        rest = payload[_DONE.size:]
        return dict(kind='done', id=rid, reason={v: k for k, v in REASONS.items()}[reason], prompt_tokens=prompt,
                    completion_tokens=completion, cached_prompt_tokens=cached, timings=dict(zip(TIMINGS, fields[6:10])),
                    tail=rest[:tail_n].decode(), text=rest[tail_n:tail_n + text_n].decode())
    if kind == ERROR:
        _, rid, code, n = _ERROR.unpack_from(payload)
        return dict(kind='error', id=rid, code=code, message=payload[_ERROR.size:_ERROR.size + n].decode())
    if kind == STATUS_REPLY:
        _, rid, n = _STATUS_REPLY.unpack_from(payload)
        return dict(kind='status', id=rid, json=payload[_STATUS_REPLY.size:_STATUS_REPLY.size + n])
    if kind == ACCEPTED:
        _, rid, created, n = _ACCEPTED.unpack_from(payload)
        return dict(kind='accepted', id=rid, created=created, name=payload[_ACCEPTED.size:_ACCEPTED.size + n].decode())
    raise ProtocolError(f'unknown reply kind {kind:#x}')


class EngineLink:
    """Serve one front end: read its frames from `reader`, answer on `writer` (binary streams).

    run() returns when the reader ends, and raises ProtocolError for a malformed frame. Either way
    it cancels the link's requests first. One thread writes frames; one thread a request reads its
    scheduler events. The outgoing queue is bounded, so a front end that stops reading fills each
    request's event queue, and the scheduler then ends those requests (its `backpressure`).
    """

    def __init__(self, scheduler, reader, writer, tokenizer=None, status=None, max_frames=1024):
        self.scheduler, self.reader, self.writer, self.tokenizer = scheduler, reader, writer, tokenizer
        self.status = status or scheduler.status
        self.requests, self.lock = {}, threading.Lock()
        self.out = queue.Queue(maxsize=max_frames)
        self.closed = threading.Event()
        self.write_error = None

    def run(self):
        writer = threading.Thread(target=self._write, daemon=True)
        writer.start()
        try:
            while True:
                payload = read_frame(self.reader)
                if payload is None:
                    return
                kind, rid, fields = parse_request(payload)
                if kind in (SUBMIT, SUBMIT_TEXT):
                    self._submit(kind, rid, **fields)
                elif kind == CANCEL:
                    with self.lock:
                        req = self.requests.get(rid)
                    if req is not None:
                        self.scheduler.cancel(req)
                else:
                    try:
                        data = json.dumps(self.status(), allow_nan=False).encode()
                    except Exception as exc:  # a status that cannot be shown must not end the link
                        self._send(error_frame(rid, FAILED, f'status: {exc}'))
                    else:
                        self._send(status_frame(rid, data))
        finally:
            with self.lock:
                active = list(self.requests.values())
            for req in active:
                self.scheduler.cancel(req)
            self.closed.set()
            self._send(None)
            writer.join(5)

    def _send(self, data):
        while True:
            try:
                self.out.put(data, timeout=0.25)
                return
            except queue.Full:
                if self.closed.is_set() and data is not None:
                    return  # the link is ending; nobody will read it

    def _write(self):
        while True:
            data = self.out.get()
            if data is None:
                return
            if self.write_error is not None:
                continue
            try:
                self.writer.write(data)
                self.writer.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self.write_error = exc

    def _submit(self, kind, rid, prompt, max_tokens, temperature, seed, ignore_eos):
        with self.lock:
            if rid in self.requests:
                raise ProtocolError(f'request id {rid} is in use')
        try:
            if kind == SUBMIT_TEXT:
                if self.tokenizer is None:
                    raise ValueError('text needs --tokenizer; tiny demo accepts token IDs')
                prompt = self.tokenizer.encode(prompt.decode('utf-8')).ids
            req = self.scheduler.submit(prompt, max_tokens, temperature=temperature, seed=seed, ignore_eos=ignore_eos)
        except BusyError as exc:
            self._send(error_frame(rid, BUSY, str(exc)))
            return
        except (ValueError, TypeError, OverflowError, UnicodeDecodeError) as exc:
            self._send(error_frame(rid, INVALID, str(exc)))
            return
        with self.lock:
            self.requests[rid] = req
        self._send(accepted_frame(rid, int(time.time()), req.id))
        threading.Thread(target=self._pump, args=(rid, req), daemon=True).start()

    def _pump(self, rid, req):
        detok, ids = IncrementalText(self.tokenizer), []
        while True:
            event = req.events.get()
            if event['type'] == 'token':
                ids.append(event['token_id'])
                try:
                    self._send(token_frame(rid, ids[-1], detok.push(ids[-1])))
                except ValueError as exc:  # a frame past MAX_PAYLOAD
                    self.scheduler.cancel(req)
                    self._end(rid, error_frame(rid, FAILED, str(exc)))
                    self._drain(req)
                    return
                continue
            if event['reason'] in ('error', 'backpressure'):
                self._end(rid, error_frame(rid, FAILED, event['error'] or event['reason']))
                return
            text = self.tokenizer.decode(ids) if self.tokenizer else ''
            try:
                data = done_frame(rid, event, detok.flush(), text)
            except ValueError as exc:
                data = error_frame(rid, FAILED, str(exc))
            self._end(rid, data)
            return

    def _end(self, rid, data):
        # The id is free before the front end can read its last frame, so it may use the id again.
        with self.lock:
            self.requests.pop(rid, None)
        self._send(data)

    @staticmethod
    def _drain(req):
        while req.events.get()['type'] != 'done':
            pass


class FrontEnd:
    """Run the Vx front end (front/build/glm-vx-front) on `address` and serve its link.

    The front end listens; the engine has no listener. Its standard input and output are the
    link, so it needs no right to reach the engine. With `tokenizer_path` (a tokenizer.json) the
    front end tokenizes text prompts itself, and the engine gets only token ids; `tokenizer`, the
    engine's copy, then only detokenizes.

    A front end that stops when close() did not ask it to (a panic, a signal, a malformed frame)
    loses its connections, and the engine cancels the requests it carried. A new one then binds the
    same address. After `max_restarts` restarts within `window` seconds the engine gives up: wait()
    returns the last exit code and `error` says why. `restarts` lists (time, exit code, error).
    """

    def __init__(self, binary, host, port, scheduler, tokenizer=None, model_name='glm-vx-tiny-random',
                 heartbeat_seconds=None, env=None, max_restarts=5, window=60.0, tokenizer_path=None):
        self.health = lambda: {'status': 'ok', 'model': model_name,
                               'backend': getattr(getattr(scheduler.model, 'backend', None), 'name', None),
                               'front_end_restarts': len(self.restarts), **scheduler.status()}
        self.argv = [str(binary), f'{host}:{port}', model_name]
        if heartbeat_seconds is not None or tokenizer_path is not None:
            self.argv.append(str(max(1, round((10.0 if heartbeat_seconds is None else heartbeat_seconds) * 1000))))
        if tokenizer_path is not None:
            self.argv.append(str(tokenizer_path))
        self.scheduler, self.tokenizer, self.env = scheduler, tokenizer, env
        self.max_restarts, self.window = max_restarts, window
        self.restarts, self.error, self.code = [], None, None
        self.lock, self.closing, self.done = threading.Lock(), threading.Event(), threading.Event()
        self._start()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _start(self):
        import subprocess
        self.process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.env)
        self.link = EngineLink(self.scheduler, self.process.stdout, self.process.stdin, self.tokenizer,
                               status=self.health)

    def _serve(self):
        """Serve the current front end until it stops; its exit code and the link's error."""
        process, error = self.process, None
        try:
            self.link.run()
        except ProtocolError as exc:
            error = exc
            process.kill()
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            return process.wait(10), error
        except Exception:  # its standard output closed, but it did not exit
            process.kill()
            return process.wait(), error

    def _run(self):
        try:
            while True:
                self.code, error = self._serve()
                if self.closing.is_set():
                    return
                now = time.monotonic()
                recent = [t for t, _, _ in self.restarts if now - t < self.window]
                if len(recent) >= self.max_restarts:
                    self.error = error or RuntimeError(f'the front end stopped {len(recent) + 1} times '
                                                       f'in {self.window:g} s, the last with {self.code}')
                    return
                self.restarts.append((now, self.code, error))
                print(json.dumps({'front_end_stopped': self.code, 'error': str(error) if error else None,
                                  'restart': len(self.restarts)}), file=sys.stderr, flush=True)
                if self.closing.wait(min(0.05 * 2 ** len(recent), 2.0)):
                    return
                with self.lock:
                    if self.closing.is_set():
                        return
                    self._start()
        finally:
            self.done.set()

    def wait(self):
        """The exit code of the last front end, once the engine stops restarting it."""
        self.done.wait()
        self.thread.join(10)
        return self.code

    def close(self):
        with self.lock:
            self.closing.set()
            process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except Exception:
                process.kill()
                process.wait()
        self.thread.join(10)
