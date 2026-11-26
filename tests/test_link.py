"""The engine link (#68, step 1; docs/ENGINE-LINK.md): frames, their checks, and the same tokens as
the HTTP server and the scheduler give."""
import http.client
import io
import json
import os
import struct
import threading
import time

import pytest

from glm_vx import link
from glm_vx.link import EngineLink, ProtocolError
from glm_vx.scheduler import Scheduler
from glm_vx.server import Server
from test_scheduler import Model, collect


class Front:
    """A front end for tests: writes frames into the link's reader pipe, reads its replies."""

    def __init__(self, scheduler, tokenizer=None, **kwargs):
        to_engine_r, self.to_engine = os.pipe()
        self.from_engine, from_engine_w = os.pipe()
        self.reader = os.fdopen(to_engine_r, 'rb')
        self.writer = os.fdopen(from_engine_w, 'wb')
        self.replies = os.fdopen(self.from_engine, 'rb')
        self.link = EngineLink(scheduler, self.reader, self.writer, tokenizer=tokenizer, **kwargs)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.link.run()
        except ProtocolError as exc:
            self.error = exc
        finally:
            self.writer.close()

    def send(self, data):
        os.write(self.to_engine, data)

    def reply(self):
        payload = link.read_frame(self.replies)
        return None if payload is None else link.parse_reply(payload)

    def until_done(self, ids):
        """Every reply up to the DONE or ERROR of each id, grouped by id."""
        got, open_ids = {i: [] for i in ids}, set(ids)
        while open_ids:
            r = self.reply()
            assert r is not None, f'the link ended with {open_ids} open'
            got[r['id']].append(r)
            if r['kind'] in ('done', 'error'):
                open_ids.discard(r['id'])
        return got

    def close(self):
        os.close(self.to_engine)
        self.thread.join(10)
        self.replies.close()
        self.reader.close()


def tokens(replies):
    assert [r['kind'] for r in replies][:1] in (['accepted'], ['error']), replies[:2]
    return [r['token_id'] for r in replies if r['kind'] == 'token']


# --------------------------------------------------------------- frames
def test_request_frames_parse_back():
    data = link.submit_frame(7, [1, 2, 3], 5, temperature=0.5, seed=11, ignore_eos=True)
    payload = link.read_frame(io.BytesIO(data))
    assert link.parse_request(payload) == (link.SUBMIT, 7, dict(prompt=[1, 2, 3], max_tokens=5, temperature=0.5,
                                                                seed=11, ignore_eos=True))
    text = link.parse_request(link.read_frame(io.BytesIO(link.submit_frame(8, 'héllo', 2))))
    assert text[:2] == (link.SUBMIT_TEXT, 8) and text[2]['prompt'] == 'héllo'.encode()
    assert link.parse_request(link.read_frame(io.BytesIO(link.cancel_frame(9)))) == (link.CANCEL, 9, {})
    assert link.parse_request(link.read_frame(io.BytesIO(link.status_request_frame(10)))) == (link.STATUS, 10, {})
    assert link.read_frame(io.BytesIO(b'')) is None


def test_reply_frames_parse_back():
    event = dict(reason='stop', prompt_tokens=3, completion_tokens=2, cached_prompt_tokens=1,
                 timings=dict(prefill_seconds=.5, decode_seconds=.25, prefix_seconds=0.0, queued_seconds=.125))
    done = link.parse_reply(link.read_frame(io.BytesIO(link.done_frame(4, event, 'ñ', 'all ñ'))))
    assert done == dict(kind='done', id=4, reason='stop', prompt_tokens=3, completion_tokens=2, cached_prompt_tokens=1,
                        timings=event['timings'], tail='ñ', text='all ñ')
    assert link.parse_reply(link.read_frame(io.BytesIO(link.token_frame(4, 99, 'é')))) == \
        dict(kind='token', id=4, token_id=99, text='é')
    assert link.parse_reply(link.read_frame(io.BytesIO(link.error_frame(4, link.BUSY, 'full')))) == \
        dict(kind='error', id=4, code=link.BUSY, message='full')


SUBMIT_HEAD = struct.Struct('<BQIdQBI')


@pytest.mark.parametrize('data', [
    struct.pack('<I', 0),                                     # an empty payload
    struct.pack('<I', link.MAX_PAYLOAD + 1),                  # too long
    b'\x05\x00',                                              # the length cut short
    struct.pack('<I', 9) + b'\x03\x01',                       # the payload cut short
    link.frame(b'\x7f'),                                      # an unknown kind
    link.frame(b'\x03' + b'\x00' * 9),                        # a cancel one byte long
    link.frame(SUBMIT_HEAD.pack(1, 1, 4, 0.0, 0, 2, 0)),      # an unknown flag bit
    link.frame(SUBMIT_HEAD.pack(1, 1, 4, 0.0, 0, 0, 3) + struct.pack('<2I', 1, 2)),  # 3 ids, 2 sent
    link.frame(SUBMIT_HEAD.pack(1, 1, 4, 0.0, 0, 0, 1) + struct.pack('<2I', 1, 2)),  # 1 id, 2 sent
    link.frame(b'\x01' + b'\x00' * 10),                       # a submit shorter than its fields
])
def test_malformed_frames_end_the_link(data):
    with pytest.raises(ProtocolError):
        payload = link.read_frame(io.BytesIO(data))
        link.parse_request(payload)


# --------------------------------------------------------------- the link with a scheduler
def test_same_tokens_as_the_scheduler_concurrent_and_sampled():
    sched, ref = Scheduler(Model(), prefill_chunk=1), Scheduler(Model(), prefill_chunk=1)
    front = Front(sched)
    cases = {1: ([1, 2, 3], 4, 0.0, 0), 2: ([3, 2, 1], 5, 0.9, 7), 3: ([5], 3, 1.5, 2**63 + 5), 4: ([7, 7], 6, 0.0, 0)}
    try:
        for rid, (prompt, n, temp, seed) in cases.items():
            front.send(link.submit_frame(rid, prompt, n, temperature=temp, seed=seed))
        got = front.until_done(cases)
        for rid, (prompt, n, temp, seed) in cases.items():
            want, event = collect(ref.submit(prompt, n, temp, seed))
            assert tokens(got[rid]) == want
            done = got[rid][-1]
            assert done['kind'] == 'done' and done['reason'] == event['reason'] == 'length'
            assert (done['prompt_tokens'], done['completion_tokens']) == (len(prompt), n)
    finally:
        front.close(); sched.close(); ref.close()
    assert front.error is None


def test_same_tokens_as_the_http_server():
    class Backend: name = 'test'
    model = Model(); model.backend = Backend()
    http_sched, link_sched = Scheduler(model), Scheduler(Model())
    server = Server(('127.0.0.1', 0), http_sched)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    front = Front(link_sched)
    bodies = [dict(prompt=[1, 2], max_tokens=4), dict(prompt=[6], max_tokens=7, temperature=0.7, seed=3),
              dict(prompt=[2, 4, 6], max_tokens=2, ignore_eos=True)]
    try:
        for rid, body in enumerate(bodies):
            c = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
            c.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
            r = c.getresponse(); status, result = r.status, json.loads(r.read()); c.close()
            assert status == 200
            front.send(link.submit_frame(rid, body['prompt'], body['max_tokens'], body.get('temperature', 0.0),
                                         body.get('seed', 0), body.get('ignore_eos', False)))
            replies = front.until_done([rid])[rid]
            assert tokens(replies) == result['choices'][0]['token_ids']
            done = replies[-1]
            assert done['reason'] == result['choices'][0]['finish_reason']
            assert {k: done[k] for k in ('prompt_tokens', 'completion_tokens')} == \
                {k: result['usage'][k] for k in ('prompt_tokens', 'completion_tokens')}
    finally:
        front.close(); server.shutdown(); server.server_close(); http_sched.close(); link_sched.close(); thread.join()


def test_bad_values_are_the_requests_error_not_the_links():
    sched = Scheduler(Model(), token_budget=12)
    front = Front(sched)
    try:
        bad = {1: link.submit_frame(1, [8], 1),                          # past the vocabulary of 8
               2: link.submit_frame(2, [1], 0),                          # max_tokens 0
               3: link.submit_frame(3, [1], 1, temperature=float('nan')),
               4: link.submit_frame(4, [1], 1, temperature=-1.0),
               5: link.submit_frame(5, [], 1),                           # no prompt
               6: link.submit_frame(6, [1] * 30, 3),                     # past the context of 32
               7: link.submit_frame(7, 'text', 1),                       # text with no tokenizer
               8: link.submit_frame(8, [1, 2], 11)}                      # past the token budget: busy
        for data in bad.values():
            front.send(data)
        got = front.until_done(bad)
        for rid in bad:
            assert [r['kind'] for r in got[rid]] == ['error'], got[rid]
            assert got[rid][0]['code'] == (link.BUSY if rid == 8 else link.INVALID), got[rid]
        front.send(link.submit_frame(9, [1, 2], 3))
        ref = Scheduler(Model())
        try: assert tokens(front.until_done([9])[9]) == collect(ref.submit([1, 2], 3))[0]
        finally: ref.close()
    finally:
        front.close(); sched.close()
    assert front.error is None


def test_an_id_in_use_ends_the_link_and_cancels_its_requests():
    release = threading.Event()
    class Slow(Model):
        def forward(self, t, c):
            release.wait(5); return super().forward(t, c)
    sched = Scheduler(Slow())
    front = Front(sched)
    try:
        front.send(link.submit_frame(1, [1], 20))
        front.send(link.submit_frame(1, [2], 20))
        front.thread.join(5)
        assert isinstance(front.error, ProtocolError) and 'in use' in str(front.error)
        release.set()
        while front.reply() is not None:  # the engine closes its side after the error
            pass
        deadline = time.monotonic() + 5
        while sched.status()['active_requests'] and time.monotonic() < deadline:
            time.sleep(.01)
        assert sched.status()['active_requests'] == 0
    finally:
        release.set(); front.reader.close(); front.replies.close(); os.close(front.to_engine); sched.close()


def test_cancel_and_status():
    entered, release = threading.Event(), threading.Event()
    class Blocking(Model):
        def forward(self, t, c):
            entered.set(); release.wait(5); return super().forward(t, c)
    sched = Scheduler(Blocking())
    front = Front(sched)
    try:
        front.send(link.submit_frame(5, [1], 20))
        accepted = front.reply()
        assert accepted['kind'] == 'accepted' and accepted['id'] == 5 and accepted['name'].startswith('cmpl-')
        assert abs(accepted['created'] - time.time()) < 60
        assert entered.wait(5)
        front.send(link.status_request_frame(6))
        status = front.reply()
        assert status['kind'] == 'status' and status['id'] == 6
        assert json.loads(status['json'])['active_requests'] == 1
        req = front.link.requests[5]
        front.send(link.cancel_frame(12345))  # not in use: ignored
        front.send(link.cancel_frame(5))
        assert req.cancelled.wait(5)
        release.set()
        done = front.until_done([5])[5][-1]
        assert done['kind'] == 'done' and done['reason'] == 'cancelled'
        front.send(link.submit_frame(5, [1], 2))  # the id is free again
        assert front.until_done([5])[5][-1]['reason'] == 'length'
    finally:
        release.set(); front.close(); sched.close()
    assert front.error is None


def test_text_prompts_and_text_replies():
    from test_openai_compat import byte_tokenizer
    from glm_vx.detok import IncrementalText
    tok = byte_tokenizer()
    class Wide(Model):
        config = {'vocab_size': 256, 'max_position_embeddings': 64, 'eos_token_id': []}
        def forward(self, t, c):
            import numpy as np
            c.append(t); out = np.zeros(256, dtype=np.float32); out[(sum(c) * 7 + 3) % 256] = 10; return out
    sched, ref = Scheduler(Wide()), Scheduler(Wide())
    front = Front(sched, tokenizer=tok)
    try:
        prompt = 'naïve ☃'
        front.send(link.submit_frame(1, prompt, 9))
        front.send(link.frame(SUBMIT_HEAD.pack(link.SUBMIT_TEXT, 2, 3, 0.0, 0, 0, 2) + b'\xff\xfe'))  # not UTF-8
        got = front.until_done([1, 2])
        want, _ = collect(ref.submit(tok.encode(prompt).ids, 9))
        assert tokens(got[1]) == want
        detok = IncrementalText(tok)
        assert [r['text'] for r in got[1] if r['kind'] == 'token'] == [detok.push(t) for t in want]
        assert got[1][-1]['tail'] == detok.flush() and got[1][-1]['text'] == tok.decode(want)
        assert got[1][-1]['prompt_tokens'] == len(tok.encode(prompt).ids)
        assert [r['kind'] for r in got[2]] == ['error'] and got[2][0]['code'] == link.INVALID
    finally:
        front.close(); sched.close(); ref.close()
    assert front.error is None


def test_the_tiny_glm_gives_the_same_tokens_over_the_link():
    """The CPU reference GLM (numpy backend), greedy and sampled, through the link and directly."""
    from test_resident_server import build, served
    sched, ref = Scheduler(build(seed=9)), Scheduler(build(seed=9))
    front = Front(sched)
    cases = {1: ([3, 17, 5, 5, 9, 200], 4, 0.0, 0), 2: ([44, 8, 2], 4, 0.8, 5)}
    try:
        for rid, (prompt, n, temp, seed) in cases.items():
            front.send(link.submit_frame(rid, prompt, n, temperature=temp, seed=seed))
        got = front.until_done(cases)
        for rid, (prompt, n, temp, seed) in cases.items():
            assert tokens(got[rid]) == served(ref, prompt, n, temp, seed)
    finally:
        front.close(); sched.close(); ref.close()
