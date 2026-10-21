"""--backend vx-gpu-resident serving: one sequence at a time over a ResidentGLM-shaped object.

CPU-only: ``FakeResident`` has ResidentGLM's serving surface (forward(token, logits=...), reset,
max_tokens, name, close) over the NumPy reference model, so these tests check the scheduler and
server wiring, not device arithmetic. The last test runs the real ResidentGLM under the CPU SIMT
emulator when its build is present."""
import http.client
import json
import threading
import time

import numpy as np
import pytest

from glm_vx import server as srv
from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.scheduler import BusyError, Scheduler
from glm_vx.server import Server, parse_args
from glm_vx.tiny import tiny_config
from gpu.resident import ResidentSequenceModel


class FakeResident:
    name = 'vx-gpu-resident'

    def __init__(self, model, *, max_tokens=4096, **kwargs):
        self.model, self.max_tokens = model, max_tokens
        self.resets, self.headless, self.closed, self.preloaded = 0, 0, False, False
        self.reset()

    def preload(self):
        self.preloaded = True
        return 123

    def reset(self):
        self.resets += 1
        self.cache, self.position = self.model.new_cache(), 0

    def forward(self, token_id, *, logits=True):
        if self.position >= self.max_tokens: raise ValueError('sequence exceeds max_tokens')
        self.position += 1
        if logits: return self.model.forward(token_id, self.cache)
        self.headless += 1
        self.model.prefill_token(token_id, self.cache)

    def close(self): self.closed = True


def build(seed=3, index_heads=16):
    config = tiny_config()
    config['index_n_heads'] = index_heads  # few indexer heads give exact score ties
    weights = tiny_weights(config, seed=seed)
    for w in weights.values(): w.flags.writeable = False
    return GlmMoeDsaModel(config, weights, NumpyBackend())


def served(scheduler, prompt, max_tokens, temperature=0, seed=0):
    req = scheduler.submit(prompt, max_tokens, temperature, seed)
    ids = []
    while True:
        event = req.events.get(timeout=30)
        if event['type'] == 'token': ids.append(event['token_id'])
        else:
            assert event['reason'] in ('stop', 'length'), event
            return ids


PROMPTS = [([3, 17, 5, 5, 9, 200, 1], 5, 0), ([44, 8], 4, 0.7), ([2, 31, 77, 3, 3], 6, 1.3)]


# --------------------------------------------------------------- arguments
@pytest.mark.parametrize('extra', [['--max-sequences', '2'], ['--prefix-cache-mib', '64'], ['--batched-prefill'],
                                   ['--max-context', '0'], ['--gpu-tuning', 'plan.json'],
                                   ['--gpu-weight-cache-mib', '1024']])
def test_resident_refuses_conflicting_flags(extra):
    with pytest.raises(SystemExit):
        parse_args(['--backend', 'vx-gpu-resident', '--tiny', *extra])


def test_resident_refuses_gguf_and_other_backends_refuse_max_context():
    with pytest.raises(SystemExit): parse_args(['--backend', 'vx-gpu-resident', '--gguf', 'x.gguf'])
    with pytest.raises(SystemExit): parse_args(['--backend', 'numpy-reference', '--tiny', '--max-context', '64'])
    with pytest.raises(SystemExit): parse_args(['--backend', 'numpy-reference', '--tiny', '--no-preload'])


def test_defaults_per_backend():
    a = parse_args(['--backend', 'vx-gpu-resident', '--checkpoint', 'dir'])
    assert (a.max_sequences, a.max_waiting, a.max_context) == (1, 32, 8192)
    a = parse_args(['--backend', 'vx-gpu-resident', '--tiny', '--max-sequences', '1', '--max-context', '64', '--max-waiting', '0'])
    assert (a.max_sequences, a.max_waiting, a.max_context) == (1, 0, 64)
    a = parse_args(['--backend', 'numpy-reference', '--tiny'])
    assert (a.max_sequences, a.max_waiting, a.max_context, a.gpu_weight_cache_mib) == (8, 0, None, 512)


def test_build_model_wraps_one_resident(monkeypatch):
    from gpu import resident
    monkeypatch.setattr(resident, 'ResidentGLM', FakeResident)
    model, backend, _, name = srv.build_model(parse_args(['--backend', 'vx-gpu-resident', '--tiny', '--max-context', '64']))
    assert isinstance(model, ResidentSequenceModel) and isinstance(backend, FakeResident)
    assert (backend.max_tokens, model.config['max_position_embeddings'], backend.name) == (64, 64, 'vx-gpu-resident')
    assert backend.preloaded
    sched = Scheduler(model, max_sequences=1, max_waiting=4)
    try:
        with pytest.raises(ValueError, match='context limit'): sched.submit([1] * 60, 5)
        assert len(served(sched, [1, 2, 3], 3)) == 3
    finally: sched.close()


def test_preload_is_optional_and_a_misfit_fails_startup(monkeypatch, capsys):
    from gpu import resident
    monkeypatch.setattr(resident, 'ResidentGLM', FakeResident)
    _, backend, _, _ = srv.build_model(parse_args(['--backend', 'vx-gpu-resident', '--tiny', '--no-preload']))
    assert not backend.preloaded
    assert json.loads(capsys.readouterr().out or '{}') == {}
    srv.build_model(parse_args(['--backend', 'vx-gpu-resident', '--tiny']))
    assert json.loads(capsys.readouterr().out)['preloaded_weight_bytes'] == 123
    class TooBig(FakeResident):
        def preload(self): raise MemoryError('preload needs 9 bytes; the device has 3 bytes free')
    monkeypatch.setattr(resident, 'ResidentGLM', TooBig)
    with pytest.raises(SystemExit, match='needs 9 bytes'):
        srv.build_model(parse_args(['--backend', 'vx-gpu-resident', '--tiny']))


# --------------------------------------------------------------- serving
def test_outputs_equal_the_numpy_reference_and_prompt_skips_lm_head():
    reference = Scheduler(build())
    resident = FakeResident(build())
    sched = Scheduler(ResidentSequenceModel(resident.model, resident), max_sequences=1, max_waiting=4)
    try:
        for prompt, n, temperature in PROMPTS:
            assert served(sched, prompt, n, temperature, 11) == served(reference, prompt, n, temperature, 11)
    finally: sched.close(); reference.close()
    assert resident.headless == sum(len(p) - 1 for p, _, _ in PROMPTS)
    assert resident.resets == 1 + len(PROMPTS)


def test_reset_between_requests():
    # Logits, not just tokens: the tiny model's argmax barely depends on earlier context.
    resident = FakeResident(build(seed=5))
    logits = []
    class Recording(ResidentSequenceModel):
        def forward(self, token_id, cache):
            logits.append(super().forward(token_id, cache)); return logits[-1]
    sched = Scheduler(Recording(resident.model, resident), max_sequences=1)
    try:
        alone = served(sched, [9, 8, 7], 6); first = logits[:]
        served(sched, [100, 3, 3, 4, 200, 17], 8); logits.clear()
        assert served(sched, [9, 8, 7], 6) == alone
        np.testing.assert_array_equal(np.stack(logits), np.stack(first))
        assert resident.position == 3 + 6 - 1
    finally: sched.close()


def test_stale_request_cannot_touch_the_next_sequence():
    resident = FakeResident(build())
    model = ResidentSequenceModel(resident.model, resident)
    old = model.new_cache(); model.forward(3, old)
    new = model.new_cache()
    with pytest.raises(RuntimeError): model.forward(4, old)
    assert resident.position == 0 and model.forward(4, new).shape == (resident.model.config['vocab_size'],)


def test_queued_request_waits_for_the_single_slot():
    resident = FakeResident(build())
    order, entered, release = [], threading.Event(), threading.Event()
    class Gated(ResidentSequenceModel):
        def forward(self, token_id, cache):
            order.append(cache); entered.set(); release.wait(5)
            return super().forward(token_id, cache)
    sched = Scheduler(Gated(resident.model, resident), max_sequences=1, max_waiting=1)
    try:
        first = sched.submit([1, 2], 3)
        assert entered.wait(5)
        second = sched.submit([4], 2)  # not rejected
        assert sched.status()['waiting_requests'] == 1 and sched.status()['active_requests'] == 1
        with pytest.raises(BusyError): sched.submit([5], 2)  # the waiting queue is bounded
        release.set()
        done = []
        for req, n in ((first, 3), (second, 2)):
            events = [req.events.get(timeout=10) for _ in range(n + 1)]
            assert events[-1]['reason'] == 'length'
            done.append(events[-1]['timings']['queued_seconds'])
        assert done[0] < 0.05 and done[1] > 0
        # The second sequence starts only after the first finishes: forwards never interleave.
        caches = [c for i, c in enumerate(order) if i == 0 or c is not order[i - 1]]
        assert len(caches) == 2
        assert sched.status()['waiting_requests'] == sched.status()['active_requests'] == 0
    finally: release.set(); sched.close()


def test_cancelled_waiting_request_is_dropped():
    gate = threading.Event()
    resident = FakeResident(build())
    class Gated(ResidentSequenceModel):
        def forward(self, token_id, cache):
            gate.wait(5); return super().forward(token_id, cache)
    sched = Scheduler(Gated(resident.model, resident), max_sequences=1, max_waiting=2)
    try:
        sched.submit([1], 2)
        waiting = sched.submit([2], 2)
        sched.cancel(waiting)
        assert waiting.events.get(timeout=1)['reason'] == 'cancelled'
        assert sched.status()['waiting_requests'] == 0
    finally: gate.set(); sched.close()


def test_default_backends_still_reject_when_full():
    sched = Scheduler(build(), max_sequences=1)
    gate = threading.Event()
    try:
        sched.model.forward = lambda t, c, f=sched.model.forward: (gate.wait(5), f(t, c))[1]
        sched.submit([1], 2)
        with pytest.raises(BusyError): sched.submit([2], 2)
    finally: gate.set(); sched.close()


def post(port, body):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    c.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse(); value = (r.status, json.loads(r.read())); c.close()
    return value


def serve_http(model, bodies):
    """Concurrent HTTP requests against one-slot serving; returns the token ids per body."""
    sched = Scheduler(model, max_sequences=1, max_waiting=8)
    server = Server(('127.0.0.1', 0), sched)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    results = [None] * len(bodies)
    def run(i): results[i] = post(server.server_port, bodies[i])
    workers = [threading.Thread(target=run, args=(i,)) for i in range(len(bodies))]
    try:
        for w in workers: w.start()
        for w in workers: w.join(120)
        health = json.loads(_get(server.server_port, '/health'))
    finally: server.shutdown(); server.server_close(); sched.close(); thread.join()
    assert health['backend'] == 'vx-gpu-resident' and health['max_sequences'] == 1
    assert all(r is not None and r[0] == 200 for r in results), results
    return [r[1]['choices'][0]['token_ids'] for r in results]


def _get(port, path):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10); c.request('GET', path)
    raw = c.getresponse().read(); c.close(); return raw


def test_concurrent_http_requests_all_complete_and_match_reference():
    resident = FakeResident(build())
    bodies = [{'prompt': p, 'max_tokens': n, 'temperature': t, 'seed': 2} for p, n, t in PROMPTS]
    got = serve_http(ResidentSequenceModel(resident.model, resident), bodies)
    reference = Scheduler(build())
    try: assert got == [served(reference, p, n, t, 2) for p, n, t in PROMPTS]
    finally: reference.close()


# --------------------------------------------------------------- real ResidentGLM, CPU SIMT lanes
def test_simt_resident_over_http_matches_reference():
    from gpu.resident import ResidentGLM
    try:
        from gpu.testing.context import Context
        ctx = Context()
    except (RuntimeError, OSError, ImportError) as exc:
        pytest.skip(f'needs the CPU SIMT build (gpu/testing/build_simt.sh): {exc}')
    model = build(seed=9)
    resident = ResidentGLM(model, ctx=ctx, max_tokens=16)
    prompts = [([3, 17, 5, 5, 9, 200], 3), ([44, 8, 2], 3)]
    try:
        got = serve_http(ResidentSequenceModel(model, resident), [{'prompt': p, 'max_tokens': n} for p, n in prompts])
    finally: resident.close()
    reference = Scheduler(build(seed=9))
    try: assert got == [served(reference, p, n) for p, n in prompts]
    finally: reference.close()


def simt_context():
    try:
        from gpu.testing.context import Context
        return Context()
    except (RuntimeError, OSError, ImportError) as exc:
        pytest.skip(f'needs the CPU SIMT build (gpu/testing/build_simt.sh): {exc}')


def test_simt_preload_uploads_everything_decode_uses_and_reports_a_misfit():
    from gpu.resident import ResidentGLM, _planned
    model = build(seed=9)
    full = ResidentGLM(model, ctx=simt_context(), max_tokens=16)
    needed = full.preload()
    keys, c = set(full._weights), model.config
    sparse = model.mlp_types.count('sparse')
    assert needed == full.weight_bytes > 0
    assert sum('.experts.' in k and k.endswith('gate_proj.weight') for k in keys) == sparse * c['n_routed_experts']
    assert full.preload() == 0  # idempotent

    class Small(type(full.ctx)):
        def memory_info(self): return (64 * 1024 + needed // 2, 1 << 40)
    part = ResidentGLM(model, ctx=Small(), max_tokens=16)
    with pytest.raises(MemoryError, match=f'needs {needed} bytes'):
        part.preload(reserve_bytes=64 * 1024)
    assert 0 < part.weight_bytes <= needed // 2 and not any(_planned(v) for v in part._weights.values())

    # Decode after preload uploads nothing more.
    full.prefill([3, 17, 5, 5, 9, 200])
    assert full.weight_bytes == needed and set(full._weights) == keys
