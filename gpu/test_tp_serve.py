"""gpu/tp_serve.py without GPUs: the lockstep protocol between rank 0 and its followers."""
import io
import os
import sys
import threading
import time

import pytest

from gpu import tp_serve


class FakeResident:
    name = 'vx-gpu-resident'
    max_tokens = 64

    def __init__(self, log):
        self.log = log

    def forward(self, token_id, *, logits=True):
        self.log.append(('forward', token_id, logits))
        return 'logits' if logits else None

    def reset(self):
        self.log.append(('reset',))

    def device_step(self, token_id=None, *, readback=True):
        self.log.append(('device_step', token_id, readback))
        return ('ticket', token_id) if readback else None

    def rewind(self, position):
        self.log.append(('rewind', position))

    def preload(self):
        self.log.append(('preload',))
        return 7

    def close(self):
        self.log.append(('close',))


class FakeTP:
    def __init__(self, log):
        self.collective = self
        self.log = log

    def finish(self):
        self.log.append(('finish',))


class FakePipe(io.BytesIO):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def write(self, data):
        self.log.append(('sent', data))
        return super().write(data)


class FakeFollower:
    def __init__(self, log):
        self.stdin = FakePipe(log)
        self.waited = False

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.waited = True
        return 0


def pipe_with(data):
    """A stdin whose fileno is a pipe holding `data`, then closed."""
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    return io.TextIOWrapper(io.FileIO(r, 'rb'))


def test_commands_spin_then_block_and_split_lines():
    # Lines come out whole however the bytes arrive; after spin_seconds without a command the reads block
    # (no CPU while the server is idle); end of input ends the commands.
    r, w = os.pipe()
    got = []

    def writer():
        os.write(w, b'F 1 1\nF 2')
        time.sleep(0.01)
        os.write(w, b' 0\nR\n')
        time.sleep(0.2)  # longer than spin_seconds: the reader is blocked in select by now
        os.write(w, b'Q\n')
        os.close(w)
    thread = threading.Thread(target=writer)
    thread.start()
    for line in tp_serve.commands(r, spin_seconds=0.05):
        got.append((line, time.perf_counter()))
    thread.join()
    os.close(r)
    assert [line for line, _ in got] == [b'F 1 1', b'F 2 0', b'R', b'Q']


def test_lockstep_sends_each_call_before_making_it():
    log = []
    followers = [FakeFollower(log), FakeFollower(log)]
    step = tp_serve.Lockstep(FakeResident(log), FakeTP(log), followers)
    assert step.preload() == 7
    step.reset()
    assert step.forward(5, logits=False) is None
    assert step.forward(9) == 'logits'
    assert step.forward(4, logits='greedy') == 'logits'
    assert step.max_tokens == 64 and step.name == 'vx-gpu-resident'
    step.close()
    step.close()  # once only
    sent = lambda line: [('sent', line)] * 2
    assert log == (sent(b'P\n') + [('preload',)] + sent(b'R\n') + [('reset',)] + sent(b'F 5 0\n') + [('forward', 5, False)]
                   + sent(b'F 9 1\n') + [('forward', 9, True)] + sent(b'F 4 2\n') + [('forward', 4, 'greedy')]
                   + sent(b'Q\n') + [('close',), ('finish',)])
    assert all(f.waited for f in followers)


def test_lockstep_sends_device_steps_and_rewinds():
    # device_decode's steps: D with the token the step starts from, S for one from the device's own token
    # (the followers never read it), W to rewind; the token is read on rank 0 only.
    log = []
    step = tp_serve.Lockstep(FakeResident(log), FakeTP(log), [FakeFollower(log)])
    assert step.device_step(12) == ('ticket', 12)
    assert step.device_step() == ('ticket', None)
    step.rewind(30)
    assert log == [('sent', b'D 12\n'), ('device_step', 12, True), ('sent', b'S\n'), ('device_step', None, True),
                   ('sent', b'W 30\n'), ('rewind', 30)]
    step.close()


def test_follower_replays_the_commands(monkeypatch):
    log = []
    monkeypatch.setattr(tp_serve, 'load_model', lambda checkpoint, layers=None: ('model', checkpoint, layers))
    built = []

    def build_rank(model, rank, ranks, max_tokens, one_shot=True, options=None):
        built.append((model, rank, ranks, max_tokens, one_shot, options))
        return FakeResident(log), FakeTP(log)
    monkeypatch.setattr(tp_serve, 'build_rank', build_rank)
    monkeypatch.setattr(sys, 'stdin', pipe_with(b'P\nR\nF 3 0\nF 4 1\nF 6 2\nD 8\nS\nW 9\nQ\nF 5 1\n'))
    tp_serve.follower_main(['--checkpoint', 'ck', '--rank', '2', '--ranks', '8', '--max-tokens', '64', '--sum', 'nccl',
                            '--layers', '4', '--option', 'moe_down2=true', '--option', 'norm_block=512'])
    assert built == [(('model', 'ck', 4), 2, 8, 64, False, dict(moe_down2=True, norm_block=512))]
    # Nothing after Q runs; the graphs close before NCCL's teardown.
    assert log == [('preload',), ('reset',), ('forward', 3, False), ('forward', 4, True), ('forward', 6, 'greedy'),
                   ('device_step', 8, False), ('device_step', None, False), ('rewind', 9), ('close',), ('finish',)]


def test_follower_rejects_an_unknown_command(monkeypatch):
    log = []
    monkeypatch.setattr(tp_serve, 'load_model', lambda checkpoint, layers=None: None)
    monkeypatch.setattr(tp_serve, 'build_rank', lambda *a, **k: (FakeResident(log), FakeTP(log)))
    monkeypatch.setattr(sys, 'stdin', pipe_with(b'X\n'))
    with pytest.raises(ValueError, match='unknown command'):
        tp_serve.follower_main(['--checkpoint', 'ck', '--rank', '1', '--ranks', '2', '--max-tokens', '8'])
    assert log == [('close',), ('finish',)]


@pytest.mark.parametrize('argv, message', [
    (['--backend', 'vx-gpu-resident', '--tiny', '--tp', '2'], '--tp loads --checkpoint'),
    (['--backend', 'vx', '--checkpoint', 'x', '--tp', '2'], '--tp applies to'),
    (['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--tp', '0'], '--tp must be positive'),
    (['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--layers', '4'], '--layers is a --tp smoke test'),
    (['--backend', 'vx-gpu-resident', '--tiny', '--tp-option', 'moe_down2=true'], '--tp-option loads --checkpoint'),
    (['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--tp', '2', '--tp-option', 'moe_down2'], 'expected KEY=VALUE'),
])
def test_server_tp_arguments(argv, message, capsys):
    from glm_vx.server import parse_args
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert message in capsys.readouterr().err


def test_server_tp_defaults():
    from glm_vx.server import parse_args
    args = parse_args(['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--tp', '8'])
    assert args.tp == 8 and args.tp_sum == 'one-shot' and args.max_sequences == 1 and args.max_context == 8192
    args = parse_args(['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--tp', '8', '--tp-option', 'moe_down2=true',
                       '--tp-option', 'fuse_router=false'])
    assert args.tp_options == dict(moe_down2=True, fuse_router=False)
