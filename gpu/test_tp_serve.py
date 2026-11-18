"""gpu/tp_serve.py without GPUs: the lockstep protocol between rank 0 and its followers."""
import io
import sys

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


def test_lockstep_sends_each_call_before_making_it():
    log = []
    followers = [FakeFollower(log), FakeFollower(log)]
    step = tp_serve.Lockstep(FakeResident(log), FakeTP(log), followers)
    assert step.preload() == 7
    step.reset()
    assert step.forward(5, logits=False) is None
    assert step.forward(9) == 'logits'
    assert step.max_tokens == 64 and step.name == 'vx-gpu-resident'
    step.close()
    step.close()  # once only
    sent = lambda line: [('sent', line)] * 2
    assert log == (sent(b'P\n') + [('preload',)] + sent(b'R\n') + [('reset',)] + sent(b'F 5 0\n') + [('forward', 5, False)]
                   + sent(b'F 9 1\n') + [('forward', 9, True)] + sent(b'Q\n') + [('close',), ('finish',)])
    assert all(f.waited for f in followers)


def test_follower_replays_the_commands(monkeypatch):
    log = []
    monkeypatch.setattr(tp_serve, 'load_model', lambda checkpoint, layers=None: ('model', checkpoint, layers))
    built = []

    def build_rank(model, rank, ranks, max_tokens, one_shot=True):
        built.append((model, rank, ranks, max_tokens, one_shot))
        return FakeResident(log), FakeTP(log)
    monkeypatch.setattr(tp_serve, 'build_rank', build_rank)
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'P\nR\nF 3 0\nF 4 1\nQ\nF 5 1\n')))
    tp_serve.follower_main(['--checkpoint', 'ck', '--rank', '2', '--ranks', '8', '--max-tokens', '64', '--sum', 'nccl',
                            '--layers', '4'])
    assert built == [(('model', 'ck', 4), 2, 8, 64, False)]
    # Nothing after Q runs; the graphs close before NCCL's teardown.
    assert log == [('preload',), ('reset',), ('forward', 3, False), ('forward', 4, True), ('close',), ('finish',)]


def test_follower_rejects_an_unknown_command(monkeypatch):
    log = []
    monkeypatch.setattr(tp_serve, 'load_model', lambda checkpoint, layers=None: None)
    monkeypatch.setattr(tp_serve, 'build_rank', lambda *a, **k: (FakeResident(log), FakeTP(log)))
    monkeypatch.setattr(sys, 'stdin', io.TextIOWrapper(io.BytesIO(b'X\n')))
    with pytest.raises(ValueError, match='unknown command'):
        tp_serve.follower_main(['--checkpoint', 'ck', '--rank', '1', '--ranks', '2', '--max-tokens', '8'])
    assert log == [('close',), ('finish',)]


@pytest.mark.parametrize('argv, message', [
    (['--backend', 'vx-gpu-resident', '--tiny', '--tp', '2'], '--tp loads --checkpoint'),
    (['--backend', 'vx', '--checkpoint', 'x', '--tp', '2'], '--tp applies to'),
    (['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--tp', '0'], '--tp must be positive'),
    (['--backend', 'vx-gpu-resident', '--checkpoint', 'x', '--layers', '4'], '--layers is a --tp smoke test'),
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
