"""Tensor-parallel serving (#34): glm_vx.server --tp N runs a ResidentGLM split over N GPUs.

The server process is rank 0: it serves HTTP, schedules and samples as with one GPU. Ranks 1..N-1
are follower processes (``python -m gpu.tp_serve``) that each build their slice of the model and
then replay rank 0's ResidentGLM calls in the same order, so every collective of a step has all its
ranks. Rank 0 holds its ResidentGLM behind Lockstep, which writes each call as one line to every
follower's stdin before making it:

    F <token> <0|1|2> forward(token, logits=False / True / 'greedy')  (followers discard what it returns)
    D <token>         device_step(token)  (device_decode=True: a step whose token the device chooses)
    S                 device_step()  (the next step, from the token the device chose; never read here)
    W <position>      rewind(position)
    R                 reset()
    P                 preload()
    Q                 close, then exit

Between commands a follower polls its pipe for GLM_VX_TP_SPIN seconds (default 0.05) before it blocks.
The logits flag must match on every rank: the lm_head's vocabulary all-gather (or 'greedy''s gather of
one pair a rank) is a collective. A
follower that exits leaves rank 0 waiting inside a collective forever, so a watchdog thread stops
the server when one does. The ranks find each other through VX_NCCL_ID_FILE (gpu.collective).
"""
import argparse
import ctypes as C
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# The kernels of the production decode step (validation/tp_real.py measures with the same).
OPTIONS = dict(device_experts=True, graph=True, fused_attention=True, mla_kernel='tc', fp8_kernels='cvt',
               matvec_kernels='x4', norm_block=1024, long_topk='ilp_hist', fuse_glue=True, index_kernel='mma',
               fold_shared=True)


def load_model(checkpoint, layers=None):
    """The host model over a checkpoint directory (weights read lazily); with `layers`, only the first
    layers (a smoke test: GLM-5.3's own per-layer patterns, meaningless output)."""
    from glm_vx.backend import NumpyBackend
    from glm_vx.checkpoint import Checkpoint
    from glm_vx.config import GLMConfig
    from glm_vx.model import GlmMoeDsaModel
    config = json.loads((Path(checkpoint) / 'config.json').read_text())
    if layers is not None:
        from validation.write_proxy import proxy_config
        config = proxy_config(layers, base=config)
    weights = Checkpoint(checkpoint, block_size=GLMConfig.from_dict(config).weight_block_size)
    return GlmMoeDsaModel(config, weights, NumpyBackend())


def parse_options(pairs):
    """KEY=VALUE strings (VALUE as JSON: true, 3, "x") as ResidentGLM keyword arguments."""
    options = {}
    for pair in pairs or ():
        key, sep, value = pair.partition('=')
        if not sep or not key.isidentifier():
            raise ValueError(f'expected KEY=VALUE, not {pair!r}')
        options[key] = json.loads(value)
    return options


def build_rank(model, rank, ranks, max_tokens, one_shot=True, options=None):
    """This rank's ResidentGLM (on GPU `rank`) and its TensorParallel; `options` override OPTIONS."""
    from gpu.collective import Collective
    from gpu.resident import ResidentGLM
    from gpu.runtime import CUDAContext
    from gpu.tp import TensorParallel
    if 'VX_DISPATCH_LIB' not in os.environ:
        raise RuntimeError('VX_DISPATCH_LIB is not set (source scripts/env.sh)')
    context = CUDAContext(rank, primary=True, minimum_compute_capability=(8, 0))
    collective = Collective(C.CDLL(os.environ['VX_DISPATCH_LIB']), context, rank, ranks)
    tp = TensorParallel(rank, ranks, collective, one_shot=one_shot)
    return ResidentGLM(model, ctx=context, max_tokens=max_tokens, tp=tp, **{**OPTIONS, **(options or {})}), tp


class Lockstep:
    """Rank 0's ResidentGLM with every call sent to the followers first; other attributes are the
    ResidentGLM's own."""

    def __init__(self, resident, tp, followers):
        self.resident, self.tp, self.followers = resident, tp, followers
        self._closed = False
        self._watchdog = threading.Thread(target=self._watch, name='tp-watchdog', daemon=True)
        self._watchdog.start()

    def __getattr__(self, name):
        if name in ('resident', 'tp', 'followers'):
            raise AttributeError(name)
        return getattr(self.resident, name)

    def _watch(self):
        while not self._closed:
            for r, follower in enumerate(self.followers, 1):
                code = follower.poll()
                if code is not None and not self._closed:
                    print(json.dumps({'error': f'tensor-parallel rank {r} exited ({code}); stopping the server'}),
                          file=sys.stderr, flush=True)
                    os._exit(1)
            time.sleep(0.5)

    def _send(self, line):
        for follower in self.followers:
            follower.stdin.write(line)
            follower.stdin.flush()

    def forward(self, token_id, *, logits=True):
        self._send(b'F %d %d\n' % (int(token_id), 2 if logits == 'greedy' else 1 if logits else 0))
        return self.resident.forward(token_id, logits=logits)

    def prefill_chunk(self, token_ids, *, logits=False):
        tokens = [int(t) for t in token_ids]
        flag = 2 if logits == 'greedy' else 1 if logits else 0
        self._send(b'C %d %s\n' % (flag, b' '.join(b'%d' % t for t in tokens)))
        return self.resident.prefill_chunk(tokens, logits=logits)

    def device_step(self, token_id=None):
        self._send(b'S\n' if token_id is None else b'D %d\n' % int(token_id))
        return self.resident.device_step(token_id)

    def rewind(self, position):
        self._send(b'W %d\n' % int(position))
        return self.resident.rewind(position)

    def reset(self):
        self._send(b'R\n')
        return self.resident.reset()

    def preload(self, *args, **kwargs):
        self._send(b'P\n')
        return self.resident.preload(*args, **kwargs)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._send(b'Q\n')
        except OSError:
            pass
        # The graphs go first: ncclCommDestroy waits for every graph that holds NCCL work.
        self.resident.close()
        self.tp.collective.finish()
        for follower in self.followers:
            try:
                follower.wait(timeout=120)
            except subprocess.TimeoutExpired:
                follower.kill()


def start(checkpoint, ranks, max_tokens, *, one_shot=True, layers=None, options=None):
    """Rank 0 in this process, ranks 1..ranks-1 as followers, every rank with OPTIONS updated by
    `options`. Returns (host model, Lockstep)."""
    tmp = tempfile.mkdtemp(prefix='glm-vx-tp-')
    os.environ['VX_NCCL_ID_FILE'] = os.path.join(tmp, 'nccl-id')  # absent until rank 0 writes it
    command = [sys.executable, '-m', 'gpu.tp_serve', '--checkpoint', str(checkpoint), '--ranks', str(ranks),
               '--max-tokens', str(max_tokens), '--sum', 'one-shot' if one_shot else 'nccl']
    command += ['--layers', str(layers)] if layers is not None else []
    for key, value in (options or {}).items():
        command += ['--option', f'{key}={json.dumps(value)}']
    followers = [subprocess.Popen(command + ['--rank', str(r)], stdin=subprocess.PIPE, env=dict(os.environ))
                 for r in range(1, ranks)]
    try:
        model = load_model(checkpoint, layers)
        resident, tp = build_rank(model, 0, ranks, max_tokens, one_shot, options)
    except BaseException:
        for follower in followers:
            follower.kill()
        raise
    return model, Lockstep(resident, tp, followers)


def commands(fd, spin_seconds=0.05):
    """rank 0's command lines from pipe `fd`, without the newline, until it closes. After each command
    the reads spin (non-blocking) for spin_seconds before they block: a follower woken from a blocking
    read can start its step late, and rank 0's first sum of the step waits for it."""
    os.set_blocking(fd, False)
    pending = b''
    idle_since = time.perf_counter()
    while True:
        while b'\n' in pending:
            line, pending = pending.split(b'\n', 1)
            yield line
            idle_since = time.perf_counter()  # the caller has finished the command
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            if time.perf_counter() - idle_since > spin_seconds:
                select.select([fd], [], [])
            continue
        if not chunk:
            return
        pending += chunk


def follower_main(argv=None):
    p = argparse.ArgumentParser(description='one follower rank of glm_vx.server --tp (started by rank 0)')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--ranks', type=int, required=True)
    p.add_argument('--max-tokens', type=int, required=True)
    p.add_argument('--sum', choices=['one-shot', 'nccl'], default='one-shot')
    p.add_argument('--layers', type=int)
    p.add_argument('--option', action='append', help='KEY=VALUE: a ResidentGLM option over OPTIONS')
    args = p.parse_args(argv)
    resident, tp = build_rank(load_model(args.checkpoint, args.layers), args.rank, args.ranks, args.max_tokens,
                              one_shot=args.sum == 'one-shot', options=parse_options(args.option))
    try:
        # GLM_VX_TP_SPIN: the seconds of non-blocking reads after a command (0: block at once).
        spin = float(os.environ.get('GLM_VX_TP_SPIN', '0.05'))
        for line in commands(sys.stdin.fileno(), spin):  # end of input: rank 0 is gone
            op = line.split()
            if op[0] == b'F':
                resident.forward(int(op[1]), logits={b'0': False, b'1': True, b'2': 'greedy'}[op[2]])
            elif op[0] == b'C':
                resident.prefill_chunk([int(t) for t in op[2:]], logits={b'0': False, b'1': True, b'2': 'greedy'}[op[1]])
            elif op[0] == b'D':
                resident.device_step(int(op[1]), readback=False)
            elif op[0] == b'S':
                resident.device_step(readback=False)
            elif op[0] == b'W':
                resident.rewind(int(op[1]))
            elif op[0] == b'R':
                resident.reset()
            elif op[0] == b'P':
                resident.preload()
            elif op[0] == b'Q':
                break
            else:
                raise ValueError(f'unknown command {line!r}')
    finally:
        resident.close()
        tp.collective.finish()


if __name__ == '__main__':
    follower_main()
