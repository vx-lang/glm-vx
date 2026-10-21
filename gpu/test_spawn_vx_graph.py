"""Vx's launches on a stream of ours, and in a CUDA graph (glm-vx #9; Vx #1986, PR #2016, which
replaced #1994).

A chain of four gpu/spawn kernels, scale -> add -> rmsnorm_rows -> softmax_rows, that Vx's own
runtime launches (lib<kernel>.so):

- eager: on a non-blocking stream of ours, bitwise equal to the run on the legacy stream
  (also with VX_CUDA_SYNC=1, which then waits on our stream);
- capture: each kernel called once, then the chain captured into a graph on our stream and
  replayed with new inputs written in place between replays; each replay bitwise equal to
  the eager run on the same inputs;
- mixed: Python launches (SpawnKernel) and Vx launches in one graph;
- refusals: VX_CUDA_SYNC=1 or VX_TIME_KERNEL during a capture must stop the program with
  Vx's message;
- first call in a capture (no warm-up): Vx (PR #2016; #1994 from d1de7d0a) stops the program and
  says to call the function once before the recording.

Every scenario runs in a child process (`python -m gpu.test_spawn_vx_graph SCENARIO`), since
Vx stops the whole program on what it refuses. Each child's stdout and stderr (Vx's
messages) is printed, always; run pytest with -s to keep it on success too.

Skipped without the libraries (gpu/spawn/build.sh with VX_DISPATCH_LIB), without
vx_cuda_set_stream in them (a Vx build without PR #2016), or without a supported GPU.
"""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPAWN = Path(__file__).parent / 'build' / 'spawn'
CHAIN = ('scale', 'add', 'rmsnorm_rows', 'softmax_rows')
ROWS, DIM = 4, 2048
N = ROWS * DIM
REPLAYS = 5
SKIP = 77  # a child's exit code for "skip", with the reason on stdout
NOT_STOPPED = 3
MISMATCH = 4


# ---- the chain -------------------------------------------------------------------------------

def host_inputs(seed):
    rng = np.random.default_rng(seed)
    return {'x': rng.normal(size=N).astype('f4'), 'y': rng.normal(size=N).astype('f4'),
            'w': rng.normal(size=DIM).astype('f4')}


def chain_buffers(context, inputs):
    """Inputs and every intermediate of the chain; outputs start as NaN bytes, so a launch that
    did not run cannot pass for one that did."""
    t = {name: context.tensor((v.size,), 'float32', v.tobytes()) for name, v in inputs.items()}
    for name in ('s', 'a', 'r', 'o'):
        t[name] = context.tensor((N,), 'float32', b'\xff' * (4 * N))
    return t


def refill(t, inputs):
    """New inputs and NaN outputs, in place: a graph keeps the pointers it recorded."""
    for name, v in inputs.items():
        t[name].upload(v.tobytes())
    for name in ('s', 'a', 'r', 'o'):
        t[name].upload(b'\xff' * (4 * N))


def run_chain(launch, t, stream, *, say=None):
    """launch(kernel, stream, **values) for each step of the chain; say(name) before each."""
    steps = (('scale', dict(out=t['s'], x=t['x'], alpha=0.37, n=N)),
             ('add', dict(out=t['a'], a=t['s'], b=t['y'], n=N)),
             ('rmsnorm_rows', dict(out=t['r'], x=t['a'], weight=t['w'], rows=ROWS, dim=DIM, eps=1e-5)),
             ('softmax_rows', dict(out=t['o'], x=t['r'], rows=ROWS, dim=DIM)))
    for name, values in steps:
        if say:
            say(name)
        launch(name, stream, **values)


def compare(label, ours, reference):
    """Bitwise, every intermediate; prints the first difference of each."""
    bad = []
    for name in ('s', 'a', 'r', 'o'):
        a = np.frombuffer(ours[name].download(), np.uint32)
        b = np.frombuffer(reference[name].download(), np.uint32)
        diff = np.flatnonzero(a != b)
        if diff.size:
            i = diff[0]
            bad.append(f'{label}: {name} differs at {diff.size} of {N} words, first [{i}] '
                       f'{a[i]:#010x} vs {b[i]:#010x}')
    for line in bad:
        print('MISMATCH', line)
    return not bad


# ---- children ----------------------------------------------------------------------------------

def skip(reason):
    print('SKIP:', reason)
    sys.exit(SKIP)


def setup():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    from gpu.spawn.vx_launch import VxKernel, VxStreamUnsupported
    kernels = {}
    for name in CHAIN:
        ptx = SPAWN / f'{name}.ptx'
        if not ptx.exists():
            skip(f'{ptx} not built')
        try:
            kernels[name] = VxKernel(ptx)
            kernels[name].set_stream
        except (FileNotFoundError, OSError, VxStreamUnsupported) as e:
            skip(str(e))
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0), primary=True)
    except CUDAUnavailable as e:
        skip('real GPU unavailable: ' + str(e))
    print(f'device {context.name}, driver {context.driver_version}', flush=True)
    return context, kernels


def vx_launcher(kernels):
    return lambda name, stream, **values: kernels[name](stream=stream, **values)


def say(prefix):
    return lambda name: print(f'  {prefix} {name}', flush=True)


def finish(ok, what):
    print(('OK ' if ok else 'FAILED ') + what, flush=True)
    sys.exit(0 if ok else MISMATCH)


def child_eager():
    context, kernels = setup()
    launch = vx_launcher(kernels)
    inputs = host_inputs(1)
    legacy, ours = chain_buffers(context, inputs), chain_buffers(context, inputs)
    stream = context.stream()  # non-blocking
    run_chain(launch, legacy, None, say=say('legacy stream:'))
    context.synchronize()
    ok = True
    for i in range(2):
        refill(ours, inputs)
        run_chain(launch, ours, stream, say=say(f'our stream, run {i}:'))
        stream.synchronize()  # our stream only: Vx's launches must be on it
        ok &= compare(f'eager on our stream, run {i}', ours, legacy)
    finish(ok, 'eager chain on our stream matches the legacy stream bitwise')


def capture_and_replay(context, launch, reference_launch, label):
    """Warm up, capture the chain on a stream of ours, replay it with new inputs written in
    place; each replay against reference_launch run eagerly on the legacy stream."""
    stream = context.stream()
    graph_t = chain_buffers(context, host_inputs(100))
    reference = chain_buffers(context, host_inputs(100))
    run_chain(launch, graph_t, stream, say=say('warm-up:'))  # each pub fn's first call loads its image
    stream.synchronize()
    with stream.capture() as graph:
        run_chain(launch, graph_t, stream, say=say(f'capturing ({label}):'))
    print('captured; replaying', flush=True)
    ok = True
    for i in range(REPLAYS):
        inputs = host_inputs(200 + i)
        refill(graph_t, inputs)  # outside the capture; an upload synchronizes the context
        refill(reference, inputs)
        run_chain(reference_launch, reference, None)
        graph.launch(stream)
        stream.synchronize()
        context.synchronize()
        ok &= compare(f'{label}: replay {i}', graph_t, reference)
    timing(launch, stream, graph, graph_t)
    return ok


def timing(launch, stream, graph, t, count=50):
    """Informational: an eager chain against a replay, on the host clock (glm-vx #9)."""
    import time
    for name, step in (('eager chain', lambda: run_chain(launch, t, stream)),
                       ('graph replay', lambda: graph.launch(stream))):
        step()
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(count):
            step()
        enqueued = time.perf_counter() - start
        stream.synchronize()
        total = time.perf_counter() - start
        print(f'timing {name}: {1e6 * enqueued / count:.1f} us to enqueue, {1e6 * total / count:.1f} us each '
              f'with the GPU ({count} in a row)', flush=True)


def child_capture():
    context, kernels = setup()
    ok = capture_and_replay(context, vx_launcher(kernels), vx_launcher(kernels), 'vx graph')
    finish(ok, f'{REPLAYS} replays of the captured Vx chain match eager bitwise')


def child_mixed():
    from gpu.spawn.launch import SpawnKernel
    context, kernels = setup()
    python = {name: SpawnKernel(context, SPAWN / f'{name}.ptx') for name in ('scale', 'rmsnorm_rows')}

    def launch(name, stream, **values):  # Python launches scale and rmsnorm_rows, Vx the others
        if name in python:
            python[name].launch(stream=stream, **values)
        else:
            kernels[name](stream=stream, **values)
    ok = capture_and_replay(context, launch, vx_launcher(kernels), 'mixed graph')
    finish(ok, f'{REPLAYS} replays of a graph of Python and Vx launches match the all-Vx eager chain bitwise')


def child_refused():
    """Run under VX_CUDA_SYNC=1 or VX_TIME_KERNEL: Vx must stop the program in the capture."""
    context, kernels = setup()
    t = chain_buffers(context, host_inputs(3))
    stream = context.stream()
    kernels['add'](stream=stream, out=t['a'], a=t['x'], b=t['y'], n=N)  # warm-up, not capturing
    stream.synchronize()
    print('warm-up done; capturing', flush=True)
    with stream.capture():
        kernels['add'](stream=stream, out=t['a'], a=t['x'], b=t['y'], n=N)
        print('NOT STOPPED: Vx returned inside the capture', flush=True)
    print('NOT STOPPED: the capture ended', flush=True)
    sys.exit(NOT_STOPPED)


def child_first_call():
    """No warm-up: the first call of the pub fn (which loads the image) inside the capture."""
    context, kernels = setup()
    t = chain_buffers(context, host_inputs(4))
    reference = chain_buffers(context, host_inputs(4))
    stream = context.stream()
    print('capturing the first call of add', flush=True)
    with stream.capture() as graph:
        kernels['add'](stream=stream, out=t['a'], a=t['x'], b=t['y'], n=N)
    print('captured; replaying', flush=True)
    graph.launch(stream)
    stream.synchronize()
    kernels['add'](out=reference['a'], a=reference['x'], b=reference['y'], n=N)
    context.synchronize()
    a = np.frombuffer(t['a'].download(), np.uint32)
    b = np.frombuffer(reference['a'].download(), np.uint32)
    finish(bool((a == b).all()), 'the first call inside a capture loaded its image, and the replay matches eager')


CHILDREN = {'eager': child_eager, 'capture': child_capture, 'mixed': child_mixed,
            'refused': child_refused, 'first-call': child_first_call}


# ---- pytest side ------------------------------------------------------------------------------

def run_child(scenario, **env):
    environment = {k: v for k, v in os.environ.items() if k not in ('VX_CUDA_SYNC', 'VX_TIME_KERNEL')}
    environment.update(env)
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT), environment.get('PYTHONPATH')]))
    try:
        proc = subprocess.run([sys.executable, '-m', 'gpu.test_spawn_vx_graph', scenario], cwd=ROOT,
                              env=environment, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        pytest.fail(f'{scenario} {env}: timed out\n--- stdout\n{e.stdout}\n--- stderr\n{e.stderr}')
    report = (f'===== {scenario} {env or ""}: exit {proc.returncode}\n--- stdout\n{proc.stdout}'
              f'--- stderr (Vx messages)\n{proc.stderr}===== end {scenario}')
    print(report, file=sys.stderr, flush=True)
    if proc.returncode == SKIP:
        pytest.skip(proc.stdout.strip().splitlines()[-1])
    return proc, report


@pytest.mark.parametrize('env', [{}, {'VX_CUDA_SYNC': '1'}], ids=['async', 'VX_CUDA_SYNC'])
def test_eager_on_our_stream_matches_the_legacy_stream(env):
    proc, report = run_child('eager', **env)
    assert proc.returncode == 0, report


def test_captured_vx_chain_replays_bitwise():
    proc, report = run_child('capture')
    assert proc.returncode == 0, report


def test_graph_of_python_and_vx_launches():
    proc, report = run_child('mixed')
    assert proc.returncode == 0, report


@pytest.mark.parametrize('env,reason', [({'VX_CUDA_SYNC': '1'}, 'VX_CUDA_SYNC=1 waits after each launch'),
                                        ({'VX_TIME_KERNEL': '1'}, 'VX_TIME_KERNEL waits for each kernel')],
                         ids=['VX_CUDA_SYNC', 'VX_TIME_KERNEL'])
def test_vx_stops_on_a_wait_during_capture(env, reason):
    proc, report = run_child('refused', **env)
    assert proc.returncode not in (0, NOT_STOPPED), 'Vx did not stop the program\n' + report
    assert 'records a CUDA graph' in proc.stderr and reason in proc.stderr, report


def test_vx_stops_on_a_first_call_inside_a_capture():
    proc, report = run_child('first-call')
    assert proc.returncode != MISMATCH, 'a wrong answer\n' + report
    assert proc.returncode != 0, 'Vx loaded its image during the capture instead of stopping\n' + report
    assert 'Call the function once before the recording' in proc.stderr, report


def test_every_library_has_vx_cuda_set_stream():
    """No GPU needed: each lib<kernel>.so resolves vx_cuda_set_stream (Vx PR #2016)."""
    from gpu.spawn.vx_launch import VxKernel
    libraries = sorted(SPAWN.glob('lib*.so'))
    if not libraries:
        pytest.skip(f'no libraries in {SPAWN} (gpu/spawn/build.sh with VX_DISPATCH_LIB)')
    for library in libraries:
        ptx = SPAWN / (library.stem[3:] + '.ptx')
        try:
            kernel = VxKernel(ptx)
        except OSError as e:
            pytest.skip(f'cannot load {library.name} here: {e}')
        kernel.set_stream  # raises VxStreamUnsupported naming PR #2016


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in CHILDREN:
        sys.exit(f'usage: python -m gpu.test_spawn_vx_graph {{{",".join(CHILDREN)}}}')
    CHILDREN[sys.argv[1]]()
