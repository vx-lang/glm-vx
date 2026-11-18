"""gpu/spawn/tp_all_reduce on real GPUs (#34, #62): the one-shot cross-rank sum over CUDA IPC.

Starts --ranks processes, each seeing every GPU (CUDA IPC maps another process's memory only on a GPU this
process can see). With --gpus N (default: as many as ranks) rank r runs on GPU r % N, so
--gpus 1 puts every rank on one GPU (the IPC handles then go through files, where NCCL would refuse
two ranks on one device; with one rank a GPU, through files too, so no NCCL is needed). Each rank:
  * --calls calls eagerly, each with new partials: every rank's x must be, bit for bit, NumPy's
    ((p0 + p1) + p2) + .. in float32 (the order ResidentGLM._tp_sum adds the gathered partials in);
  * the same through a CUDA graph of --graph-calls calls, replayed --replays times with new inputs
    (the epochs and flags carry over: no reset between replays);
  * the time of one call, from a graph of --time-calls calls (wall time / calls).
Ranks sharing a GPU are separate processes, so their kernels time-slice rather than run together: a
rank's blocks spin until the others get the GPU. That checks the bits, not the time; time on one rank a GPU.
Needs the gpu/spawn build of tp_all_reduce (gpu/spawn/build.sh, a vxc with core::sync::atomic).

--norm checks and times gpu/spawn/tp_sum_rmsnorm instead (#63, docs/TP-SERVING-PLAN.md item 3) in
--norm-blocks blocks: the same partials summed with a residual and RMSNorm (residual and weight the same on
every rank, as a layer's are): residual_out must be NumPy's rank-order sum + residual and out its RMSNorm in
the hand kernels' order at 1024 threads (residual_rmsnorm8's), bit for bit, eager and under graph replay.

  python -m validation.tp_all_reduce_check --ranks 2 --gpus 1 --output run/tp_all_reduce.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def partials(seed, ranks, n):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=n).astype('f4') for _ in range(ranks)]


def rank_order_sum(parts):
    total = parts[0].copy()
    for p in parts[1:]:
        total = total + p
    return total


def norm_inputs(seed, n):
    """The residual and RMSNorm weight of a --norm call, the same on every rank."""
    rng = np.random.default_rng(10 ** 6 + seed)
    return rng.normal(size=n).astype('f4'), rng.normal(size=n).astype('f4')


def norm_reference(seed, ranks, n, eps):
    from gpu.test_spawn_vx_host import block_partials, block_sum
    residual, weight = norm_inputs(seed, n)
    v = rank_order_sum(partials(seed, ranks, n)) + residual
    total = block_sum(block_partials((v * v)[None], 1024), 1024)
    inv = np.float32(1) / np.sqrt(total / np.float32(n) + np.float32(eps))
    return v, (v * inv[:, None] * weight)[0]


def norm_main(rank, ranks, n, calls, graph_calls, replays, time_calls, handles, ptx_dir, device, blocks, eps=1e-5):
    """--norm: tp_sum_rmsnorm's bits against NumPy, then its time a call."""
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    from gpu.tp import PeerSum, handles_by_files
    context = CUDAContext(device, minimum_compute_capability=(8, 0))
    stream = context.stream()
    peers = PeerSum(context, rank, ranks, n, handles_by_files(handles, rank, ranks), norm_blocks=blocks)
    kernel = SpawnKernel(context, Path(ptx_dir) / 'tp_sum_rmsnorm.ptx')
    failures = []

    def tensors():
        return [context.tensor((n,)) for _ in range(5)]  # x, residual, weight, out, residual_out

    def load(t, seed):
        residual, weight = norm_inputs(seed, n)
        for tensor, value in zip(t, (partials(seed, ranks, n)[rank], residual, weight)):
            tensor.upload(value.tobytes())

    def launch(t):
        kernel.launch(stream=stream, **peers.norm_arguments(t[0], t[1], t[2], t[3], t[4], eps))

    def check(name, t, seed):
        want_residual, want_out = norm_reference(seed, ranks, n, eps)
        for what, tensor, want in (('residual_out', t[4], want_residual), ('out', t[3], want_out)):
            got = np.frombuffer(tensor.download(), 'f4')
            if not np.array_equal(got.view('u4'), want.view('u4')):
                failures.append(f'{name} {what}: {int(np.sum(got.view("u4") != want.view("u4")))} of {n} differ')

    t = tensors()
    for call in range(calls):
        load(t, call)
        launch(t)
        stream.synchronize()
        check(f'eager call {call}', t, call)
    sets = [tensors() for _ in range(graph_calls)]
    with stream.capture() as graph:
        for u in sets:
            launch(u)
    for replay in range(replays):
        for i, u in enumerate(sets):
            load(u, 1000 + 100 * replay + i)
        graph.launch(stream)
        stream.synchronize()
        for i, u in enumerate(sets):
            check(f'replay {replay} call {i}', u, 1000 + 100 * replay + i)
    timing = None
    if time_calls:
        with stream.capture() as timed:
            for _ in range(time_calls):
                launch(t)
        timed.launch(stream)
        stream.synchronize()
        start = time.perf_counter()
        timed.launch(stream)
        stream.synchronize()
        timing = (time.perf_counter() - start) / time_calls * 1e6
    print(json.dumps(dict(rank=rank, kernel='tp_sum_rmsnorm', blocks=blocks, failures=failures[:20], failed=len(failures),
                          us_per_call=timing)), flush=True)
    return 1 if failures else 0


def rank_main(rank, ranks, n, calls, graph_calls, replays, time_calls, handles, ptx_dir, device):
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    from gpu.tp import PeerSum, handles_by_files
    context = CUDAContext(device, minimum_compute_capability=(8, 0))
    stream = context.stream()
    peers = PeerSum(context, rank, ranks, n, handles_by_files(handles, rank, ranks))
    kernel = SpawnKernel(context, Path(ptx_dir) / 'tp_all_reduce.ptx')
    failures = []

    def check(name, x, seed):
        want = rank_order_sum(partials(seed, ranks, n))
        got = np.frombuffer(x.download(), 'f4')
        if not np.array_equal(got.view('u4'), want.view('u4')):
            bad = int(np.sum(got.view('u4') != want.view('u4')))
            failures.append(f'{name}: {bad} of {n} differ')

    x = context.tensor((n,))
    for call in range(calls):
        x.upload(partials(call, ranks, n)[rank].tobytes())
        kernel.launch(stream=stream, **peers.arguments(x))
        stream.synchronize()
        check(f'eager call {call}', x, call)

    xs = [context.tensor((n,)) for _ in range(graph_calls)]
    with stream.capture() as graph:
        for t in xs:
            kernel.launch(stream=stream, **peers.arguments(t))
    for replay in range(replays):
        for i, t in enumerate(xs):
            t.upload(partials(1000 + 100 * replay + i, ranks, n)[rank].tobytes())
        graph.launch(stream)
        stream.synchronize()
        for i, t in enumerate(xs):
            check(f'replay {replay} call {i}', t, 1000 + 100 * replay + i)

    timing = None
    if time_calls:
        with stream.capture() as timed:
            for _ in range(time_calls):
                kernel.launch(stream=stream, **peers.arguments(x))
        timed.launch(stream)
        stream.synchronize()
        start = time.perf_counter()
        timed.launch(stream)
        stream.synchronize()
        timing = (time.perf_counter() - start) / time_calls * 1e6
    print(json.dumps(dict(rank=rank, failures=failures[:20], failed=len(failures), us_per_call=timing)), flush=True)
    return 1 if failures else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ranks', type=int, default=2)
    p.add_argument('--gpus', type=int)
    p.add_argument('--n', type=int, default=6144)
    p.add_argument('--calls', type=int, default=50)
    p.add_argument('--graph-calls', type=int, default=157)  # a GLM-5.3 decode step's sums: 2 x 78 + 1
    p.add_argument('--replays', type=int, default=5)
    p.add_argument('--time-calls', type=int, default=1000)
    p.add_argument('--ptx-dir', default=str(Path(__file__).resolve().parents[1] / 'gpu' / 'build' / 'spawn'))
    p.add_argument('--output', type=Path)
    p.add_argument('--norm', action='store_true', help='check and time tp_sum_rmsnorm (the sum with the RMSNorm after it)')
    p.add_argument('--norm-blocks', type=int, default=8)
    p.add_argument('--rank', type=int, help=argparse.SUPPRESS)
    p.add_argument('--handles', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.rank is not None and args.norm:
        return norm_main(args.rank, args.ranks, args.n, args.calls, args.graph_calls, args.replays, args.time_calls,
                         args.handles, args.ptx_dir, args.rank % (args.gpus or args.ranks), args.norm_blocks)
    if args.rank is not None:
        return rank_main(args.rank, args.ranks, args.n, args.calls, args.graph_calls, args.replays, args.time_calls,
                         args.handles, args.ptx_dir, args.rank % (args.gpus or args.ranks))
    if args.output is not None and args.output.exists():
        p.error('output exists; preserve evidence')
    gpus = args.gpus or args.ranks
    with tempfile.TemporaryDirectory() as handles:
        processes = [subprocess.Popen([sys.executable, '-m', 'validation.tp_all_reduce_check', '--rank', str(r),
                                       '--ranks', str(args.ranks), '--n', str(args.n), '--calls', str(args.calls),
                                       '--graph-calls', str(args.graph_calls), '--replays', str(args.replays),
                                       '--time-calls', str(args.time_calls), '--ptx-dir', args.ptx_dir,
                                       '--handles', handles, '--gpus', str(gpus), '--norm-blocks', str(args.norm_blocks)]
                                      + (['--norm'] if args.norm else []),
                                      stdout=subprocess.PIPE, text=True)
                     for r in range(args.ranks)]
        outputs = [process.communicate(timeout=1800)[0] for process in processes]
    results = []
    for r, (process, output) in enumerate(zip(processes, outputs)):
        sys.stdout.write(output)
        lines = [line for line in output.splitlines() if line.startswith('{')]
        results.append(json.loads(lines[-1]) if lines else dict(rank=r, failures=[f'exit {process.returncode}'], failed=1))
    ok = all(not result['failed'] for result in results) and all(pr.returncode == 0 for pr in processes)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(ranks=args.ranks, gpus=gpus, n=args.n, norm=args.norm, norm_blocks=args.norm_blocks,
                                               ok=ok, results=results), indent=2) + '\n')
    print('tp_all_reduce_check:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
