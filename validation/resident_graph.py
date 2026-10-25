"""ResidentGLM graph replay (graph=True, #26) against the per-launch path, on a real GPU (#28).

Both run with the routed experts on the device (device_experts=True, every expert uploaded), one
after the other in their own context. Reports whether the logits are bitwise equal (the
expectation: the same kernels in the same order), and per token:

  * wall-clock ms per token of each path, steady state (a warm pass, reset, then the timed pass);
  * for the per-launch path, the host time to enqueue one step's layers and the GPU time of those
    launches (CUDA events), and the kernel launches per step;
  * for graph replay, the GPU time of one replay of the step graph (CUDA events over --replays).

  python -m validation.resident_graph --output run/graph-f32-small.json [--format f32|int4|fp8|bf16]
      [--geometry small|full] [--layers 4] [--tokens 48]

A single run on random weights; a measurement of this script on this machine, not a benchmark.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel
from validation.gpu_vs_official import CONFIG, full_geometry, random_weights
from validation.resident_compare import bf16_source, fp8_source, int4_source


def events_ms(resident, enqueue, repeats):
    """Median host ms to enqueue and GPU ms (events) of `enqueue()` on the resident stream."""
    ctx, stream = resident.ctx, resident.stream
    host, gpu = [], []
    for _ in range(repeats):
        start, end = ctx.event(), ctx.event()
        stream.synchronize()
        start.record(stream)
        t0 = time.perf_counter()
        enqueue()
        host.append((time.perf_counter() - t0) * 1e3)
        end.record(stream)
        end.synchronize()
        gpu.append(start.elapsed_ms(end))
        start.close()
        end.close()
    return statistics.median(host), statistics.median(gpu)


def measure(model, tokens, *, graph, replays):
    from gpu.resident import ResidentGLM
    resident = ResidentGLM(model, max_tokens=len(tokens) + 1, device_experts=True, graph=graph)
    try:
        t0 = time.perf_counter()
        for t in tokens:  # uploads every weight (and captures the step graph at the second token)
            resident.forward(t)
        warm_s = time.perf_counter() - t0
        resident.reset()
        start = time.perf_counter()
        got = np.stack([resident.forward(t) for t in tokens])
        report = dict(ms_per_token=(time.perf_counter() - start) / len(tokens) * 1e3, warm_pass_s=warm_s,
                      weights_gib=resident.weight_bytes / 1024 ** 3)
        position = resident.position - 1  # rerun the last step's layers in place: same rows, same values
        if graph:
            graph_exec = resident._step_graph
            host, gpu = events_ms(resident, lambda: [graph_exec.launch(resident.stream) for _ in range(replays)], 5)
            report.update(replay_host_ms=host / replays, replay_gpu_ms=gpu / replays)
        else:
            before = sum(resident.launches.values())
            resident._layers(position)
            resident.stream.synchronize()
            report['launches_per_step'] = sum(resident.launches.values()) - before
            host, gpu = events_ms(resident, lambda: resident._layers(position), 20)
            report.update(step_enqueue_host_ms=host, step_gpu_ms=gpu)
        return got, report
    finally:
        resident.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--format', choices=['f32', 'int4', 'fp8', 'bf16'], default='f32')
    p.add_argument('--geometry', choices=['small', 'full'], default='small')
    p.add_argument('--layers', type=int, default=4, help='layers for --geometry full')
    p.add_argument('--tokens', type=int, default=48)
    p.add_argument('--replays', type=int, default=50)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    config = CONFIG if args.geometry == 'small' else full_geometry(args.layers)
    weights = random_weights(config, args.seed)
    for w in weights.values():
        w.flags.writeable = False
    source = weights if args.format == 'f32' else {'int4': int4_source, 'fp8': fp8_source, 'bf16': bf16_source}[args.format](weights)
    model = GlmMoeDsaModel(config, source, NumpyBackend())
    tokens = [int(t) for t in np.random.default_rng(100 + args.seed).integers(0, config['vocab_size'], args.tokens)]
    want, eager = measure(model, tokens, graph=False, replays=args.replays)
    got, graph = measure(model, tokens, graph=True, replays=args.replays)
    report = dict(
        format=args.format, geometry=args.geometry, layers=config['num_hidden_layers'], hidden=config['hidden_size'],
        tokens=len(tokens), bitwise_equal=bool(np.array_equal(got, want)),
        max_abs_diff=float(np.abs(got - want).max()), top1_equal=bool(np.array_equal(got.argmax(-1), want.argmax(-1))),
        per_launch=eager, graph=graph, speedup=eager['ms_per_token'] / graph['ms_per_token'],
        note='single run, random weights, every routed expert resident (device_experts); ms_per_token includes the '
             'step-input upload, the final norm and lm_head and the logits download; not a benchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['bitwise_equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
