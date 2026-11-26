"""One tensor-parallel rank of GLM-5.3 on one GPU, a few layers deep: where prefill and decode spend their time, layer by
layer and kernel by kernel, for iterating on kernels before an 8-GPU run (#63).

The model is GLM-5.3's geometry cut to --layers layers (4: layers 0-2 dense with full indexers, layer 3 MoE with a
shared indexer), index_topk 2048, random weights in the checkpoint's formats (block FP8 Linears; lm_head, each MoE
router's gate and each indexer's weights_proj in bfloat16), split as rank 0 of 8 by gpu/tp.ShardedModel: every kernel
gets one rank's shapes under --tp 8. The two all-reduces a layer are local no-ops (SoloCollective logs their sizes; NCCL
time is estimated, not measured). ResidentGLM runs with gpu/tp_serve.py's OPTIONS plus the served options of
docs/TP-SERVING-PLAN.md (SERVED below), and --options' JSON over them.

Cases:
  * W1: a prompt of --w1 tokens at position 0 (prefill_chunk), --reps timed calls, then one with CUDA events around
    each layer's attention and FFN (GPU ms, host enqueue ms, launches by kernel name), then a cProfile of one call;
  * W3-like: a 2048-token chunk at each --long position (the chunks before it prefilled first, as a server would),
    timed and instrumented the same way;
  * decode: --decode steps after the W1 prompt (each forward() with its logits downloaded, as tp_real times them), the
    first two dropped (graph capture);
  * verify (#63, item 8: MTP's verify step): forward_many(greedy=True) of each --verify token count at the position
    after the decode steps, --reps timed steps a count after the graph's capture (each step's tokens downloaded, so
    it is done; its inputs uploaded, as forward_many does without MTP's device inputs). The step against a decode
    step says what a round of drafts must save; --options '{"many_moe": "grouped"}' against the default is the A/B
    for the routed experts read once.
Each case's last logits are saved (w1-logits.npy, long-<start>-logits.npy, decode-logits.npy), so two builds or two
option sets can be compared on the same random model with --compare. summary.txt adds estimates for all 78 layers
(3 dense, 18 MoE with a full indexer, 57 MoE with a shared one) from the instrumented parts.

This is for diagnosis, not a claim: TTFT and TPOT are measured end to end on the 8 GPUs, through the server and
SGLang's client (docs/TP-SERVING-PLAN.md). Random weights also exaggerate some sensitivities: near-uniform indexer
scores put many keys at the top-k cutoff, so a small change in the indexer's inputs moves a W3-like chunk's logits
far more than the real checkpoint's would. Check numerics that matter on the real model.

  PYTHONPATH=$PWD python -m validation.rank_profile --output run/a [--layers 4] [--long 2048] [--decode 32]
      [--verify 1,2,3,4] [--options '{"moe_mma": "vx16"}'] [--nsys | --nsys-long | --nsys-verify N]
  python -m validation.rank_profile --compare run/a run/b
  validation/rank_profile_ab.sh REPO_A REPO_B OUT [ROUNDS]   # two builds, alternating rounds, then --compare
"""
import argparse
import cProfile
import ctypes
import io
import json
import pstats
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

SERVED = dict(moe_down2=True, fuse_router=True, device_decode=True, moe_mma=True, fuse_mla_q=True, prefill_rows=2048,
              prefill_gemm='cublaslt', prefill_sum='nccl', prefill_moe='grouped', prefill_attention='mha_tc',
              topk_bitmap=True)
# GLM-5.3: 78 layers; 0-2 dense with full indexers, 18 MoE layers with full indexers, 57 MoE layers with shared ones.
LAYERS = dict(dense=3, moe_full=18, moe_shared=57)


class SoloCollective:
    """Rank 0 of `ranks`, alone on one GPU: all_reduce_sum keeps this rank's partial sums, all_gather puts this
    rank's part first. Each call is logged as (operation, elements, dtype)."""

    def __init__(self, ranks):
        self.ranks, self.calls = ranks, []

    def all_reduce_sum(self, tensor, *, stream, out=None):
        self.calls.append(('all_reduce_sum', int(tensor.size), str(tensor.dtype)))
        if out is not None and out is not tensor:
            out.copy_from(tensor, stream=stream)
        return tensor if out is None else out

    def all_gather(self, tensor, out, *, stream):
        self.calls.append(('all_gather', int(tensor.size), str(tensor.dtype)))
        out.view((tensor.size,)).copy_from(tensor, stream=stream)


class Recorder:
    """Logs every kernel launch, cuBLASLt matmul (with m x n x k) and device copy or memset while `on`."""

    def __init__(self):
        self.on, self.log = False, []
        from gpu import cublaslt, runtime
        rec = self
        launch = runtime.Kernel.launch

        def logged_launch(kernel, *a, **kw):
            if rec.on:
                rec.log.append(('kernel', kernel.name))
            return launch(kernel, *a, **kw)
        runtime.Kernel.launch = logged_launch
        for name in ('Fp8BlockMatmul', 'Bf16Matmul'):  # Bf16Matmul: builds that have it
            cls = getattr(cublaslt, name, None)
            if cls is None:
                continue
            call = cls.__call__

            def logged_matmul(mm, *a, _call=call, _kind=name, **kw):
                if rec.on:
                    rec.log.append(('cublaslt', f'{_kind} {mm.m}x{mm.n}x{mm.k}'))
                return _call(mm, *a, **kw)
            cls.__call__ = logged_matmul

    def driver(self, driver):
        call, rec = driver.call, self

        def logged(name, *a, **kw):
            if rec.on and name.startswith(('cuMemcpy', 'cuMemset')):
                rec.log.append(('op', name))
            return call(name, *a, **kw)
        driver.call = logged


def checkpoint_formats(weights):
    """validation/resident_compare.fp8_source's block-FP8 Linears, with the 16-bit weights of GLM-5.3's checkpoint in
    bfloat16 as it stores them (lm_head, each MoE router's gate, each indexer's weights_proj: docs/model.md), so
    ResidentGLM takes the paths the served model takes (a float32 gate would route a token at a time)."""
    from glm_vx.quant import BF16Weights, BlockFP8Weights
    from validation.write_checkpoint import UNQUANTIZED, fp8_quantize, to_bf16_bits
    linears = {name[:-len('.weight')]: fp8_quantize(w) for name, w in weights.items()
               if w.ndim == 2 and name.endswith('.weight') and not any(u in name for u in UNQUANTIZED)}
    rest = {n: w for n, w in weights.items() if n[:-len('.weight')] not in linears}
    halves = {}
    for name in [n for n in rest if rest[n].ndim == 2 and (n == 'lm_head.weight' or n.endswith('.mlp.gate.weight')
                                                           or n.endswith('.indexer.weights_proj.weight'))]:
        halves[name[:-len('.weight')]] = to_bf16_bits(rest.pop(name))
    return BlockFP8Weights(BF16Weights(rest, halves), linears)


def build(layers, seed, max_tokens):
    from glm_vx.backend import NumpyBackend
    from glm_vx.model import GlmMoeDsaModel
    from validation.gpu_vs_official import full_geometry, random_weights
    config = dict(full_geometry(layers, topk=2048), max_position_embeddings=max_tokens)
    return GlmMoeDsaModel(config, checkpoint_formats(random_weights(config, seed)), NumpyBackend())


def timed_chunk(r, tokens, start):
    r.position = start
    t = time.perf_counter()
    r.prefill_chunk(tokens)
    r.ctx.synchronize()
    return time.perf_counter() - t


def layered_chunk(r, rec, tokens, start):
    """One prefill_chunk with events at each layer's attention and FFN; returns per-part GPU ms, host ms, launches."""
    parts = []

    def wrap(name, method):
        def wrapped(layer, *a, **kw):
            e0 = r.ctx.event()
            e0.record(r.stream)
            n0, t0 = len(rec.log), time.perf_counter()
            result = method(layer, *a, **kw)
            host = time.perf_counter() - t0
            e1 = r.ctx.event()
            e1.record(r.stream)
            parts.append(dict(part=name, layer=layer, events=(e0, e1), host_ms=1e3 * host, launches=rec.log[n0:]))
            return result
        return wrapped
    r._prefill_attention = wrap('attention', r._prefill_attention)
    r._prefill_feed_forward = wrap('ffn', r._prefill_feed_forward)
    try:
        r.position = start
        begin, end = r.ctx.event(), r.ctx.event()
        rec.log.clear()
        rec.on = True
        t = time.perf_counter()
        begin.record(r.stream)
        r.prefill_chunk(tokens)
        end.record(r.stream)
        host_total = time.perf_counter() - t
        r.ctx.synchronize()
        wall = time.perf_counter() - t
        rec.on = False
        launches = list(rec.log)
    finally:
        rec.on = False
        del r._prefill_attention, r._prefill_feed_forward
    for p in parts:
        e0, e1 = p.pop('events')
        p['gpu_ms'] = e0.elapsed_ms(e1)
        kinds = Counter(k for k, _ in p['launches'])
        p['kernels'], p['cublaslt'], p['ops'] = kinds['kernel'], kinds['cublaslt'], kinds['op']
        p['by_name'] = Counter(n for _, n in p['launches']).most_common()
        del p['launches']
    return dict(wall_ms=1e3 * wall, host_enqueue_ms=1e3 * host_total, gpu_span_ms=begin.elapsed_ms(end),
                launches=len(launches), launch_kinds=dict(Counter(k for k, _ in launches)), parts=parts,
                by_name=Counter(n for _, n in launches).most_common())


def estimate(parts, key):
    """All 78 layers from a 4-layer case's parts: attention as layers 0-2 (full indexers) or 3 (shared), the FFN as
    layers 0-2 (dense) or 3 (MoE). None unless every part of layers 0-3 is there."""
    by = {(p['layer'], p['part']): p[key] for p in parts}
    if not all((l, q) in by for l in range(4) for q in ('attention', 'ffn')):
        return None
    full = statistics.mean(by[(l, 'attention')] for l in range(3))
    dense = statistics.mean(by[(l, 'ffn')] for l in range(3))
    return (LAYERS['dense'] * (full + dense) + LAYERS['moe_full'] * (full + by[(3, 'ffn')])
            + LAYERS['moe_shared'] * (by[(3, 'attention')] + by[(3, 'ffn')]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--output', type=Path)
    p.add_argument('--compare', nargs=2, type=Path, metavar=('A', 'B'), help='compare two runs\' outputs instead')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--w1', type=int, default=1000)
    p.add_argument('--long', default='2048', help='start positions of the W3-like 2048-token chunks ("" none)')
    p.add_argument('--decode', type=int, default=32, help='decode steps timed after the W1 prompt (0 none)')
    p.add_argument('--verify', default='1,2,3,4', help='forward_many token counts timed after the decode steps ("" none)')
    p.add_argument('--options', default='{}', help="JSON over ResidentGLM's options, e.g. '{\"moe_mma\": \"vx16\"}'")
    p.add_argument('--nsys', action='store_true', help='one W1 call between cuProfilerStart and cuProfilerStop')
    p.add_argument('--nsys-long', action='store_true', help='the first --long chunk instead of W1')
    p.add_argument('--nsys-verify', type=int, default=0, help='one verify step of this many tokens instead of W1')
    args = p.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.output is None:
        p.error('--output is required (or --compare A B)')
    args.output.mkdir(parents=True, exist_ok=True)
    import gc
    from gpu.resident import ResidentGLM
    from gpu.tp import TensorParallel
    from gpu.tp_serve import OPTIONS
    options = {**OPTIONS, **SERVED, **json.loads(args.options)}
    longs = [int(s) for s in args.long.split(',') if s]
    verifies = [int(s) for s in args.verify.split(',') if s]
    max_tokens = max([args.w1 + args.decode + 1 + max(verifies + [args.nsys_verify, 0])] + [s + 2048 for s in longs])
    rec = Recorder()
    t0 = time.time()
    model = build(args.layers, args.seed, max_tokens)
    print(f'model ready in {time.time() - t0:.0f} s', flush=True)
    solo = SoloCollective(8)
    r = ResidentGLM(model, max_tokens=max_tokens, tp=TensorParallel(0, 8, solo), **options)
    rec.driver(r.ctx.driver)
    tokens = [int(t) for t in np.random.default_rng(7).integers(0, model.config['vocab_size'], max_tokens)]
    report = dict(layers=args.layers, options=options, rank='0 of 8 (ShardedModel); sums are no-ops')
    gc.collect()
    gc.disable()  # a collection inside a timed loop once doubled a kernel's time (2026-11-20)
    try:
        w1 = tokens[:args.w1]
        for _ in range(2):
            timed_chunk(r, w1, 0)
        solo.calls.clear()
        walls = [1e3 * timed_chunk(r, w1, 0) for _ in range(args.reps)]
        calls = list(solo.calls[:len(solo.calls) // args.reps])
        report['w1'] = dict(tokens=args.w1, start=0, wall_ms=walls, wall_median_ms=statistics.median(walls),
                            collective_calls=calls, layered=layered_chunk(r, rec, w1, 0))
        print('w1 wall ms', [round(w) for w in walls], flush=True)
        r.position = 0
        np.save(args.output / 'w1-logits.npy', np.asarray(r.prefill_chunk(w1, logits=True), np.float32))
        if args.decode:
            steps, logits = [], None
            for t in tokens[args.w1:args.w1 + args.decode]:
                start = time.perf_counter()
                out = r.forward(t)  # the downloaded logits: the step is done
                steps.append(1e3 * (time.perf_counter() - start))
                logits = out if logits is None else logits
            np.save(args.output / 'decode-logits.npy', np.asarray(logits, np.float32))
            kept = steps[2:] if len(steps) > 2 else steps  # the first steps capture graphs
            report['decode'] = dict(steps=args.decode, start=args.w1, step_ms=steps,
                                    step_median_ms=statistics.median(kept), step_min_ms=min(kept))
            print(f"decode step ms median {report['decode']['step_median_ms']:.3f}", flush=True)
        if verifies:
            base = args.w1 + args.decode
            report['verify'] = dict(start=base, counts={})
            for n in verifies:
                step = tokens[base:base + n]
                before = Counter(r.launches)
                for i in range(3):  # eager, the graph's capture, a replay
                    r.position = base
                    r.forward_many(step, greedy=True)
                    if i == 0:  # what one step launches (a replay counts none)
                        eager = Counter(r.launches) - before
                walls = []
                for _ in range(args.reps):
                    r.position = base
                    start = time.perf_counter()
                    r.forward_many(step, greedy=True)  # the tokens downloaded: the step is done
                    walls.append(1e3 * (time.perf_counter() - start))
                report['verify']['counts'][str(n)] = dict(step_ms=walls, step_median_ms=statistics.median(walls),
                                                          step_min_ms=min(walls), launches_a_step=dict(eager),
                                                          kernels_a_step=sum(eager.values()))
                print(f'verify {n} tokens: step ms median {statistics.median(walls):.3f}', flush=True)
            r.position = base
        prof = cProfile.Profile()
        r.position = 0
        prof.enable()
        r.prefill_chunk(w1)
        r.ctx.synchronize()
        prof.disable()
        for key in ('tottime', 'cumulative'):
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats(key).print_stats(40)
            (args.output / f'cprofile-w1-{key}.txt').write_text(s.getvalue())
        report['long'] = []
        filled = 0
        for start in longs:
            while filled < start:  # the chunks before, as a server would prefill them
                n = min(2048, start - filled)
                timed_chunk(r, tokens[filled:filled + n], filled)
                filled += n
            chunk = tokens[start:start + 2048]
            timed_chunk(r, chunk, start)  # warm
            wall = 1e3 * timed_chunk(r, chunk, start)
            report['long'].append(dict(tokens=2048, start=start, wall_ms=wall, layered=layered_chunk(r, rec, chunk, start)))
            r.position = start
            np.save(args.output / f'long-{start}-logits.npy', np.asarray(r.prefill_chunk(chunk, logits=True), np.float32))
            filled = start + 2048
            print(f'long start {start} wall ms {wall:.0f}', flush=True)
        if args.nsys or args.nsys_long or args.nsys_verify:
            if args.nsys_long:  # the first long chunk again, its chunks before still in the caches
                timed_chunk(r, tokens[longs[0]:longs[0] + 2048], longs[0])
            base, step = args.w1 + args.decode, tokens[args.w1 + args.decode:args.w1 + args.decode + args.nsys_verify]
            for _ in range(3 if args.nsys_verify else 0):  # eager, capture, a replay: the profiled step is a replay
                r.position = base
                r.forward_many(step, greedy=True)
            cuda = ctypes.CDLL('libcuda.so.1')
            with r.ctx._activate():
                cuda.cuProfilerStart()
            if args.nsys_verify:
                r.position = base
                r.forward_many(step, greedy=True)
            elif args.nsys_long:
                timed_chunk(r, tokens[longs[0]:longs[0] + 2048], longs[0])
            else:
                timed_chunk(r, w1, 0)
            r.ctx.synchronize()
            with r.ctx._activate():
                cuda.cuProfilerStop()
    finally:
        gc.enable()
        (args.output / 'rank_profile.json').write_text(json.dumps(report, indent=1, default=str) + '\n')
        r.close()
    summarize(report, args.output)


def summarize(report, out):
    L = []
    w = report['w1']
    lay = w['layered']
    L.append(f"W1: {w['tokens']} tokens at 0, rank 0 of 8, {report['layers']} layers. wall median {w['wall_median_ms']:.1f} ms "
             f"{[round(x, 1) for x in w['wall_ms']]}")
    L.append(f"  instrumented call: wall {lay['wall_ms']:.1f} ms, host enqueue {lay['host_enqueue_ms']:.1f} ms, GPU span "
             f"{lay['gpu_span_ms']:.1f} ms, launches {lay['launches']} {lay['launch_kinds']}")
    L.append(f"  collective calls a chunk: {Counter((c[0], c[1], c[2]) for c in w['collective_calls']).most_common()}")
    if 'decode' in report:
        d = report['decode']
        L.append(f"decode: {d['steps']} steps after the W1 prompt, {report['layers']} layers: median {d['step_median_ms']:.3f} ms, "
                 f"min {d['step_min_ms']:.3f} ms (first two dropped)")
    for n, v in report.get('verify', {}).get('counts', {}).items():
        ratio = f", {v['step_median_ms'] / report['decode']['step_median_ms']:.2f}x a decode step" if 'decode' in report else ''
        L.append(f"verify: forward_many of {n} tokens at {report['verify']['start']}, {report['layers']} layers: median "
                 f"{v['step_median_ms']:.3f} ms, min {v['step_min_ms']:.3f} ms{ratio}; {v['kernels_a_step']} launches")
    for case in [w] + report.get('long', []):
        lay = case['layered']
        L.append(f"\nchunk of {case['tokens']} at {case['start']}: per part, GPU ms (events) / host enqueue ms / launches "
                 f"(kernels, cuBLASLt, copies)")
        for prt in lay['parts']:
            L.append(f"  layer {prt['layer']} {prt['part']:9} GPU {prt['gpu_ms']:8.2f}  host {prt['host_ms']:8.2f}  "
                     f"launches {prt['kernels']:5} {prt['cublaslt']:3} {prt['ops']:4}  top: "
                     + ', '.join(f'{n} {c}' for n, c in prt['by_name'][:4]))
        if case is not w:
            L.append(f"  wall {case['wall_ms']:.1f} ms; instrumented: host enqueue {lay['host_enqueue_ms']:.1f}, GPU span "
                     f"{lay['gpu_span_ms']:.1f}, launches {lay['launches']}")
        gpu, host = estimate(lay['parts'], 'gpu_ms'), estimate(lay['parts'], 'host_ms')
        if gpu is not None:
            L.append(f"  estimate for 78 layers (3 dense, 18 MoE full indexer, 57 MoE shared): GPU {gpu:.0f} ms, "
                     f"host enqueue {host:.0f} ms")
    width = Counter(c[1] for c in w['collective_calls'] if c[0] == 'all_reduce_sum')
    if width:
        mb = max(width) * 4 / 1e6
        # NCCL ring all-reduce on 8 GPUs: 2 (n - 1) / n of the bytes over the bus; 350 GB/s bus bandwidth assumed.
        per = 2 * 7 / 8 * mb / 350e3 * 1e3
        L.append(f"\nNCCL, assumed: 2 all-reduces a layer of {mb:.1f} MB (W1) at 350 GB/s bus bandwidth: {per:.3f} ms each, "
                 f"{2 * 78 * per:.0f} ms for 78 layers")
    (out / 'summary.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def compare(a, b):
    """Two runs side by side: wall and decode medians, each part's GPU ms, the 78-layer estimates, and how far b's
    saved logits are from a's (relative to a's largest |logit|) and whether their argmax agrees."""
    ra, rb = (json.loads((d / 'rank_profile.json').read_text()) for d in (a, b))
    L = [f'A = {a}', f'B = {b}']
    differ = {k: (ra['options'].get(k), rb['options'].get(k)) for k in set(ra['options']) | set(rb['options'])
              if ra['options'].get(k) != rb['options'].get(k)}
    L.append(f'options that differ (A, B): {differ or "none"}')
    L.append(f"W1 wall median ms: A {ra['w1']['wall_median_ms']:.1f}  B {rb['w1']['wall_median_ms']:.1f}")
    if 'decode' in ra and 'decode' in rb:
        L.append(f"decode step median ms: A {ra['decode']['step_median_ms']:.3f}  B {rb['decode']['step_median_ms']:.3f}")
    for n in ra.get('verify', {}).get('counts', {}):
        va, vb = ra['verify']['counts'][n], rb.get('verify', {}).get('counts', {}).get(n)
        if vb is not None:
            L.append(f"verify step of {n} tokens, median ms: A {va['step_median_ms']:.3f}  B {vb['step_median_ms']:.3f}")
    cases = [('W1', ra['w1'], rb['w1'])] + [(f"chunk at {x['start']}", x, y) for x, y in zip(ra.get('long', []), rb.get('long', []))]
    for name, x, y in cases:
        L.append(f"\n{name}: wall ms A {x['layered']['wall_ms']:.1f}  B {y['layered']['wall_ms']:.1f}; per part GPU ms (A -> B)")
        for pa, pb in zip(x['layered']['parts'], y['layered']['parts']):
            L.append(f"  layer {pa['layer']} {pa['part']:9} {pa['gpu_ms']:8.2f} -> {pb['gpu_ms']:8.2f}   launches "
                     f"{pa['kernels'] + pa['cublaslt']} -> {pb['kernels'] + pb['cublaslt']}")
        ga, gb = estimate(x['layered']['parts'], 'gpu_ms'), estimate(y['layered']['parts'], 'gpu_ms')
        if ga is not None and gb is not None:
            L.append(f"  78-layer GPU estimate: A {ga:.0f} ms  B {gb:.0f} ms")
    L.append('')
    for f in sorted(p.name for p in a.glob('*-logits.npy')):
        if not (b / f).exists():
            continue
        la, lb = np.load(a / f), np.load(b / f)
        drift = float(np.abs(lb - la).max()) / float(np.abs(la).max())
        L.append(f'{f}: max |B - A| / max |A| = {drift:.3e}; argmax A {int(la.argmax())} B {int(lb.argmax())}; '
                 f'B finite {bool(np.isfinite(lb).all())}')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
