"""Tensor-parallel decode on real GPUs against its one-GPU emulation (#56, #62).

A GLM-5.3-wide proxy (validation.gpu_vs_official.full_geometry: 64 heads, hidden 6144, random weights,
FP8 Linears, bf16 router and lm_head, 16 routed experts) split over --ranks ranks (gpu/tp.py):

  1. reference: the ranks as threads of one process on GPU 0, sharing its context, each step through
     graph=True's launches run eagerly (never captured), the all-gather a device copy between the
     ranks' tensors (gpu/testing/collective.DeviceJob). Its logits per step.
  2. run: one process a rank, rank r on GPU r (every GPU visible: CUDA IPC maps memory only on a GPU
     the process can see), joined by Vx's NCCL collectives (gpu/collective.py), graph=True replayed;
     with --one-shot the per-layer sums through gpu/spawn/tp_all_reduce over IPC peer buffers.
     Every rank's logits per step, then the time of --time steps.

Every rank's logits must equal the reference's, bit for bit, at every step: the cross-rank sum is in
rank order, so n GPUs give what one GPU emulating them gives. --ranks 1 skips the reference and times
one GPU.

  python -m validation.tp_decode_check --ranks 2 --layers 4 --context 64 --output run/tp2.json
"""
import argparse
import ctypes as C
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

OPTIONS = dict(device_experts=True, graph=True, fused_attention=True, mla_kernel='tc', fp8_kernels='cvt',
               matvec_kernels='x4', norm_block=1024, fold_shared=True)


def proxy_model(layers, moe_from):
    from glm_vx.backend import NumpyBackend
    from glm_vx.model import GlmMoeDsaModel
    from validation.gpu_vs_official import full_geometry, random_weights
    from validation.resident_compare import fp8_source
    config = full_geometry(layers, topk=2048)
    config['indexer_types'] = ['full' if i % 4 == 0 else 'shared' for i in range(layers)]
    config['mlp_layer_types'] = ['dense' if i < moe_from else 'sparse' for i in range(layers)]
    config['n_routed_experts'] = 16
    weights = random_weights(config, 1)
    return GlmMoeDsaModel(config, fp8_source(weights, bf16_head=True, bf16_router=True), NumpyBackend())


def model_for(args):
    """The proxy, or with --checkpoint the real checkpoint's first --layers layers (its own per-layer
    patterns: GLM-5.3's first 3 are dense), read lazily from its shards."""
    if args.checkpoint is None:
        return proxy_model(args.layers, args.moe_from)
    from glm_vx.backend import NumpyBackend
    from glm_vx.checkpoint import Checkpoint
    from glm_vx.config import GLMConfig
    from glm_vx.model import GlmMoeDsaModel
    from validation.write_proxy import proxy_config
    config = proxy_config(args.layers, base=json.loads((args.checkpoint / 'config.json').read_text()))
    checkpoint = Checkpoint(args.checkpoint, block_size=GLMConfig.from_dict(config).weight_block_size)
    return GlmMoeDsaModel(config, checkpoint, NumpyBackend())


def tokens_for(model, count):
    return [int(t) for t in np.random.default_rng(7).integers(0, int(model.config['vocab_size']), count)]


def reference_main(args, out):
    """The ranks as threads on GPU 0: every step's logits (rank 0's; all ranks' must agree)."""
    from gpu.resident import ResidentGLM
    from gpu.runtime import CUDAContext
    from gpu.testing.collective import DeviceJob, run_ranks
    from gpu.tp import TensorParallel
    model = model_for(args)
    tokens = tokens_for(model, args.context)
    context = CUDAContext(0, minimum_compute_capability=(8, 0))
    job = DeviceJob(args.ranks)

    def rank_main(rank):
        r = ResidentGLM(model, ctx=context, max_tokens=args.context + 8,
                        tp=TensorParallel(rank, args.ranks, job.member(rank)), **OPTIONS)
        r._capture = False
        try:
            return np.stack([r.forward(t) for t in tokens])
        finally:
            r.close()
    logits = run_ranks(job, rank_main)
    for rank in range(1, args.ranks):
        if not np.array_equal(logits[rank].view(np.uint32), logits[0].view(np.uint32)):
            raise SystemExit(f'reference: rank {rank} differs from rank 0')
    np.save(out, logits[0])
    context.close()


def rank_main(args, rank, out):
    from gpu.collective import Collective
    from gpu.resident import ResidentGLM
    from gpu.runtime import CUDAContext
    from gpu.tp import TensorParallel
    model = model_for(args)
    tokens = tokens_for(model, args.context + args.time)
    context = CUDAContext(rank, primary=True, minimum_compute_capability=(8, 0))
    tp = None
    if args.ranks > 1:
        collective = Collective(C.CDLL(os.environ['VX_DISPATCH_LIB']), context, rank, args.ranks)
        tp = TensorParallel(rank, args.ranks, collective, one_shot=args.one_shot)
    r = ResidentGLM(model, ctx=context, max_tokens=args.context + args.time + 8, tp=tp, **OPTIONS)
    logits = np.stack([r.forward(t) for t in tokens[:args.context]])
    np.save(out, logits)
    timing = None
    if args.time:
        for t in tokens[args.context:args.context + 2]:  # the long step graphs warm
            r.forward(t)
        context.synchronize()
        start = time.perf_counter()
        for t in tokens[args.context + 2:args.context + args.time]:
            r.forward(t)
        context.synchronize()
        timing = (time.perf_counter() - start) / max(1, args.time - 2) * 1e6
    print(json.dumps(dict(rank=rank, us_per_step=timing, replays=r.replays,
                          sums=r.launches.get('tp_all_gather', 0) + r.launches.get('tp_all_reduce', 0))), flush=True)
    r.close()
    if tp is not None:
        tp.collective.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ranks', type=int, default=2)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--moe-from', type=int, default=3, help='the first MoE layer (GLM-5.3: 3)')
    p.add_argument('--context', type=int, default=48, help='steps compared with the reference')
    p.add_argument('--time', type=int, default=50, help='steps timed after them')
    p.add_argument('--one-shot', action='store_true', help='TensorParallel(one_shot=True): gpu/spawn/tp_all_reduce')
    p.add_argument('--checkpoint', type=Path, help="a real checkpoint (zai-org/GLM-5.3's layout): its first --layers layers")
    p.add_argument('--skip-reference', action='store_true', help='time the ranks only (for a profiler): no bits compared')
    p.add_argument('--output', type=Path)
    p.add_argument('--role', help=argparse.SUPPRESS)
    p.add_argument('--out', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.role == 'reference':
        return reference_main(args, args.out)
    if args.role is not None:
        return rank_main(args, int(args.role), args.out)
    if args.output is not None and args.output.exists():
        p.error('output exists; preserve evidence')
    if args.ranks > 1 and 'VX_DISPATCH_LIB' not in os.environ:
        p.error('VX_DISPATCH_LIB is not set (source scripts/env.sh)')
    command = [sys.executable, '-m', 'validation.tp_decode_check', '--ranks', str(args.ranks), '--layers', str(args.layers),
               '--moe-from', str(args.moe_from), '--context', str(args.context), '--time', str(args.time)]
    command += ['--one-shot'] if args.one_shot else []
    command += ['--checkpoint', str(args.checkpoint)] if args.checkpoint is not None else []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        reference = None
        if args.ranks > 1 and not args.skip_reference:
            done = subprocess.run(command + ['--role', 'reference', '--out', str(tmp / 'reference.npy')])
            if done.returncode:
                raise SystemExit(f'reference failed: exit {done.returncode}')
            reference = np.load(tmp / 'reference.npy')
        environment = dict(os.environ, VX_NCCL_ID_FILE=str(tmp / 'nccl-id'))
        processes = [subprocess.Popen(command + ['--role', str(r), '--out', str(tmp / f'rank{r}.npy')], env=environment,
                                      stdout=subprocess.PIPE, text=True) for r in range(args.ranks)]
        outputs = [process.communicate()[0] for process in processes]
        results = []
        for r, (process, output) in enumerate(zip(processes, outputs)):
            sys.stdout.write(output)
            lines = [line for line in output.splitlines() if line.startswith('{')]
            result = json.loads(lines[-1]) if process.returncode == 0 and lines else dict(rank=r, error=f'exit {process.returncode}')
            if 'error' not in result and reference is not None:
                got = np.load(tmp / f'rank{r}.npy')
                differ = np.flatnonzero((got.view(np.uint32) != reference.view(np.uint32)).any(axis=1))
                result['steps_differing'] = [int(i) for i in differ[:10]]
                result['bit_exact'] = differ.size == 0
            results.append(result)
    ok = all('error' not in r and r.get('bit_exact', True) for r in results)
    summary = dict(ranks=args.ranks, layers=args.layers, checkpoint=args.checkpoint and str(args.checkpoint), context=args.context, one_shot=args.one_shot,
                   compared=reference is not None, ok=ok, results=results)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))
    print('tp_decode_check:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
