"""router_bf16 with its old tail (two passes of ranks, OLD_PTX) and its new one (warp argmax rounds, gpu/build/spawn):
the same indices, weights and logits bit for bit on 200 random inputs (some with tied scores), then the time of each
(CUDA events after an L2 flush, median of 50)."""
import json, sys
from pathlib import Path
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us
old_ptx = Path(sys.argv[1]); spawn = Path('gpu/build/spawn')
ctx = CUDAContext(minimum_compute_capability=(8, 0)); stream = ctx.stream()
rng = np.random.default_rng(0)
E, H = 256, 6144
kernels = {'old': SpawnKernel(ctx, old_ptx), 'new': SpawnKernel(ctx, spawn / 'router_bf16.ptx')}
mismatch, cases = [], 0
for case in range(200):
    w = rng.standard_normal((E, H)).astype(np.float32)
    if case % 4 == 1:  # duplicated rows: tied logits, ties to the lower id
        w[rng.integers(0, E, 40)] = w[rng.integers(0, E, 1)]
    gate = (w.view(np.uint32) >> 16).astype(np.uint16)
    x = (rng.standard_normal(H) * (0.02 if case % 2 else 0.2)).astype(np.float32)
    bias = (rng.standard_normal(E) * 0.01 if case % 3 else np.zeros(E)).astype(np.float32)
    k = [8, 8, 6, 1][case % 4]
    out = {}
    for name, kern in kernels.items():
        t = lambda a, d: ctx.tensor((a.size,), d, np.ascontiguousarray(a).tobytes())
        idx, wt, lg = ctx.tensor((8,), 'int32', bytes(32)), ctx.tensor((8,), 'float32', bytes(32)), ctx.tensor((E,), 'float32')
        kern.launch(stream=stream, indices=idx, weights=wt, logits=lg, gate=ctx.tensor((E * H // 2,), 'uint32', gate.tobytes()),
                    x=t(x, 'float32'), bias=t(bias, 'float32'), experts=E, cols=H, k=k, scale=2.5,
                    arrivals=ctx.tensor((1,), 'int32', bytes(4)))
        stream.synchronize()
        out[name] = (np.frombuffer(idx.download(), np.int32)[:k].copy(), np.frombuffer(wt.download(), np.uint32)[:k].copy(),
                     np.frombuffer(lg.download(), np.uint32).copy())
    cases += 1
    if not all(np.array_equal(a, b) for a, b in zip(out['old'], out['new'])):
        mismatch.append(dict(case=case, k=k, old=out['old'][0].tolist(), new=out['new'][0].tolist()))
args = None
times = {}
gate = (rng.standard_normal(E * H).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
t = lambda a, d: ctx.tensor((a.size,), d, np.ascontiguousarray(a).tobytes())
args = dict(indices=ctx.tensor((8,), 'int32'), weights=ctx.tensor((8,), 'float32'), logits=ctx.tensor((E,), 'float32'),
            gate=ctx.tensor((E * H // 2,), 'uint32', gate.tobytes()), x=t(rng.standard_normal(H).astype(np.float32), 'float32'),
            bias=t(np.zeros(E, np.float32), 'float32'), experts=E, cols=H, k=8, scale=2.5, arrivals=ctx.tensor((1,), 'int32', bytes(4)))
for name, kern in kernels.items():
    times[name] = event_us(ctx, stream, spawn, lambda: kern.launch(stream=stream, **args), 50)
print(json.dumps(dict(cases=cases, mismatches=len(mismatch), first=mismatch[:3], us=times)))
