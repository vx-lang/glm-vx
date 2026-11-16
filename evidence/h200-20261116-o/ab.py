"""A/B of top-k / scores kernels: cold (L2 flushed, median of 20) and warm (50 back to back, mean), us.
  python ab.py KERNEL [KERNEL ...]   (topk kernels: name with tokens, k; run from the repo root)"""
import sys
import numpy as np
from pathlib import Path
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us

spawn = Path('gpu/build/spawn')
ctx = CUDAContext(minimum_compute_capability=(8, 0))
stream = ctx.stream()
rng = np.random.default_rng(1)
names = sys.argv[1:]
kernels = {n: SpawnKernel(ctx, spawn / f'{n}.ptx') for n in names}
print('tokens ' + ' '.join(f'{n + " cold":>22} {n + " warm":>22}' for n in names))
for tokens in (2049, 3001, 4096, 8192, 32768, 131072):
    row = [f'{tokens:6d}']
    if names[0].startswith('topk'):
        s = rng.normal(size=tokens).astype('f4')
        d = ctx.tensor((tokens,), 'int32', s.view('i4').tobytes())
        out = ctx.tensor((2048,), 'int32')
        want = np.sort(np.argsort(-s, kind='stable')[:2048])
        args = dict(out_ids=out, scores=d, tokens=tokens, k=2048)
    else:
        heads, dim = 32, 128
        q = ctx.tensor((heads * dim,), 'float32', rng.normal(size=heads * dim).astype('f4').tobytes())
        keys = ctx.tensor((tokens * dim,), 'float32', rng.normal(size=tokens * dim).astype('f4').tobytes())
        w = ctx.tensor((heads,), 'float32', (rng.normal(size=heads) * heads ** -0.5).astype('f4').tobytes())
        out = ctx.tensor((tokens,), 'float32')
        args = dict(out=out, q=q, keys=keys, weights=w, tokens=tokens, heads=heads, dim=dim, scale=dim ** -0.5)
    for n, k in kernels.items():
        launch = lambda: k.launch(stream=stream, **args)
        launch(); stream.synchronize()
        if n.startswith('topk'):
            assert np.array_equal(np.frombuffer(out.download(), 'i4'), want), n
        cold = event_us(ctx, stream, spawn, launch, 20)
        start, end = ctx.event().record(stream), ctx.event()
        for _ in range(50):
            launch()
        end.record(stream)
        row.append(f'{cold:22.1f} {start.elapsed_ms(end) * 1e3 / 50:22.1f}')
    print(' '.join(row), flush=True)
ctx.close()
