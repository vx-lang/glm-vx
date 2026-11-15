"""Graph time of mla_tc_at cut after each phase, one split (4 blocks) and 64 splits (256 blocks)."""
import sys
from pathlib import Path
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us
H, R, RO, T, SM = 64, 512, 64, 4096, 64
spawn, fdir = Path('gpu/build/spawn'), Path(sys.argv[1])
ctx = CUDAContext(minimum_compute_capability=(8, 0)); stream = ctx.stream()
rng = np.random.default_rng(1)
up = lambda a: ctx.tensor(a.shape, str(a.dtype), np.ascontiguousarray(a).tobytes())
bf = lambda n: (rng.normal(size=n).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
q_t, qr_t = up(bf(4 * 1024 * 8).reshape(-1, 8)), up(bf(4 * 128 * 8).reshape(-1, 8))
kv, kr = up(bf(T * R).reshape(T, R)), up(bf(T * RO).reshape(T, RO))
num, sts = ctx.tensor((4 * SM * 16, R)), ctx.tensor((4 * SM, 32))
for count in (32, 2048):
    sel = up(rng.permutation(T)[:count].astype('i4'))
    splits = (count + 31) // 32
    st = up(np.array([T - 1, T, count, splits], 'i4'))
    allv = dict(numerator=num, stats=sts, q_t=q_t, qr_t=qr_t, cache_latent=kv, cache_rope=kr, selected=sel, step=st,
                heads=H, rank=R, rope_dim=RO, selection_stride=0, splits_max=SM, scale=0.04)
    line = []
    for name in ('mla_tcp1_at', 'mla_tcp2_at', 'mla_tcp3_at', 'mla_tc_at'):
        k = SpawnKernel(ctx, fdir / f'{name}.ptx')
        v = {p['name']: allv[p['name']] for p in k.manifest['params'] if p['name'] in allv}
        f = lambda: k.launch(stream=stream, **v)
        f(); stream.synchronize()
        with stream.capture() as g:
            f()
        line.append(f"{name[6:-3] or 'full'} {event_us(ctx, stream, spawn, lambda: g.launch(stream), 50):5.1f}")
        g.close()
    print(f'{count:5d} tokens ({splits} splits):', '  '.join(line), flush=True)
k = SpawnKernel(ctx, spawn / 'add.ptx'); n = 32
a = ctx.tensor((n,)); vals = dict(out=ctx.tensor((n,)), a=a, b=a, n=n)
with stream.capture() as g:
    k.launch(stream=stream, **vals)
print('empty-ish graph (add 32):', f'{event_us(ctx, stream, spawn, lambda: g.launch(stream), 50):.1f}')
ctx.close()
