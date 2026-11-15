"""mla_tc_at (bf16 cache, tensor cores) + merge against mla_flash_at (f32 cache) + merge, GLM-5.3 MLA
shape, CUDA events with L2 flushed, median of 30; outputs against float64 attention.
  python tc_sweep.py DIR"""
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
def to_bf16(x):
    u = np.asarray(x, 'f4').view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
f32 = lambda b: (b.astype(np.uint32) << 16).view('f4')
q, qr = rng.normal(size=(H, R)).astype('f4'), rng.normal(size=(H, RO)).astype('f4')
kvb, krb = to_bf16(rng.normal(size=(T, R))), to_bf16(rng.normal(size=(T, RO)))
kv, kr = f32(kvb), f32(krb)
scale = float(np.float32((R + RO) ** -0.5))
K = {n: SpawnKernel(ctx, fdir / f'{n}.ptx') for n in ('mla_tc_at', 'mla_flash_at', 'mla_flash_merge_at', 'mla_q_bf16_at')}
q_t, qr_t = ctx.tensor((H // 16, 1024, 8), 'uint16'), ctx.tensor((H // 16, 128, 8), 'uint16')
base = dict(q_latent=up(q.reshape(-1)), q_rope=up(qr.reshape(-1)), heads=H, rank=R, rope_dim=RO, selection_stride=0,
            splits_max=SM, scale=scale)
caches = {'mla_tc_at': dict(cache_latent=up(kvb), cache_rope=up(krb), q_t=q_t.view((H // 16 * 1024, 8)), qr_t=qr_t.view((H // 16 * 128, 8))),
          'mla_flash_at': dict(cache_latent=up(kv.reshape(-1)), cache_rope=up(kr.reshape(-1)))}
for count in (256, 1000, 2048):
    sel_np = rng.permutation(T)[:count].astype('i4'); sel = up(sel_np)
    ids = sel_np.tolist()
    s = (kv[ids].astype(np.float64) @ q.T.astype(np.float64) + kr[ids].astype(np.float64) @ qr.T.astype(np.float64)) * scale
    p = np.exp(s - s.max(0)); want = (p.T @ kv[ids].astype(np.float64)) / p.sum(0)[:, None]
    for name, per in (('mla_flash_at', 8), ('mla_tc_at', 32)):
        splits = min(SM, max(1, (count + per - 1) // per)); st = up(np.array([T - 1, T, count, splits], 'i4'))
        num, sts, out = ctx.tensor((H // 16, SM, 16 * R)), ctx.tensor((H // 16, SM, 32)), ctx.tensor((H, R))
        fv = dict(base, numerator=num, stats=sts, selected=sel, step=st, **caches[name])
        if name == 'mla_tc_at':
            fv = {k: v for k, v in fv.items() if k not in ('q_latent', 'q_rope')}
        mv = dict(out=out, numerator=num, stats=sts, step=st, heads=H, rank=R, splits_max=SM)
        k, m = K[name], K['mla_flash_merge_at']
        qv = dict(q_t=q_t, qr_t=qr_t, q_latent=base['q_latent'], q_rope=base['q_rope'], heads=H, rank=R, rope_dim=RO)
        prep = (lambda: K['mla_q_bf16_at'].launch(stream=stream, **qv)) if name == 'mla_tc_at' else (lambda: None)
        both = lambda: (prep(), k.launch(stream=stream, **fv), m.launch(stream=stream, **mv))
        both(); stream.synchronize(); got = np.frombuffer(out.download(), 'f4').reshape(H, R)
        a = event_us(ctx, stream, spawn, lambda: k.launch(stream=stream, **fv), 30)
        if name == 'mla_tc_at':
            print(f'      mla_q_bf16_at {event_us(ctx, stream, spawn, prep, 30):5.1f} us', flush=True)
        b = event_us(ctx, stream, spawn, both, 30)
        print(f'{count:5d} {name:13s} splits {splits:2d}  kernel {a:5.1f}  +prep+merge {b:5.1f}   max|err| vs f64 {np.abs(got - want).max():.2e} (max|out| {np.abs(want).max():.2f})', flush=True)
ctx.close()
