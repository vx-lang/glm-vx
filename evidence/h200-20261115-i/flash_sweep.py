"""MLA decode on the GPU, GLM-5.3 shape (64 heads, rank 512, rope 64, 4096 cached tokens): mla_flash_at
(4 heads a subgroup) and mla_flash2_at (2) + mla_flash_merge_at, against mla_chunks_at and mla_heads_at.
Outputs checked against mla_chunks_at (1e-4) and float64 attention. CUDA events, L2 flushed, median of 30.
  python flash_sweep.py FLASH_DIR"""
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
q, qr = rng.normal(size=(H, R)).astype('f4'), rng.normal(size=(H, RO)).astype('f4')
kv, kr = rng.normal(size=(T, R)).astype('f4'), rng.normal(size=(T, RO)).astype('f4')
scale = float(np.float32(R ** -0.5))
common = dict(q_latent=up(q.reshape(-1)), q_rope=up(qr.reshape(-1)), cache_latent=up(kv.reshape(-1)), cache_rope=up(kr.reshape(-1)),
              heads=H, rank=R, rope_dim=RO, selection_stride=0, splits_max=SM, scale=scale)
chunks, heads = SpawnKernel(ctx, spawn / 'mla_chunks_at.ptx'), SpawnKernel(ctx, spawn / 'mla_heads_at.ptx')
flash = {n: SpawnKernel(ctx, fdir / f'{n}.ptx') for n in ('mla_flash2_at', 'mla_flash3_at', 'mla_flash3r_at')}
merge = SpawnKernel(ctx, fdir / 'mla_flash_merge_at.ptx')
def f64(sel, count):
    ids = [int(i) for i in sel[:count]]
    s = (kv[ids].astype(np.float64) @ q.T.astype(np.float64) + kr[ids].astype(np.float64) @ qr.T.astype(np.float64)) * scale
    p = np.exp(s - s.max(0)); return (p.T @ kv[ids].astype(np.float64)) / p.sum(0)[:, None]
for count in (256, 1000, 2048):
    sel_np = rng.permutation(T)[:count].astype('i4'); sel = up(sel_np); want64 = f64(sel_np, count)
    step = lambda splits: up(np.array([T - 1, T, count, splits], 'i4'))
    rows = []
    for per in (64, 128, 256):  # the old kernels at their usual and better splits
        st = step(min(16, max(1, (count + per - 1) // per)))
        for name, k, g in (('chunks', chunks, 1), ('heads', heads, 4)):
            v = dict(common, out=ctx.tensor((H // g, g * R)), numerator=ctx.tensor((H // g, SM, g * R)), stats=ctx.tensor((H // g, SM, 2 * g)),
                     selected=sel, step=st, arrivals=up(np.zeros(H // g, 'i4')), threads=512)
            k.launch(stream=stream, **v); stream.synchronize()
            out = np.frombuffer(v['out'].download(), 'f4').reshape(H, R)
            if name == 'chunks' and per == 64: ref = out
            rows.append((f'{name} {per}/split', event_us(ctx, stream, spawn, lambda: k.launch(stream=stream, **v), 30), out))
    for per in (8, 16, 32):
        splits = min(SM, max(1, (count + per - 1) // per)); st = step(splits)
        for name, k in flash.items():
            num, sts, out = ctx.tensor((H // 16, SM, 16 * R)), ctx.tensor((H // 16, SM, 32)), ctx.tensor((H, R))
            fv = dict(common, numerator=num, stats=sts, selected=sel, step=st)
            mv = dict(out=out, numerator=num, stats=sts, step=st, heads=H, rank=R, splits_max=SM)
            both = lambda: (k.launch(stream=stream, **fv), merge.launch(stream=stream, **mv))
            both(); stream.synchronize()
            rows.append((f'{name[4:]} {per}/split ({splits})', event_us(ctx, stream, spawn, both, 30),
                         np.frombuffer(out.download(), 'f4').reshape(H, R)))
    print(f'## {count} selected tokens')
    for label, us, out in rows:
        d = np.abs(out - ref).max(); e = np.abs(out - want64).max() / np.abs(want64).max()
        print(f'{label:26s} {us:8.1f} us   max|diff vs chunks| {d:.2e}   rel err vs f64 {e:.2e}', flush=True)
ctx.close()
