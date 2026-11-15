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
    sel = up(rng.permutation(T)[:count].astype('i4'))
    for per in (8, 16, 32):
        splits = min(SM, max(1, (count + per - 1) // per)); st = up(np.array([T - 1, T, count, splits], 'i4'))
        for name, k in flash.items():
            num, sts, out = ctx.tensor((H // 16, SM, 16 * R)), ctx.tensor((H // 16, SM, 32)), ctx.tensor((H, R))
            fv = dict(common, numerator=num, stats=sts, selected=sel, step=st)
            mv = dict(out=out, numerator=num, stats=sts, step=st, heads=H, rank=R, splits_max=SM)
            a = event_us(ctx, stream, spawn, lambda: k.launch(stream=stream, **fv), 30)
            b = event_us(ctx, stream, spawn, lambda: merge.launch(stream=stream, **mv), 30)
            print(f'{count:5d} {name[4:]:10s} {per:3d}/split ({splits:2d})  flash {a:6.1f}  merge {b:6.1f}', flush=True)
ctx.close()
