"""prep + mla_tc{,2}_at + mla_flash_merge_at as CUDA graphs, GLM-5.3 shape; 32 * C tokens a split."""
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
qb, qrb = f32(to_bf16(q)), f32(to_bf16(qr))
scale = float(np.float32((R + RO) ** -0.5))
K = {n: SpawnKernel(ctx, fdir / f'{n}.ptx') for n in ('mla_tc_at', 'mla_tc2_at', 'mla_flash_merge_at', 'mla_q_bf16_at')}
q_t, qr_t = ctx.tensor((H // 16, 1024, 8), 'uint16'), ctx.tensor((H // 16, 128, 8), 'uint16')
ql, qrd = up(q.reshape(-1)), up(qr.reshape(-1))
c16 = dict(cache_latent=up(kvb), cache_rope=up(krb))
num, sts, out = ctx.tensor((H // 16, SM, 16 * R)), ctx.tensor((H // 16, SM, 32)), ctx.tensor((H, R))
for count in (256, 1000, 2048):
    sel_np = rng.permutation(T)[:count].astype('i4'); sel = up(sel_np); ids = sel_np.tolist()
    s = (kv[ids].astype(np.float64) @ qb.T.astype(np.float64) + kr[ids].astype(np.float64) @ qrb.T.astype(np.float64)) * scale
    p = np.exp(s - s.max(0)); want = (p.T @ kv[ids].astype(np.float64)) / p.sum(0)[:, None]
    line = []
    for name, C in (('mla_tc_at', 1), ('mla_tc2_at', 2)):
        splits = min(SM, max(1, -(-count // (32 * C)))); st = up(np.array([T - 1, T, count, splits], 'i4'))
        mv = dict(out=out, numerator=num, stats=sts, step=st, heads=H, rank=R, splits_max=SM)
        tv = dict(q_t=q_t.view((H // 16 * 1024, 8)), qr_t=qr_t.view((H // 16 * 128, 8)), numerator=num.view((H // 16 * SM * 16, R)),
                  stats=sts.view((H // 16 * SM, 32)), selected=sel, step=st, heads=H, rank=R, rope_dim=RO, selection_stride=0,
                  splits_max=SM, scale=scale, **c16)
        seq = lambda: (K['mla_q_bf16_at'].launch(stream=stream, q_t=q_t, qr_t=qr_t, q_latent=ql, q_rope=qrd, heads=H, rank=R, rope_dim=RO),
                       K[name].launch(stream=stream, **tv), K['mla_flash_merge_at'].launch(stream=stream, **mv))
        seq(); stream.synchronize()
        with stream.capture() as g:
            seq()
        g.launch(stream); stream.synchronize()
        err = np.abs(np.frombuffer(out.download(), 'f4').reshape(H, R) - want).max()
        line.append(f'C={C} ({splits:2d} splits) {event_us(ctx, stream, spawn, lambda: g.launch(stream), 50):5.1f} us err {err:.1e}')
        g.close()
    print(f'{count:5d}:', ' | '.join(line), flush=True)
ctx.close()
