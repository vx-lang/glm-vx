"""MLA decode sequences as CUDA graphs (as ResidentGLM replays them), GLM-5.3 shape: flash (f32 cache):
mla_flash_at + mla_flash_merge_at; tc (bf16 cache): mla_q_bf16_at + mla_tc128_at + mla_flash_merge_at.
Each graph replay timed with CUDA events after an L2 flush, median of 50.  python graph_sweep.py DIR"""
import sys
from pathlib import Path
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us
H, R, RO, T, SM = 64, 512, 64, 4096, 64
spawn, fdir = Path('gpu/build/spawn'), Path(sys.argv[1])
tc_name = sys.argv[2] if len(sys.argv) > 2 else 'mla_tc128_at'
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
K = {n: SpawnKernel(ctx, fdir / f'{n}.ptx') for n in (tc_name, 'mla_flash_at', 'mla_flash_merge_at', 'mla_q_bf16_at')}
q_t, qr_t = ctx.tensor((H // 16, 1024, 8), 'uint16'), ctx.tensor((H // 16, 128, 8), 'uint16')
ql, qrd = up(q.reshape(-1)), up(qr.reshape(-1))
c32 = dict(cache_latent=up(kv.reshape(-1)), cache_rope=up(kr.reshape(-1)))
c16 = dict(cache_latent=up(kvb), cache_rope=up(krb))
num, sts, out = ctx.tensor((H // 16, SM, 16 * R)), ctx.tensor((H // 16, SM, 32)), ctx.tensor((H, R))
for count in (256, 1000, 2048):
    sel_np = rng.permutation(T)[:count].astype('i4'); sel = up(sel_np); ids = sel_np.tolist()
    s = (kv[ids].astype(np.float64) @ q.T.astype(np.float64) + kr[ids].astype(np.float64) @ qr.T.astype(np.float64)) * scale
    p = np.exp(s - s.max(0)); want = (p.T @ kv[ids].astype(np.float64)) / p.sum(0)[:, None]
    for path, per in (('flash', 8), ('tc', 32)):
        splits = min(SM, max(1, (count + per - 1) // per)); st = up(np.array([T - 1, T, count, splits], 'i4'))
        mv = dict(out=out, numerator=num, stats=sts, step=st, heads=H, rank=R, splits_max=SM)
        common = dict(numerator=num, stats=sts, selected=sel, step=st, heads=H, rank=R, rope_dim=RO, selection_stride=0,
                      splits_max=SM, scale=scale)
        if path == 'flash':
            seq = lambda: (K['mla_flash_at'].launch(stream=stream, q_latent=ql, q_rope=qrd, **c32, **common),
                           K['mla_flash_merge_at'].launch(stream=stream, **mv))
        else:
            seq = lambda: (K['mla_q_bf16_at'].launch(stream=stream, q_t=q_t, qr_t=qr_t, q_latent=ql, q_rope=qrd, heads=H, rank=R, rope_dim=RO),
                           K[tc_name].launch(stream=stream, q_t=q_t.view((H // 16 * 1024, 8)), qr_t=qr_t.view((H // 16 * 128, 8)), **c16,
                                             **dict(common, numerator=num.view((H // 16 * SM * 16, R)), stats=sts.view((H // 16 * SM, 32)))),
                           K['mla_flash_merge_at'].launch(stream=stream, **mv))
        if path == 'tc':
            pre = lambda: K['mla_q_bf16_at'].launch(stream=stream, q_t=q_t, qr_t=qr_t, q_latent=ql, q_rope=qrd, heads=H, rank=R, rope_dim=RO)
            tck = lambda: K[tc_name].launch(stream=stream, q_t=q_t.view((H // 16 * 1024, 8)), qr_t=qr_t.view((H // 16 * 128, 8)), **c16, **dict(common, numerator=num.view((H // 16 * SM * 16, R)), stats=sts.view((H // 16 * SM, 32))))
            line = []
            for pn, fn in (('prep', lambda: pre()), ('prep+tc', lambda: (pre(), tck()))):
                fn(); stream.synchronize()
                with stream.capture() as g1:
                    fn()
                line.append(f"{pn} {event_us(ctx, stream, spawn, lambda: g1.launch(stream), 50):5.1f}")
                g1.close()
            print('       ', '  '.join(line), flush=True)
        seq(); stream.synchronize()
        with stream.capture() as graph:
            seq()
        graph.launch(stream); stream.synchronize()
        got = np.frombuffer(out.download(), 'f4').reshape(H, R)
        us = event_us(ctx, stream, spawn, lambda: graph.launch(stream), 50)
        print(f'{count:5d} {path:5s} splits {splits:2d}  graph {us:5.1f} us   max|err| vs f64 {np.abs(got - want).max():.2e}', flush=True)
        graph.close()
ctx.close()
