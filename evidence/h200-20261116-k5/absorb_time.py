"""GLM-5.3's two MLA absorbs from kv_b_proj's FP8 codes, each kernel one CUDA graph timed with CUDA events
after an L2 flush, median of 50 (an almost empty graph is ~3 us).  python absorb_time.py DIR NAME..."""
import sys
from pathlib import Path
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us
H, NOPE, VAL, RANK, QK = 64, 192, 256, 512, 256
P = NOPE + VAL
spawn, d = Path('gpu/build/spawn'), Path(sys.argv[1])
ctx = CUDAContext(minimum_compute_capability=(8, 9)); stream = ctx.stream()
rng = np.random.default_rng(1)
up = lambda a: ctx.tensor(a.shape, str(a.dtype), np.ascontiguousarray(a).tobytes())
codes = rng.integers(0, 256, size=(H * P, RANK), dtype=np.uint8); codes[(codes & 0x7F) == 0x7F] = 0
scales = rng.uniform(0.001, 0.1, size=(H * P // 128, RANK // 128)).astype('f4')
q, lat = rng.normal(size=H * QK).astype('f4'), rng.normal(size=H * RANK).astype('f4')
e = codes.astype(np.int32); sgn = np.where(e & 0x80, -1.0, 1.0); ex = (e >> 3) & 15; m = e & 7
val = sgn * np.where(ex == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (ex - 7.0))
dense = val * np.repeat(np.repeat(scales.astype(np.float64), 128, 0), 128, 1)
want_q = np.concatenate([q[h * QK:h * QK + NOPE].astype(np.float64) @ dense[h * P:h * P + NOPE] for h in range(H)])
want_v = np.concatenate([dense[h * P + NOPE:(h + 1) * P] @ lat[h * RANK:(h + 1) * RANK].astype(np.float64) for h in range(H)])
w, s, xq, xv = up(codes.reshape(-1)).view((codes.size // 4,), 'uint32'), up(scales.reshape(-1)), up(q), up(lat)
oq, ov = ctx.tensor((H * RANK,)), ctx.tensor((H * VAL,))
for name in sys.argv[2:]:
    k = SpawnKernel(ctx, d / f'{name}.ptx')
    if 'cols' in name:
        fn = lambda: k.launch(stream=stream, out=oq, weight=w, scales=s, x=xq, groups=H, cols=RANK, depth=NOPE, period=P, offset=0, x_stride=QK)
        out, want = oq, want_q
    else:
        fn = lambda: k.launch(stream=stream, out=ov, weight=w, scales=s, x=xv, rows=H * VAL, cols=RANK, group=VAL, x_stride=RANK, period=P, offset=NOPE)
        out, want = ov, want_v
    fn(); stream.synchronize()
    with stream.capture() as g:
        fn()
    g.launch(stream); stream.synchronize()
    got = np.frombuffer(out.download(), 'f4')
    us = event_us(ctx, stream, spawn, lambda: g.launch(stream), 50)
    print(f'{name:28s} {us:6.1f} us   rel err {np.abs(got - want).max() / np.abs(want).max():.1e}', flush=True)
    g.close()
ctx.close()
