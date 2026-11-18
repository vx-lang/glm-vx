"""fp8_moe_cvt_down against fp8_moe_cvt_down2 on one H200: bits and CUDA-event time, GLM-5.3's MoE down at
tensor-parallel slices (hidden 6144, intermediate 2048 / tp a rank, 256 routed experts + the folded shared one)."""
import sys
from pathlib import Path
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us
SPAWN = Path('gpu/build/spawn')
D2 = Path(sys.argv[1]) if len(sys.argv) > 1 else SPAWN
rng = np.random.default_rng(0)
ctx = CUDAContext(int(sys.argv[2]) if len(sys.argv) > 2 else 0, minimum_compute_capability=(8, 9)); stream = ctx.stream()
up = lambda a: ctx.tensor(a.shape, a.dtype.name, data=np.ascontiguousarray(a))
def codes(n):
    b = rng.integers(0, 256, n, dtype=np.uint8); b[(b & 0x7F) == 0x7F] = 0; return up(b.view(np.uint32))
fl = lambda n, lo=-1, hi=1: up(rng.uniform(lo, hi, n).astype(np.float32))
cs = lambda r, c: ((r + 127) // 128) * ((c + 127) // 128)
kernels = [("down", SpawnKernel(ctx, SPAWN / "fp8_moe_cvt_down.ptx")), ("down2", SpawnKernel(ctx, D2 / "fp8_moe_cvt_down2.ptx")), ("down4", SpawnKernel(ctx, D2 / "fp8_moe_cvt_down4.ptx"))]
E_, H = 257, 6144
print('tp | cols | k | ' + ' | '.join(f'{n} us' for n, _ in kernels) + ' | same bits')
for tp in (8, 4):
    I = 2048 // tp
    d, s = codes(E_ * H * I), fl(E_ * cs(H, I), .5, 2)
    for K in (9, 8):
        ids = up(np.concatenate([rng.permutation(256)[:K - 1], [256]]).astype(np.int32) if K == 9 else rng.permutation(256)[:K].astype(np.int32))
        a, wt = fl(K * I), fl(K, 0, 1)
        cells, outs = [], []
        for name, k in kernels:
            out = ctx.tensor((H,))
            launch = lambda: k.launch(stream=stream, out=out, down=d, scales=s, act=a, weights=wt, ids=ids, k=K, experts=E_, rows=H, cols=I)
            launch(); stream.synchronize(); outs.append(bytes(out.download()))
            cells.append(f'{event_us(ctx, stream, SPAWN, launch, 50):6.1f}')
        print(f'{tp} | {I} | {K} | ' + ' | '.join(cells) + f' | {outs[0] == outs[1]}', flush=True)
ctx.close()
