"""router_bf16 alone on the GPU at GLM-5.3's shapes (256 experts x 6144, BF16 gate): with its last-block tail (k = 8)
and without (k = 0), CUDA events after an L2 flush, median of 50; and under graph replay of 75 launches."""
import json, sys
from pathlib import Path
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from validation.moe_profile import event_us
spawn = Path('gpu/build/spawn')
ctx = CUDAContext(minimum_compute_capability=(8, 0)); stream = ctx.stream()
rng = np.random.default_rng(0)
E, H = 256, 6144
gate = (rng.standard_normal(E * H).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
t = lambda a, d: ctx.tensor((a.size,), d, np.ascontiguousarray(a).tobytes())
args = dict(indices=ctx.tensor((8,), 'int32'), weights=ctx.tensor((8,), 'float32'), logits=ctx.tensor((E,), 'float32'),
            gate=ctx.tensor((E * H // 2,), 'uint32', gate.tobytes()), x=t(rng.standard_normal(H).astype(np.float32), 'float32'),
            bias=t(np.zeros(E, np.float32), 'float32'), experts=E, cols=H, scale=2.5,
            arrivals=ctx.tensor((1,), 'int32', bytes(4)))
kern = SpawnKernel(ctx, spawn / 'router_bf16.ptx')
out = {}
for k in (8, 0):
    kern.launch(stream=stream, k=k, **args); stream.synchronize()
    out[f'k{k}_us'] = event_us(ctx, stream, spawn, lambda: kern.launch(stream=stream, k=k, **args), 50)
print(json.dumps(out))
