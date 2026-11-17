"""index_scores_tc_at alone at GLM-5.3's indexer shape, tokens from argv, for ncu."""
import sys
import numpy as np
from gpu.runtime import CUDAContext
from gpu.spawn.launch import SpawnKernel
from pathlib import Path
tokens = int(sys.argv[1]); cap = (tokens + 127) // 128 * 128
ctx = CUDAContext()
rng = np.random.default_rng(1)
def upload(a):
    return ctx.tensor(a.shape, str(a.dtype), a.tobytes())
k = SpawnKernel(ctx, Path('gpu/build/spawn/index_scores_tc_at.ptx'))
args = dict(out=upload(np.zeros(cap, 'f4')), q=upload(rng.normal(size=(32, 128)).astype('f4')),
            keys=upload(rng.normal(size=(cap, 128)).astype('f4')), weights=upload(rng.normal(size=32).astype('f4')),
            step=upload(np.array([tokens - 1, tokens, 0, 0], np.int32)), max_tokens=cap, heads=32, dim=128, scale=128 ** -0.5)
for _ in range(3):
    k.launch(**args)
ctx.synchronize()
print('ok')
