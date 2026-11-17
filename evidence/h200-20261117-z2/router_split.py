"""Pieces of the router at 256 x 6144: each one CUDA graph after an L2 flush (median of 50, us)."""
import ctypes as C
from pathlib import Path
import numpy as np
from validation.moe_profile import event_us
def main():
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    build = Path('gpu/build'); spawn = build / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 0)); stream = ctx.stream()
    hand_router = ctx.load_ptx(build / 'kernels.ptx').kernel('glm_vx_gpu_router')
    rows_f32 = ctx.load_ptx(build / 'matvec_rows.ptx').kernel('glm_vx_gpu_matvec_rows')
    m4 = SpawnKernel(ctx, spawn / 'bf16_matvec4.ptx'); import sys; fused = SpawnKernel(ctx, Path(sys.argv[1]) / 'router_bf16.ptx' if len(sys.argv) > 1 else spawn / 'router_bf16.ptx')
    copy = SpawnKernel(ctx, spawn / 'copy.ptx')
    rng = np.random.default_rng(5)
    for E in (256, 16):
        w = (rng.normal(size=(E, 6144)).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
        dense = (w.astype(np.uint32) << 16).view('f4'); x = rng.normal(size=6144).astype('f4'); b = (rng.normal(size=E) * .01).astype('f4')
        up = lambda a: ctx.tensor(a.shape, str(a.dtype), np.ascontiguousarray(a).tobytes())
        dw, dd, dx, db = up(w.reshape(-1)), up(dense.reshape(-1)), up(x), up(b)
        words = dw.view((E * 3072,), 'uint32'); lg = ctx.tensor((E,)); ids = ctx.tensor((8,), 'int32'); wt = ctx.tensor((8,))
        arr = ctx.tensor((1,), 'int32', np.zeros(1, np.int32).tobytes()); one = ctx.tensor((32,))
        v = {'empty copy (graph floor)': lambda: copy.launch(stream=stream, out=one, x=dx.view((32,), 'float32'), n=32),
             'hand router alone': lambda: hand_router.launch(1, 32, [ids, wt, lg, db, C.c_int32(E), C.c_int32(8), C.c_float(2.5)], stream=stream),
             'matvec_rows f32 alone': lambda: rows_f32.launch(E, 1024, [lg, dd, dx, C.c_int32(E), C.c_int32(6144)], stream=stream),
             'bf16_matvec4 alone': lambda: m4.launch(stream=stream, out=lg, weight=words, x=dx, rows=E, cols=6144),
             'router_bf16 k=0 (projection only)': lambda: fused.launch(stream=stream, indices=ids, weights=wt, logits=lg, gate=words, x=dx, bias=db, experts=E, cols=6144, k=0, scale=2.5, arrivals=arr),
             'router_bf16 k=8': lambda: fused.launch(stream=stream, indices=ids, weights=wt, logits=lg, gate=words, x=dx, bias=db, experts=E, cols=6144, k=8, scale=2.5, arrivals=arr)}
        for name, f in v.items():
            f(); stream.synchronize()
            with stream.capture() as g:
                f()
            print(f'{E:4d} {name:36s} {event_us(ctx, stream, spawn, lambda: g.launch(stream), repeats=50):7.1f}', flush=True)
if __name__ == '__main__':
    main()
