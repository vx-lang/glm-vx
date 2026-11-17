"""The router's projection at 256 x 6144 (and 16): each variant one CUDA graph after an L2 flush, median of 50 (us)."""
import ctypes as C
import sys
from pathlib import Path
import numpy as np
from validation.moe_profile import event_us
def main():
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    build = Path('gpu/build'); spawn = build / 'spawn'; extra = Path(sys.argv[1])
    ctx = CUDAContext(minimum_compute_capability=(8, 0)); stream = ctx.stream()
    hand_router = ctx.load_ptx(build / 'kernels.ptx').kernel('glm_vx_gpu_router')
    hand_bf16 = ctx.load_ptx(build / 'bf16.ptx').kernel('glm_vx_gpu_bf16_matvec')
    rows_f32 = ctx.load_ptx(build / 'matvec_rows.ptx').kernel('glm_vx_gpu_matvec_rows')
    m4 = SpawnKernel(ctx, spawn / 'bf16_matvec4.ptx')
    rows16 = SpawnKernel(ctx, extra / 'rows' / 'bf16_matvec_rows.ptx')
    fused = SpawnKernel(ctx, extra / 'rb5' / 'router_bf16.ptx'); fused_nf = SpawnKernel(ctx, extra / 'rb5nf' / 'router_bf16.ptx')
    copy = SpawnKernel(ctx, spawn / 'copy.ptx')
    rng = np.random.default_rng(5)
    for E in (256, 16):
        w = (rng.normal(size=(E, 6144)).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
        dense = (w.astype(np.uint32) << 16).view('f4'); x = rng.normal(size=6144).astype('f4'); b = (rng.normal(size=E) * .01).astype('f4')
        up = lambda a: ctx.tensor(a.shape, str(a.dtype), np.ascontiguousarray(a).tobytes())
        dw, dd, dx, db = up(w.reshape(-1)), up(dense.reshape(-1)), up(x), up(b)
        words = dw.view((E * 3072,), 'uint32'); lg = ctx.tensor((E,)); ref = ctx.tensor((E,))
        ids = ctx.tensor((8,), 'int32'); wt = ctx.tensor((8,)); arr = ctx.tensor((1,), 'int32', np.zeros(1, np.int32).tobytes()); one = ctx.tensor((32,))
        router = lambda: hand_router.launch(1, 32, [ids, wt, lg, db, C.c_int32(E), C.c_int32(8), C.c_float(2.5)], stream=stream)
        rows = lambda: rows16.launch(stream=stream, out=lg, weight=words, x=dx, rows=E, cols=6144)
        m4l = lambda: m4.launch(stream=stream, out=lg, weight=words, x=dx, rows=E, cols=6144)
        fz = lambda k: (lambda: fused.launch(stream=stream, indices=ids, weights=wt, logits=lg, gate=words, x=dx, bias=db, experts=E, cols=6144, k=k, scale=2.5, arrivals=arr))
        fnf = lambda: fused_nf.launch(stream=stream, indices=ids, weights=wt, logits=lg, gate=words, x=dx, bias=db, experts=E, cols=6144, k=8, scale=2.5, arrivals=arr)
        # bits: bf16_matvec_rows against the hand bf16_matvec
        hand_bf16.launch(-(-E // 8), 256, [ref, dw, dx, C.c_int32(E), C.c_int32(6144)], stream=stream); rows(); stream.synchronize()
        same = np.array_equal(np.frombuffer(lg.download(), np.uint32), np.frombuffer(ref.download(), np.uint32))
        print(f'{E}: bf16_matvec_rows == hand bf16_matvec bit for bit: {same}', flush=True)
        v = {'graph floor (empty copy)': lambda: copy.launch(stream=stream, out=one, x=dx.view((32,), 'float32'), n=32),
             'hand router alone': router,
             'matvec_rows f32 alone': lambda: rows_f32.launch(E, 1024, [lg, dd, dx, C.c_int32(E), C.c_int32(6144)], stream=stream),
             'bf16_matvec4 alone': m4l,
             'bf16_matvec_rows alone': rows,
             'bf16_matvec4 + hand router (today)': lambda: (m4l(), router()),
             'bf16_matvec_rows + hand router': lambda: (rows(), router()),
             'router_bf16 fused (rb5)': fz(8),
             'router_bf16 fused, no .sys fences': fnf}
        for name, f in v.items():
            f(); stream.synchronize()
            with stream.capture() as g:
                f()
            print(f'{E:4d} {name:38s} {event_us(ctx, stream, spawn, lambda: g.launch(stream), repeats=50):7.1f}', flush=True)
if __name__ == '__main__':
    main()
