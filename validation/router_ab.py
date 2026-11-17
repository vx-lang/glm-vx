"""A MoE layer's router, the launches against fuse_router's one: each variant captured as a CUDA graph
and timed after an L2 flush (moe_profile.event_us), the median of --repeats. GLM-5.3's router: a
[256, 6144] BF16 gate, k = 8, sigmoid scores plus a bias.

  f32 rows + router      matvec_rows on the gate as float32 (twice the bytes), then the hand router
  bf16 + router          the hand bf16_matvec (warp a row, eight rows a block), then the hand router
  bf16_matvec4 + router  matvec_kernels='x4''s projection, then the hand router (ResidentGLM's default today)
  router_bf16            fuse_router: a block a row in the hand bf16_matvec's order, the router in the last
                         block

Checked: router_bf16's logits are the hand bf16_matvec's bit for bit, and its ids the hand router's (the
two sigmoids differ in the last bits, ex2.approx against libdevice's expf, so a near tie could differ:
reported, not fatal).

  python -m validation.router_ab --output run/router-ab.md [--experts 256 16]
"""
import argparse
import ctypes as C
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

COLS, K = 6144, 8


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--experts', type=int, nargs='+', default=[256, 16])
    args = p.parse_args()
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    build = Path(__file__).resolve().parent.parent / 'gpu' / 'build'
    spawn = build / 'spawn'
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    hand_router = ctx.load_ptx(build / 'kernels.ptx').kernel('glm_vx_gpu_router')
    hand_bf16 = ctx.load_ptx(build / 'bf16.ptx').kernel('glm_vx_gpu_bf16_matvec')
    rows_f32 = ctx.load_ptx(build / 'matvec_rows.ptx').kernel('glm_vx_gpu_matvec_rows')
    bf16_matvec4 = SpawnKernel(ctx, spawn / 'bf16_matvec4.ptx')
    fused = SpawnKernel(ctx, spawn / 'router_bf16.ptx')
    rng = np.random.default_rng(5)
    rows = ['| experts | f32 rows + router | bf16 + router | bf16_matvec4 + router | router_bf16 | ids differ |',
            '|---|---|---|---|---|---|']
    for experts in args.experts:
        w = (rng.normal(size=(experts, COLS)).astype('f4').view(np.uint32) >> 16).astype(np.uint16)
        dense = (w.astype(np.uint32) << 16).view('f4')
        x = rng.normal(size=COLS).astype('f4')
        bias = (rng.normal(size=experts) * 0.01).astype('f4')
        up = lambda a, dtype=None: ctx.tensor(a.shape, dtype or str(a.dtype), np.ascontiguousarray(a).tobytes())
        dw, dd, dx, db = up(w.reshape(-1)), up(dense.reshape(-1)), up(x), up(bias)
        words = dw.view((experts * COLS // 2,), 'uint32')
        logits = [ctx.tensor((experts,)) for _ in range(2)]
        ids = [ctx.tensor((K,), 'int32') for _ in range(2)]
        weights = [ctx.tensor((K,)) for _ in range(2)]
        arrivals = ctx.tensor((1,), 'int32', np.zeros(1, np.int32).tobytes())
        router = lambda i: hand_router.launch(1, 32, [ids[i], weights[i], logits[i], db, C.c_int32(experts), C.c_int32(K),
                                                      C.c_float(2.5)], stream=stream)
        variants = {
            'f32 rows + router': lambda: (rows_f32.launch(experts, 1024, [logits[1], dd, dx, C.c_int32(experts),
                                                                          C.c_int32(COLS)], stream=stream), router(1)),
            'bf16 + router': lambda: (hand_bf16.launch(-(-experts // 8), 256, [logits[1], dw, dx, C.c_int32(experts),
                                                                               C.c_int32(COLS)], stream=stream), router(1)),
            'bf16_matvec4 + router': lambda: (bf16_matvec4.launch(stream=stream, out=logits[1], weight=words, x=dx,
                                                                  rows=experts, cols=COLS), router(1)),
            'router_bf16': lambda: fused.launch(stream=stream, indices=ids[0], weights=weights[0], logits=logits[0],
                                                gate=words, x=dx, bias=db, experts=experts, cols=COLS, k=K, scale=2.5,
                                                arrivals=arrivals),
        }
        # The check: router_bf16 against the hand bf16_matvec and the hand router.
        variants['bf16 + router']()
        variants['router_bf16']()
        stream.synchronize()
        got = lambda t, dtype: np.frombuffer(t.download(), dtype)
        if not np.array_equal(got(logits[0], np.uint32), got(logits[1], np.uint32)):
            raise SystemExit(f'{experts}: router_bf16\'s logits are not the hand bf16_matvec\'s')
        differ = int((got(ids[0], np.int32) != got(ids[1], np.int32)).sum())
        if np.frombuffer(arrivals.download(), np.int32)[0] != 0:
            raise SystemExit('arrivals not back to 0')
        times = []
        for name, launch in variants.items():
            launch()  # load the kernels before the capture
            stream.synchronize()
            with stream.capture() as graph:
                launch()
            times.append(event_us(ctx, stream, spawn, lambda: graph.launch(stream), repeats=args.repeats))
        rows.append(f'| {experts} | ' + ' | '.join(f'{t:.1f}' for t in times) + f' | {differ} |')
        print(rows[-1], flush=True)
    rows += ['', f'{COLS} columns, k = {K}; each variant one CUDA graph, timed after an L2 flush, median of '
             f'{args.repeats}, microseconds; ids differ: slots where router_bf16 and the hand router disagree.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
