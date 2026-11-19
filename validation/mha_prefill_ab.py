"""A prompt's dense attention, mha_prefill (float32 on the CUDA cores) against mha_prefill_tc (bf16 on the tensor
cores), each launch alone for GLM-5.3 at TP8: 8 heads, nope 192, rope 64, value 256. CUDA events after an L2 flush
(moe_profile.event_us), the median of --repeats. Each output is checked against float64 attention: mha_prefill_tc's
over bf16-rounded inputs, to bf16 rounding. TFLOP/s counts the causal products, 2 * (256 + 256) a (query, key, head).

  python -m validation.mha_prefill_ab --output run/mha-prefill.md [--chunks 0:2048 1024:1024 ..] [--spawn DIR]

A chunk is start:queries, the keys 0 .. start + queries.
"""
import argparse
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

HEADS = 8
CHUNKS = ['0:512', '0:1024', '0:2048', '1024:1024', '2000:48']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--chunks', nargs='+', default=CHUNKS)
    p.add_argument('--spawn', type=Path, default=Path(__file__).resolve().parent.parent / 'gpu' / 'build' / 'spawn')
    args = p.parse_args()
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    from gpu.test_spawn_rows_host import mha_float64, mha_tc_inputs
    spawn = args.spawn.resolve()
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    cores, tc = SpawnKernel(ctx, spawn / 'mha_prefill.ptx'), SpawnKernel(ctx, spawn / 'mha_prefill_tc.ptx')
    upload = lambda a: ctx.tensor(a.shape, str(a.dtype), a.tobytes())
    rows = ['| start | queries | keys | mha_prefill | mha_prefill_tc | speedup | tc TFLOP/s |',
            '|---|---|---|---|---|---|---|']
    for chunk in args.chunks:
        start, queries = (int(x) for x in chunk.split(':'))
        q, kv, k_rope = mha_tc_inputs(HEADS, start, queries, seed=start + queries)
        inputs = dict(q=upload(q), kv=upload(kv), k_rope=upload(k_rope), start=start, queries=queries, heads=HEADS,
                      nope=192, rot=64, value=256, scale=np.float32(256 ** -0.5))
        o_cores, o_tc = upload(np.zeros((queries, HEADS, 256), 'f4')), upload(np.zeros((queries, HEADS * 256), 'f4'))
        run_cores = lambda: cores.launch(stream=stream, o=o_cores, **inputs)
        run_tc = lambda: tc.launch(stream=stream, o=o_tc, **inputs)
        run_cores(), run_tc()
        stream.synchronize()
        want = mha_float64(q, kv, k_rope, start, queries)
        got = np.frombuffer(o_tc.download(), 'f4').reshape(queries, HEADS, 256)
        err = float(np.abs(got - want).max())
        if err > 5e-3 * max(1.0, float(np.abs(want).max())):
            raise SystemExit(f'{chunk}: mha_prefill_tc is {err:.3g} from float64')
        t_cores = event_us(ctx, stream, spawn, run_cores, args.repeats)
        t_tc = event_us(ctx, stream, spawn, run_tc, args.repeats)
        flops = 2 * 512 * HEADS * sum(start + i + 1 for i in range(queries))
        rows.append(f'| {start} | {queries} | {start + queries} | {t_cores:.1f} | {t_tc:.1f} | {t_cores / t_tc:.1f}x | '
                    f'{flops / t_tc / 1e6:.0f} |')
        print(rows[-1], flush=True)
    rows += ['', f'Microseconds, median of {args.repeats}; mha_prefill_tc checked against float64; PTX from {spawn}.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
