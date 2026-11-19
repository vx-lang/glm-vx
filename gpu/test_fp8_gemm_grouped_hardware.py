"""The grouped block-FP8 GEMM of the routed experts on an H100 or H200 (compute capability 9.0):
the device plan against gpu.fp8_gemm.grouped_tiles, C (f32, and bfloat16 bits) against grouped_reference
within its bound,
and CUDA-event timing of the plan and the GEMM.

Skipped without a GPU, on any other compute capability, or when gpu/spawn/fp8_gemm/build.sh has not
built gpu/build/spawn/fp8_gemm/. Each case appends a record to $GLM_VX_FP8_GROUPED_JSON, default
gpu/build/fp8_gemm_grouped_hardware.json.

    VXC=... PTXAS=... gpu/spawn/fp8_gemm/build.sh
    OMP_NUM_THREADS=16 python -m pytest -q gpu/test_fp8_gemm_grouped_hardware.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from gpu import fp8_gemm as F

OUT_JSON = Path(os.environ.get('GLM_VX_FP8_GROUPED_JSON', Path(__file__).parent / 'build' / 'fp8_gemm_grouped_hardware.json'))
GUARD = 64
EXPERTS = 256
# Per rank at TP8: gate_up (N = 2 x 256, K = 6144) and down (N = 6144, K = 256). Rows = chunk tokens x 9.
MODEL = [(512, 6144), (6144, 256)]
CHUNKS = [256, 2048]
# (rows, experts, n, k): small cases with empty experts, a tile that crosses into the next expert,
# one row, and every row in one expert.
SMALL = [(300, 8, 256, 512), (1, 4, 128, 128), (65, 2, 128, 256), (200, 16, 384, 1024)]
RECORDS = []


@pytest.fixture(scope='module')
def grouped():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(9, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        if ctx.compute_capability != (9, 0):
            pytest.skip(f'the FP8 GEMM is sm_90a code; this GPU is {ctx.compute_capability}')
        if not (F.BUILD / 'fp8_gemm_grouped_m64.ptx').exists():
            pytest.skip(f'{F.BUILD} not built: run gpu/spawn/fp8_gemm/build.sh')
        yield ctx, F.FP8GroupedGEMM(ctx)
    finally:
        ctx.close()
        if RECORDS:
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps({'device': ctx.name, 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                            'records': RECORDS}, indent=1) + '\n')


def upload(ctx, array):
    return ctx.tensor((array.size,), str(array.dtype), np.ascontiguousarray(array).tobytes())


def run_case(grouped, rows, experts, n, k, *, seed, timed, bf16=False):
    ctx, kernels = grouped
    a, a_scale, b, b_scale, offsets = F.random_grouped_case(rows, experts, n, k, seed)
    tiles_max = F.grouped_tiles_max(rows, experts)
    # A NaN in every element of C and of the guards, so an element not written stays NaN.
    if bf16:
        sentinel = np.full(rows * n + 2 * GUARD, 0x7FC1, np.uint16)
    else:
        sentinel = np.empty(rows * n + 2 * GUARD, np.float32)
        sentinel.view(np.uint32)[:] = 0x7FC00123
    backing = upload(ctx, sentinel)
    out = backing.view((rows * n,), offset=GUARD * sentinel.itemsize)
    tiles = upload(ctx, np.full(1 + 3 * tiles_max, -1, np.int32))
    da, dsa, db, dsb, doff = (upload(ctx, x) for x in (a, a_scale, b, b_scale, offsets))
    record = {'rows': rows, 'experts': experts, 'n': n, 'k': k, 'bf16': bf16, 'tiles_max': tiles_max,
              'experts_with_rows': int((np.diff(offsets) > 0).sum())}
    with ctx.stream() as stream:
        kernels.plan(doff, tiles, rows=rows, experts=experts, stream=stream)
        kernels(da, dsa, db, dsb, out, tiles, rows=rows, n=n, k=k, experts=experts, stream=stream)
        stream.synchronize()
        table = np.frombuffer(tiles.download(), np.int32)
        raw = np.frombuffer(backing.download(), sentinel.dtype)
        if timed:
            for _ in range(3):
                kernels(da, dsa, db, dsb, out, tiles, rows=rows, n=n, k=k, experts=experts, stream=stream)
            reps = 20
            for name, call in [('plan', lambda: kernels.plan(doff, tiles, rows=rows, experts=experts, stream=stream)),
                               ('gemm', lambda: kernels(da, dsa, db, dsb, out, tiles, rows=rows, n=n, k=k,
                                                        experts=experts, stream=stream))]:
                start, end = ctx.event(), ctx.event()
                start.record(stream)
                for _ in range(reps):
                    call()
                end.record(stream)
                end.synchronize()
                record[f'{name}_us'] = start.elapsed_ms(end) * 1e3 / reps
            # Bytes the GEMM must read at least: the weights and scales of each expert with rows, A,
            # a_scale, and C written once.
            used = record['experts_with_rows']
            least = (used * (n * k + 4 * (n // F.BLOCK_N) * -(-k // F.BLOCK_K)) + rows * k
                     + 4 * rows * -(-k // F.BLOCK_K) + sentinel.itemsize * rows * n)
            record['gbps'] = least / (record['gemm_us'] * 1e-6) / 1e9
    want = F.grouped_tiles(offsets, tiles_max)
    count = int(want[0])
    assert table[0] == count, f'plan has {table[0]} entries, want {count}'
    assert (table[1:1 + 3 * count] == want[1:1 + 3 * count]).all(), 'plan entries differ from grouped_tiles'
    assert raw[:GUARD].tobytes() == sentinel[:GUARD].tobytes(), 'wrote before C'
    assert raw[-GUARD:].tobytes() == sentinel[-GUARD:].tobytes(), 'wrote after C'
    got = raw[GUARD:-GUARD].reshape(rows, n)
    got = (F.bf16_value(got) if bf16 else got).astype(np.float64)
    ref, bound = F.grouped_reference(a, a_scale, b, b_scale, offsets, bf16=bf16)
    err = np.abs(got - ref)
    finite = np.isfinite(got)
    record.update({'max_err_over_bound': float(np.max(np.where(finite, err / bound, np.inf))),
                   'max_rel_err': float(err.max() / max(np.abs(ref).max(), 1e-300))})
    RECORDS.append(record)
    assert finite.all(), f'{int((~finite).sum())} elements of C were not written'
    bad = err > bound
    assert not bad.any(), f'{int(bad.sum())} of {rows * n} outside the bound: {record}'
    return record


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('rows,experts,n,k', SMALL)
def test_small_cases(grouped, rows, experts, n, k, bf16):
    run_case(grouped, rows, experts, n, k, seed=rows + experts + n + k, timed=False, bf16=bf16)


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('chunk', CHUNKS)
@pytest.mark.parametrize('n,k', MODEL)
def test_prefill_shapes(grouped, chunk, n, k, bf16):
    run_case(grouped, chunk * 9, EXPERTS, n, k, seed=chunk + n + k, timed=True, bf16=bf16)
