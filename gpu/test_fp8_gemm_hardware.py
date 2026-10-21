"""The block-FP8 GEMM on an H100 (compute capability 9.0): correctness against the f64 reference
within gpu.fp8_gemm's bound, and CUDA-event timing.

Skipped without a GPU, on any other compute capability (the kernels are sm_90a code), or when
gpu/spawn/fp8_gemm/build.sh has not built gpu/build/spawn/fp8_gemm/. Each case appends a record
(time, TFLOP/s, GB/s, the measured error against the bound and the effective accumulation
precision) to a JSON file: $GLM_VX_FP8_GEMM_JSON, default gpu/build/fp8_gemm_hardware.json.

    VXC=... PTXAS=... gpu/spawn/fp8_gemm/build.sh
    OMP_NUM_THREADS=16 python -m pytest -q gpu/test_fp8_gemm_hardware.py
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from gpu import fp8_gemm as F

OUT_JSON = Path(os.environ.get('GLM_VX_FP8_GEMM_JSON', Path(__file__).parent / 'build' / 'fp8_gemm_hardware.json'))
GUARD = 64     # sentinel elements on each side of C: the masked epilogue must not touch them

# GLM-5.3's FP8 Linears, (N, K): q_a, kv_a, o_proj-like, the dense MLP's gate/up and down.
# lm_head (154880 x 6144) is BF16 in the checkpoint and is not here.
MODEL = [(2048, 6144), (576, 6144), (6144, 2048), (12288, 6144), (6144, 12288)]
DECODE_M = [1, 7, 64, 128, 256]
TAILS = [  # (m, n, k): K tails (k % 128 != 0), N tails, M tails, odd sizes
    (5, 2048, 6160), (64, 576, 1040), (130, 1000, 6144), (65, 130, 400), (255, 300, 2064), (1, 128, 16),
]
RECORDS = []


@pytest.fixture(scope='module')
def gemm():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(9, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        if ctx.compute_capability != (9, 0):
            pytest.skip(f'the FP8 GEMM is sm_90a code; this GPU is {ctx.compute_capability}')
        if not (F.BUILD / 'fp8_gemm_m64.ptx').exists():
            pytest.skip(f'{F.BUILD} not built: run gpu/spawn/fp8_gemm/build.sh')
        yield ctx, F.FP8GEMM(ctx)
    finally:
        ctx.close()
        if RECORDS:
            manifests = {p.stem: json.loads(p.read_text()) for p in sorted(F.BUILD.glob('*.json'))}
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps({
                'device': ctx.name, 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'kernels': {k: {'ptx_sha256': v.get('ptx_sha256'), 'vxc': v.get('vxc'),
                                'ptxas_log': v.get('ptxas_log'), 'dynamic_shared_bytes': v.get('dynamic_shared_bytes')}
                            for k, v in manifests.items()},
                'eps_mma_bound': F.EPS_MMA, 'records': RECORDS}, indent=1) + '\n')


def upload(ctx, array):
    return ctx.tensor((array.size,), str(array.dtype), np.ascontiguousarray(array).tobytes())


def run_case(gemm, m, n, k, *, bf16, timed, seed):
    ctx, kernels = gemm
    a, a_scale, b, b_scale = F.random_case(m, n, k, seed)
    dtype = 'uint16' if bf16 else 'float32'
    sentinel = np.full(m * n + 2 * GUARD, 0x7FC1 if bf16 else np.nan, dtype=np.uint16 if bf16 else np.float32)
    if not bf16:
        sentinel.view(np.uint32)[:] = 0x7FC00123
    backing = upload(ctx, sentinel)
    out = backing.view((m * n,), offset=GUARD * sentinel.itemsize)
    da, dsa, db, dsb = upload(ctx, a), upload(ctx, a_scale), upload(ctx, b), upload(ctx, b_scale)
    with ctx.stream() as stream:
        p = kernels(da, dsa, db, dsb, out, m=m, n=n, k=k, stream=stream)
        stream.synchronize()
        raw = np.frombuffer(backing.download(), dtype=sentinel.dtype)
        record = {'m': m, 'n': n, 'k': k, 'out': dtype, 'variant': p.variant.name, 'splits': p.splits,
                  'split_blocks': p.split_blocks, 'grid': p.grid, 'block': p.block}
        if timed:
            for _ in range(3):
                kernels(da, dsa, db, dsb, out, m=m, n=n, k=k, stream=stream)
            start, end = ctx.event(), ctx.event()
            reps = 20
            start.record(stream)
            for _ in range(reps):
                kernels(da, dsa, db, dsb, out, m=m, n=n, k=k, stream=stream)
            end.record(stream)
            end.synchronize()
            seconds = start.elapsed_ms(end) / 1e3 / reps
            record.update({'time_us': seconds * 1e6, 'tflops': 2 * m * n * k / seconds / 1e12,
                           'gbps': F.bytes_moved(m, n, k, out_bytes=2 if bf16 else 4) / seconds / 1e9})
    assert raw[:GUARD].tobytes() == sentinel[:GUARD].tobytes(), 'wrote before C'
    assert raw[-GUARD:].tobytes() == sentinel[-GUARD:].tobytes(), 'wrote after C'
    got = raw[GUARD:-GUARD].reshape(m, n)
    got = F.bf16_value(got).astype(np.float64) if bf16 else got.astype(np.float64)
    ref, bound, mma = F.reference(a, a_scale, b, b_scale, splits=p.splits, bf16=bf16, terms=True)
    err = np.abs(got - ref)
    finite = np.isfinite(got)
    record.update({'max_err_over_bound': float(np.max(np.where(finite, err / bound, np.inf))),
                   'median_err_over_bound': float(np.median(err / bound)),
                   'effective_eps_mma': float(np.max(err / np.maximum(mma, 1e-300))),
                   'max_rel_err': float(err.max() / max(np.abs(ref).max(), 1e-300)),
                   'c_sha256': hashlib.sha256(got.tobytes()).hexdigest()[:16]})
    RECORDS.append(record)
    assert finite.all(), f'{int((~finite).sum())} elements of C were not written'
    bad = err > bound
    assert not bad.any(), (f'{int(bad.sum())} of {m * n} outside the bound; worst at {np.unravel_index(np.argmax(err / bound), err.shape)}: '
                           f'{record}')
    return record


@pytest.mark.parametrize('m', DECODE_M)
@pytest.mark.parametrize('n,k', MODEL)
def test_model_shapes(gemm, m, n, k):
    run_case(gemm, m, n, k, bf16=True, timed=True, seed=m * 31 + n + k)


@pytest.mark.parametrize('m,n,k', TAILS)
@pytest.mark.parametrize('bf16', [False, True])
def test_tails(gemm, m, n, k, bf16):
    run_case(gemm, m, n, k, bf16=bf16, timed=False, seed=m + n * 3 + k)


@pytest.mark.parametrize('splits', [1, 2, 5])
def test_split_k_agrees(gemm, splits):
    """Every split count gives C within the bound; the reduction is in a fixed order, so a
    second run is bitwise the same."""
    ctx, kernels = gemm
    m, n, k = 7, 576, 6144
    a, a_scale, b, b_scale = F.random_case(m, n, k, 5)
    da, dsa, db, dsb = upload(ctx, a), upload(ctx, a_scale), upload(ctx, b), upload(ctx, b_scale)
    out = ctx.tensor((m * n,), 'float32')
    runs = []
    for _ in range(2):
        kernels(da, dsa, db, dsb, out, m=m, n=n, k=k, splits=splits)
        ctx.synchronize()
        runs.append(np.frombuffer(out.download(), np.float32).reshape(m, n))
    assert runs[0].tobytes() == runs[1].tobytes()
    ref, bound = F.reference(a, a_scale, b, b_scale, splits=splits)
    assert np.all(np.abs(runs[0] - ref) <= bound)


@pytest.mark.parametrize('variant', ['m64', 'm128'])
def test_variants_agree_on_small_m(gemm, variant):
    ctx, kernels = gemm
    m, n, k = 33, 384, 1024
    a, a_scale, b, b_scale = F.random_case(m, n, k, 9)
    da, dsa, db, dsb = upload(ctx, a), upload(ctx, a_scale), upload(ctx, b), upload(ctx, b_scale)
    out = ctx.tensor((m * n,), 'float32')
    kernels(da, dsa, db, dsb, out, m=m, n=n, k=k, variant=variant, splits=1)
    ctx.synchronize()
    ref, bound = F.reference(a, a_scale, b, b_scale)
    assert np.all(np.abs(np.frombuffer(out.download(), np.float32).reshape(m, n) - ref) <= bound)
