"""The indexer's scores for a prefill chunk on an H100 or H200 (#63, gpu/index_chunk.py): index_q_bf16_at
bit for bit against the NumPy rounding, index_scores_chunk_at against float64 sums of the same bf16
values within gpu.index_chunk.bound, nothing written past each query's position, the _hist_at twin's
counts against the bins of the scores it wrote, and CUDA-event timing at the prefill shapes.

Skipped without a GPU, or when gpu/spawn/index_mma/build.sh has not built the kernels. Each case
appends a record to $GLM_VX_INDEX_CHUNK_JSON, default gpu/build/index_chunk_hardware.json.

    VXC=... PTXAS=... gpu/spawn/index_mma/build.sh
    OMP_NUM_THREADS=16 python -m pytest -q gpu/test_index_chunk_hardware.py
"""
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from gpu import index_chunk as X

OUT_JSON = Path(os.environ.get('GLM_VX_INDEX_CHUNK_JSON', Path(__file__).parent / 'build' / 'index_chunk_hardware.json'))
SENTINEL = np.uint32(0x7FC00123)
SCALE = 128 ** -0.5
RECORDS = []


@pytest.fixture(scope='module')
def kernels():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as error:
        pytest.skip('real GPU unavailable: ' + str(error))
    try:
        if not (X.BUILD / 'index_scores_chunk_at.ptx').exists():
            pytest.skip(f'{X.BUILD} not built: run gpu/spawn/index_mma/build.sh')
        yield ctx, X.IndexChunkScores(ctx)
    finally:
        ctx.close()
        if RECORDS:
            OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
            OUT_JSON.write_text(json.dumps({'device': ctx.name, 'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                            'records': RECORDS}, indent=1) + '\n')


def upload(ctx, array):
    array = np.ascontiguousarray(array)
    return ctx.tensor(array.shape, str(array.dtype), array.tobytes())


def run(kernels, rows, start, cap, *, seed, hist=False, timed=False):
    ctx, scores = kernels
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(rows, X.HEADS * X.DIM)).astype('f4')
    keys = rng.normal(size=(cap, X.DIM)).astype('f4')
    weights = (rng.normal(size=(rows, X.HEADS)) * X.HEADS ** -0.5).astype('f4')
    key_words = X.bf16_words(keys)
    n = start + rows
    out = upload(ctx, np.full((rows, n), SENTINEL, np.uint32).view(np.float32))
    q_words = upload(ctx, np.full((rows, X.HEADS * X.DIM // 2), 7, np.int32))
    dhist = upload(ctx, np.zeros(rows * 4096, np.uint32)) if hist else None
    dq, dk, dw = upload(ctx, q), upload(ctx, key_words), upload(ctx, weights)
    record = {'rows': rows, 'start': start, 'max_tokens': cap, 'hist': hist}
    with ctx.stream() as stream:
        scores(out, q_words, dk, dw, rows=rows, start=start, max_tokens=cap, scale=SCALE, q=dq, hist=dhist, stream=stream)
        stream.synchronize()
        got_words = np.frombuffer(q_words.download(), np.int32).reshape(rows, -1)
        got = np.frombuffer(out.download(), np.float32).reshape(rows, n)
        counts = np.frombuffer(dhist.download(), np.uint32).reshape(rows, 4096) if hist else None
        if timed:
            reps = 10
            start_event, end_event = ctx.event(), ctx.event()
            # A gen-2 collection while the loop launches lets the stream run dry, and the events time the gap
            # as the kernel's (gpu/test_fp8_gemm_grouped_hardware.py saw up to 2x).
            gc.collect()
            gc.disable()
            try:
                start_event.record(stream)
                for _ in range(reps):
                    scores(out, q_words, dk, dw, rows=rows, start=start, max_tokens=cap, scale=SCALE, hist=dhist,
                           stream=stream)
                end_event.record(stream)
                end_event.synchronize()
            finally:
                gc.enable()
            record['us'] = start_event.elapsed_ms(end_event) * 1e3 / reps
            # Multiply-adds of the written elements: 32 heads x 128 dims each.
            pairs = rows * start + rows * (rows + 1) // 2
            record['tflops'] = 2 * X.HEADS * X.DIM * pairs / (record['us'] * 1e-6) / 1e12
    RECORDS.append(record)
    np.testing.assert_array_equal(got_words, X.bf16_words(q))
    future = np.arange(n)[None, :] > start + np.arange(rows)[:, None]
    assert (got.view(np.uint32)[future] == SENTINEL).all(), 'wrote a score past a query\'s position'
    # The float64 reference of every row costs rows * n * 4096 multiply-adds; check a sample of rows.
    sample = np.unique(np.r_[np.arange(min(rows, 66)), rows - 1, np.linspace(0, rows - 1, 16).astype(int)])
    for row in sample:
        want, bound = X.reference_row(got_words[row], key_words, weights[row], start + row, SCALE)
        err = np.abs(got[row, :start + row + 1] - want)
        assert (err <= bound).all(), f'row {row}: max error {err.max()}'
    if hist:
        for row in range(rows):
            want_counts = np.bincount(X.topk_bins(got[row, :start + row + 1]), minlength=4096)
            np.testing.assert_array_equal(counts[row], want_counts, err_msg=f'row {row}')
    return record


@pytest.mark.parametrize('rows,start,cap', [(1, 0, 64), (70, 230, 320), (333, 4000, 8192), (2048, 0, 2048)])
def test_small_and_edge_chunks(kernels, rows, start, cap):
    run(kernels, rows, start, cap, seed=rows + start)


def test_histogram_counts_each_rows_scores(kernels):
    run(kernels, 200, 3000, 4096, seed=11, hist=True)


# Prefill chunks of 2048 tokens at 8K and 32K of context, timed.
@pytest.mark.parametrize('start', [6144, 30720])
def test_prefill_chunks(kernels, start):
    record = run(kernels, 2048, start, start + 2048, seed=start, timed=True)
    assert record['us'] > 0
