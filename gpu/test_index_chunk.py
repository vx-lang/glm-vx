"""index_scores_chunk_at without a GPU (#63): the kernel's lanes, emulated from the PTX that
gen_index_chunk.py writes (gpu.index_chunk.emulate), against float64 sums of the same bf16 values; and
the generated sources are what the generator writes."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gpu import index_chunk as X

HERE = Path(__file__).parent


def case(rows, start, cap, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(rows, X.HEADS * X.DIM)).astype('f4')
    keys = rng.normal(size=(cap, X.DIM)).astype('f4')
    weights = (rng.normal(size=(rows, X.HEADS)) * X.HEADS ** -0.5).astype('f4')
    return X.bf16_words(q), X.bf16_words(keys), weights


# (rows, start, max_tokens): one query; a chunk at the start of the cache (all blocks on the diagonal);
# a chunk whose last key is the cache's last row; a query tile cut short (70 = 64 + 6) behind two key tiles.
@pytest.mark.parametrize('rows,start,cap', [(1, 0, 8), (40, 0, 40), (3, 300, 303), (70, 230, 320)])
def test_emulated_kernel_matches_float64_on_bf16_values(rows, start, cap):
    q, keys, weights = case(rows, start, cap, rows + start)
    scale = 128 ** -0.5
    got = X.emulate(q, keys, weights, start, scale)
    want = X.reference(q, keys, weights, start, scale)
    written = ~np.isnan(want)
    np.testing.assert_array_equal(~np.isnan(got), written)
    err = np.abs(got[written] - want[written])
    assert (err <= X.bound(q, keys, weights, start, scale)[written]).all(), err.max()


def test_kernel_writes_nothing_when_the_chunk_does_not_fit_the_cache():
    q, keys, weights = case(5, 10, 14, 3)
    got = X.emulate(q, keys, weights, 10, 0.1, max_tokens=14)
    assert np.isnan(got).all()


def test_reference_is_causal_and_relu_after_scale():
    q = np.zeros((2, X.HEADS * X.DIM), 'f4')
    q[:, 0] = [1, -1]
    keys = np.zeros((3, X.DIM), 'f4')
    keys[:, 0] = [2, 3, 4]
    weights = np.zeros((2, X.HEADS), 'f4')
    weights[:, 0] = [0.5, 2]
    got = X.reference(X.bf16_words(q), X.bf16_words(keys), weights, start=1, scale=0.25)
    np.testing.assert_array_equal(got[0, :2], [0.25, 0.375])
    np.testing.assert_array_equal(got[1], [0, 0, 0])
    assert np.isnan(got[0, 2])


def test_generated_sources_are_current():
    result = subprocess.run([sys.executable, str(HERE / 'spawn/index_mma/gen_index_chunk.py'), '--check'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reference_row_is_a_row_of_reference():
    q, keys, weights = case(6, 20, 30, 9)
    full = X.reference(q, keys, weights, 20, 0.3)
    for row in range(6):
        scores, bound = X.reference_row(q[row], keys, weights[row], 20 + row, 0.3)
        np.testing.assert_allclose(scores, full[row, :21 + row], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bound, X.bound(q, keys, weights, 20, 0.3)[row, :21 + row], rtol=1e-12)
