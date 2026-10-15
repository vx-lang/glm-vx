import copy
import pickle

import numpy as np
import pytest

from glm_vx.kvcache import RowBuffer


def rows(n, width=5, seed=0):
    return [r for r in np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)]


def test_append_grows_past_capacity_and_stays_contiguous():
    b, ref = RowBuffer(), rows(100)
    for r in ref:
        b.append(r)
    assert len(b) == 100
    assert b.array.shape == (100, 5) and b.array.flags.c_contiguous
    np.testing.assert_array_equal(b.array, np.stack(ref))


def test_append_copies_and_validates():
    b = RowBuffer()
    r = np.arange(4, dtype=np.float32)
    b.append(r)
    r[:] = -1
    np.testing.assert_array_equal(b[0], [0, 1, 2, 3])
    with pytest.raises(ValueError):
        b.append(np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        b.append(np.zeros((2, 4), np.float32))


def test_indexing_views_and_negative_indices():
    b = RowBuffer(rows(10))
    assert b[-1] is not None
    np.testing.assert_array_equal(b[-1], b[9])
    assert b[0].flags.writeable
    assert not np.shares_memory(b[0], b[1])
    b[0][:] = 7  # in-place mutation reaches the buffer (tests rely on this)
    assert (b.array[0] == 7).all()
    with pytest.raises(IndexError):
        b[10]
    with pytest.raises(IndexError):
        b[-11]
    np.testing.assert_array_equal(b[2:5], b.array[2:5])
    with pytest.raises(TypeError):
        b['x']


def test_truncation_is_the_rollback_form_only():
    b = RowBuffer(rows(8))
    keep = b.array[:5].copy()
    del b[5:]
    assert len(b) == 5
    np.testing.assert_array_equal(b.array, keep)
    b.append(np.ones(5, np.float32))  # reuse after rollback overwrites the freed slot
    assert len(b) == 6 and (b[5] == 1).all()
    del b[100:]  # beyond the end is a no-op, like list
    assert len(b) == 6
    del b[0:]
    assert len(b) == 0
    for bad in (slice(None, 3), slice(0, None, 2), 3):
        with pytest.raises(TypeError):
            del b[bad]


def test_equality_with_lists_and_buffers():
    assert RowBuffer() == []
    assert RowBuffer(rows(3)) == rows(3)
    assert RowBuffer(rows(3)) != rows(4)
    assert RowBuffer(rows(3)) == RowBuffer(rows(3))
    assert RowBuffer(rows(3)) != RowBuffer(rows(3, seed=1))
    assert not RowBuffer() and RowBuffer(rows(1))


def test_numpy_interop_and_empty_shapes():
    b = RowBuffer(rows(4, width=6))
    assert np.shape(b) == (4, 6)
    np.testing.assert_array_equal(np.asarray(b), b.array)
    assert np.asarray(b, dtype=np.float64).dtype == np.float64
    assert np.shape(RowBuffer()) == (0,)  # width unknown until the first row
    assert RowBuffer().array.shape == (0, 0) and RowBuffer().nbytes == 0
    assert b.nbytes == 4 * 6 * 4


def test_gather_order_bounds_and_copy():
    b = RowBuffer(rows(20))
    idx = np.array([7, 3, 3, 19, 0])
    got = b.gather(idx)
    np.testing.assert_array_equal(got, b.array[idx])
    got[:] = 0
    assert not (b.array[7] == 0).all()  # gather returns a copy
    assert b.gather([]).shape == (0, 5)
    for bad in ([20], [-1]):
        with pytest.raises(IndexError):
            b.gather(bad)


def test_pickle_and_copy_are_independent_and_compact():
    b = RowBuffer(rows(5))
    assert b._data.shape[0] >= 16  # spare capacity exists...
    blob = pickle.dumps(b)
    assert len(blob) < 5 * 5 * 4 + 400  # ...but is not serialized
    c = pickle.loads(blob)
    assert c == b and not np.shares_memory(c.array, b.array)
    for d in (copy.copy(b), copy.deepcopy(b)):
        assert d == b and not np.shares_memory(d.array, b.array)
        d[0][:] = 99
        assert not (b[0] == 99).all()
    e = pickle.loads(pickle.dumps(RowBuffer()))
    assert e == [] and len(e) == 0


def test_from_array_owns_its_data():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = RowBuffer.from_array(a)
    a[:] = 0
    np.testing.assert_array_equal(b.array, np.arange(12).reshape(3, 4))
    b.append(np.ones(4, np.float32))
    assert len(b) == 4
    assert len(RowBuffer.from_array(np.empty((0, 4), np.float32))) == 0
    with pytest.raises(ValueError):
        RowBuffer.from_array(np.zeros(4))


def test_randomized_equivalence_with_a_python_list():
    rng = np.random.default_rng(42)
    b, ref = RowBuffer(), []
    for step in range(600):
        op = rng.integers(0, 10)
        if op < 7 or not ref:
            row = rng.normal(size=3).astype(np.float32)
            b.append(row)
            ref.append(row.copy())
        elif op < 9:
            cut = int(rng.integers(0, len(ref) + 1))
            del b[cut:]
            del ref[cut:]
        else:
            i = int(rng.integers(-len(ref), len(ref)))
            np.testing.assert_array_equal(b[i], ref[i])
        assert len(b) == len(ref)
    assert b == ref
    if ref:
        np.testing.assert_array_equal(b.array, np.stack(ref))
        idx = rng.integers(0, len(ref), 7)
        np.testing.assert_array_equal(b.gather(idx), np.stack([ref[i] for i in idx]))
