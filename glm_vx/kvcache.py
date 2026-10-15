"""Contiguous per-layer KV storage.

`RowBuffer` replaces the per-token Python lists the cache used to hold. Rows live
in one preallocated float32 array that doubles when full, so the used prefix is
always a single C-contiguous [n, width] matrix: selected rows are gathered with
one fancy-index instead of `np.stack` over a list, the buffer uploads to a device
in one copy, and snapshot/handoff serialization is a plain slice.

It deliberately keeps the list surface the engine already relies on (`append`,
`len`, integer/slice indexing, `del x[n:]` rollback, iteration, `== []`, pickling)
so call sites that only append, truncate and index need no change.
"""
import numpy as np


class RowBuffer:
    __slots__ = ('_data', '_n')

    MIN_CAPACITY = 16

    def __init__(self, rows=None):
        self._data = None
        self._n = 0
        if rows is not None:
            for row in rows:
                self.append(row)

    @classmethod
    def from_array(cls, array):
        """Own a copy of a [n, width] float32 matrix (no per-row Python objects)."""
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError('from_array expects a matrix')
        buffer = cls()
        if len(array):
            buffer._data = np.empty((max(len(array), cls.MIN_CAPACITY), array.shape[1]), dtype=np.float32)
            buffer._data[:len(array)] = array
            buffer._n = len(array)
        return buffer

    def append(self, row):
        row = np.asarray(row, dtype=np.float32)
        if row.ndim != 1:
            raise ValueError('rows must be vectors')
        if self._data is None:
            self._data = np.empty((self.MIN_CAPACITY, row.shape[0]), dtype=np.float32)
        elif row.shape[0] != self._data.shape[1]:
            raise ValueError(f'row width {row.shape[0]} differs from buffer width {self._data.shape[1]}')
        if self._n == len(self._data):
            grown = np.empty((2 * len(self._data), self._data.shape[1]), dtype=np.float32)
            grown[:self._n] = self._data[:self._n]
            self._data = grown
        self._data[self._n] = row
        self._n += 1

    @property
    def array(self):
        """View of the used rows: C-contiguous [n, width] (shape (0, 0) when empty)."""
        if self._data is None:
            return np.empty((0, 0), dtype=np.float32)
        return self._data[:self._n]

    @property
    def nbytes(self):
        return self._n * (0 if self._data is None else self._data.shape[1]) * 4

    def gather(self, indices):
        """Copy of the selected rows, in the order given: [len(indices), width]."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= self._n):
            raise IndexError('gather index out of range')
        return self.array[indices]

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.array[index]
        if isinstance(index, (int, np.integer)):
            if not -self._n <= index < self._n:
                raise IndexError('row index out of range')
            return self._data[index % self._n if index < 0 else index]
        raise TypeError('RowBuffer supports integer and slice indices only')

    def __delitem__(self, index):
        """Truncate: only `del buffer[n:]` (the rollback form) is supported."""
        if not isinstance(index, slice) or index.step not in (None, 1) or index.stop is not None:
            raise TypeError('RowBuffer supports only truncation `del buffer[n:]`')
        start = 0 if index.start is None else index.start
        if start < 0:
            start = max(0, self._n + start)
        self._n = min(self._n, start)

    def __iter__(self):
        for i in range(self._n):
            yield self._data[i]

    def __bool__(self):
        return self._n > 0

    def __eq__(self, other):
        if isinstance(other, RowBuffer):
            return self._n == other._n and bool(np.array_equal(self.array, other.array))
        if isinstance(other, (list, tuple)):
            return len(other) == self._n and all(np.array_equal(a, b) for a, b in zip(self, other))
        return NotImplemented

    __hash__ = None

    def __array__(self, dtype=None, copy=None):
        out = self.array if self._data is not None else np.empty((0,), dtype=np.float32)
        if dtype is not None and np.dtype(dtype) != out.dtype:
            return out.astype(dtype)
        return out.copy() if copy else out

    def __copy__(self):
        return RowBuffer.from_array(self.array) if self._data is not None else RowBuffer()

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __reduce__(self):
        # Pickle only the used rows, never the spare capacity.
        return (_restore, (None if self._data is None else self.array.copy(),))

    def __repr__(self):
        return f'RowBuffer(n={self._n}, width={None if self._data is None else self._data.shape[1]})'


def _restore(array):
    return RowBuffer() if array is None else RowBuffer.from_array(array)
