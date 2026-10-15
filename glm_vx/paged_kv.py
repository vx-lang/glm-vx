"""Paged KV storage: fixed-size blocks from a shared pool, per-sequence block tables.

The CPU reference for M2's paged cache. `PagedRows` has the same surface as
`RowBuffer` (append, len, indexing, `del x[n:]` truncation, `gather`, `array`), so
`LayerCache` can hold either. What paging adds:

* a bounded pool (`BlockPool`) shared by many sequences, so capacity is a block
  count, not per-sequence preallocation, and exhaustion is an explicit error;
* reference-counted blocks, so `fork()` shares a prefix between sequences with
  copy-on-write on the first write to a shared block (prefix reuse without copying
  the prompt's cache);
* fixed block size, which maps one-to-one onto device allocations later.

Blocks are never zeroed on reuse; only the used prefix of a sequence is readable.
"""
import numpy as np


class PoolExhausted(RuntimeError):
    pass


class BlockPool:
    def __init__(self, block_rows, width, capacity_blocks):
        if block_rows < 1 or width < 1 or capacity_blocks < 1:
            raise ValueError('block_rows, width and capacity_blocks must be positive')
        self.block_rows, self.width, self.capacity = int(block_rows), int(width), int(capacity_blocks)
        self.storage = np.empty((self.capacity, self.block_rows, self.width), dtype=np.float32)
        self.refs = np.zeros(self.capacity, dtype=np.int64)
        self._free = list(range(self.capacity - 1, -1, -1))

    @property
    def free_blocks(self):
        return len(self._free)

    @property
    def used_blocks(self):
        return self.capacity - len(self._free)

    def allocate(self):
        if not self._free:
            raise PoolExhausted(f'KV block pool exhausted ({self.capacity} blocks of {self.block_rows} rows)')
        block = self._free.pop()
        self.refs[block] = 1
        return block

    def retain(self, block):
        if self.refs[block] < 1:
            raise ValueError('retain of a free block')
        self.refs[block] += 1

    def release(self, block):
        if self.refs[block] < 1:
            raise ValueError('double free of a KV block')
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self._free.append(block)

    def check(self):
        """Invariant: every block is either free (ref 0, on the free list once) or held."""
        free = sorted(self._free)
        assert len(set(free)) == len(free), 'duplicate block on free list'
        assert all(self.refs[b] == 0 for b in free), 'free block with references'
        assert sum(1 for b in range(self.capacity) if self.refs[b] == 0) == len(free), 'leaked block'


class PagedRows:
    __slots__ = ('pool', 'table', '_n')

    def __init__(self, pool, rows=None):
        self.pool, self.table, self._n = pool, [], 0
        if rows is not None:
            for row in rows:
                self.append(row)

    def _writable_tail(self):
        """Block that receives the next row, copying it first if another sequence shares it."""
        offset = self._n % self.pool.block_rows
        if offset == 0:
            self.table.append(self.pool.allocate())
        else:
            tail = self.table[-1]
            if self.pool.refs[tail] > 1:
                fresh = self.pool.allocate()
                self.pool.storage[fresh] = self.pool.storage[tail]
                self.pool.release(tail)
                self.table[-1] = fresh
        return self.table[-1], offset

    def append(self, row):
        row = np.asarray(row, dtype=np.float32)
        if row.shape != (self.pool.width,):
            raise ValueError(f'row must have shape ({self.pool.width},), got {row.shape}')
        block, offset = self._writable_tail()
        self.pool.storage[block, offset] = row
        self._n += 1

    def fork(self):
        """New sequence sharing every block of this one (copy-on-write on append)."""
        other = PagedRows(self.pool)
        other.table = list(self.table)
        other._n = self._n
        for block in self.table:
            self.pool.retain(block)
        return other

    def free(self):
        for block in self.table:
            self.pool.release(block)
        self.table, self._n = [], 0

    @property
    def nbytes(self):
        return self._n * self.pool.width * 4

    @property
    def array(self):
        """Contiguous copy of the used rows (paged data is not contiguous by design)."""
        if self._n == 0:
            return np.empty((0, 0), dtype=np.float32)
        return self.pool.storage[self.table].reshape(-1, self.pool.width)[:self._n]

    def gather(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= self._n):
            raise IndexError('gather index out of range')
        table = np.asarray(self.table, dtype=np.int64)
        block_rows = self.pool.block_rows
        return self.pool.storage[table[indices // block_rows], indices % block_rows]

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.array[index]
        if isinstance(index, (int, np.integer)):
            if not -self._n <= index < self._n:
                raise IndexError('row index out of range')
            index = index % self._n if index < 0 else index
            block, offset = self.table[index // self.pool.block_rows], index % self.pool.block_rows
            if self.pool.refs[block] > 1:
                # A view of a shared block could be mutated through, bypassing COW.
                return self.pool.storage[block, offset].copy()
            return self.pool.storage[block, offset]
        raise TypeError('PagedRows supports integer and slice indices only')

    def __delitem__(self, index):
        if not isinstance(index, slice) or index.step not in (None, 1) or index.stop is not None:
            raise TypeError('PagedRows supports only truncation `del rows[n:]`')
        start = 0 if index.start is None else index.start
        if start < 0:
            start = max(0, self._n + start)
        if start >= self._n:
            return
        keep_blocks = -(-start // self.pool.block_rows)
        for block in self.table[keep_blocks:]:
            self.pool.release(block)
        del self.table[keep_blocks:]
        self._n = start

    def __iter__(self):
        for i in range(self._n):
            yield self[i]

    def __bool__(self):
        return self._n > 0

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return len(other) == self._n and all(np.array_equal(a, b) for a, b in zip(self, other))
        if hasattr(other, 'array') and hasattr(other, 'gather'):
            return len(other) == self._n and (self._n == 0 or bool(np.array_equal(self.array, other.array)))
        return NotImplemented

    __hash__ = None

    def __array__(self, dtype=None, copy=None):
        out = self.array if self._n else np.empty((0,), dtype=np.float32)
        if dtype is not None and np.dtype(dtype) != out.dtype:
            return out.astype(dtype)
        return out.copy() if copy else out

    def __reduce__(self):
        # Snapshots (prefix cache, handoff) are pool-independent: they restore as a
        # contiguous RowBuffer holding the used rows.
        return (_restore_rows, (self.array.copy() if self._n else None,))

    def __copy__(self):
        return self.fork()

    def __deepcopy__(self, memo):
        copy = PagedRows(self.pool)
        for i in range(self._n):
            copy.append(self[i])
        return copy


def _restore_rows(array):
    from .kvcache import RowBuffer
    return RowBuffer() if array is None else RowBuffer.from_array(array)


class KVPools:
    """The three block pools a model's cache draws from (latents, rope keys, index keys)."""

    def __init__(self, rank, rope_dim, index_dim, block_rows, blocks):
        self.latents = BlockPool(block_rows, rank, blocks)
        self.rope_keys = BlockPool(block_rows, rope_dim, blocks)
        self.index_keys = BlockPool(block_rows, index_dim, blocks)

    @classmethod
    def for_model(cls, model, block_rows, blocks):
        return cls(model.rank, model.rot, model.index_dim, block_rows, blocks)

    def check(self):
        for pool in (self.latents, self.rope_keys, self.index_keys):
            pool.check()

    @property
    def used_blocks(self):
        return self.latents.used_blocks + self.rope_keys.used_blocks + self.index_keys.used_blocks


def fork_cache(cache):
    """Share every block of a paged RequestCache; either side may then append or roll back."""
    from .model import LayerCache, RequestCache
    return RequestCache([LayerCache(latents=l.latents.fork(), rope_keys=l.rope_keys.fork(),
                                    index_keys=l.index_keys.fork(),
                                    selected_indices=None if l.selected_indices is None else l.selected_indices.copy())
                         for l in cache.layers], cache.position)


def free_cache(cache):
    for layer in cache.layers:
        for rows in (layer.latents, layer.rope_keys, layer.index_keys):
            rows.free()
    cache.position = 0
