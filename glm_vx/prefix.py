"""Instance-scoped, byte-bounded CPU prompt snapshots.

A snapshot is read-only copies of the cache's arrays: each layer's latents, rope
keys, index keys and DSA selections, the absolute position and the final prompt
logits. There is no serialization, so nothing here can load code (#68: no pickle
on the serving path). Each hit copies them into a new cache. This is
prompt reuse, not paged KV or fused prefill. Weights must be immutable while a
scheduler runs; increment model.prefix_cache_revision after any in-place edit.
The budget includes retained Python entry/key/array storage, not live request
caches or temporary copy workspace.
"""
from dataclasses import dataclass
import json
import sys
import threading

import numpy as np
from .kvcache import RowBuffer
from .model import LayerCache, RequestCache


def _size(value):
    """Conservative accounting: repeated immutable references are charged again. An array
    that owns its data counts its bytes (ndarray.__sizeof__)."""
    return sys.getsizeof(value) + (sum(_size(v) for v in value) if isinstance(value, tuple) else 0)


def _frozen(array):
    array = np.array(array, copy=True)  # owns its bytes, so a later decode cannot change them
    array.setflags(write=False)
    return array


def _rows(buffer):
    """The used rows of a RowBuffer or PagedRows, as an owned matrix; None when there are none."""
    array = np.asarray(buffer)
    return _frozen(array) if array.ndim == 2 and len(array) else None


def _snapshot(cache, logits):
    return (cache.position, _frozen(logits),
            tuple((_rows(l.latents), _rows(l.rope_keys), _rows(l.index_keys),
                   None if l.selected_indices is None else _frozen(l.selected_indices))
                  for l in cache.layers))


def _restore(snapshot):
    """A new cache (contiguous RowBuffers, whatever the stored cache used) and logits."""
    position, logits, layers = snapshot
    rows = lambda a: RowBuffer() if a is None else RowBuffer.from_array(a)
    return RequestCache([LayerCache(latents=rows(a), rope_keys=rows(r), index_keys=rows(i),
                                    selected_indices=None if s is None else s.copy())
                         for a, r, i, s in layers], position), logits.copy()


@dataclass(frozen=True)
class Hit:
    cache: RequestCache
    logits: np.ndarray
    tokens: int


class PrefixCache:
    def __init__(self, model, budget_bytes=0):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError('prefix cache budget must be a nonnegative integer')
        self.model = model
        self.budget = budget_bytes
        self.entries = ()  # immutable LRU sequence, oldest first
        self.retained_bytes = 0
        self.lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.rejected = 0
        self.namespace = self._namespace()

    def _namespace(self):
        # Same checkpoint name is insufficient. Bind the actual model, backend,
        # weights owner, complete config and explicitly managed weight revision.
        # The constructor resolves these values outside config. Execution mode
        # and resolved architecture must not reuse snapshots from another path.
        resolved = {name: getattr(self.model, name, None) for name in (
            'n_layers', 'hidden', 'heads', 'rank', 'nope', 'rot', 'value_dim',
            'index_heads', 'index_dim', 'index_topk', 'eps', 'theta',
            'indexer_types', 'mlp_types', 'packed_weights')}
        config = json.dumps(dict(config=self.model.config, resolved=resolved),
                            sort_keys=True, separators=(',', ':'))
        return (id(self.model), id(getattr(self.model, 'weights', None)),
                id(getattr(self.model, 'backend', None)), config,
                str(getattr(self.model, 'prefix_cache_revision', 0)))

    def _refresh(self):
        current = self._namespace()
        if current != self.namespace:
            self.entries = ()
            self.retained_bytes = 0
            self.namespace = current

    def lookup(self, tokens):
        if not self.budget: return None
        tokens = tuple(tokens)
        with self.lock:
            self._refresh()
            candidates = [(len(key), i) for i, (key, _) in enumerate(self.entries)
                          if len(key) <= len(tokens) and tokens[:len(key)] == key]
            if not candidates:
                self.misses += 1
                return None
            length, index = max(candidates)
            entry = self.entries[index]
            self.entries = self.entries[:index] + self.entries[index+1:] + (entry,)
            cache, logits = _restore(entry[1])
            self.hits += 1
            return Hit(cache, logits, length)

    def store(self, tokens, cache, logits):
        if not self.budget: return False
        key = tuple(tokens)
        if not isinstance(cache, RequestCache) or cache.position != len(key):
            raise ValueError('prefix snapshot requires complete prompt RequestCache at exact position')
        logits = np.asarray(logits)
        if logits.shape != (self.model.config['vocab_size'],) or not np.all(np.isfinite(logits)):
            raise ValueError('prefix snapshot requires finite final prompt logits')
        # Reject obviously oversized snapshots before copying them. Scalar/tuple
        # overhead is included in the final exact check.
        raw_bytes = logits.nbytes + sum(
            getattr(layer, name).nbytes for layer in cache.layers
            for name in ('latents', 'rope_keys', 'index_keys'))
        if raw_bytes > self.budget:
            with self.lock: self.rejected += 1
            return False
        entry = (key, _snapshot(cache, logits))
        with self.lock:
            self._refresh()
            if _size((entry,)) > self.budget:
                self.rejected += 1
                return False
            entries = tuple(old for old in self.entries if old[0] != key)
            while entries and _size(entries + (entry,)) > self.budget:
                entries = entries[1:]
                self.evictions += 1
            self.entries = entries + (entry,)
            self.retained_bytes = _size(self.entries)
            assert self.retained_bytes <= self.budget
            return True

    def clear(self):
        with self.lock:
            self.entries = ()
            self.retained_bytes = 0

    def status(self):
        with self.lock:
            return {'budget_bytes': self.budget, 'retained_bytes': self.retained_bytes,
                    'entries': len(self.entries), 'hits': self.hits, 'misses': self.misses,
                    'evictions': self.evictions, 'rejected': self.rejected}
