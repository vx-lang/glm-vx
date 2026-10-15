import pickle

import numpy as np
import pytest

from glm_vx.kvcache import RowBuffer
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from glm_vx.paged_kv import BlockPool, KVPools, PagedRows, PoolExhausted, fork_cache, free_cache
from glm_vx.tiny import tiny_config
from glm_vx.backend import NumpyBackend


def rows(n, width=4, seed=0):
    return list(np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32))


def pool(block_rows=4, blocks=16, width=4):
    return BlockPool(block_rows, width, blocks)


def test_append_gather_and_block_table_layout():
    p = pool()
    r = PagedRows(p, rows(10))
    assert len(r) == 10 and len(r.table) == 3 and p.used_blocks == 3
    np.testing.assert_array_equal(r.array, np.stack(rows(10)))
    idx = np.array([9, 0, 4, 4, 3])
    np.testing.assert_array_equal(r.gather(idx), np.stack(rows(10))[idx])
    np.testing.assert_array_equal(r[-1], rows(10)[9])
    with pytest.raises(IndexError):
        r.gather([10])
    p.check()


def test_truncation_frees_whole_blocks_only():
    p = pool()
    r = PagedRows(p, rows(10))
    del r[6:]
    assert len(r) == 6 and len(r.table) == 2 and p.used_blocks == 2
    r.append(np.ones(4, np.float32))  # partially filled block is reused in place
    assert len(r) == 7 and p.used_blocks == 2
    del r[0:]
    assert len(r) == 0 and p.used_blocks == 0
    p.check()


def test_exhaustion_is_explicit_and_recoverable():
    p = pool(block_rows=2, blocks=3)
    r = PagedRows(p, rows(6))
    with pytest.raises(PoolExhausted):
        r.append(np.zeros(4, np.float32))
    assert len(r) == 6  # failed append changed nothing
    del r[4:]
    r.append(np.zeros(4, np.float32))
    p.check()


def test_fork_shares_prefix_and_copies_on_write():
    p = pool()
    a = PagedRows(p, rows(6))
    b = a.fork()
    assert p.used_blocks == 2  # nothing copied yet
    snapshot = a.array.copy()
    b.append(np.full(4, 7, np.float32))  # tail block is shared and partial -> copied
    assert p.used_blocks == 3
    np.testing.assert_array_equal(a.array, snapshot)  # a unaffected
    assert len(a) == 6 and len(b) == 7
    np.testing.assert_array_equal(b.array[:6], snapshot)
    a.append(np.full(4, 9, np.float32))  # a's tail is now exclusively a's again
    assert (b[6] == 7).all() and (a[6] == 9).all()
    a.free(); b.free()
    assert p.used_blocks == 0
    p.check()


def test_views_of_shared_blocks_cannot_write_through():
    p = pool()
    a = PagedRows(p, rows(4))
    b = a.fork()
    row = b[0]
    row[:] = 123
    np.testing.assert_array_equal(a[0], rows(4)[0])
    np.testing.assert_array_equal(b[0], rows(4)[0])


def test_double_free_and_bad_retain_detected():
    p = pool()
    block = p.allocate()
    p.release(block)
    with pytest.raises(ValueError):
        p.release(block)
    with pytest.raises(ValueError):
        p.retain(block)


def test_pickle_restores_as_pool_independent_contiguous_rows():
    p = pool()
    r = PagedRows(p, rows(9))
    restored = pickle.loads(pickle.dumps(r))
    assert isinstance(restored, RowBuffer) and restored == r
    assert pickle.loads(pickle.dumps(PagedRows(p))) == []


def test_randomized_equivalence_with_rowbuffer_and_pool_invariants():
    rng = np.random.default_rng(7)
    p = pool(block_rows=3, blocks=64)
    seqs = [(PagedRows(p), RowBuffer())]
    for step in range(1500):
        i = int(rng.integers(len(seqs)))
        paged, ref = seqs[i]
        op = int(rng.integers(0, 12))
        if op < 6:
            if p.free_blocks > 2:
                row = rng.normal(size=4).astype(np.float32)
                paged.append(row); ref.append(row)
        elif op < 8:
            cut = int(rng.integers(0, len(ref) + 1))
            del paged[cut:]; del ref[cut:]
        elif op < 9 and len(seqs) < 6 and len(ref) and p.free_blocks > 6:
            seqs.append((paged.fork(), RowBuffer.from_array(ref.array)))
        elif op < 10 and len(seqs) > 1:
            paged.free(); seqs.pop(i)
        elif len(ref):
            idx = rng.integers(0, len(ref), 5)
            np.testing.assert_array_equal(paged.gather(idx), ref.gather(idx))
        for pg, rf in seqs:
            assert len(pg) == len(rf)
        if step % 50 == 0:
            for pg, rf in seqs:
                assert pg == rf
            p.check()
            held = {}
            for pg, _ in seqs:
                for b in pg.table:
                    held[b] = held.get(b, 0) + 1
            assert all(p.refs[b] == c for b, c in held.items())  # refcounts match tables
            assert p.used_blocks == len(held)
    for pg, _ in seqs:
        pg.free()
    assert p.used_blocks == 0
    p.check()


@pytest.fixture(scope='module')
def tiny_model():
    config = tiny_config()
    weights = tiny_weights(config, seed=3)
    for w in weights.values():
        w.flags.writeable = False
    return GlmMoeDsaModel(config, weights, NumpyBackend())


def run(model, cache, tokens):
    return np.stack([model.forward(t, cache) for t in tokens])


def test_model_logits_are_bit_identical_with_paged_cache(tiny_model):
    tokens = [3, 17, 5, 5, 9, 200, 1, 44, 8, 2, 31]  # longer than index_topk=4: sparse path
    expected = run(tiny_model, tiny_model.new_cache(), tokens)
    pools = KVPools.for_model(tiny_model, block_rows=3, blocks=256)
    cache = tiny_model.new_cache(pools)
    np.testing.assert_array_equal(run(tiny_model, cache, tokens), expected)
    free_cache(cache)
    assert pools.used_blocks == 0
    pools.check()


def test_forked_prefix_decodes_like_independent_runs(tiny_model):
    prompt, a_tail, b_tail = [3, 17, 5, 9, 200, 1, 44], [8, 2, 31], [60, 61, 62, 63]
    pools = KVPools.for_model(tiny_model, block_rows=3, blocks=512)
    base = tiny_model.new_cache(pools)
    run(tiny_model, base, prompt)
    used_after_prompt = pools.used_blocks
    a, b = fork_cache(base), fork_cache(base)
    assert pools.used_blocks == used_after_prompt  # fork shares every block
    got_a, got_b = run(tiny_model, a, a_tail), run(tiny_model, b, b_tail)
    np.testing.assert_array_equal(got_a, run(tiny_model, tiny_model.new_cache(), prompt + a_tail)[len(prompt):])
    np.testing.assert_array_equal(got_b, run(tiny_model, tiny_model.new_cache(), prompt + b_tail)[len(prompt):])
    rerun = fork_cache(base)  # base itself was not disturbed by the other forks
    np.testing.assert_array_equal(run(tiny_model, rerun, a_tail), got_a)
    for c in (a, b, rerun, base):
        free_cache(c)
    assert pools.used_blocks == 0
    pools.check()
