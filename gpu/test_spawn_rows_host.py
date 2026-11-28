"""The prefill rows kernels (#63, item 5b) on the CPU through Vx's own runtime (gpu/spawn/host.py), against NumPy
and against the decode kernels they repeat a row at a time: fp8_quant_rows (fp8_quant_groups' codes and scales),
router_logits_rows + router_topk_rows (router_bf16's bits a token), moe_combine_rows, bf16_rows (the rows for cuBLASLt's
bfloat16 matmul).

Skipped unless gpu/spawn/build.sh made the libraries (VX_DISPATCH_LIB set)."""
import numpy as np
import pytest

from gpu import fp8_quant
from gpu.test_spawn_vx_host import F, bits_equal, run  # noqa: F401  (run: the module's fixture)


@pytest.mark.parametrize('rows,cols,gather', [(5, 384, False), (7, 6144, True), (1, 128, True), (40, 4224, False)])
def test_fp8_quant_rows_is_quant_groups_row_major(run, rows, cols, gather):
    rng = np.random.default_rng(rows * cols)
    source = rows + 3 if gather else rows
    x = (rng.normal(size=(source, cols)) * rng.choice([1e-3, 1.0, 300.0], size=(source, 1))).astype(F)
    x[0, :128] = 0.0  # an all-zero group: the 1e-10 floor
    x[-1, 5] = np.nan
    x[-1, 7] = -1e6  # past 448 times the scale's other values: clamped
    src = rng.permutation(source)[:rows].astype(np.int32) if gather else np.zeros(1, np.int32)
    got = run('fp8_quant_rows', codes=np.zeros((rows, cols), np.uint8), scales=np.zeros((rows, cols // 128), F), x=x,
              src=src, rows=rows, cols=cols, gather=int(gather))
    picked = x[src] if gather else x
    # The hand kernel's max skips a NaN (vxk_quant_max), NumPy's would give the group a NaN scale: the NaN alone
    # gets the NaN code.
    codes, scales = fp8_quant.quant_groups(np.where(np.isnan(picked), F(0), picked))  # scales [cols / 128, rows]
    codes[np.isnan(picked)] = 127
    np.testing.assert_array_equal(got['codes'], codes)
    bits_equal(got['scales'], scales.T)


def test_bf16_rows_round_each_value_to_nearest_even(run):
    rng = np.random.default_rng(11)
    count, cols, rows = 5, 6144, 8  # rows past count keep their old bits (cuBLASLt's padding, not read back)
    x = (rng.normal(size=(rows, cols)) * rng.choice([1e-30, 1e-3, 1.0, 1e37], size=(rows, 1))).astype(F)
    x[0, :4] = [0.0, -0.0, np.inf, -np.inf]
    x[1, :3] = (np.array([0x3F808000, 0x3F818000, 0x3F80FFFF], np.uint32)).view(F)  # ties to even, and below a tie
    got = run('bf16_rows', out=np.full((rows, cols), 0xABCD, np.uint16), x=x, count=count, cols=cols)['out']
    np.testing.assert_array_equal(got[:count], bf16_bits(x[:count]))
    assert got[1, 0] == 0x3F80 and got[1, 1] == 0x3F82 and got[1, 2] == 0x3F81
    assert (got[count:] == 0xABCD).all()


def test_moe_combine_rows_sums_in_the_given_order(run):
    rng = np.random.default_rng(7)
    tokens, cols, kk, rows = 6, 300, 9, 60
    y = rng.normal(size=(rows, cols)).astype(F)
    slots = rng.integers(0, rows, size=(tokens, kk)).astype(np.int32)
    weights = rng.random(size=(tokens, kk)).astype(F)
    got = run('moe_combine_rows', out=np.zeros((tokens, cols), F), y=y, slots=slots, weights=weights, tokens=tokens,
              cols=cols, kk=kk)
    want = np.zeros((tokens, cols), F)
    for j in range(kk):
        want = (want + weights[:, j:j + 1] * y[slots[:, j]]).astype(F)
    bits_equal(got['out'], want)


@pytest.mark.parametrize('experts,cols,k,tokens', [(256, 6144, 8, 3), (33, 520, 8, 4), (7, 136, 3, 2)])
def test_router_rows_are_router_bf16_a_token(run, experts, cols, k, tokens):
    rng = np.random.default_rng(experts + cols)
    w = (rng.normal(size=(experts, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=(tokens, cols)).astype(F)
    x[-1] = x[0]  # two tokens with one row: the same choice
    bias = (rng.normal(size=experts) * 0.1).astype(F)
    gate = w.view(np.uint32).reshape(-1)
    logits = run('router_logits_rows', logits=np.zeros((tokens, experts), F), gate=gate, x=x, tokens=tokens,
                 experts=experts, cols=cols)['logits']
    stride = k + 1
    got = run('router_topk_rows', indices=np.full((tokens, stride), -7, np.int32), weights=np.full((tokens, stride), 3.0, F),
              logits=logits, bias=bias, tokens=tokens, experts=experts, k=k, scale=2.5)
    for t in range(tokens):
        one = run('router_bf16', indices=np.full(stride, -7, np.int32), weights=np.full(stride, 3.0, F),
                  logits=np.zeros(experts, F), gate=gate, x=x[t], bias=bias, experts=experts, cols=cols, k=k, scale=2.5,
                  arrivals=np.zeros(1, np.uint32))
        bits_equal(logits[t], one['logits'])
        np.testing.assert_array_equal(got['indices'][t], one['indices'])
        bits_equal(got['weights'][t], one['weights'])
    assert (got['indices'][:, k] == -7).all() and (got['weights'][:, k] == 3.0).all()


def bf16_bits(x):
    """float32 to bfloat16 bits, nearest even (Vx's `as bf16`)."""
    b = np.asarray(x, F).view(np.uint32).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def test_kv_rows_prefill_bf16_is_kv_rows_bf16_at_a_row(run):
    # The chunk's cache rows: each row's RMSNorm (rmsnorm_rows' sums with `threads` lanes) and rope, as bf16, at
    # start + t; against decode's kv_rows_bf16_at on the same row (its latent normalized by rmsnorm_rows), bit for bit.
    rng = np.random.default_rng(11)
    rows, rank, rot, start, cap, threads = 5, 512, 64, 3, 12, 256
    compressed = rng.normal(size=(rows, rank + rot)).astype(F)
    weight = rng.normal(size=rank).astype(F)
    angles = rng.normal(size=(rows, rot // 2)).astype(F)
    steps = np.zeros((rows, 128), F)
    steps[:, :rot // 2], steps[:, rot // 2:rot] = np.cos(angles), np.sin(angles)
    got = run('kv_rows_prefill_bf16', latents=np.zeros((cap, rank), np.uint16), rope_keys=np.zeros((cap, rot), np.uint16),
              compressed=compressed, weight=weight, steps=steps, start=start, rows=rows, rank=rank, rot=rot, eps=1e-6,
              threads=threads)
    for t in range(rows):
        normed = run('rmsnorm_rows', out=np.zeros((1, rank), F), x=compressed[t, :rank].copy(), weight=weight, rows=1,
                     dim=rank, eps=1e-6, threads=threads)['out'][0]
        one = run('kv_rows_bf16_at', latents=np.zeros((cap, rank), np.uint16), rope_keys=np.zeros((cap, rot), np.uint16),
                  latent_row=normed, rope_x=compressed[t, rank:].copy(), cosine=steps[t, :rot // 2].copy(),
                  sine=steps[t, rot // 2:rot].copy(), step=np.array([start + t, 0, 0, 0], np.int32), rank=rank, rot=rot)
        np.testing.assert_array_equal(got['latents'][start + t], one['latents'][start + t])
        np.testing.assert_array_equal(got['rope_keys'][start + t], one['rope_keys'][start + t])
    assert not got['latents'][:start].any() and not got['latents'][start + rows:].any()
    out = run('rows_bf16_f32', out=np.zeros((cap, rank), F), x=got['latents'], rows=cap, cols=rank)['out']
    np.testing.assert_array_equal(bf16_bits(out), got['latents'])


def test_q_rows_and_mha_prefill_are_causal_attention(run):
    # q_rows_prefill: each (token, head)'s nope values, then its rope values rotated as the rope cache's (packed
    # [even, odd]); mha_prefill: query i at start + i over keys 0..start + i, against float64 NumPy.
    rng = np.random.default_rng(13)
    heads, nope, rot, value, start, queries = 3, 24, 8, 16, 5, 4
    keys = start + queries
    query = rng.normal(size=(queries, heads * (nope + rot))).astype(F)
    angles = rng.normal(size=(queries, rot // 2)).astype(F)
    steps = np.zeros((queries, 64), F)
    steps[:, :rot // 2], steps[:, rot // 2:rot] = np.cos(angles), np.sin(angles)
    q = run('q_rows_prefill', q_out=np.zeros((queries, heads, nope + rot), F), query=query, steps=steps, rows=queries,
            heads=heads, nope=nope, rot=rot)['q_out']
    raw = query.reshape(queries, heads, nope + rot)
    even, odd = raw[:, :, nope::2], raw[:, :, nope + 1::2]
    c, s = steps[:, None, :rot // 2], steps[:, None, rot // 2:rot]
    want_q = np.concatenate([raw[:, :, :nope], even * c - odd * s, odd * c + even * s], -1)
    np.testing.assert_allclose(q, want_q, rtol=1e-6, atol=1e-6)
    kv = rng.normal(size=(keys, heads, nope + value)).astype(F)
    k_rope = rng.normal(size=(keys, rot)).astype(F)
    scale = F((nope + rot) ** -0.5)
    o = run('mha_prefill', o=np.zeros((queries, heads, value), F), q=q, kv=kv, k_rope=k_rope, start=start, queries=queries,
            heads=heads, nope=nope, rot=rot, value=value, scale=scale)['o']
    want = np.zeros((queries, heads, value))
    for i in range(queries):
        n = start + i + 1
        for h in range(heads):
            score = (kv[:n, h, :nope].astype(np.float64) @ q[i, h, :nope] + k_rope[:n] @ q[i, h, nope:]) * scale
            p = np.exp(score - score.max())
            want[i, h] = p @ kv[:n, h, nope:] / p.sum()
    np.testing.assert_allclose(o, want, rtol=2e-5, atol=2e-6)


def test_index_scores_and_topk_rows_are_each_querys_stable_top_k(run):
    # index_scores_rows: query i's scores over keys 0..start + i (sum_h w * relu(q.k * scale)); topk_select_rows: the
    # stable top-k set of each row in position order, nothing for a row with tokens <= k. Ties: repeated keys.
    rng = np.random.default_rng(17)
    heads, dim, start, queries, k = 4, 16, 9, 6, 5
    keys = rng.normal(size=(start + queries, dim)).astype(F)
    keys[3] = keys[7]  # equal scores: the lower position first
    q = rng.normal(size=(queries, heads, dim)).astype(F)
    w = rng.random(size=(queries, heads)).astype(F)
    scale = F(dim ** -0.5)
    cols = 32
    scores = run('index_scores_rows', scores=np.full((queries, cols), -7.0, F), q=q, keys=keys, weights=w, start=start,
                 queries=queries, heads=heads, dim=dim, scale=scale)['scores']
    for i in range(queries):
        n = start + i + 1
        want = (np.maximum((q[i].astype(np.float64) @ keys[:n].T) * scale, 0) * w[i][:, None]).sum(0)
        np.testing.assert_allclose(scores[i, :n], want, rtol=2e-5, atol=1e-6)
        assert (scores[i, n:] == -7.0).all()
    ids = run('topk_select_rows', out_ids=np.full((queries, k), -1, np.int32), scores=scores.view(np.int32), start=start,
              rows=queries, k=k)['out_ids']
    for i in range(queries):
        n = start + i + 1
        np.testing.assert_array_equal(ids[i], np.sort(np.argsort(-scores[i, :n], kind='stable')[:k]))


def test_mha_prefill_sparse_attends_to_each_querys_selection(run):
    rng = np.random.default_rng(19)
    heads, nope, rot, value, start, queries, topk = 2, 8, 4, 8, 3, 5, 4
    keys = start + queries
    q = rng.normal(size=(queries, heads, nope + rot)).astype(F)
    kv = rng.normal(size=(keys, heads, nope + value)).astype(F)
    k_rope = rng.normal(size=(keys, rot)).astype(F)
    selected = np.zeros((queries, topk), np.int32)
    for i in range(queries):
        if start + i + 1 > topk:
            selected[i] = np.sort(rng.choice(start + i + 1, topk, replace=False))
    scale = F((nope + rot) ** -0.5)
    o = run('mha_prefill_sparse', o=np.zeros((queries, heads, value), F), q=q, kv=kv, k_rope=k_rope, selected=selected,
            start=start, queries=queries, heads=heads, nope=nope, rot=rot, value=value, topk=topk, scale=scale)['o']
    for i in range(queries):
        n = start + i + 1
        pick = np.arange(n) if n <= topk else selected[i]
        for h in range(heads):
            score = (kv[pick, h, :nope].astype(np.float64) @ q[i, h, :nope] + k_rope[pick] @ q[i, h, nope:]) * scale
            p = np.exp(score - score.max())
            np.testing.assert_allclose(o[i, h], p @ kv[pick, h, nope:] / p.sum(), rtol=2e-5, atol=2e-6)


def mha_float64(q, kv, k_rope, start, queries, nope=192):
    """Causal attention in float64 over bf16-rounded q, keys and values: query i at start + i sees keys
    0..start + i. o [queries, heads, value]."""
    rounded = lambda x: (bf16_bits(x).astype(np.uint32) << 16).view(F).astype(np.float64)
    q, kv, k_rope = rounded(q), rounded(kv), rounded(k_rope)
    heads, keys, scale = q.shape[1], start + queries, q.shape[2] ** -0.5
    seen = np.arange(keys)[None, :] <= start + np.arange(queries)[:, None]
    o = np.zeros((queries, heads, kv.shape[2] - nope))
    for h in range(heads):
        score = (q[:, h, :nope] @ kv[:keys, h, :nope].T + q[:, h, nope:] @ k_rope[:keys].T) * scale
        p = np.exp(np.where(seen, score, -np.inf) - np.where(seen, score, -np.inf).max(1, keepdims=True))
        o[:, h] = p @ kv[:keys, h, nope:] / p.sum(1, keepdims=True)
    return o


def mha_tc_inputs(heads, start, queries, seed, rising=False):
    """q [queries, heads, 256], kv [keys, heads, 448] and k_rope [keys, 64] of normal values. With rising, every
    query's rope part is one vector r and key j's is 3 r j / keys, so later keys score higher and a query's max
    rises with each tile of keys."""
    rng = np.random.default_rng(seed)
    keys = start + queries
    q = rng.normal(size=(queries, heads, 256)).astype(F)
    kv = rng.normal(size=(keys, heads, 448)).astype(F)
    k_rope = rng.normal(size=(keys, 64)).astype(F)
    if rising:
        r = rng.normal(size=64).astype(F)
        q[:, :, 192:] = r
        k_rope = (3 * r[None, :] * (np.arange(keys, dtype=F) / keys)[:, None]).astype(F)
    return q, kv, k_rope


def run_mha_tc(run, q, kv, k_rope, start, queries, rows=None, nope=192, fill=0.0):
    heads = q.shape[1]
    o = np.full((rows or queries, heads * 256), fill, F)
    return run('mha_prefill_tc', o=o, q=q, kv=kv, k_rope=k_rope, start=start, queries=queries, heads=heads, nope=nope,
               rot=64, value=256, scale=F(256 ** -0.5))['o']


# mha_prefill_tc: mha_prefill's attention for GLM-5.3's shapes (nope 192, rope 64, value 256) on the tensor cores,
# 64 queries and 64 keys a tile. Its products are of bf16 values with f32 sums, and P is rounded to bf16 for P V, so
# it agrees with float64 attention over bf16-rounded inputs to bf16 rounding, not bit for bit.
@pytest.mark.parametrize('heads,start,queries,rising', [
    (2, 0, 70, False),  # query 0 sees one key; two query tiles, the second partly past queries
    (1, 100, 5, False),  # one query tile over two key tiles, the second partly past the queries
    (3, 37, 130, False),  # three query tiles over three key tiles, none of them on a multiple of 64
    (1, 0, 1, False),
    (2, 200, 40, True),  # the max of each query rises with each of its four key tiles
])
def test_mha_prefill_tc_is_causal_attention_to_bf16_rounding(run, heads, start, queries, rising):
    q, kv, k_rope = mha_tc_inputs(heads, start, queries, seed=heads * 1000 + start + queries, rising=rising)
    got = run_mha_tc(run, q, kv, k_rope, start, queries).reshape(queries, heads, 256)
    want = mha_float64(q, kv, k_rope, start, queries)
    np.testing.assert_allclose(got, want, rtol=0, atol=5e-3 * max(1.0, float(np.abs(want).max())))


def test_mha_prefill_tc_writes_its_tiles_rows_only_and_nothing_off_its_shape(run):
    # o with more rows than queries: the last tile (rows 64..127) writes all its rows, the rows after it keep their
    # values. With nope 128 the kernel writes nothing.
    heads, start, queries = 1, 5, 70
    q, kv, k_rope = mha_tc_inputs(heads, start, queries, seed=23)
    got = run_mha_tc(run, q, kv, k_rope, start, queries, rows=140, fill=7.0)
    want = mha_float64(q, kv, k_rope, start, queries).reshape(queries, 256)
    np.testing.assert_allclose(got[:queries], want, rtol=0, atol=5e-3 * max(1.0, float(np.abs(want).max())))
    assert np.isfinite(got[queries:128]).all() and (got[128:] == 7.0).all()
    assert (run_mha_tc(run, q, kv, k_rope, start, queries, nope=128, fill=7.0) == 7.0).all()


@pytest.mark.parametrize('tokens,kk,experts', [(5, 3, 7), (40, 9, 257), (300, 9, 257)])
def test_moe_group_kernels_are_a_stable_sort_by_expert(run, tokens, kk, experts):
    # moe_group_count / _offsets / _place (Vx's std::nn counting sort) and moe_group_finish: the pairs grouped by
    # expert in pair order, the starts, each pair's grouped row, and each token's grouped rows ranked by expert id (with
    # its router slot).
    rng = np.random.default_rng(tokens)
    ids = np.stack([rng.choice(experts - 1, kk - 1, replace=False) for _ in range(tokens)]).astype(np.int32)
    ids = np.concatenate([ids, np.full((tokens, 1), experts - 1, np.int32)], 1)  # the shared expert last
    rows = tokens * kk
    blocks = -(-rows // 256)
    flat = ids.reshape(-1).copy()
    counts = run('moe_group_count', counts=np.zeros((blocks, experts), np.int32), ids=flat, n=rows, experts=experts)['counts']
    got = run('moe_group_offsets', base=np.zeros((blocks, experts), np.int32), starts=np.zeros(experts + 1, np.int32),
              tile_expert=np.zeros(rows, np.int32), order=np.zeros(rows, np.int32), counts=counts, n=rows, experts=experts,
              align=1)
    placed = run('moe_group_place', order=got['order'], row_of=np.full(rows, -1, np.int32), ids=flat, base=got['base'],
                 n=rows)
    want = np.argsort(flat, kind='stable')
    np.testing.assert_array_equal(placed['order'], want)
    np.testing.assert_array_equal(got['starts'], np.searchsorted(flat[want], np.arange(experts + 1)))
    slot_of = np.empty(rows, np.int64)
    slot_of[want] = np.arange(rows)
    np.testing.assert_array_equal(placed['row_of'], slot_of)
    fin = run('moe_group_finish', slots=np.full((tokens, kk), -1, np.int32), picks=np.full((tokens, kk), -1, np.int32),
              row_of=placed['row_of'], ids=flat, tokens=tokens, kk=kk)
    by_id = np.argsort(ids, axis=1, kind='stable')
    np.testing.assert_array_equal(fin['picks'].reshape(tokens, kk), by_id)
    np.testing.assert_array_equal(fin['slots'].reshape(tokens, kk), slot_of.reshape(tokens, kk)[np.arange(tokens)[:, None], by_id])
