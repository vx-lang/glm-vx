"""A numpy model of gpu/spawn/index_mma/topk_select_ilp_at.vx, pass by pass as the kernel does it, against
np.argsort: the list path and the path that reads the scores again, on the GPU test's cases.

  python -m validation.topk_ilp_model
"""
import numpy as np

def keys(scores):
    b = scores.view(np.uint32).astype(np.uint64)
    key = b ^ 0x80000000
    key = np.where(b >= 0x80000000, 0xFFFFFFFF - b, key)
    key = np.where(b == 0x80000000, 0x80000000, key)
    key = np.where((b & 0x7FFFFFFF) > 0x7F800000, 0, key)
    return key.astype(np.uint64)

def model(scores, k, cap=2048):
    tokens = len(scores)
    need = min(k, tokens)
    key = keys(scores)
    chunk = (tokens + 4095) // 4096 * 128
    # A
    hist = np.bincount((key >> 20).astype(np.int64), minlength=4096)
    above = np.concatenate([np.cumsum(hist[::-1])[::-1], [0]])  # above[d] = bins d..4095
    lanes = [l for l in range(1024) if above[4 * l] >= need and above[4 * l + 4] < need]
    assert len(lanes) == 1, lanes
    l = lanes[0]
    d = max(j for j in range(4) if above[4 * l + j] >= need)
    B = 4 * l + d
    take = need - above[B + 1]
    # B
    w_of = np.arange(tokens) // chunk
    bin_ = key >> 20
    wsum_b = np.bincount(w_of[bin_ > B], minlength=32)
    listed = (bin_ == B).sum() <= cap
    cand = np.flatnonzero(bin_ == B)  # the list, or the scores again: the same keys
    # rounds
    K, left = B << 20, take
    for r in range(2):
        shift = 10 - 10 * r
        hi = shift + 10
        ck = key[cand]
        m = (ck >> hi) == (K >> hi)
        h2 = np.bincount(((ck[m] >> shift) & 1023).astype(np.int64), minlength=1024)
        a = np.cumsum(h2[::-1])[::-1]
        ds = [dd for dd in range(1024) if a[dd] >= left and a[dd] - h2[dd] < left]
        assert len(ds) == 1, (r, ds)
        dd = ds[0]
        K, left = K + (dd << shift), left - (a[dd] - h2[dd])
    ck, cw = key[cand], w_of[cand]
    wa = wsum_b + np.bincount(cw[ck > K], minlength=32)
    wt = np.bincount(cw[ck == K], minlength=32)
    # D
    out = np.full(k, -7, np.int64)
    for g in range(32):
        na, nt = wa[:g].sum(), wt[:g].sum()
        for p in range(g * chunk, min(tokens, (g + 1) * chunk)):
            a_, e_ = key[p] > K, key[p] == K
            if a_ or (e_ and nt < left):
                out[na + min(nt, left)] = p
            na += a_; nt += e_
    return out, listed

cases = [(2049, 2048, True), (3001, 2048, False), (8192, 2048, False), (70000, 2048, True), (9000, 16, True),
         (100000, 2048, 'equal'), (5000, 2048, 'few'), (131072, 2048, False), (131072, 2048, 'narrow'),
         (40000, 1000, 'near'), (4099, 2048, 'narrow'), (6000, 1, False), (6000, 4000, True)]
for tokens, k, ties in cases:
    rng = np.random.default_rng(tokens * 7 + k)
    if ties == 'equal': scores = np.zeros(tokens, 'f4')
    elif ties == 'few': scores = rng.integers(0, 2, size=tokens).astype('f4')
    elif ties == 'narrow': scores = (5 + 0.01 * rng.normal(size=tokens)).astype('f4')
    elif ties == 'near': scores = np.where(rng.random(tokens) < 0.05, 4 + 0.5 * rng.random(tokens), rng.random(tokens)).astype('f4')
    else: scores = rng.integers(-50, 50, size=tokens).astype('f4') if ties else rng.normal(size=tokens).astype('f4')
    if tokens > 9: scores[[3, 5, 7, 9]] = np.nan, -np.inf, -0.0, np.inf
    want = np.sort(np.argsort(-scores, kind='stable')[:k])
    got, listed = model(scores, k)
    print(tokens, k, ties, 'list' if listed else 'scores', 'OK' if np.array_equal(got, want) else 'WRONG')
