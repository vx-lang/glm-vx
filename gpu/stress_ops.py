"""Concurrency stress for every GPUBackend op against independent oracles.

Exists because a full-model comparison gave different (wrong) answers for identical
inputs when several processes shared one GPU. Run N copies at once; any MISMATCH
line names the op. Needs a real GPU; never part of serving.

  for s in 1 2 3 4 5 6; do python -m gpu.stress_ops --seed $s --iters 300 & done; wait
"""
import argparse
import sys

import numpy as np

from glm_vx.backend import NumpyBackend


def mla_oracle(q, qr, kv, kr, scale):
    s = (q.astype('f8') @ kv.astype('f8').T + qr.astype('f8') @ kr.astype('f8').T) * scale
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ kv.astype('f8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--ops', nargs='*', default=None)
    p.add_argument('--readonly-weights', action='store_true',
                   help='mark weights read-only so the backend caches them (as the model does)')
    args = p.parse_args()
    from gpu.backend import GPUBackend
    b, ref = GPUBackend(), NumpyBackend()
    rng = np.random.default_rng(args.seed)
    n = lambda *s: rng.normal(size=s).astype('f4')
    bad = {}

    def check(op, got, want, rtol=3e-4, atol=3e-4):
        if not np.allclose(got, want, rtol=rtol, atol=atol):
            bad[op] = bad.get(op, 0) + 1
            print(f'MISMATCH {op} seed={args.seed} max_err={float(np.max(np.abs(np.asarray(got) - np.asarray(want)))):.3g}', flush=True)

    def ro(a):
        if args.readonly_weights:
            a.flags.writeable = False
        return a

    cases = {}
    cases['matvec'] = lambda: (lambda w, x: check('matvec', b.matvec(ro(w), x), w.astype('f8') @ x.astype('f8')))(n(96, 70), n(70))
    for rows, cols in ((96, 64), (64, 96), (128, 128), (33, 32), (257, 512)):
        cases[f'matvec_{rows}x{cols}'] = (lambda r, c: lambda: (lambda w, x: check(f'matvec_{r}x{c}', b.matvec(ro(w), x), w.astype('f8') @ x.astype('f8')))(n(r, c), n(c)))(rows, cols)
    cases['rmsnorm'] = lambda: (lambda x, w: check('rmsnorm', b.rmsnorm(x, ro(w), 1e-5), ref.rmsnorm(x, w, 1e-5)))(n(256), n(256))
    cases['softmax'] = lambda: (lambda x: check('softmax', b.softmax(x), ref.softmax(x)))(n(512))
    cases['swiglu'] = lambda: (lambda g, u: check('swiglu', b.swiglu(g, u), ref.swiglu(g, u)))(n(384), n(384))
    cases['rope'] = lambda: (lambda x: check('rope', b.rope(x, int(rng.integers(0, 2000)), 10000.0), ref.rope(x, 0, 10000.0)) if False else
                             (lambda pos: check('rope', b.rope(x, pos, 10000.0), ref.rope(x, pos, 10000.0)))(int(rng.integers(0, 2000))))(n(16))
    def route():
        logits, bias = n(8), (rng.normal(size=8) * .1).astype('f4')
        gi, gw = b.route(logits, ro(bias), 2, 2.5)
        ri, rw = ref.route(logits, bias, 2, 2.5)
        check('route', np.asarray(gw)[np.argsort(gi)], np.asarray(rw)[np.argsort(ri)])
        if sorted(np.asarray(gi).tolist()) != sorted(np.asarray(ri).tolist()):
            check('route_ids', [1], [0])
    cases['route'] = route
    def mlp():
        g, u, d, x = n(96, 64), n(96, 64), n(64, 96), n(64)
        h = ref.swiglu((g.astype('f8') @ x).astype('f4'), (u.astype('f8') @ x).astype('f4'))
        check('mlp', b.mlp(ro(g), ro(u), ro(d), x), d.astype('f8') @ h.astype('f8'), 1e-3, 1e-3)
    cases['mlp'] = mlp
    def index_scores():
        heads, tokens, dim = 16, int(rng.integers(1, 300)), 64
        q, k, w = n(heads, dim), n(tokens, dim), n(heads)
        scale = dim ** -.5
        want = (np.maximum(q.astype('f8') @ k.astype('f8').T * scale, 0) * w.astype('f8')[:, None]).sum(0)
        check('index_scores', b.index_scores(q, k, w, scale), want, 1e-3, 1e-3)
    cases['index_scores'] = index_scores
    def mla():
        heads, rank, rdim = 8, 64, 16
        count = int(rng.integers(1, 700))
        q, qr, kv, kr = n(heads, rank) * .3, n(heads, rdim) * .3, n(count, rank), n(count, rdim)
        check('compressed_attention', b.compressed_attention(q, qr, kv, kr, .1), mla_oracle(q, qr, kv, kr, .1), 1e-3, 1e-3)
    cases['compressed_attention'] = mla
    names = args.ops or list(cases)
    for _ in range(args.iters):
        for name in names:
            cases[name]()
    print(f'done seed={args.seed} ops={names} mismatches={bad or 0}')
    b.close()
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
