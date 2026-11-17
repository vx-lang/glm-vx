"""The indexer's scores and top-k, old against new (#32), each launch alone: CUDA events after an L2
flush (moe_profile.event_us), the median of --repeats, k = 2048, 32 heads of 128.

  scores        index_scores_{mma,fp8}_at, and its _hist_at twin, which also counts the bins
  topk_select_at          six passes over the scores (long_topk='position', the default)
  topk_select_ilp_at      three passes (long_topk='ilp')
  topk_select_ilp_hist_at two passes, from the scores kernel's bins (long_topk='ilp_hist'); it zeroes the
                          bins, so it is timed after a copy of them, less the copy alone
  topk_mb                 topk_mb_list_at + topk_mb_write_at, the same over blocks of 8192 positions
                          (past topk_blocks_from), timed the same way

Every output is checked: the twins' scores are equal bit for bit, the bins are numpy's, and each top-k
gives np.sort(np.argsort(-scores, kind='stable')[:k]). The last columns are a step's indexer before
(scores + topk_select_at) and after (scores_hist + topk_select_ilp_hist_at).

  python -m validation.topk_hist_ab --output run/topk-hist.md [--kernel mma|fp8|mma16] [--spawn DIR]

--spawn times the PTX in another build directory (default gpu/build/spawn), such as one built by a
different vxc.
"""
import argparse
from pathlib import Path

import numpy as np

from validation.moe_profile import event_us

K = 2048
HEADS, DIM = 32, 128
TOKENS = [3001, 8192, 32768, 65536, 131072, 200000]


def bins(scores):
    b = np.asarray(scores, 'f4').view(np.uint32).astype(np.uint64)
    key = np.where(b >= 0x80000000, 0xFFFFFFFF - b, b ^ 0x80000000)
    key = np.where(b == 0x80000000, 0x80000000, key)
    key = np.where((b & 0x7FFFFFFF) > 0x7F800000, 0, key)
    return np.bincount((key >> 20).astype(np.int64), minlength=4096).astype(np.uint32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--tokens', type=int, nargs='+', default=TOKENS)
    p.add_argument('--kernel', choices=['mma', 'fp8', 'mma16'], default='mma')
    p.add_argument('--spawn', type=Path, default=Path(__file__).resolve().parent.parent / 'gpu' / 'build' / 'spawn')
    args = p.parse_args()
    from gpu.runtime import CUDAContext
    from gpu.spawn.launch import SpawnKernel
    spawn = args.spawn.resolve()
    ctx = CUDAContext(minimum_compute_capability=(8, 0))
    stream = ctx.stream()
    mma = spawn / 'index_mma'
    scores_k = SpawnKernel(ctx, mma / f'index_scores_{args.kernel}_at.ptx')
    hist_k = SpawnKernel(ctx, mma / f'index_scores_{args.kernel}_hist_at.ptx')
    topk = {name: SpawnKernel(ctx, path) for name, path in
            [('topk_select_at', spawn / 'topk_select_at.ptx'), ('topk_select_ilp_at', mma / 'topk_select_ilp_at.ptx'),
             ('topk_select_ilp_hist_at', mma / 'topk_select_ilp_hist_at.ptx'),
             ('topk_mb_list_at', mma / 'topk_mb_list_at.ptx'), ('topk_mb_write_at', mma / 'topk_mb_write_at.ptx')]}
    z = lambda n, dtype: ctx.tensor((n,), dtype, np.zeros(n, dtype).tobytes())
    mb = dict(wcnt=z(4096, np.int32), ckey=z(16384, np.uint32), cpos=z(16384, np.int32), count=z(1, np.uint32),
              offs=z(8192, np.int32), sel=z(4, np.uint32))
    arrivals = z(1, np.int32), z(1, np.int32)
    copy = SpawnKernel(ctx, spawn / 'copy.ptx')
    time = lambda launch: event_us(ctx, stream, spawn, launch, repeats=args.repeats)
    rng = np.random.default_rng(1)
    rows = ['| tokens | scores | scores_hist | topk_select_at | ilp | ilp_hist | topk_mb | before | after (one block) |'
            ' after (blocks) |', '|---|---|---|---|---|---|---|---|---|---|']
    for tokens in args.tokens:
        q = rng.normal(size=(HEADS, DIM)).astype('f4')
        keys = (rng.normal(size=(tokens, DIM)) * rng.uniform(0.1, 10, size=(tokens, 1))).astype('f4')
        weights = (rng.normal(size=HEADS) * HEADS ** -0.5).astype('f4')
        step = ctx.tensor((4,), 'int32', np.array([tokens - 1, tokens, K, 1], 'i4').tobytes())
        common = dict(q=ctx.tensor(q.shape, 'float32', q.tobytes()), weights=ctx.tensor((HEADS,), 'float32', weights.tobytes()),
                      step=step, max_tokens=tokens, heads=HEADS, dim=DIM, scale=float(np.float32(DIM ** -0.5)))
        if args.kernel == 'mma':
            common['keys'] = ctx.tensor(keys.shape, 'float32', keys.tobytes())
        elif args.kernel == 'mma16':  # index_key_bf16_at's words: the key rows rounded to nearest even
            u = keys.view(np.uint32).astype(np.uint64)
            words = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16).view(np.int32).reshape(tokens, 64)
            common['keys'] = ctx.tensor((tokens, 64), 'int32', words.tobytes())
        else:
            from gpu.test_spawn_hardware import fp8_rows
            kc, ks = fp8_rows(keys, bf16_first=True)
            common['codes'] = ctx.tensor((tokens, 32), 'int32', kc.view(np.int32).reshape(tokens, 32).tobytes())
            common['scales'] = ctx.tensor((tokens,), 'float32', ks.tobytes())
        twin, out = ctx.tensor((tokens,), 'float32'), ctx.tensor((tokens,), 'float32')
        hist = ctx.tensor((4096,), 'uint32', np.zeros(4096, np.uint32).tobytes())
        scores_k.launch(stream=stream, out=twin, **common)
        hist_k.launch(stream=stream, out=out, hist=hist, **common)
        stream.synchronize()
        scores = np.frombuffer(out.download(), 'f4')
        if not np.array_equal(scores.view('u4'), np.frombuffer(twin.download(), 'u4')):
            raise SystemExit(f'{tokens}: the _hist_at scores are not its twin\'s')
        counted = np.frombuffer(hist.download(), np.uint32).copy()
        if not np.array_equal(counted, bins(scores)):
            raise SystemExit(f'{tokens}: the bins are not numpy\'s')
        saved = ctx.tensor((4096,), 'uint32', counted.tobytes())
        want = np.sort(np.argsort(-scores, kind='stable')[:K])
        as_int = out.view((tokens,), 'int32')
        ids = ctx.tensor((K,), 'int32')
        restore = lambda: copy.launch(stream=stream, out=hist.view((4096,), 'float32'), x=saved.view((4096,), 'float32'),
                                      n=4096)
        runs = {'topk_select_at': lambda: topk['topk_select_at'].launch(stream=stream, out_ids=ids, scores=as_int,
                                                                        step=step, k=K),
                'topk_select_ilp_at': lambda: topk['topk_select_ilp_at'].launch(stream=stream, out_ids=ids, scores=as_int,
                                                                                step=step, k=K),
                'topk_select_ilp_hist_at': lambda: (restore(), topk['topk_select_ilp_hist_at'].launch(
                    stream=stream, out_ids=ids, scores=as_int, hist=hist, step=step, k=K)),
                'topk_mb': lambda: (restore(), topk['topk_mb_list_at'].launch(
                    stream=stream, scores=as_int, hist=hist, step=step, max_tokens=tokens, k=K, arrivals=arrivals[0], **mb),
                    topk['topk_mb_write_at'].launch(stream=stream, out_ids=ids, scores=as_int, hist=hist, offs=mb['offs'],
                                                    sel=mb['sel'], step=step, max_tokens=tokens, k=K,
                                                    arrivals=arrivals[1]))}
        for name, run in runs.items():
            run()
            stream.synchronize()
            if not np.array_equal(np.frombuffer(ids.download(), 'i4'), want):
                raise SystemExit(f'{tokens}: {name} is not the stable set')
        t = [time(lambda: scores_k.launch(stream=stream, out=twin, **common)),
             time(lambda: (restore(), hist_k.launch(stream=stream, out=out, hist=hist, **common))) - time(restore)]
        t += [time(runs['topk_select_at']), time(runs['topk_select_ilp_at']),
              time(runs['topk_select_ilp_hist_at']) - time(restore), time(runs['topk_mb']) - time(restore)]
        t += [t[0] + t[2], t[1] + t[4], t[1] + t[5]]
        rows.append(f'| {tokens} | ' + ' | '.join(f'{x:.1f}' for x in t) + ' |')
        print(rows[-1], flush=True)
    rows += ['', f'index_scores_{args.kernel}; scores_hist and ilp_hist less a 4096-word copy timed alone; every '
             f'output checked; median of {args.repeats}; PTX from {spawn}.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    main()
