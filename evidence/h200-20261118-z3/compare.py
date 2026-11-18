"""glm-vx against SGLang 0.5.22 at batch 1: one decode step of the 9-layer proxy (3 dense + 6 MoE layers; the
full indexer on layers 0-2 and 6, the other 5 share), GPU kernel time inside the CUDA graph.

Ours from three step profiles a context: d1 (a dense layer with the full indexer), d1-logits (the same with
the final norm, the bf16 lm_head and the logits' download), m2 (d1's layer and a MoE one, 16 routed
experts: a token reads 8 + the shared one, as GLM-5.3's 256). A shared layer runs no indexer: its cost
(INDEXER, from d1 at the context against d1 at 1000, where the indexer is skipped (#43) but the key row is
still stored) is taken off 5 layers. ROUTER: the router at 256 experts costs about 2 us more a MoE layer
than at 16 (evidence/h200-20261117-z2/router_split). SGLang: its decode step's kernel sum
(sglang/profile/*decode*_kern_sum.csv).

  python evidence/h200-20261118-z3/compare.py
"""
import csv
from pathlib import Path

E = Path(__file__).resolve().parent
# Per full-indexer layer (us): the indexer's wq_b (a 4th fp8_matvec_cvt_tiles), weights_proj (matvec_rows),
# its rope and scale, index_key_at, the scores and the top-k (abdiff of d1-1000 against d1-<ctx>).
INDEXER = {1000: 3.2, 3001: 8.3 + 3.6 + 1.5 + 1.2 + 2.5 + 4.7 + 12.5, 32768: 8.7 + 3.6 + 1.5 + 1.2 + 3.0 + 9.7 + 28.0}
ROUTER = 2.0


def kernels(name):
    rows = [r for r in csv.DictReader(open(E / f'{name}.2_cuda_gpu_trace.csv')) if r.get('Name')]
    kern = sum(int(r['Duration (ns)']) for r in rows if '[CUDA' not in r['Name']) / 1e3
    copies = sum(int(r['Duration (ns)']) for r in rows if '[CUDA memcpy Device-to-Host' in r['Name']) / 1e3
    return kern, copies


def sglang(ctx):
    rows = list(csv.DictReader(open(E / 'sglang' / 'profile' / f'sglang-fp8-decode-bs1-in{ctx}_cuda_gpu_kern_sum.csv')))
    key = [c for c in rows[0] if 'Total Time' in c][0]
    return sum(float(r[key]) for r in rows) / 1e3


def main():
    print('| context | dense layer | MoE layer | head | glm-vx, 9-layer proxy | SGLang | glm-vx / SGLang |')
    print('|---|---|---|---|---|---|---|')
    for ctx in (1000, 3001, 32768):
        d1, _ = kernels(f'd1-{ctx}')
        dl, copy = kernels(f'd1-{ctx}-logits')
        m2, _ = kernels(f'm2-{ctx}')
        dense, moe, head = d1, m2 - d1 + ROUTER, dl - d1 + copy
        ours = 3 * dense + 6 * moe + head - 5 * INDEXER[ctx]
        theirs = sglang(ctx)
        print(f'| {ctx} | {dense:.1f} | {moe:.1f} | {head:.1f} | {ours:.0f} | {theirs:.0f} | {ours / theirs:.2f} |')


if __name__ == '__main__':
    main()
