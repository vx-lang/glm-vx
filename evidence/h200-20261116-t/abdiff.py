"""Per-kernel sums of nsys cuda_gpu_trace CSVs side by side: abdiff.py A.csv B.csv"""
import csv, sys
from collections import defaultdict
def load(p):
    d = defaultdict(float); c = defaultdict(int)
    for r in csv.DictReader(open(p)):
        n = r.get('Name') or ''
        if not n or n.startswith('[CUDA'):
            continue
        n = n.split('(')[0][:60]; d[n] += float(r['Duration (ns)']) / 1e3; c[n] += 1
    return d, c
(a, ca), (b, cb) = load(sys.argv[1]), load(sys.argv[2])
print(f'total {sum(a.values()):8.1f} -> {sum(b.values()):8.1f} us ({len(sys.argv[1:])} traces)')
for n in sorted(set(a) | set(b), key=lambda n: -max(a.get(n, 0), b.get(n, 0))):
    if abs(a.get(n, 0) - b.get(n, 0)) >= 0.3:
        print(f'{a.get(n, 0):7.1f} x{ca.get(n, 0):<3} {b.get(n, 0):7.1f} x{cb.get(n, 0):<3} {n}')
