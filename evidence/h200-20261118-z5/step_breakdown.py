"""GPU 0's last decode steps from an nsys cuda_gpu_trace CSV: kernels per step by name, the span and the gap before it."""
import csv, sys, collections
path, device = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '(0)'
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        if r['Device'].endswith(device) and r['Name'] and not r['Name'].startswith('[CUDA'):
            rows.append((int(r['Start (ns)']), int(r['Duration (ns)']), r['Name']))
rows.sort()
heads = [i for i, (s, d, n) in enumerate(rows) if 'bf16_matvec4' in n and d > 30000]  # the lm_head ends a step
print(len(rows), 'kernels on GPU', device, ';', len(heads), 'steps with a head')
for k in (-3, -2):
    a, b = heads[k - 1] + 1, heads[k] + 1
    step = rows[a:b]
    first = next(i for i, (s, d, n) in enumerate(step) if 'AllGather' not in n)
    body = step[first:]
    span = body[-1][0] + body[-1][1] - body[0][0]
    busy = sum(d for s, d, n in body)
    gap = body[0][0] - (step[first - 1][0] + step[first - 1][1]) if first else body[0][0] - (rows[a - 1][0] + rows[a - 1][1])
    print(f'step {k}: {len(body)} kernels, span {span / 1e3:.0f} us, busy {busy / 1e3:.0f} us, host gap before it {gap / 1e3:.0f} us')
    total = collections.defaultdict(lambda: [0, 0])
    for s, d, n in body:
        total[n[:48]][0] += d; total[n[:48]][1] += 1
    for n, (d, c) in sorted(total.items(), key=lambda x: -x[1][0])[:22]:
        print(f'   {d / 1e3:8.1f} us {c:4d}x {d / c / 1e3:6.1f}  {n}')
