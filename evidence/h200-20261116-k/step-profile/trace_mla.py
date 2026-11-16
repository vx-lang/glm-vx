import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ks = [r for r in rows if r.get('Name') and not r['Name'].startswith('[CUDA')]
print(len(ks), 'kernels; total', round(sum(float(r['Duration (ns)']) for r in ks) / 1e3, 1), 'us')
for r in ks:
    n = r['Name']
    d = float(r['Duration (ns)']) / 1e3
    if any(k in n for k in ('mla', 'matvec_heads', 'cols_cvt', 'store_row', 'rope', 'q_bf16', 'topk', 'index', 'kv')):
        print(f'{d:7.1f}  {n[:80]}')
