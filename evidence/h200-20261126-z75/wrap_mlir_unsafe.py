"""Vx main (E8016) wants an mlir! block with ensures clauses inside unsafe { }. Wraps each helper body
`return mlir!(<header with ensures>) {` ... `};` (the header may take two lines) as `return unsafe { mlir!(...) { ... } };`,
in .vx sources and in the generators' templates (.py, .vx.in; in an f-string template the braces are doubled).
Arguments: files or directories. Prints each file it changed and how many bodies."""
import pathlib, re, sys
files = []
for a in sys.argv[1:]:
    p = pathlib.Path(a)
    files += sorted(f for f in p.rglob('*') if f.suffix in ('.vx', '.py', '.in')) if p.is_dir() else [p]
total = 0
for path in files:
    lines = path.read_text().split('\n'); n = 0; i = 0
    while i < len(lines):
        if re.match(r'^\s*return mlir!\(', lines[i]) and 'unsafe' not in lines[i]:
            h = i
            while not lines[h].rstrip().endswith('{'):
                h += 1
            doubled = lines[h].rstrip().endswith('{{')
            close = re.compile(r'^\s*\}\};\s*$' if doubled else r'^\s*\};\s*$')
            j = h + 1
            while j < len(lines) and not close.match(lines[j]):
                j += 1
            if j == len(lines):
                sys.exit(f'{path}:{i + 1}: no closing line for this mlir! body')
            if 'ensures' in ' '.join(lines[i:h + 1]):
                o, c = ('{{', '}}') if doubled else ('{', '}')
                lines[i] = lines[i].replace('return mlir!(', f'return unsafe {o} mlir!(', 1)
                lines[j] = lines[j].replace(f'{c};', f'{c} {c};', 1); n += 1
            i = j
        i += 1
    if n:
        path.write_text('\n'.join(lines)); print(path, n); total += n
print('bodies wrapped:', total)
