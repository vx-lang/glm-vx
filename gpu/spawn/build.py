#!/usr/bin/env python3
"""Build a glm-vx kernel through Vx's own GPU path (spawn on), with no extern adapter.

vxc lowers the one `spawn on(Topology::Device)` region in SOURCE to a gpu.func and stores its
PTX in the dispatch payload (`image=`). This script compiles with --emit-llvm, takes that PTX,
renames the entry from vxc's counter name (vx_npu_kernel_N) to the name declared in SOURCE, and
checks the parameter list against the declared signature, since vxc does not document it yet
(Vx #251). The declaration is one comment line:

    // spawn-kernel: NAME(p: i32, x: f32[?], out: mut f32[?])

Parameters are the region's captures in first-use order (an assignment reads its right side
first). A scalar is passed by value; a rank-1 tensor is a memref descriptor of five u64 (base
pointer, aligned pointer, offset, size, stride). The check matches scalar widths and descriptor
layout, and requires that exactly the `mut` tensors are stored to, which catches two tensors
declared in the wrong order when only one is written.
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

SCALARS = {'i32': '.u32', 'i64': '.u64', 'f32': '.f32'}
DECL = re.compile(r'^//\s*spawn-kernel:\s*(\w+)\((.*)\)\s*$', re.M)
INCLUDE = re.compile(r'^//\s*spawn-include:\s*(\S+)\s*$', re.M)


def signature(source: str):
    found = DECL.findall(source)
    if len(found) != 1:
        raise ValueError('source needs exactly one "// spawn-kernel: NAME(...)" line')
    name, text = found[0]
    params = []
    for item in filter(None, (p.strip() for p in text.split(','))):
        pname, kind = (s.strip() for s in item.split(':'))
        mutable = kind.startswith('mut ')
        kind = kind.removeprefix('mut ').strip()
        if kind in SCALARS and not mutable:
            params.append({'name': pname, 'kind': 'scalar', 'type': kind})
        elif re.fullmatch(r'(f32|u16|u8|i32)\[\?\]', kind):
            params.append({'name': pname, 'kind': 'memref', 'element': kind[:-3], 'rank': 1, 'mut': mutable})
        else:
            raise ValueError(f'unsupported parameter {item!r}')
    return name, params


def payload_images(ir: str):
    images = {}
    for match in re.finditer(r'@(\w+)_str\("(.*?)"\)', ir, re.S):
        raw = re.sub(r'\\([0-9A-Fa-f]{2})', lambda x: chr(int(x.group(1), 16)), match.group(2))
        for field in raw.split('\0'):
            if field.startswith('image='):
                images[match.group(1)] = field[len('image='):]
    return images


def check_params(ptx: str, entry: str, params):
    head = re.search(rf'\.entry {entry}\((.*?)\)', ptx, re.S)
    if not head:
        raise ValueError(f'no .entry {entry} in PTX')
    actual = [re.match(r'\s*\.param\s+(\.\w+)(\s+\.ptr)?', line) for line in head[1].split(',')]
    expected, owner = [], []
    for index, p in enumerate(params):
        if p['kind'] == 'scalar':
            expected.append((SCALARS[p['type']], False)); owner.append(index)
        else:
            expected += [('.u64', True), ('.u64', True), ('.u64', False), ('.u64', False), ('.u64', False)]
            owner += [index] * 5
    got = [(m[1].replace('.b', '.u'), bool(m[2])) for m in actual]
    if got != expected:
        raise ValueError(f'PTX parameters {got} do not match the declared signature {expected}')
    # The aligned pointer (second descriptor field) is the one the kernel dereferences.
    body = ptx[head.end():]
    stored = set()
    for index, p in enumerate(params):
        if p['kind'] != 'memref':
            continue
        slot = owner.index(index) + 1
        reg = re.search(rf'ld\.param\.b64\s+(%rd\d+), \[{entry}_param_{slot}\];', body)
        if not reg:
            if p['mut']:
                raise ValueError(f'mut tensor {p["name"]} is never loaded')
            continue
        global_reg = re.search(rf'cvta\.to\.global\.u64\s+(%rd\d+), {re.escape(reg[1])};', body)
        base = global_reg[1] if global_reg else reg[1]
        # Follow address arithmetic one step: add.s64 %rdX, %base, ...
        addresses = {base} | set(re.findall(rf'add\.s64\s+(%rd\d+), {re.escape(base)},', body))
        if any(re.search(rf'st\.global\.\w+\s+\[{re.escape(a)}(\+\d+)?\]', body) for a in addresses):
            stored.add(p['name'])
    declared = {p['name'] for p in params if p['kind'] == 'memref' and p['mut']}
    if stored != declared:
        raise ValueError(f'kernel stores to {sorted(stored)} but the signature marks {sorted(declared)} mut')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--machine', type=Path, default=Path(__file__).with_name('machine.vx'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vxc', default=os.environ.get('VXC', 'vxc'))
    parser.add_argument('--ptxas', help='assemble with this NVIDIA ptxas; failure is fatal')
    parser.add_argument('--arch', default='sm_80')
    args = parser.parse_args()
    try:
        source = args.source.read_text()
        name, params = signature(source)
        # `// spawn-include: FILE` pastes FILE (beside SOURCE) in front: shared device helpers.
        included = [args.source.with_name(f).read_text() for f in INCLUDE.findall(source)]
        env = dict(os.environ, VX_GPU_CHIP=args.arch)
        with tempfile.TemporaryDirectory(prefix='vx-spawn-src-') as temp:
            combined = Path(temp, args.source.name)
            combined.write_text(''.join(included) + source)
            result = subprocess.run([args.vxc, '--machine', str(args.machine.resolve()), str(combined), '--emit-llvm'],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if result.returncode:
            raise ValueError(f'vxc failed ({result.returncode}):\n{result.stderr}')
        images = payload_images(result.stdout)
        if len(images) != 1:
            raise ValueError(f'expected one device image, got {sorted(images)}; a region outside the device '
                             'allow-list falls back to the host silently (Vx #1945)')
        (vx_name, ptx), = images.items()
        if not re.search(rf'^\.target {args.arch}\b', ptx, re.M):
            raise ValueError(f'PTX is not for {args.arch}')
        check_params(ptx, vx_name, params)
        ptx = re.sub(rf'\b{vx_name}(?!\d)', name, ptx)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {'schema': 'glm-vx-spawn-v1', 'entry': name, 'vx_entry': vx_name, 'params': params,
                    'arch': args.arch, 'source_sha256': hashlib.sha256((''.join(included) + source).encode()).hexdigest(),
                    'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                    'vxc': subprocess.run([args.vxc, '--version'], text=True, stdout=subprocess.PIPE).stdout.strip(),
                    'assembly_validated': False, 'gpu_execution_validated': False}
        if args.ptxas:
            with tempfile.TemporaryDirectory(prefix='vx-spawn-') as temp:
                path = Path(temp, 'k.ptx'); path.write_text(ptx)
                assembled = subprocess.run([args.ptxas, f'-arch={args.arch}', '-v', str(path), '-o', str(Path(temp, 'k.cubin'))],
                                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if assembled.returncode:
                    raise ValueError(f'ptxas failed:\n{assembled.stderr}')
                evidence.update({'assembly_validated': True, 'ptxas_log': assembled.stderr.strip().splitlines()})
        args.output.write_text(ptx)
        args.output.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(f'{args.output}: {name} ({len(params)} params) from {vx_name}')
        return 0
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
