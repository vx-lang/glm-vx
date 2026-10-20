#!/usr/bin/env python3
"""Build a glm-vx kernel through Vx's own GPU path (spawn on), with no extern adapter.

`vxc --emit-ptx` writes the PTX of the one `spawn on(Topology::Device)` region in SOURCE, with
the entry named after the `pub fn` that holds it, and a manifest of its parameters: the
function's parameters, in declared order (Vx docs/lang/abi.md section 3.5). A scalar is passed
by value; a tensor of rank r is a memref descriptor of 3 + 2r u64 (base pointer, aligned
pointer, offset, sizes, strides).

This script checks vxc's manifest against the PTX it wrote: the parameter widths and
descriptor layout, that exactly the mutable tensors are stored to, and that each tensor is
loaded at its element width. It writes the PTX and our manifest (.json) beside it.
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

SCALARS = {'i32': '.u32', 'i64': '.u64', 'f32': '.f32'}
ELEMENT_BITS = {'f32': 32, 'i32': 32, 'u16': 16, 'u8': 8}
INCLUDE = re.compile(r'^//\s*spawn-include:\s*(\S+)\s*$', re.M)


def kernel_params(manifest_params):
    """vxc's parameter list in our form. vxc's `element` is the MLIR type (u16 is i16), so the
    element comes from the Vx type, `&Tensor<u16, [?], Memory::HBM>`."""
    params = []
    for p in manifest_params:
        if p['kind'] == 'scalar':
            if p['type'] not in SCALARS or p['words'] != 1:
                raise ValueError(f'unsupported scalar parameter {p}')
            params.append({'name': p['name'], 'kind': 'scalar', 'type': p['type']})
        elif p['kind'] == 'tensor':
            element = re.match(r'&(?:mut )?Tensor<(\w+),', p['type'])
            if not element or element[1] not in ELEMENT_BITS or p['rank'] != 1 or p['words'] != 5:
                raise ValueError(f'unsupported tensor parameter {p}')
            if p['mutable'] != p['type'].startswith('&mut '):
                raise ValueError(f'manifest mutable flag disagrees with the type: {p}')
            params.append({'name': p['name'], 'kind': 'memref', 'element': element[1], 'rank': 1, 'mut': p['mutable']})
        else:
            raise ValueError(f'unknown parameter kind {p}')
    return params


def derived(body: str, base: str):
    """Registers computed from BASE by 64-bit address arithmetic (add/sub/mov/cvta), followed
    to a fixed point, so a load through base + row offset + column offset is still traced."""
    steps = [(m[1], set(re.findall(r'%rd\d+', m[2]))) for m in
             re.finditer(r'^\s*(?:add|sub|mov|cvta)[.\w]*\s+(%rd\d+),([^;]*);', body, re.M)]
    found, grew = {base}, True
    while grew:
        grew = False
        for dst, srcs in steps:
            if dst not in found and srcs & found:
                found.add(dst); grew = True
    return found


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
        raise ValueError(f'PTX parameters {got} do not match the vxc manifest {expected}')
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
        addresses = derived(body, base)
        if any(re.search(rf'st\.global\.\w+\s+\[{re.escape(a)}(\+\d+)?\]', body) for a in addresses):
            stored.add(p['name'])
        # Element width: loads through this tensor must be as wide as its element in the manifest,
        # which catches a manifest whose read-only tensors of different types are out of place.
        widths = {int(w) for a in addresses
                  for w in re.findall(rf'ld\.global(?:\.nc)?\.[bsuf](\d+)\s+%\w+, \[{re.escape(a)}(?:\+\d+)?\]', body)}
        if widths - {ELEMENT_BITS[p['element']]}:
            raise ValueError(f'{p["name"]} is {p["element"]} in the manifest but loaded {sorted(widths)} bits wide')
    declared = {p['name'] for p in params if p['kind'] == 'memref' and p['mut']}
    if stored != declared:
        raise ValueError(f'kernel stores to {sorted(stored)} but the manifest marks {sorted(declared)} mut')


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
        # `// spawn-include: FILE` pastes FILE (beside SOURCE) in front: shared device helpers.
        included = [args.source.with_name(f).read_text() for f in INCLUDE.findall(source)]
        env = dict(os.environ, VX_GPU_CHIP=args.arch)
        with tempfile.TemporaryDirectory(prefix='vx-spawn-src-') as temp:
            combined = Path(temp, args.source.name)
            combined.write_text(''.join(included) + source)
            emitted = Path(temp, 'kernels.ptx')
            result = subprocess.run([args.vxc, '--machine', str(args.machine.resolve()), '--emit-ptx', str(combined),
                                     '-o', str(emitted)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            if result.returncode:
                raise ValueError(f'vxc failed ({result.returncode}):\n{result.stderr}')
            ptx = emitted.read_text()
            vx_manifest = json.loads(emitted.with_suffix('.json').read_text())
        kernels = vx_manifest['kernels']
        if not kernels:
            raise ValueError('vxc wrote no GPU kernel; a region outside the device allow-list runs on the host (Vx #1945)')
        if len(kernels) > 1:
            raise ValueError(f'expected one GPU kernel per file, got {[k["name"] for k in kernels]}')
        name = kernels[0]['name']
        if vx_manifest['target'] != args.arch or not re.search(rf'^\.target {args.arch}\b', ptx, re.M):
            raise ValueError(f'PTX is not for {args.arch}')
        params = kernel_params(kernels[0]['params'])
        check_params(ptx, name, params)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {'schema': 'glm-vx-spawn-v2', 'entry': name, 'params': params, 'vx_params': kernels[0]['params'],
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
        print(f'{args.output}: {name} ({len(params)} params)')
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
