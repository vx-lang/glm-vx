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

With --library, it also compiles the same source to a host object (vxc --action emit-obj;
needs llvm-config on PATH) and links it with Vx's dispatch library into a shared library
whose `pub fn` Vx's own runtime launches (gpu/spawn/vx_launch.py).
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

SCALARS = {'i32': '.u32', 'i64': '.u64', 'f32': '.f32'}
ELEMENT_BITS = {'f32': 32, 'i32': 32, 'u16': 16, 'u8': 8}
INCLUDE = re.compile(r'^//\s*spawn-include:\s*(\S+)\s*$', re.M)


def split_top(text: str):
    """Split on commas outside <> and [] (Tensor<f32, [?], Memory::HBM> has commas inside)."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += ch in '<[' and 1 or ch in '>]' and -1 or 0
        if ch == ',' and depth == 0:
            parts.append(text[start:i]); start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def ptxas_target(ptx_arch: str, ptxas_arch) -> str:
    """The GPU ptxas assembles for: the real device, which may be newer than the PTX target
    (an H100 runs sm_80 PTX through the driver JIT, so ptxas checks it for sm_90), never older."""
    target = ptxas_arch or ptx_arch
    match, base = re.fullmatch(r'sm_(\d+)a?', target), re.fullmatch(r'sm_(\d+)a?', ptx_arch)
    if not match or not base:
        raise ValueError(f'arch must look like sm_90; got {target!r} / {ptx_arch!r}')
    if int(match[1]) < int(base[1]):
        raise ValueError(f'ptxas arch {target} is older than the PTX target {ptx_arch}')
    # Arch-specific PTX (sm_90a: wgmma, setmaxnreg) runs only on that exact chip.
    if ptx_arch.endswith('a') and target != ptx_arch:
        raise ValueError(f'{ptx_arch} PTX assembles only for {ptx_arch}, not {target}')
    return target


def machine_chip(machine: Path):
    """The `chip:` of the machine file's topology (Vx #1982), or None."""
    found = re.search(r'^\s*chip:\s*(sm_\w+)', machine.read_text(), re.M)
    return found[1] if found else None


def check_dynamic_shared(ptx, nbytes):
    """The manifest's dynamic_shared_bytes (Vx #1984, PR #1995) agrees with the PTX: an
    `.extern .shared` array exactly when it is positive, since a launch passes that size."""
    if type(nbytes) is not int or nbytes < 0:
        raise ValueError(f'manifest dynamic_shared_bytes must be a nonnegative integer, got {nbytes!r}')
    extern = re.search(r'^\s*\.extern \.shared\b', ptx, re.M) is not None
    if extern != (nbytes > 0):
        raise ValueError(f'the manifest says {nbytes} dynamic shared bytes, but the PTX '
                         f'{"declares" if extern else "has no"} .extern .shared memory')
    return nbytes


def raise_ptx_version(ptx: str, version: str):
    """PTX with its `.version` raised to VERSION, and the version vxc wrote. vxc pins sm_90a to PTX
    8.0, and fence.proxy.tensormap::generic (which a tensor map in global memory needs, CUDA
    programming guide 4.12.2.2) needs 8.3; a newer version accepts every older instruction, so only
    the header changes. Never lower it."""
    found = re.search(r'^\.version (\d+)\.(\d+)\s*$', ptx, re.M)
    wanted = re.fullmatch(r'(\d+)\.(\d+)', version or '')
    if not found or not wanted:
        raise ValueError(f'cannot raise the PTX version to {version!r}: no .version line or a bad version')
    old, new = (int(found[1]), int(found[2])), (int(wanted[1]), int(wanted[2]))
    if new < old:
        raise ValueError(f'--ptx-version {version} is older than the {old[0]}.{old[1]} vxc wrote')
    return ptx[:found.start()] + f'.version {version}' + ptx[found.end():], f'{old[0]}.{old[1]}'


def same_kernel(a: str, b: str) -> bool:
    """Two PTX files hold the same kernel apart from its entry name and label numbering."""
    def body(ptx):
        ptx = ptx[ptx.index('.entry'):]
        entry = re.match(r'\.entry (\w+)', ptx)[1]
        return re.sub(r'\$L__BB\d+_', '$L__BB_', re.sub(rf'\b{entry}(?=\b|_param)', 'K', ptx))
    return body(a) == body(b)


def build_library(args, combined_text, name, params, env, ptx):
    """The shared library Vx launches from. Its `pub fn` takes every parameter of the function,
    so they must be exactly the kernel's, in the manifest's order, for the call to line up."""
    declared = re.search(rf'^pub fn {name}\((.*?)\)\s*->', combined_text, re.M | re.S)
    names = [item.split(':', 1)[0].strip() for item in split_top(declared[1])] if declared else None
    if names != [p['name'] for p in params]:
        raise ValueError(f'pub fn {name} takes {names}, but the kernel uses {[p["name"] for p in params]}')
    if not args.dispatch_lib or not Path(args.dispatch_lib).is_file():
        raise ValueError('--library needs --dispatch-lib (or VX_DISPATCH_LIB): the libvx_cuda_dispatch.so vxc was built with')
    dispatch = Path(args.dispatch_lib).resolve()
    with tempfile.TemporaryDirectory(prefix='vx-spawn-lib-') as temp:
        combined, obj = Path(temp, args.source.name), Path(temp, 'k.o')
        combined.write_text(combined_text)
        # The optimization level as for --emit-ptx: the device image follows it, and the PTX checked
        # above must be the PTX this library launches.
        result = subprocess.run([args.vxc, '--machine', str(args.machine.resolve()), str(combined),
                                 '--action', 'emit-obj', '-o', str(obj)], text=True, capture_output=True, env=env)
        if result.returncode or not obj.exists():
            raise ValueError(f'vxc emit-obj failed ({result.returncode}):\n{result.stderr}')
        args.library.parent.mkdir(parents=True, exist_ok=True)
        linked = subprocess.run([args.cc, '-shared', str(obj), str(dispatch), f'-Wl,-rpath,{dispatch.parent}',
                                 '-o', str(args.library)], text=True, capture_output=True)
        if linked.returncode:
            raise ValueError(f'linking {args.library} failed:\n{linked.stderr}')
    # The library carries its kernel in the dispatch payload (`image=`); it must be the PTX that
    # was checked, or a test of Vx's launch would test other code.
    images = re.findall(rb'image=(.*?)\x00', args.library.read_bytes(), re.S)
    if len(images) != 1 or not same_kernel(images[0].decode(), ptx):
        args.library.unlink()
        raise ValueError(f'{args.library.name} does not carry the kernel that --emit-ptx wrote')
    return {'library': args.library.name, 'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'dispatch_lib': str(dispatch)}


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
    parser.add_argument('--arch', help="PTX target; default the machine file's chip:, else sm_80")
    parser.add_argument('--ptxas-arch', help='GPU for --ptxas to assemble for (default --arch), e.g. sm_90 on an H100')
    parser.add_argument('--library', type=Path, help='also build this shared library for Vx to launch from')
    parser.add_argument('--dispatch-lib', default=os.environ.get('VX_DISPATCH_LIB'))
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--ptx-version', help="raise the PTX's .version to this (vxc pins sm_90a to 8.0; "
                        'fence.proxy.tensormap needs 8.3). Not with --library: Vx would load the 8.0 image')
    args = parser.parse_args()
    try:
        if args.ptx_version and args.library:
            raise ValueError('--ptx-version and --library: the library would carry the PTX vxc wrote')
        # vxc takes the chip from the machine file; --arch (as VX_GPU_CHIP) overrides it.
        env = dict(os.environ)
        env.pop('VX_GPU_CHIP', None)
        if args.arch:
            env['VX_GPU_CHIP'] = args.arch
        args.arch = args.arch or machine_chip(args.machine) or 'sm_80'
        ptxas_arch = ptxas_target(args.arch, args.ptxas_arch)
        source = args.source.read_text()
        # `// spawn-include: FILE` pastes FILE (relative to SOURCE's directory, so probes/ can
        # name ../device.vx) in front: shared device helpers.
        included = [(args.source.parent / f).read_text() for f in INCLUDE.findall(source)]
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
        dynamic_shared = check_dynamic_shared(ptx, kernels[0].get('dynamic_shared_bytes', 0))
        vxc_ptx_sha = hashlib.sha256(ptx.encode()).hexdigest()
        raised = None
        if args.ptx_version:
            ptx, raised = raise_ptx_version(ptx, args.ptx_version)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {'schema': 'glm-vx-spawn-v2', 'entry': name, 'params': params, 'vx_params': kernels[0]['params'],
                    'launch': kernels[0].get('launch'), 'dynamic_shared_bytes': dynamic_shared,
                    'arch': args.arch, 'source_sha256': hashlib.sha256((''.join(included) + source).encode()).hexdigest(),
                    'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                    'vxc': subprocess.run([args.vxc, '--version'], text=True, stdout=subprocess.PIPE).stdout.strip(),
                    'assembly_validated': False, 'gpu_execution_validated': False}
        if raised:
            evidence['ptx_version'] = {'vxc': raised, 'raised_to': args.ptx_version, 'vxc_ptx_sha256': vxc_ptx_sha}
        if args.ptxas:
            with tempfile.TemporaryDirectory(prefix='vx-spawn-') as temp:
                path = Path(temp, 'k.ptx'); path.write_text(ptx)
                assembled = subprocess.run([args.ptxas, f'-arch={ptxas_arch}', '-v', str(path), '-o', str(Path(temp, 'k.cubin'))],
                                           text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if assembled.returncode:
                    raise ValueError(f'ptxas failed:\n{assembled.stderr}')
                evidence.update({'assembly_validated': True, 'ptxas_arch': ptxas_arch, 'ptxas_log': assembled.stderr.strip().splitlines()})
        if args.library:
            evidence.update(build_library(args, ''.join(included) + source, name, params, env, ptx))
        args.output.write_text(ptx)
        args.output.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(f'{args.output}: {name} ({len(params)} params)')
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
