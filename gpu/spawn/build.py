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
ELEMENT_BITS = {'f32': 32, 'i32': 32, 'i64': 64, 'u32': 32, 'u16': 16, 'u8': 8, 'bf16': 16}
INCLUDE = re.compile(r'^//\s*spawn-include:\s*(\S+)\s*$', re.M)
# `// spawn-vxc: -fFEATURE`: a vxc feature flag for this kernel only (Vx's -f values).
VXC_FLAG = re.compile(r'^//\s*spawn-vxc:\s*(-f[\w-]+)\s*$', re.M)
# `// spawn-variant: SUFFIX INCLUDE [MACHINE]`: one build of the source per line (--variant SUFFIX), with
# INCLUDE pasted after the spawn-includes, the entry renamed <entry>SUFFIX (`-`: no suffix), and
# MACHINE (relative to the source) in place of --machine. glm-vx #38 builds the FP8 conversion
# kernels so: an sm_90 one with Hopper's e4m3x2 -> f16x2 instruction, and a `_ref` one with an exact
# integer decode that the host tests run and that must give the same bits.
VARIANT = re.compile(r'^//\s*spawn-variant:\s*(\S+)\s+(\S+)(?:\s+(\S+))?\s*$', re.M)
# `// spawn-maxnreg: N`: `.maxnreg N` on the entry, a register budget for ptxas. Without one, ptxas
# picks 64-72 registers for the FP8 conversion kernels and moves their weight loads below the first
# conversions, so a warp has one or two loads in flight; at 88-104 every load of a tile issues first
# (glm-vx #38: o_proj 44.6 -> 36.4 us, the routed experts' gate_up 105.5 -> 76.9 us on the H200,
# the same bits). Vx has no way to say it yet.
MAXNREG = re.compile(r'^//\s*spawn-maxnreg:\s*(\d+)\b', re.M)


def variant_of(source: str, suffix: str):
    """(include, machine or None) of the source's `spawn-variant` line for suffix."""
    for found in VARIANT.finditer(source):
        if found[1] == suffix:
            return found[2], found[3]
    raise ValueError(f'no `// spawn-variant: {suffix} ...` line in the source')


def rename_entry(source: str, suffix: str) -> str:
    """The source with its `pub fn glm_vx_spawn_...(` entry renamed <entry><suffix> (`-`: unchanged)."""
    if suffix == '-':
        return source
    renamed, count = re.subn(r'\bpub fn (glm_vx_spawn_\w+)\(', lambda m: f'pub fn {m[1]}{suffix}(', source, count=1)
    if not count:
        raise ValueError('no `pub fn glm_vx_spawn_...(` entry to rename')
    return renamed


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


def set_maxnreg(ptx: str, entry: str, registers: int) -> str:
    """PTX with `.maxnreg registers` on entry, after its parameter list."""
    if not 16 <= registers <= 255:
        raise ValueError(f'spawn-maxnreg {registers} is outside 16..255')
    found = re.search(rf'^\.visible \.entry {entry}\($.*?^\)$', ptx, re.M | re.S)
    if not found:
        raise ValueError(f'no parameter list of .entry {entry} in the PTX')
    if re.search(r'^\.maxnreg\b', ptx, re.M):
        raise ValueError('the PTX already has a .maxnreg')
    return ptx[:found.end()] + f'\n.maxnreg {registers}' + ptx[found.end():]


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
    # The arrival count is the kernel's, not the function's: Vx's host side makes it (Vx #2188).
    kernel_names = [p['name'] for p in params if p['kind'] != 'arrivals']
    if names != kernel_names:
        raise ValueError(f'pub fn {name} takes {names}, but the kernel uses {kernel_names}')
    if not args.dispatch_lib or not Path(args.dispatch_lib).is_file():
        raise ValueError('--library needs --dispatch-lib (or VX_DISPATCH_LIB): the libvx_cuda_dispatch.so vxc was built with')
    dispatch = Path(args.dispatch_lib).resolve()
    with tempfile.TemporaryDirectory(prefix='vx-spawn-lib-') as temp:
        combined, obj = Path(temp, args.source.name), Path(temp, 'k.o')
        combined.write_text(combined_text)
        # The optimization level as for --emit-ptx: the device image follows it, and the PTX checked
        # above must be the PTX this library launches.
        result = subprocess.run([args.vxc, '--machine', str(args.machine.resolve()), *VXC_FLAG.findall(combined_text),
                                 str(combined), '--action', 'emit-obj', '-o', str(obj)], text=True, capture_output=True, env=env)
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
    element comes from the Vx type, `&Tensor<u16, [?], Memory::HBM>`. A kernel with code after a
    group of blocks (Vx #2187) takes the groups' arrival counts last, as `arrivals` (Vx #2188):
    `elements` u32s, zeroed before the first launch, which each group's last pass zeroes again."""
    params = []
    for index, p in enumerate(manifest_params):
        if p['kind'] == 'arrivals':
            if index != len(manifest_params) - 1 or p['element'] != 'i32' or p['rank'] != 1 or p['words'] != 5 \
               or not p.get('zero_before_first_launch'):
                raise ValueError(f'unsupported arrivals parameter {p}')
            params.append({'name': p['name'], 'kind': 'arrivals', 'element': 'i32', 'rank': 1,
                           'elements': p['elements']})
            continue
        if p['kind'] == 'scalar':
            if p['type'] not in SCALARS or p['words'] != 1:
                raise ValueError(f'unsupported scalar parameter {p}')
            params.append({'name': p['name'], 'kind': 'scalar', 'type': p['type']})
        elif p['kind'] == 'tensor':
            element = re.match(r'&(?:mut )?Tensor<(\w+),', p['type'])
            if not element or element[1] not in ELEMENT_BITS or p['rank'] not in (1, 2, 3) or p['words'] != 3 + 2 * p['rank']:
                raise ValueError(f'unsupported tensor parameter {p}')
            if p['mutable'] != p['type'].startswith('&mut '):
                raise ValueError(f'manifest mutable flag disagrees with the type: {p}')
            params.append({'name': p['name'], 'kind': 'memref', 'element': element[1], 'rank': p['rank'], 'mut': p['mutable']})
        else:
            raise ValueError(f'unknown parameter kind {p}')
    return params


def derived(body: str, base: str):
    """Registers computed from BASE by 64-bit address arithmetic (add/sub/mov/cvta), followed
    to a fixed point, so a load through base + row offset + column offset is still traced."""
    steps = [(m[1], set(re.findall(r'%rd\d+', m[2]))) for m in
             re.finditer(r'^\s*(?:add|sub|mov|cvta)[.\w]*\s+(%rd\d+),([^;]*);', body, re.M)]
    # mad.lo a, b, c, base: only the addend is an address (a row base plus row * stride).
    steps += [(m[1], {m[2]}) for m in
              re.finditer(r'^\s*mad\.lo\.[su]64\s+(%rd\d+),[^;]*,\s*(%rd\d+);', body, re.M)]
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
    actual = [re.match(r'\s*\.param\s+(\.\w+)(\s+\.ptr(?:\s+\.align\s+(\d+))?)?', line) for line in head[1].split(',')]
    expected, owner = [], []
    for index, p in enumerate(params):
        if p['kind'] == 'scalar':
            expected.append((SCALARS[p['type']], False)); owner.append(index)
        else:
            words = 3 + 2 * p['rank']
            expected += [('.u64', True), ('.u64', True)] + [('.u64', False)] * (words - 2)
            owner += [index] * words
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
        # The machine file's HBM alignment (Vx #2375): a `&` tensor's pointer is `.align 16`, so LLVM
        # may load four words at once (ld.global.nc.v4), and a launch must pass a pointer that is.
        p['align'] = int(actual[slot][3] or 1)
        reg = re.search(rf'ld\.param\.b64\s+(%rd\d+), \[{entry}_param_{slot}\];', body)
        if not reg:
            if p['mut']:
                raise ValueError(f'mut tensor {p["name"]} is never loaded')
            continue
        global_reg = re.search(rf'cvta\.to\.global\.u64\s+(%rd\d+), {re.escape(reg[1])};', body)
        base = global_reg[1] if global_reg else reg[1]
        addresses = derived(body, base)
        # An atomic read-modify-write (core::sync::atomic) changes the tensor as a store does.
        if any(re.search(rf'st\.global(?:\.v[24])?\.\w+\s+\[{re.escape(a)}([+-]\d+)?\]', body)
               or re.search(rf'(?:atom|red)\.global(?:\.\w+)+\s+(?:%\w+,\s*)?\[{re.escape(a)}([+-]\d+)?\]', body)
               for a in addresses):
            stored.add(p['name'])
        # Element width: loads through this tensor must be as wide as its element in the manifest,
        # which catches a manifest whose read-only tensors of different types are out of place.
        widths = {int(w) for a in addresses
                  for w in re.findall(rf'ld\.global(?:\.nc)?(?:\.v[24])?\.[bsuf](\d+)\s+(?:%\w+|\{{[^}}]*\}}), \[{re.escape(a)}(?:[+-]\d+)?\]', body)}
        # vxc may load an i64 as two 32-bit halves, and bf16 as words of two (ld.global.nc.v4.b32).
        allowed = {ELEMENT_BITS[p['element']]} | ({32} if p['element'] in ('i64', 'bf16') else set())
        if widths - allowed:
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
    parser.add_argument('--variant', help="build the source's `// spawn-variant: SUFFIX ...` line SUFFIX (`-`: no suffix)")
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
        source = args.source.read_text()
        extra = []
        if args.variant:
            include, machine = variant_of(source, args.variant)
            extra = [(args.source.parent / include).read_text()]
            if machine:
                args.machine = args.source.parent / machine
            source = rename_entry(source, args.variant)
        args.arch = args.arch or machine_chip(args.machine) or 'sm_80'
        ptxas_requested = args.ptxas_arch
        if args.variant and ptxas_requested and ptxas_target(ptxas_requested, None) and \
                int(re.fullmatch(r'sm_(\d+)a?', ptxas_requested)[1]) < int(re.fullmatch(r'sm_(\d+)a?', args.arch)[1]):
            ptxas_requested = args.arch  # a variant newer than the device (sm_90 on an A100 pod): check it for itself
        ptxas_arch = ptxas_target(args.arch, ptxas_requested)
        # `// spawn-include: FILE` pastes FILE (relative to SOURCE's directory, so probes/ can
        # name ../device.vx) in front: shared device helpers.
        included = [(args.source.parent / f).read_text() for f in INCLUDE.findall(source)] + extra
        with tempfile.TemporaryDirectory(prefix='vx-spawn-src-') as temp:
            combined = Path(temp, args.source.name)
            combined.write_text(''.join(included) + source)
            emitted = Path(temp, 'kernels.ptx')
            flags = VXC_FLAG.findall(source)
            result = subprocess.run([args.vxc, '--machine', str(args.machine.resolve()), *flags, '--emit-ptx',
                                     str(combined), '-o', str(emitted)], text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env)
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
        vxc_ptx, vxc_ptx_sha = ptx, hashlib.sha256(ptx.encode()).hexdigest()
        raised = None
        if args.ptx_version:
            ptx, raised = raise_ptx_version(ptx, args.ptx_version)
        budgets = [int(n) for n in MAXNREG.findall(source)]
        if len(budgets) > 1:
            raise ValueError('more than one `// spawn-maxnreg:` line')
        if budgets:
            ptx = set_maxnreg(ptx, name, budgets[0])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        evidence = {'schema': 'glm-vx-spawn-v2', 'entry': name, 'params': params, 'vx_params': kernels[0]['params'],
                    'launch': kernels[0].get('launch'), 'dynamic_shared_bytes': dynamic_shared,
                    'arch': args.arch, 'vxc_flags': flags, 'maxnreg': budgets[0] if budgets else None, 'source_sha256': hashlib.sha256((''.join(included) + source).encode()).hexdigest(),
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
            # vxc builds the library's image itself, so it has no spawn-maxnreg budget: the same kernel
            # as vxc's PTX, which the launch tests run through Vx's own path.
            evidence.update(build_library(args, ''.join(included) + source, name, params, env, vxc_ptx))
        args.output.write_text(ptx)
        args.output.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(f'{args.output}: {name} ({len(params)} params)')
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
