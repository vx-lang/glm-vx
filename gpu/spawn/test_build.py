"""build.py contract: vxc's manifest in our form and the check of it against the PTX. The
integration tests compile real Vx and run only when VXC points at a vxc with --emit-ptx."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('spawn_build', HERE / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

# vxc --emit-ptx output for add.vx: out, a, b as rank-1 memref descriptors, then n.
ADD_PTX = '.visible .entry K(\n' + ''.join(f'''\t.param .u64 .ptr .align 1 K_param_{5 * m},
\t.param .u64 .ptr .align 1 K_param_{1 + 5 * m},
\t.param .u64 K_param_{2 + 5 * m},
\t.param .u64 K_param_{3 + 5 * m},
\t.param .u64 K_param_{4 + 5 * m},
''' for m in range(3)) + '''\t.param .u32 K_param_15
)
{
\tld.param.b64 \t%rd4, [K_param_1];
\tcvta.to.global.u64 \t%rd1, %rd4;
\tld.param.b64 \t%rd5, [K_param_6];
\tcvta.to.global.u64 \t%rd2, %rd5;
\tld.param.b64 \t%rd6, [K_param_11];
\tcvta.to.global.u64 \t%rd3, %rd6;
\tld.param.b32 \t%r2, [K_param_15];
\tadd.s64 \t%rd8, %rd2, %rd7;
\tld.global.b32 \t%r7, [%rd8];
\tadd.s64 \t%rd9, %rd3, %rd7;
\tld.global.b32 \t%r8, [%rd9];
\tadd.rn.f32 \t%r9, %r7, %r8;
\tadd.s64 \t%rd10, %rd1, %rd7;
\tst.global.b32 \t[%rd10], %r9;
\tret;
}
'''


def vx_param(name, vx_type):
    """One parameter as vxc's manifest writes it."""
    if 'Tensor<' in vx_type:
        element = vx_type.split('<')[1].split(',')[0]
        return {'name': name, 'type': vx_type, 'mutable': vx_type.startswith('&mut '), 'kind': 'tensor',
                'element': {'u16': 'i16', 'u8': 'i8'}.get(element, element), 'rank': 1, 'words': 5}
    return {'name': name, 'type': vx_type, 'mutable': False, 'kind': 'scalar', 'element': vx_type, 'words': 1}


def params(text):
    """'out: mut f32[?], a: f32[?], n: i32' -> our parameter list, through kernel_params."""
    result = []
    for item in text.split(','):
        name, kind = (s.strip() for s in item.split(':'))
        if kind.endswith('[?]'):
            ref = '&mut ' if kind.startswith('mut ') else '&'
            kind = f'{ref}Tensor<{kind.removeprefix("mut ")[:-3]}, [?], Memory::HBM>'
        result.append(vx_param(name, kind))
    return build.kernel_params(result)


def test_kernel_params_takes_the_element_from_the_vx_type():
    ps = build.kernel_params([vx_param('w', '&Tensor<u16, [?], Memory::HBM>'), vx_param('out', '&mut Tensor<f32, [?], Memory::HBM>'),
                              vx_param('n', 'i32')])
    assert [(p['kind'], p.get('element'), p.get('mut')) for p in ps] == [('memref', 'u16', False), ('memref', 'f32', True),
                                                                         ('scalar', None, None)]


@pytest.mark.parametrize('change', [{'mutable': True}, {'rank': 2, 'words': 7}, {'type': '&Tensor<f16, [?], Memory::HBM>'}])
def test_kernel_params_refuses_what_launch_cannot_pass(change):
    with pytest.raises(ValueError):
        build.kernel_params([vx_param('x', '&Tensor<f32, [?], Memory::HBM>') | change])


def test_ptxas_checks_the_device_arch():
    # sm_80 PTX is assembled for the real device (sm_90 on an H100), never for an older one.
    assert build.ptxas_target('sm_80', None) == 'sm_80'
    assert build.ptxas_target('sm_80', 'sm_90') == 'sm_90'
    with pytest.raises(ValueError, match='older than the PTX target'):
        build.ptxas_target('sm_90', 'sm_80')
    with pytest.raises(ValueError, match='look like sm_90'):
        build.ptxas_target('sm_80', '90')
    # sm_90a PTX (wgmma) loads only on Hopper: assembled for sm_90a itself, never plain sm_90.
    assert build.ptxas_target('sm_90a', None) == 'sm_90a'
    with pytest.raises(ValueError, match='only for sm_90a'):
        build.ptxas_target('sm_90a', 'sm_90')


def test_machine_chip(tmp_path):
    assert build.machine_chip(HERE / 'machine.vx') is None
    (tmp_path / 'h100.vx').write_text('Topology Device {\n  arch: nvptx64,\n  chip: sm_90a,\n}\n')
    assert build.machine_chip(tmp_path / 'h100.vx') == 'sm_90a'


def test_add_ptx_matches_manifest():
    build.check_params(ADD_PTX, 'K', params('out: mut f32[?], a: f32[?], b: f32[?], n: i32'))


@pytest.mark.parametrize('text', [
    'n: i32, out: mut f32[?], a: f32[?], b: f32[?]',   # scalar in the wrong slot
    'out: mut f32[?], a: f32[?], b: f32[?], n: i64',   # scalar width
    'a: f32[?], out: mut f32[?], b: f32[?], n: i32',   # tensors swapped: stores land on a
    'out: mut f32[?], a: f32[?], b: f32[?]',          # missing parameter
    'out: mut f32[?], a: u16[?], b: f32[?], n: i32',   # element width
])
def test_add_ptx_rejects_wrong_manifest(text):
    with pytest.raises(ValueError):
        build.check_params(ADD_PTX, 'K', params(text))


VXC = os.environ.get('VXC')
needs_vxc = pytest.mark.skipif(not VXC or not shutil.which(VXC), reason='set VXC to a vxc with --emit-ptx')


def run_build(source, output):
    return subprocess.run([sys.executable, str(HERE / 'build.py'), '--source', str(source),
                           '--machine', str(HERE / 'machine.vx'), '--output', str(output)], text=True, capture_output=True)


@needs_vxc
def test_builds_add_from_vx(tmp_path):
    out = tmp_path / 'add.ptx'
    result = run_build(HERE / 'add.vx', out)
    assert result.returncode == 0, result.stderr
    assert '.entry glm_vx_spawn_add(' in out.read_text()
    assert [p['name'] for p in json.loads(out.with_suffix('.json').read_text())['params']] == ['out', 'a', 'b', 'n']


@needs_vxc
@pytest.mark.parametrize('kernel,first,second,message', [
    ('add', 'out', 'a', 'mut'),
    ('int4_matvec', 'packed', 'scales', 'bits wide'),
    ('fp8_matvec', 'weight', 'scales', 'bits wide'),
])
def test_check_catches_a_manifest_out_of_place(tmp_path, kernel, first, second, message):
    """The PTX check is independent of vxc's manifest: swap two of its entries and it refuses."""
    out = tmp_path / f'{kernel}.ptx'
    result = run_build(HERE / f'{kernel}.vx', out)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(out.with_suffix('.json').read_text())
    ps = manifest['params']
    i, j = [p['name'] for p in ps].index(first), [p['name'] for p in ps].index(second)
    swapped = list(ps)
    swapped[i], swapped[j] = ps[j], ps[i]
    with pytest.raises(ValueError, match=message):
        build.check_params(out.read_text(), manifest['entry'], swapped)


@needs_vxc
def test_refuses_two_kernels_in_one_file(tmp_path):
    source = (HERE / 'copy.vx').read_text()
    second = source[source.index('pub fn'):].replace('glm_vx_spawn_copy', 'glm_vx_spawn_copy_again')
    (tmp_path / 'copy.vx').write_text(source + '\n' + second)
    shutil.copy(HERE / 'device.vx', tmp_path)
    result = run_build(tmp_path / 'copy.vx', tmp_path / 'k.ptx')
    assert result.returncode == 1 and 'one GPU kernel per file' in result.stderr, result.stderr


@needs_vxc
def test_every_kernel_has_a_launch_shape_that_launch_reads(tmp_path):
    from gpu.spawn.launch import launch_size
    for source in sorted(HERE.glob('*.vx')):
        if source.stem in ('machine', 'device'):
            continue
        out = tmp_path / f'{source.stem}.ptx'
        result = run_build(source, out)
        assert result.returncode == 0, result.stderr
        manifest = json.loads(out.with_suffix('.json').read_text())
        launch = manifest['launch']
        assert launch, source.stem
        values = {p['name']: 3 for p in manifest['params'] if p['kind'] == 'scalar'}
        for size in launch['grid'] + [launch['block']]:
            assert launch_size(size, values) >= 1, (source.stem, size)


DISPATCH = os.environ.get('VX_DISPATCH_LIB')
needs_dispatch = pytest.mark.skipif(not (VXC and DISPATCH and shutil.which('llvm-config')),
                                    reason='set VXC, VX_DISPATCH_LIB and put llvm-config on PATH')


def run_library(source, output):
    return subprocess.run([sys.executable, str(HERE / 'build.py'), '--source', str(source), '--machine', str(HERE / 'machine.vx'),
                           '--output', str(output), '--library', str(output.with_name('lib' + output.stem + '.so'))],
                          text=True, capture_output=True)


@needs_dispatch
def test_library_carries_the_checked_kernel(tmp_path):
    result = run_library(HERE / 'add.vx', tmp_path / 'add.ptx')
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / 'add.json').read_text())
    assert manifest['library'] == 'libadd.so' and (tmp_path / 'libadd.so').exists()
    assert manifest['launch'] == {'block': 256, 'grid': [{'computed': '((n - 1) / 256) + 1'}]}


@needs_dispatch
def test_library_refuses_a_parameter_the_kernel_does_not_use(tmp_path):
    # The pub fn would take `unused`, the kernel would not: the C call would not line up.
    source = (HERE / 'copy.vx').read_text().replace(', n: i32)', ', n: i32, unused: i32)')
    assert 'unused: i32' in source
    (tmp_path / 'copy.vx').write_text(source)
    shutil.copy(HERE / 'device.vx', tmp_path)
    result = run_library(tmp_path / 'copy.vx', tmp_path / 'copy.ptx')
    assert result.returncode == 1 and 'but the kernel uses' in result.stderr, result.stderr
