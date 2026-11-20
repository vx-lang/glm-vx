"""build.py contract: vxc's manifest in our form and the check of it against the PTX. The
integration tests compile real Vx and run only when VXC points at a vxc with --emit-ptx."""
import importlib.util
import json
import os
import re
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


@pytest.mark.parametrize('change', [{'mutable': True}, {'rank': 4, 'words': 11}, {'rank': 2, 'words': 5},
                                    {'type': '&Tensor<f64, [?], Memory::HBM>'}])
def test_kernel_params_refuses_what_launch_cannot_pass(change):
    with pytest.raises(ValueError):
        build.kernel_params([vx_param('x', '&Tensor<f32, [?], Memory::HBM>') | change])


def test_kernel_params_takes_a_rank_2_tensor():
    p = vx_param('out', '&mut Tensor<f32, [?, ?], Memory::HBM>') | {'rank': 2, 'words': 7}
    assert build.kernel_params([p])[0]['rank'] == 2


def test_kernel_params_takes_an_fp8_tile_tensor():
    # A lane matmul's first tile in HBM (Vx #3694): e4m3 codes, [tiles, 16, 256] (gpu/spawn/moe_mma/fp8_moe_vxmm16_down_at).
    p = vx_param('down', '&Tensor<f8e4m3, [?, 16, 256], Memory::HBM>') | {'rank': 3, 'words': 9}
    assert [(q['element'], q['rank']) for q in build.kernel_params([p])] == [('f8e4m3', 3)]


def test_kernel_params_takes_a_rank_3_tensor():
    p = vx_param('out', '&mut Tensor<f32, [?, ?, ?], Memory::HBM>') | {'rank': 3, 'words': 9}
    assert build.kernel_params([p])[0]['rank'] == 3


ARRIVALS = {'element': 'i32', 'elements': {'param': 'heads'}, 'kind': 'arrivals', 'name': 'arrivals', 'rank': 1,
            'words': 5, 'zero_before_first_launch': True}


def test_kernel_params_takes_the_arrival_count_last():
    ps = build.kernel_params([vx_param('heads', 'i32'), ARRIVALS])
    assert ps[1] == {'name': 'arrivals', 'kind': 'arrivals', 'element': 'i32', 'rank': 1, 'elements': {'param': 'heads'}}


@pytest.mark.parametrize('params', [[ARRIVALS, vx_param('heads', 'i32')], [ARRIVALS | {'zero_before_first_launch': False}],
                                    [ARRIVALS | {'rank': 2, 'words': 7}]])
def test_kernel_params_refuses_an_arrival_count_launch_cannot_pass(params):
    with pytest.raises(ValueError):
        build.kernel_params(params)


def test_rank_2_descriptor_is_seven_words():
    # out: a rank-2 descriptor (pointer, aligned pointer, offset, 2 sizes, 2 strides), then n.
    ptx = ('.visible .entry K(\n\t.param .u64 .ptr .align 1 K_param_0,\n\t.param .u64 .ptr .align 1 K_param_1,\n'
           + ''.join(f'\t.param .u64 K_param_{i},\n' for i in range(2, 7)) + '\t.param .u32 K_param_7\n)\n{\n'
           '\tld.param.b64 \t%rd4, [K_param_1];\n\tcvta.to.global.u64 \t%rd1, %rd4;\n'
           '\tst.global.b32 \t[%rd1], %r9;\n\tret;\n}\n')
    out = vx_param('out', '&mut Tensor<f32, [?, ?], Memory::HBM>') | {'rank': 2, 'words': 7}
    build.check_params(ptx, 'K', build.kernel_params([out, vx_param('n', 'i32')]))
    with pytest.raises(ValueError, match='do not match'):
        build.check_params(ptx, 'K', params('out: mut f32[?], n: i32'))


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


def test_a_row_base_plus_row_times_stride_is_traced():
    # A rank-2 write out[row][col]: the address is the addend of a mad, not a factor.
    body = '\tmad.lo.s64 \t%rd9, %rd7, %rd8, %rd1;\n\tmad.lo.s64 \t%rd12, %rd1, %rd10, %rd11;\n'
    assert build.derived(body, '%rd1') == {'%rd1', '%rd9'}


def test_an_fp8_tensor_may_be_loaded_in_words():
    # ADD_PTX loads a with ld.global.b32: four e4m3 codes (or two f16) a word, as a tile's 16-byte loads do.
    build.check_params(ADD_PTX, 'K', params('out: mut f32[?], a: f8e4m3[?], b: f32[?], n: i32'))
    build.check_params(ADD_PTX, 'K', params('out: mut f32[?], a: f16[?], b: f32[?], n: i32'))
    with pytest.raises(ValueError, match='bits wide'):
        build.check_params(ADD_PTX, 'K', params('out: mut f32[?], a: u8[?], b: f32[?], n: i32'))


def test_an_i64_tensor_may_be_loaded_in_32_bit_halves():
    ptx = ADD_PTX.replace('\tld.global.b32 \t%r7, [%rd8];', '\tld.global.b32 \t%r7, [%rd8];\n\tld.global.b32 \t%r6, [%rd8+4];')
    build.check_params(ptx, 'K', params('out: mut f32[?], a: i64[?], b: f32[?], n: i32'))
    with pytest.raises(ValueError, match='bits wide'):
        build.check_params(ptx, 'K', params('out: mut f32[?], a: u16[?], b: f32[?], n: i32'))


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


def run_build(source, output, variant=None):
    return subprocess.run([sys.executable, str(HERE / 'build.py'), '--source', str(source),
                           '--machine', str(HERE / 'machine.vx'), '--output', str(output),
                           *(['--variant', variant] if variant else [])], text=True, capture_output=True)


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
    from gpu.spawn.build import VARIANT
    for source in sorted(HERE.glob('*.vx')):
        text = source.read_text()
        if not re.search(r'^pub fn glm_vx_spawn_', text, re.M):
            continue  # machine*.vx, device.vx and the variant includes, as build.sh skips them
        for variant in [v[1] for v in VARIANT.finditer(text)] or [None]:
            out = tmp_path / f'{source.stem}{variant or ""}.ptx'
            result = run_build(source, out, variant)
            assert result.returncode == 0, result.stderr
            manifest = json.loads(out.with_suffix('.json').read_text())
            launch = manifest['launch']
            assert launch, (source.stem, variant)
            # 128: heads / 16 >= 1, and fp8_matvec_cols_cvt's cols / 128 (a block a 128 column block).
            values = {p['name']: 128 for p in manifest['params'] if p['kind'] == 'scalar'}
            for size in launch['grid'] + [launch['block']]:
                assert launch_size(size, values) >= 1, (source.stem, variant, size)


@needs_vxc
def test_a_variant_renames_the_entry_and_takes_its_include_and_machine(tmp_path):
    # glm-vx #38: fp8_matvec_cvt_tiles builds as sm_90 (Hopper's conversion) and as _ref (sm_80, integers).
    source = HERE / 'fp8_matvec_cvt_tiles.vx'
    for variant, entry, target in (('-', 'glm_vx_spawn_fp8_matvec_cvt_tiles', 'sm_90'),
                                   ('_ref', 'glm_vx_spawn_fp8_matvec_cvt_tiles_ref', 'sm_80')):
        out = tmp_path / f'v{variant}.ptx'
        result = run_build(source, out, variant)
        assert result.returncode == 0, result.stderr
        ptx = out.read_text()
        assert f'.entry {entry}(' in ptx and re.search(rf'^\.target {target}\b', ptx, re.M)
        assert ('cvt.rn.f16x2.e4m3x2' in ptx) == (target == 'sm_90')
    assert run_build(source, tmp_path / 'x.ptx', '_nope').returncode != 0


def test_set_maxnreg_puts_the_budget_after_the_entry_parameters():
    ptx = '.version 8.0\n.target sm_90\n.visible .entry K(\n\t.param .u64 K_param_0\n)\n{\n\tret;\n}\n'
    out = build.set_maxnreg(ptx, 'K', 96)
    assert out == ptx.replace(')\n{', ')\n.maxnreg 96\n{')
    for bad in (lambda: build.set_maxnreg(ptx, 'J', 96), lambda: build.set_maxnreg(out, 'K', 96),
                lambda: build.set_maxnreg(ptx, 'K', 8), lambda: build.set_maxnreg(ptx, 'K', 256)):
        with pytest.raises(ValueError):
            bad()


@needs_vxc
def test_spawn_maxnreg_gives_the_entry_a_register_budget(tmp_path):
    """fp8_matvec_cvt_tiles' `// spawn-maxnreg: 96` reaches the PTX and the manifest, in each variant;
    without the line there is no .maxnreg (glm-vx #38)."""
    source = (HERE / 'fp8_matvec_cvt_tiles.vx').read_text()
    assert re.search(r'^// spawn-maxnreg: 96\b', source, re.M)
    for f in ('device.vx', 'fp8_cvt_sm90.vx', 'fp8_cvt_ref.vx', 'machine-sm90.vx'):
        shutil.copy(HERE / f, tmp_path)
    (tmp_path / 'with.vx').write_text(source)
    (tmp_path / 'without.vx').write_text(re.sub(r'^// spawn-maxnreg:.*\n', '', source, flags=re.M))
    for name, variant, budget in (('with', '-', 96), ('with', '_ref', 96), ('without', '-', None)):
        out = tmp_path / f'{name}{variant}.ptx'
        result = run_build(tmp_path / f'{name}.vx', out, variant)
        assert result.returncode == 0, result.stderr
        assert re.findall(r'^\.maxnreg (\d+)$', out.read_text(), re.M) == ([str(budget)] if budget else [])
        assert json.loads(out.with_suffix('.json').read_text())['maxnreg'] == budget


@needs_vxc
def test_spawn_vxc_passes_a_feature_flag(tmp_path):
    """topk_radix's `// spawn-vxc: -fweak-relaxed-atomics` keeps its atomics relaxed: no fence after
    the shared-memory counts. Only the arrival's fences, before and after its atomic, remain."""
    source = (HERE / 'topk_radix.vx').read_text()
    assert '// spawn-vxc: -fweak-relaxed-atomics' in source
    shutil.copy(HERE / 'device.vx', tmp_path)
    fenced = {}
    for name, text in (('weak', source), ('default', source.replace('// spawn-vxc: -fweak-relaxed-atomics\n', ''))):
        (tmp_path / f'{name}.vx').write_text(text)
        result = run_build(tmp_path / f'{name}.vx', tmp_path / f'{name}.ptx')
        assert result.returncode == 0, result.stderr
        ptx = (tmp_path / f'{name}.ptx').read_text()
        fenced[name] = ptx.count('fence.')
        assert json.loads((tmp_path / f'{name}.json').read_text())['vxc_flags'] == (['-fweak-relaxed-atomics'] if name == 'weak' else [])
    shared = (tmp_path / 'weak.ptx').read_text().split('atom.shared.add.u32')[1].split('\n')[1]
    assert 'fence' not in shared
    assert fenced['weak'] == 4 and fenced['default'] > fenced['weak'], fenced


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
    assert manifest['launch'] == {'block': 256, 'grid': [{'computed': '((n - 1) / 256) + 1'}], 'grid_loops': True}


@needs_dispatch
def test_library_refuses_a_parameter_the_kernel_does_not_use(tmp_path):
    # The pub fn would take `unused`, the kernel would not: the C call would not line up.
    source = (HERE / 'copy.vx').read_text().replace(', n: i32)', ', n: i32, unused: i32)')
    assert 'unused: i32' in source
    (tmp_path / 'copy.vx').write_text(source)
    shutil.copy(HERE / 'device.vx', tmp_path)
    result = run_library(tmp_path / 'copy.vx', tmp_path / 'copy.ptx')
    assert result.returncode == 1 and 'but the kernel uses' in result.stderr, result.stderr


def test_dynamic_shared_bytes_must_match_the_ptx():
    """Vx #1984 (PR #1995): an .extern .shared array exactly when the manifest gives a size."""
    extern = '.version 8.0\n.target sm_90a\n.extern .shared .align 16 .b8 __dynamic_shmem__0[];\n'
    assert build.check_dynamic_shared(extern, 163840) == 163840
    assert build.check_dynamic_shared(ADD_PTX, 0) == 0
    with pytest.raises(ValueError, match='has no .extern .shared'):
        build.check_dynamic_shared(ADD_PTX, 4096)
    with pytest.raises(ValueError, match='declares .extern .shared'):
        build.check_dynamic_shared(extern, 0)
    with pytest.raises(ValueError, match='nonnegative integer'):
        build.check_dynamic_shared(ADD_PTX, -1)


def test_raise_ptx_version_changes_only_the_header():
    ptx = '//\n.version 8.0\n.target sm_90a\n.address_size 64\n\tfence.proxy.tensormap::generic.acquire.sys [%rd1], 128;\n'
    raised, old = build.raise_ptx_version(ptx, '8.3')
    assert old == '8.0'
    assert raised == ptx.replace('.version 8.0', '.version 8.3')
    assert build.raise_ptx_version(ptx, '8.0')[0] == ptx
    for bad in ('7.8', '8', 'x'):
        with pytest.raises(ValueError):
            build.raise_ptx_version(ptx, bad)
    with pytest.raises(ValueError):
        build.raise_ptx_version('.target sm_90a\n', '8.3')
