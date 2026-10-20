"""build.py contract: signature parsing and the PTX parameter check. The integration tests
compile real Vx and run only when VXC points at a vxc with --machine support."""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('spawn_build', HERE / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

# vxc 390c2534 output for add.vx: n, then a, b, out as rank-1 memref descriptors.
ADD_PTX = '''.visible .entry K(
\t.param .u32 K_param_0,
''' + ''.join(f'''\t.param .u64 .ptr .align 1 K_param_{1 + 5 * m},
\t.param .u64 .ptr .align 1 K_param_{2 + 5 * m},
\t.param .u64 K_param_{3 + 5 * m},
\t.param .u64 K_param_{4 + 5 * m},
\t.param .u64 K_param_{5 + 5 * m}{"," if m < 2 else ""}
''' for m in range(3)) + ''')
{
\tld.param.b32 \t%r2, [K_param_0];
\tld.param.b64 \t%rd4, [K_param_2];
\tcvta.to.global.u64 \t%rd1, %rd4;
\tld.param.b64 \t%rd5, [K_param_7];
\tcvta.to.global.u64 \t%rd2, %rd5;
\tld.param.b64 \t%rd6, [K_param_12];
\tcvta.to.global.u64 \t%rd3, %rd6;
\tadd.s64 \t%rd8, %rd1, %rd7;
\tld.global.b32 \t%r7, [%rd8];
\tadd.s64 \t%rd9, %rd2, %rd7;
\tld.global.b32 \t%r8, [%rd9];
\tadd.rn.f32 \t%r9, %r7, %r8;
\tadd.s64 \t%rd10, %rd3, %rd7;
\tst.global.b32 \t[%rd10], %r9;
\tret;
}
'''


def params(text):
    return build.signature(f'// spawn-kernel: k({text})\n')[1]


def test_signature_parses_scalars_and_tensors():
    name, ps = build.signature('// spawn-kernel: glm_vx_spawn_add(n: i32, a: f32[?], out: mut f32[?])\n')
    assert name == 'glm_vx_spawn_add'
    assert [p['kind'] for p in ps] == ['scalar', 'memref', 'memref']
    assert [p.get('mut') for p in ps] == [None, False, True]


@pytest.mark.parametrize('source', ['', '// spawn-kernel: a()\n// spawn-kernel: b()\n'])
def test_signature_requires_one_declaration(source):
    with pytest.raises(ValueError):
        build.signature(source)


def test_add_ptx_matches_declared_order():
    build.check_params(ADD_PTX, 'K', params('n: i32, a: f32[?], b: f32[?], out: mut f32[?]'))


@pytest.mark.parametrize('text', [
    'out: mut f32[?], a: f32[?], b: f32[?], n: i32',   # scalar in the wrong slot
    'n: i64, a: f32[?], b: f32[?], out: mut f32[?]',   # scalar width
    'n: i32, out: mut f32[?], a: f32[?], b: f32[?]',   # tensors swapped: stores land on b
    'n: i32, a: f32[?], b: f32[?]',                    # missing parameter
])
def test_add_ptx_rejects_wrong_signature(text):
    with pytest.raises(ValueError):
        build.check_params(ADD_PTX, 'K', params(text))


VXC = os.environ.get('VXC')
needs_vxc = pytest.mark.skipif(not VXC or not shutil.which(VXC), reason='set VXC to a vxc with --machine')


@needs_vxc
def test_builds_add_from_vx(tmp_path):
    out = tmp_path / 'add.ptx'
    result = subprocess.run([sys.executable, str(HERE / 'build.py'), '--source', str(HERE / 'add.vx'), '--output', str(out)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    ptx = out.read_text()
    assert '.entry glm_vx_spawn_add(' in ptx and 'vx_npu_kernel' not in ptx


@needs_vxc
@pytest.mark.parametrize('kernel,right,wrong,message', [
    ('add', 'a: f32[?], b: f32[?], out: mut f32[?]', 'out: mut f32[?], a: f32[?], b: f32[?]', 'mut'),
    ('int4_matvec', 'scales: u16[?], scale_kind: i32, packed: u8[?]', 'packed: u8[?], scale_kind: i32, scales: u16[?]', 'bits wide'),
    ('fp8_matvec', 'scales: f32[?], weight: u8[?]', 'weight: u8[?], scales: f32[?]', 'bits wide'),
])
def test_refuses_a_signature_in_the_wrong_order(tmp_path, kernel, right, wrong, message):
    source = (HERE / f'{kernel}.vx').read_text()
    assert right in source
    (tmp_path / f'{kernel}.vx').write_text(source.replace(right, wrong))
    shutil.copy(HERE / 'device.vx', tmp_path)
    result = subprocess.run([sys.executable, str(HERE / 'build.py'), '--source', str(tmp_path / f'{kernel}.vx'),
                             '--machine', str(HERE / 'machine.vx'), '--output', str(tmp_path / 'k.ptx')],
                            text=True, capture_output=True)
    assert result.returncode == 1 and message in result.stderr, result.stderr
