"""The compiler keeps the front end confined (#68, steps 2 and 4).

Each case adds one module to a copy of the package `front` (front/src), imports it from main.vx and
builds with `--no-trusted-code front`, as front/build.sh does. The build must fail with the given
error. With Vx #4136, vxc checks every function of the package, so nothing has to call the module.

- E5003: trusted code (`unsafe`, `unsafe fn`, an `extern` block, `unsafe impl`) in the package.
- E8014: a capability built outside std (`Files {}`, `Network {}`, `Environment {}`, `Stdout {}`).
- A handler that opens a file or a connection without the right does not type-check: `File::open`
  takes `&Files` and `TcpStream::connect` takes `&Network`, and nothing in `front` can make one.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

FRONT = Path(__file__).resolve().parents[1] / 'front'
VXC = os.environ.get('VXC') or shutil.which('vxc')
pytestmark = pytest.mark.skipif(not VXC or not (FRONT / 'build' / 'glm-vx-front').exists(),
                                reason='needs VXC and front/build.sh')

CASES = {
    'unsafe_block': ('E5003', 'pub fn f() -> i32 {\n  return unsafe { 1 };\n}\n'),
    'unsafe_fn': ('E5003', 'pub unsafe fn f() -> i32 {\n  return 1;\n}\n'),
    'extern_block': ('E5003', 'extern "C" {\n  pub fn getpid() -> i32;\n}\n'),
    'unsafe_impl': ('E5003', 'struct Raw {\n  p : *mut i8,\n}\n\nunsafe impl Send for Raw {}\n'),
    'forged_files': ('E8014', 'import std::fs;\n\npub fn f() -> Files {\n  return Files {};\n}\n'),
    'forged_network': ('E8014', 'import std::net;\n\npub fn f() -> Network {\n  return Network {};\n}\n'),
    'forged_environment': ('E8014', 'import std::env;\n\npub fn f() -> Environment {\n  return Environment {};\n}\n'),
    'forged_stdout': ('E8014', 'import std::io;\n\npub fn f() -> Stdout {\n  return Stdout {};\n}\n'),
    'open_without_files': ('Error', 'import std::fs;\nimport std::string;\n\n'
                           'pub fn f(name : &String) -> bool {\n  match File::open(name, OpenMode::Read) {\n'
                           '    Result<File, FileError>::Ok(_f) => { return true; }\n'
                           '    Result<File, FileError>::Err(_e) => { return false; }\n  }\n}\n'),
    'connect_without_network': ('Error', 'import std::net;\nimport std::string;\n\n'
                                'pub fn f(address : &String) -> bool {\n  match TcpStream::connect(address) {\n'
                                '    Result<TcpStream, NetError>::Ok(_s) => { return true; }\n'
                                '    Result<TcpStream, NetError>::Err(_e) => { return false; }\n  }\n}\n'),
}


def build(tmp_path, module=None, source=None):
    src = tmp_path / 'src'
    shutil.copytree(FRONT / 'src', src)
    main = (FRONT / 'main.vx').read_text()
    if module:
        (src / f'{module}.vx').write_text(source)
        main = f'import front::{module};\n' + main
    (tmp_path / 'main.vx').write_text(main)
    return subprocess.run([VXC, str(tmp_path / 'main.vx'), '--package', f'front={src}', '--no-trusted-code', 'front',
                           '--action', 'emit-obj', '-o', str(tmp_path / 'main.o')],
                          capture_output=True, text=True, timeout=900)


def test_the_front_end_builds_with_no_trusted_code(tmp_path):
    out = build(tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_package_refuses(case, tmp_path):
    code, source = CASES[case]
    out = build(tmp_path, f'evil_{case}', source)
    text = out.stdout + out.stderr
    assert out.returncode != 0, text
    assert code in text and f'evil_{case}.vx' in text, text


EXPORTED = ('import std::string;\n\npub extern fn glm_vx_front_g(p : *const u8, n : i32) -> i32\n'
            'requires valid(p, n)\nrequires initialized(p, n)\n{\n  match String::from_bytes(p, n) {\n'
            '    Result<String, Utf8Error>::Ok(s) => { return s.len(); }\n'
            '    Result<String, Utf8Error>::Err(_e) => { return -1; }\n  }\n}\n')


@pytest.mark.xfail(strict=True, reason='Vx #4144: --no-trusted-code lets an exported fn with pointer requires through, '
                   'though --audit lists them as trusted. The audit test below catches it in front/.')
def test_the_package_refuses_an_exported_fn_whose_requires_c_does_not_prove(tmp_path):
    out = build(tmp_path, 'evil_exported', EXPORTED)
    assert out.returncode != 0 and 'evil_exported.vx' in out.stdout + out.stderr


def test_the_audit_lists_no_trusted_code_outside_core_and_std():
    """`vxc --audit` of the front end: only std's facts and the counts of core and std."""
    out = subprocess.run([VXC, str(FRONT / 'main.vx'), '--package', f'front={FRONT / "src"}', '--no-trusted-code',
                          'front', '--audit'], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = [l for l in out.stdout.splitlines() if l.strip() and not l.startswith('[')]
    assert lines and lines[-1].startswith('core and std:'), lines
    for line in lines[:-1]:
        assert '/stdlib/' in line, line  # each listed fact is in std, not in front/ or main.vx
