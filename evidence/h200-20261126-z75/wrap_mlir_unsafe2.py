"""The two mlir! blocks with ensures that are not a helper's return value (after wrap_mlir_unsafe.py), for Vx main's
E8016: moe_mma's `let sm: i64 = mlir!(...) { ... };` (fp8_moe_mma16p_gate_up_at.vx and its generator gen_moe_mma.py)
and fp8_gemm's even_pitch! macro, whose expansion is an mlir! block (fp8_gemm_grouped.vx.in and the generated
fp8_gemm_grouped_m64.vx). Run from the repository root."""
import pathlib, re
def edit(path, fn):
    p = pathlib.Path(path); old = p.read_text(); new = fn(old)
    assert new != old, f'{path}: nothing changed'; p.write_text(new); print('changed', path)
def let_sm(s):
    # `let sm: i64 = mlir!(` ... the first `};` line after it (plain text, or a Python string piece "...};\n")
    i = s.index('let sm: i64 = mlir!(')
    fstring = re.match(r'\s*f["\']', s[s.rindex('\n', 0, i) + 1:i]) is not None  # an f-string template doubles its braces
    s = s[:i] + ('let sm: i64 = unsafe {{ mlir!(' if fstring else 'let sm: i64 = unsafe { mlir!(') + s[i + len('let sm: i64 = mlir!('):]
    m = re.compile(r'^(\s*"?\s*)\};', re.M).search(s, i)
    return s[:m.start()] + m.group(1) + '} };' + s[m.end():]
def even_pitch(s):
    i = s.index('macro_rules! even_pitch')
    j = s.index('        mlir!( inputs : (%o = $t', i)
    s = s[:j] + '        unsafe { mlir!( inputs : (%o = $t' + s[j + len('        mlir!( inputs : (%o = $t'):]
    m = re.compile(r'^        \}\n    \};', re.M).search(s, j)
    return s[:m.start()] + '        } }\n    };' + s[m.end():]
for f in ('gpu/spawn/moe_mma/fp8_moe_mma16p_gate_up_at.vx', 'gpu/spawn/moe_mma/gen_moe_mma.py'): edit(f, let_sm)
for f in ('gpu/spawn/fp8_gemm/fp8_gemm_grouped.vx.in', 'gpu/spawn/fp8_gemm/fp8_gemm_grouped_m64.vx'): edit(f, even_pitch)
