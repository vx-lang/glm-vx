"""Print a Vx device function whose mlir! block runs one K-block of wgmma on a 32-f32 REG tensor.

gen_wgmma.py NAME SHAPE_TYPES STEPS TRANS FIRST
  SHAPE_TYPES: e.g. m64n64k16.f32.bf16.bf16
  STEPS: wgmma instructions per call; step s adds s * 256 bytes (16 encoded) to both descriptors
  TRANS: 1 to pass imm-trans-a, imm-trans-b (bf16/f16 only), 0 for fp8
  FIRST: 'acc' (scale-d 1 on every step: D += A B) or 'zero' (scale-d 0 on step 0: D = A B)
"""
import sys

name, shape, steps, trans, first = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == '1', sys.argv[5]
N = 32
ST = '!llvm.struct<(' + ', '.join(['f32'] * N) + ')>'
FT = ', '.join(['f32'] * N)
L = []
L.append(f'fn {name}(acc: &mut Tensor<f32, [32], Memory::REG>, a_desc: i64, b_desc: i64) on Topology::Device -> i32 {{')
L.append('  return mlir!( inputs : (%acc = acc : memref<32xf32>, %ad = a_desc : i64, %bd = b_desc : i64), returns : i32,')
L.append('                dialects : ["llvm", "memref", "arith"]) {')
for i in range(N):
    L.append(f'    %i{i} = arith.constant {i} : index')
for i in range(N):
    L.append(f'    %c{i} = memref.load %acc[%i{i}] : memref<32xf32>')


def tied(prefix):
    return ', '.join(f'%{prefix}{i}' for i in range(N))


outs = ','.join(['=f'] * N)
ties = ','.join(str(i) for i in range(N))
regs = ', '.join(f'${i}' for i in range(N))
# wgmma.fence orders the register accesses of acc before the wgmma reads them (PTX ISA 9.7.15.7.1).
L.append(f'    %f = llvm.inline_asm has_side_effects "wgmma.fence.sync.aligned;", "{outs},{ties}" {tied("c")} : ({FT}) -> {ST}')
for i in range(N):
    L.append(f'    %f{i} = llvm.extractvalue %f[{i}] : {ST}')
prev = 'f'
for s in range(steps):
    L.append(f'    %off{s} = arith.constant {16 * s} : i64')
    L.append(f'    %ad{s} = arith.addi %ad, %off{s} : i64')
    L.append(f'    %bd{s} = arith.addi %bd, %off{s} : i64')
    scale = 0 if (first == 'zero' and s == 0) else 1
    L.append(f'    %sd{s} = arith.constant {scale} : i32')
    tail = ', 1, 1, 0, 0' if trans else ', 1, 1'
    asm = (f'{{ .reg .pred p; setp.ne.b32 p, ${N + 2}, 0; wgmma.mma_async.sync.aligned.{shape} '
           f'{{{regs}}}, ${N}, ${N + 1}, p{tail}; }}')
    L.append(f'    %w{s} = llvm.inline_asm has_side_effects "{asm}", "{outs},l,l,r,{ties}" '
             f'%ad{s}, %bd{s}, %sd{s}, {tied(prev)} : (i64, i64, i32, {FT}) -> {ST}')
    for i in range(N):
        L.append(f'    %w{s}_{i} = llvm.extractvalue %w{s}[{i}] : {ST}')
    prev = f'w{s}_'
# commit_group + wait_group 0 (PTX ISA 9.7.15.7.2, 9.7.15.7.3): acc is passed through, so nothing
# reads it before the wait.
L.append(f'    %d = llvm.inline_asm has_side_effects "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;", '
         f'"{outs},{ties}" {tied(prev)} : ({FT}) -> {ST}')
for i in range(N):
    L.append(f'    %d{i} = llvm.extractvalue %d[{i}] : {ST}')
    L.append(f'    memref.store %d{i}, %acc[%i{i}] : memref<32xf32>')
L.append('    %z = arith.constant 0 : i32')
L.append('    macro.yield %z : i32')
L.append('  };')
L.append('}')
print('\n'.join(L))
