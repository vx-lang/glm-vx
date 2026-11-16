"""Variants of fp8_matvec_heads_cvt: SG subgroups a block, R rows a warp (pairs). python genv.py SPAWN_DIR"""
import sys
from pathlib import Path
header = '// spawn-include: device.vx\n// spawn-variant: - fp8_cvt_sm90.vx machine-sm90.vx\n\n'
def body(sg, r, name):
    rows_block = sg * r
    pairs = r // 2
    sums = '\n'.join(f'          let mut s{k}: f32 = 0.0;\n          let mut t{k}: f32 = 0.0;' for k in range(pairs))
    calls = '\n'.join(f'          fp8t_tiles_rows_ab_at!(s{k}, t{k}, weight, scales, x, p + {2 * k}, p + {2 * k + 1}, row < (rows as i64), n, lane, xo);' for k in range(pairs))
    stores = '\n'.join(f'          a[w * {r} + {2 * k}][lane] = s{k};\n          a[w * {r} + {2 * k + 1}][lane] = t{k};' for k in range(pairs))
    tot = '\n'.join(f'        let v{j}: f32 = lanes::sum(&a[w * {r} + {j}], 32);' for j in range(r))
    wr = '\n'.join(f'        if ok && rb * {rows_block} + w * {r} + {j} < rows {{ out[rb * {rows_block} + w * {r} + {j}] = v{j}; }}' for j in range(r))
    return f'''pub fn glm_vx_spawn_{name}(out: &mut Tensor<f32, [?], Memory::HBM>, weight: &Tensor<u32, [?], Memory::HBM>,
                                         scales: &Tensor<f32, [?], Memory::HBM>, x: &Tensor<f32, [?], Memory::HBM>,
                                         rows: i32, cols: i32, group: i32, x_stride: i32, period: i32, offset: i32) -> void {{
  spawn on(Topology::Device) {{
    for rb in 0..(rows - 1) / {rows_block} + 1 across blocks {{
      let mut a = unsafe {{ Tensor<f32, [{rows_block}, 32], Memory::SMEM>::uninit() }};
      for w in 0..{sg} across subgroups {{
        for lane in 0..32 across lanes {{
          let ok: bool = group > 0 && group % {r} == 0 && rows % group == 0 && x_stride >= 0 && offset >= 0
                         && period >= offset + group;
          let row: i64 = (rb * {rows_block} + w * {r}) as i64;
          let gd: i64 = if ok {{ group as i64 }} else {{ 1 }};
          let g: i64 = row / gd;
          let p: i64 = g * (period as i64) + (offset as i64) + (row - g * gd);
          let xo: i64 = g * (x_stride as i64);
          let n: i32 = if ok {{ cols }} else {{ 1 }};
{sums}
{calls}
{stores}
        }}
{tot}
        let ok: bool = group > 0 && group % {r} == 0 && rows % group == 0 && x_stride >= 0 && offset >= 0
                       && period >= offset + group && cols % 16 == 0;
{wr}
      }}
    }}
  }}
  return;
}}
'''
for combo in [(4, 4), (2, 4), (4, 8), (2, 8), (8, 8), (4, 4, 96), (4, 8, 128), (2, 8, 128)]:
    sg, r = combo[:2]; mr = combo[2] if len(combo) > 2 else 0
    name = f'fp8_heads_s{sg}r{r}' + (f'm{mr}' if mr else '')
    (Path(sys.argv[1]) / f'{name}.vx').write_text(header.replace('\n\n', f'\n// spawn-maxnreg: {mr}\n\n' if mr else '\n\n') + body(sg, r, name))
    print(name, end=' ')
