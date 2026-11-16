"""Variants of fp8_matvec_cols_cvt: SG subgroups a block, U rows a step (loads first). python gen.py SPAWN_DIR"""
import sys, re
from pathlib import Path
src = Path(sys.argv[1]) / 'fp8_matvec_cols_cvt.vx'
text = src.read_text()
head = text[:text.index('pub fn')]
macro = head[head.index('macro_rules! cols_row4'):]
header = '// spawn-include: device.vx\n// spawn-variant: - fp8_cvt_sm90.vx machine-sm90.vx\n\n'
def body(sg, u, name):
    loads = []
    for i in range(u):
        o = f' + {i * sg}' if i else ''
        loads.append(f'            let u{i}: u32 = weight[(p{o}) * rw + word];')
        loads.append(f'            let s{i}: f32 = scales[((p{o}) / 128) * sc + cb];')
        loads.append(f'            let x{i}: f32 = x[xb + d{o}];')
    fmas = [f'            cols_row4!(a0, a1, a2, a3, u{i}, x{i} * s{i});' for i in range(u)]
    return f'''pub fn glm_vx_spawn_{name}(out: &mut Tensor<f32, [?], Memory::HBM>, weight: &Tensor<u32, [?], Memory::HBM>,
                                        scales: &Tensor<f32, [?], Memory::HBM>, x: &Tensor<f32, [?], Memory::HBM>,
                                        groups: i32, cols: i32, depth: i32, period: i32, offset: i32, x_stride: i32) -> void {{
  spawn on(Topology::Device) {{
    for b in 0..groups * (cols / 128) across blocks {{
      let mut part = unsafe {{ Tensor<f32, [{sg}, 4, 32], Memory::SMEM>::uninit() }};
      for w in 0..{sg} across subgroups {{
        for lane in 0..32 across lanes {{
          let ok: bool = cols % 128 == 0 && cols > 0 && depth >= 0 && offset >= 0 && period >= offset + depth && x_stride >= 0;
          let chunks: i32 = if ok {{ cols / 128 }} else {{ 1 }};
          let g: i64 = (b / chunks) as i64;
          let cb: i64 = (b % chunks) as i64;
          let rw: i64 = (chunks * 32) as i64;
          let sc: i64 = chunks as i64;
          let word: i64 = cb * 32 + (lane as i64);
          let base: i64 = g * (period as i64) + (offset as i64);
          let xb: i64 = g * (x_stride as i64);
          let n: i64 = if ok {{ depth as i64 }} else {{ 0 }};
          let mut a0: f32 = 0.0;
          let mut a1: f32 = 0.0;
          let mut a2: f32 = 0.0;
          let mut a3: f32 = 0.0;
          let mut d: i64 = w as i64;
          while d + {(u - 1) * sg} < n {{
            let p: i64 = base + d;
{chr(10).join(loads)}
{chr(10).join(fmas)}
            d = d + {u * sg};
          }}
          while d < n {{
            let p: i64 = base + d;
            let u0: u32 = weight[p * rw + word];
            let s0: f32 = scales[(p / 128) * sc + cb];
            let x0: f32 = x[xb + d];
            cols_row4!(a0, a1, a2, a3, u0, x0 * s0);
            d = d + {sg};
          }}
          part[w][0][lane] = a0;
          part[w][1][lane] = a1;
          part[w][2][lane] = a2;
          part[w][3][lane] = a3;
        }}
      }}
      for t in 0..128 across lanes {{
        let ok: bool = cols % 128 == 0 && cols > 0 && depth >= 0 && offset >= 0 && period >= offset + depth && x_stride >= 0;
        let l: i32 = t / 4;
        let q: i32 = t % 4;
        let mut sum: f32 = 0.0;
        for s in 0..{sg} {{
          sum = sum + part[s][q][l];
        }}
        if ok {{ out[b * 128 + t] = sum; }}
      }}
    }}
  }}
  return;
}}
'''
for combo in [(16, 12), (24, 8), (12, 16), (32, 6), (16, 12, 128), (24, 8, 128), (32, 6, 64)]:
    sg, u = combo[:2]; mr = combo[2] if len(combo) > 2 else 0
    name = f'fp8_cols_s{sg}u{u}' + (f'm{mr}' if mr else '')
    (Path(sys.argv[1]) / f'{name}.vx').write_text(header.replace('\n\n', f'\n// spawn-maxnreg: {mr}\n\n' if mr else '\n\n') + macro + body(sg, u, name))
    print(name, end=' ')
