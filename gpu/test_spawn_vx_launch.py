"""gpu/spawn kernels launched by Vx's own runtime (the kernel's `pub fn` in lib<kernel>.so)
against the same PTX launched from Python with the same shape: the outputs must be bitwise
equal. test_spawn_hardware.py ties that PTX to the hand kernels. Skipped without a supported
GPU or without the libraries (gpu/spawn/build.sh with VX_DISPATCH_LIB). Run once more with
VX_CUDA_SYNC=1, so that Vx also waits after each launch."""
from pathlib import Path

import numpy as np
import pytest

from gpu.runtime import CUDAContext, CUDAUnavailable
from gpu.spawn.launch import SpawnKernel
from gpu.spawn.vx_launch import VxKernel

SPAWN = Path(__file__).parent / 'build' / 'spawn'


@pytest.fixture(scope='module')
def context():
    try:
        ctx = CUDAContext(minimum_compute_capability=(8, 0), primary=True)
    except CUDAUnavailable as e:
        pytest.skip('real GPU unavailable: ' + str(e))
    try:
        yield ctx
    finally:
        ctx.close()


def normal(rng, n, scale=1.0):
    return (rng.normal(size=n) * scale).astype('f4')


def bf16(rng, n):
    return (normal(rng, n).view(np.uint32) >> 16).astype(np.uint16)


def fp8(rng, n):
    codes = rng.integers(0, 256, size=n, dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0  # no NaN codes
    return codes


def int4(rng, rows, cols):
    packed = rng.integers(0, 256, size=(rows, -(-cols // 2)), dtype=np.uint8)
    if cols % 2:
        packed[:, -1] &= 0x0F
    return packed.reshape(-1)


def int4_scales(rng, n, kind):
    magnitudes = rng.uniform(0.001, 0.1, size=n).astype('f4')
    return (magnitudes.view(np.uint32) >> 16).astype(np.uint16) if kind else magnitudes.astype(np.float16).view(np.uint16)


def routing(rng, m, experts):
    if not experts:
        return np.zeros(1, 'i4')
    ids = rng.integers(0, experts, size=m).astype('i4')
    ids[0], ids[-1] = -1, experts  # out of range: zeros
    return ids


def outputs(*sizes, fill=None):
    return [np.zeros(s, 'f4') if fill is None else fill.copy() for s in sizes]


def mla_partial(rng, heads=4, rank=64, rope=64, tokens=300, count=37, splits=4, stride=37):
    selected = rng.integers(0, tokens, size=heads * count).astype('i4')
    selected[1], selected[-1] = -1, tokens + 5
    return dict(heads=heads, rank=rank, rope_dim=rope, tokens=tokens, count=count, selection_stride=stride,
                splits=splits, selected=selected, cache_latent=normal(rng, tokens * rank), q_latent=normal(rng, heads * rank),
                q_rope=normal(rng, heads * rope), cache_rope=normal(rng, tokens * rope), scale=rank ** -0.5,
                threads=-(-rank // 32) * 32, numerator=np.zeros((heads * splits, rank), 'f4'),
                stats=np.zeros((heads * splits, 2), 'f4'))


def mla_merge(rng, heads=4, rank=64, splits=4):
    stats = normal(rng, heads * splits * 2)
    stats[1::2] = np.abs(stats[1::2]) + 0.5  # denominators
    return dict(heads=heads, rank=rank, splits=splits, stats=stats, numerator=normal(rng, heads * splits * rank),
                out=np.zeros((heads, rank), 'f4'))


def mla_decode(rng, heads=4, rank=64, rope=64, tokens=300, count=37, splits=4, stride=37):
    values = mla_partial(rng, heads, rank, rope, tokens, count, splits, stride)
    values.update(numerator=np.zeros((heads, splits, rank), 'f4'), stats=np.zeros((heads, splits, 2), 'f4'),
                  out=np.full((heads, rank), 7, 'f4'), arrivals=np.zeros(heads, 'i4'))
    return values


def mla_decode_at(rng, heads=4, rank=64, rope=64, tokens=300, count=37, splits=2, splits_max=4, stride=37):
    values = mla_partial(rng, heads, rank, rope, tokens, count, splits, stride)
    for name in ('tokens', 'count', 'splits'):
        del values[name]
    values.update(numerator=np.zeros((heads, splits_max, rank), 'f4'), stats=np.zeros((heads, splits_max, 2), 'f4'),
                  out=np.full((heads, rank), 7, 'f4'), arrivals=np.zeros(heads, 'i4'), splits_max=splits_max,
                  step=np.array([tokens - 1, tokens, count, splits], 'i4'))
    return values


def expert_weights(rng, fmt, experts, rows, cols):
    """One projection for all experts in `fmt`, named as the expert_matvec kernels name them."""
    if fmt == 'bf16':
        return dict(weight=bf16(rng, experts * rows * cols))
    if fmt in ('fp8', 'fp8x4'):  # fp8x4: the codes viewed as u32 (fp8_moe4_*), cols % 4 == 0
        codes = fp8(rng, experts * rows * cols)
        return dict(weight=codes.view('u4') if fmt == 'fp8x4' else codes,
                    scales=rng.uniform(0.001, 0.1, size=experts * -(-rows // 128) * -(-cols // 128)).astype('f4'))
    if fmt == 'int4':
        packed = rng.integers(0, 256, size=(experts, rows, -(-cols // 2)), dtype=np.uint8)
        if cols % 2:
            packed[..., -1] &= 0x0F
        return dict(packed=packed.reshape(-1), scales=int4_scales(rng, experts * rows * -(-cols // 128), 1), scale_kind=1)
    return dict(weight=normal(rng, experts * rows * cols, 0.2))


def moe_ids(rng, experts, k):
    ids = rng.choice(experts, size=k, replace=False).astype('i4')
    ids[-1] = experts + 3  # out of range: adds nothing
    return ids


def expert_matvec(fmt, rows=130, cols=257, experts=6, k=4, slot=1):
    def case(rng):
        return dict(out=np.full(rows, 7, 'f4'), x=normal(rng, cols), ids=moe_ids(rng, experts, k), k=k, slot=slot,
                    experts=experts, rows=rows, cols=cols, **expert_weights(rng, fmt, experts, rows, cols))
    return case


def moe_gate_up(fmt, inter=130, cols=257, experts=6, k=4):
    def case(rng):
        gate, up = (expert_weights(rng, fmt, experts, inter, cols) for _ in range(2))
        extra = {'scale_kind': gate.pop('scale_kind')} if fmt == 'int4' else {}
        up.pop('scale_kind', None)
        named = {('gate' if n in ('weight', 'packed') else 'gate_scales'): v for n, v in gate.items()}
        named.update({('up' if n in ('weight', 'packed') else 'up_scales'): v for n, v in up.items()})
        return dict(act=np.full(k * inter, 7, 'f4'), x=normal(rng, cols), ids=moe_ids(rng, experts, k), k=k,
                    experts=experts, inter=inter, up_row=0, stride=inter, cols=cols, **named, **extra)
    return case


def moe_down(fmt, rows=130, cols=257, experts=6, k=4):
    def case(rng):
        down = expert_weights(rng, fmt, experts, rows, cols)
        named = {('down' if n in ('weight', 'packed') else n): v for n, v in down.items()}
        return dict(out=np.full(rows, 7, 'f4'), act=normal(rng, k * cols), weights=rng.uniform(0.01, 1, size=k).astype('f4'),
                    ids=moe_ids(rng, experts, k), k=k, experts=experts, rows=rows, cols=cols, **named)
    return case


def topk(rng, tokens=5000, k=2048):
    from gpu.topk import spawn_buffers
    scores = rng.integers(-4, 4, size=tokens).astype('f4')
    scores[3], scores[7] = np.nan, -0.0
    scratch, order_a, order_b = (np.zeros(s, t) for s, t in spawn_buffers(tokens, k))
    return dict(tokens=tokens, k=k, scratch=scratch, order_a=order_a, order_b=order_b, scores=scores.view('i4'),
                out_ids=np.zeros(min(k, tokens), 'i4'))


# One case per kernel, at sizes with a tail; scalars are Python numbers, tensors numpy arrays
# (of the parameter's rank: a kernel whose writes the race proof checks by row writes a rank-2 out).
CASES = {
    'add': lambda r: dict(n=100003, a=normal(r, 100003), b=normal(r, 100003), out=np.zeros(100003, 'f4')),
    'copy': lambda r: dict(n=1000, x=normal(r, 1000), out=np.zeros(1000, 'f4')),
    'scale': lambda r: dict(n=1000, x=normal(r, 1000), alpha=0.37, out=np.zeros(1000, 'f4')),
    'scaled_add': lambda r: dict(slot=5, weights=normal(r, 8), n=6144, x=normal(r, 6144), out=normal(r, 6144)),
    'swiglu': lambda r: dict(n=12288, gate=normal(r, 12288, 6), up=normal(r, 12288), out=np.zeros(12288, 'f4')),
    'rope_glm': lambda r: dict(heads=16, dim=64, x=normal(r, 1024), cosine=normal(r, 32), sine=normal(r, 32),
                               out=np.zeros((16, 64), 'f4')),
    'rope_strided': lambda r: dict(heads=16, dim=64, x_stride=192, out_stride=128, x=normal(r, 16 * 192),
                                   cosine=normal(r, 32), sine=normal(r, 32), out=normal(r, 16 * 128).reshape(16, 128)),
    'matvec': lambda r: dict(rows=7, cols=131, weight=normal(r, 7 * 131), x=normal(r, 131), out=np.zeros(7, 'f4')),
    'matvec_heads': lambda r: dict(heads=8, rows=37, cols=131, w_stride=37 * 131 + 5, x_stride=140,
                                   weight=normal(r, 8 * (37 * 131 + 5)), x=normal(r, 8 * 140), out_stride=40,
                                   out=normal(r, 8 * 40).reshape(8, 40)),
    'index_scores': lambda r: dict(tokens=37, heads=16, dim=128, q=normal(r, 16 * 128), keys=normal(r, 37 * 128),
                                   scale=128 ** -0.5, weights=normal(r, 16, 0.25), out=np.zeros(37, 'f4')),
    'bf16_matvec': lambda r: dict(rows=96, cols=6144, weight=bf16(r, 96 * 6144), x=normal(r, 6144), out=np.zeros(96, 'f4')),
    'int4_matvec': lambda r: dict(rows=3, cols=257, scales=int4_scales(r, 3 * 3, 1), scale_kind=1,
                                  packed=int4(r, 3, 257), x=normal(r, 257), out=np.zeros(3, 'f4')),
    'fp8_matvec': lambda r: dict(rows=300, cols=6144, scales=r.uniform(0.001, 0.1, size=3 * 48).astype('f4'),
                                 weight=fp8(r, 300 * 6144), x=normal(r, 6144), out=np.zeros(300, 'f4')),
    'fp8_matvec4': lambda r: dict(rows=300, cols=6144, scales=r.uniform(0.001, 0.1, size=3 * 48).astype('f4'),
                                  weight=fp8(r, 300 * 6144).view('u4'), x=normal(r, 6144), out=np.zeros(300, 'f4')),
    'bf16_matmul': lambda r: dict(rows_m=6, rows_n=300, cols=257, experts=3, expert_ids=routing(r, 6, 3),
                                  weight=bf16(r, 3 * 300 * 257), x=normal(r, 6 * 257), out=np.zeros((6, 300), 'f4')),
    'int4_matmul': lambda r: dict(rows_m=6, rows_n=300, cols=257, experts=3, expert_ids=routing(r, 6, 3),
                                  scales=int4_scales(r, 3 * 300 * 3, 0), scale_kind=0, packed=int4(r, 3 * 300, 257),
                                  x=normal(r, 6 * 257), out=np.zeros((6, 300), 'f4')),
    'fp8_matmul': lambda r: dict(rows_m=6, rows_n=300, cols=257, experts=3, expert_ids=routing(r, 6, 3),
                                 weight=fp8(r, 3 * 300 * 257), x=normal(r, 6 * 257),
                                 scales=r.uniform(0.001, 0.1, size=3 * 3 * 3).astype('f4'), out=np.zeros((6, 300), 'f4')),
    'gemm_f32': lambda r: dict(m=33, n=65, k=256, a=normal(r, 33 * 256), b=normal(r, 256 * 65), out=np.full((33, 65), 7, 'f4')),
    'rmsnorm_rows': lambda r: dict(rows=5, dim=6144, eps=1e-5, x=normal(r, 5 * 6144), weight=normal(r, 6144),
                                   threads=256, out=np.zeros((5, 6144), 'f4')),
    'residual_rmsnorm': lambda r: dict(rows=7, dim=333, eps=1e-5, x=normal(r, 7 * 333), residual=normal(r, 7 * 333),
                                       residual_out=np.zeros((7, 333), 'f4'), weight=normal(r, 333), threads=96,
                                       out=np.zeros((7, 333), 'f4')),
    'layernorm': lambda r: dict(rows=9, dim=333, eps=1e-6, x=normal(r, 9 * 333) + 3, weight=normal(r, 333),
                                bias=normal(r, 333), threads=64, out=np.zeros((9, 333), 'f4')),
    'softmax_rows': lambda r: dict(rows=4, dim=2048, x=normal(r, 4 * 2048, 3), threads=1024, out=np.zeros((4, 2048), 'f4')),
    'router': lambda r: dict(experts=256, k=8, indices=np.zeros(8, 'i4'), logits=normal(r, 256),
                             bias=normal(r, 256, 0.1), weights=np.zeros(8, 'f4'), scale=2.5),
    'topk': topk,
    'topk_sort': lambda r: dict(tokens=1500, k=2048, scores=r.integers(-4, 4, size=1500).astype('f4').view('i4'),
                                out_ids=np.zeros(1500, 'i4')),
    'topk_sort_at': lambda r: dict(k=64, step=np.array([0, 700, 0, 0], 'i4'),
                                   scores=r.normal(size=1024).astype('f4').view('i4'), out_ids=np.zeros(64, 'i4')),
    'mla_partial': mla_partial,
    'mla_merge': mla_merge,
    'mla_decode': mla_decode,
    'mla_decode_at': mla_decode_at,
    'mla_chunks_at': mla_decode_at,
    'expert_scaled_add': lambda r: dict(out=normal(r, 333), x=normal(r, 333), weights=normal(r, 8),
                                        ids=moe_ids(r, 256, 8), k=8, slot=3, n=333),
    **{f'{p}expert_matvec': expert_matvec(f) for p, f in (('', 'f32'), ('bf16_', 'bf16'), ('fp8_', 'fp8'), ('int4_', 'int4'))},
    **{f'{p}moe_gate_up': moe_gate_up(f) for p, f in (('', 'f32'), ('bf16_', 'bf16'), ('fp8_', 'fp8'), ('int4_', 'int4'))},
    **{f'{p}moe_down': moe_down(f) for p, f in (('', 'f32'), ('bf16_', 'bf16'), ('fp8_', 'fp8'), ('int4_', 'int4'))},
    'fp8_moe4_gate_up': moe_gate_up('fp8x4', cols=260),
    'fp8_moe4_down': moe_down('fp8x4', cols=132),
}


def on_device(context, values):
    return {k: context.tensor(v.shape, str(v.dtype), v.tobytes()) if isinstance(v, np.ndarray) else v
            for k, v in values.items()}


def test_every_kernel_has_a_case():
    built = {p.stem for p in SPAWN.glob('*.ptx')}
    assert built <= set(CASES), built - set(CASES)


@pytest.mark.parametrize('name', sorted(CASES))
def test_vx_launch_matches_python_launch(context, name):
    ptx = SPAWN / f'{name}.ptx'
    if not ptx.exists():
        pytest.skip(f'{ptx} not built')
    try:
        vx = VxKernel(ptx)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(str(e))
    python = SpawnKernel(context, ptx)
    mutable = [p['name'] for p in python.params if p.get('mut')]
    ours, theirs = (on_device(context, CASES[name](np.random.default_rng(len(name)))) for _ in range(2))
    python.launch(**ours)  # the manifest's shape, which is Vx's
    vx(**theirs)
    context.synchronize()
    for out in mutable:
        a = np.frombuffer(ours[out].download(), dtype=ours[out].dtype)
        b = np.frombuffer(theirs[out].download(), dtype=theirs[out].dtype)
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8), err_msg=f'{name}.{out}')


def test_vx_launch_refuses_a_tensor_from_another_context(context):
    ptx = SPAWN / 'add.ptx'
    try:
        vx = VxKernel(ptx)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(str(e))
    with CUDAContext() as own:
        x = own.tensor((4,))
        with pytest.raises(ValueError):
            vx(n=4, a=x, b=x, out=x)
