"""The block-FP8 GEMM (gpu/fp8_gemm.py, gpu/spawn/fp8_gemm/) without a GPU.

- Tensor maps: encode_tensor_map_2d against a fake driver (the arguments cuTensorMapEncodeTiled
  gets, the 64-byte host alignment, cuda.h's checks).
- The kernel's index math: gpu.fp8_gemm.simulate replays the TMA swizzle, the wgmma descriptors,
  the register fragments, the scale promotion, split-K and the masked epilogue, against the f64
  reference, and mutations of the descriptor or k step must fail.
- The numerical contract: the bound holds for an emulated reduced-precision accumulator.
- The host wrapper: plan, argument checks, and the launches it makes (fake driver).
- The generated Vx sources are up to date and agree with the host's constants.
gpu/test_fp8_gemm_hardware.py runs the kernels on an H100.
"""
import ctypes as C
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gpu import fp8_gemm as F
from gpu.runtime import (CUDAContext, CUDADriver, CUDAUnavailable, DeviceTensor, TENSOR_MAP_BYTES,
                         encode_tensor_map_2d)
from gpu.test_runtime import FakeLibrary, value

HERE = Path(__file__).parent
SOURCES = HERE / 'spawn' / 'fp8_gemm'


# --- tensor maps -----------------------------------------------------------------------------------

class TensorMapLibrary(FakeLibrary):
    """FakeLibrary whose cuTensorMapEncodeTiled writes its arguments into the 128-byte map."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.encoded = []

    def dispatch(self, name, args):
        if name == 'cuTensorMapEncodeTiled':
            target, dtype, rank, address, dims, strides, box, steps, interleave, swizzle, l2, oob = args
            target, rank = value(target), value(rank)
            fields = {'target': target, 'dtype': value(dtype), 'rank': rank, 'address': value(address),
                      'dims': [dims[i] for i in range(rank)], 'strides': [strides[i] for i in range(rank - 1)],
                      'box': [box[i] for i in range(rank)], 'steps': [steps[i] for i in range(rank)],
                      'interleave': value(interleave), 'swizzle': value(swizzle), 'l2': value(l2), 'oob': value(oob)}
            self.encoded.append(fields)
            blob = struct.pack('<QQQQQII', fields['address'], *fields['dims'], *fields['strides'],
                               fields['box'][0] | fields['box'][1] << 16, fields['swizzle'], fields['l2'])
            C.memmove(target, blob.ljust(TENSOR_MAP_BYTES, b'\0'), TENSOR_MAP_BYTES)
            return 0
        if name == 'cuDeviceGetAttribute' and value(args[1]) == 16:
            C.cast(args[0], C.POINTER(C.c_int32))[0] = 132
            return 0
        result = super().dispatch(name, args)
        if name == 'cuMemAlloc_v2':        # cuMemAlloc's 256-byte alignment
            self.next_address = -(-self.next_address // 256) * 256
        return result


def decode_map(raw):
    address, d0, d1, stride, box, swizzle, l2 = struct.unpack_from('<QQQQQII', raw)
    return {'address': address, 'dims': [d0, d1], 'stride': stride, 'box': [box & 0xFFFF, box >> 16],
            'swizzle': swizzle, 'l2': l2}


def test_tensor_map_arguments():
    library = TensorMapLibrary()
    driver = CUDADriver(_library=library)
    raw = encode_tensor_map_2d(driver, 0x7000_0100, 576, 6144, 6144, 128, 128)
    assert len(raw) == TENSOR_MAP_BYTES
    f = library.encoded[-1]
    assert f['target'] % 64 == 0                    # host CUtensorMap 64-byte aligned
    assert (f['dtype'], f['rank'], f['address']) == (0, 2, 0x7000_0100)
    assert f['dims'] == [6144, 576] and f['strides'] == [6144]      # innermost (k) first, bytes
    assert f['box'] == [128, 128] and f['steps'] == [1, 1]
    assert (f['interleave'], f['swizzle'], f['l2'], f['oob']) == (0, 3, 3, 0)   # none, 128B, 256B, zeros
    assert decode_map(raw)['address'] == 0x7000_0100


@pytest.mark.parametrize('kwargs,error', [
    ({'address': 0x1008}, 'aligned'),
    ({'row_stride': 6152 + 4}, 'multiple of 16'),
    ({'row_stride': 16, 'cols': 6144}, 'at least a row'),
    ({'box_cols': 256}, 'at most 128'),
    ({'box_cols': 8}, 'multiple of 16'),
    ({'box_rows': 257}, 'between'),
    ({'rows': 0}, 'between'),
    ({'swizzle': '256B'}, 'swizzle must be'),
])
def test_tensor_map_checks(kwargs, error):
    args = {'address': 0x1000, 'rows': 64, 'cols': 6144, 'row_stride': 6144, 'box_rows': 64, 'box_cols': 128}
    args.update(kwargs)
    with pytest.raises(ValueError, match=error):
        encode_tensor_map_2d(CUDADriver(_library=TensorMapLibrary()), **args)


def test_tensor_map_needs_the_driver_symbol():
    driver = CUDADriver(_library=TensorMapLibrary())
    driver.cuTensorMapEncodeTiled = None
    with pytest.raises(CUDAUnavailable):
        encode_tensor_map_2d(driver, 0x1000, 64, 128, 128, 64, 128)


# --- the kernel's index math ---------------------------------------------------------------------------

CASES = [  # m, n, k, variant, splits, bf16: tails in every dimension, both variants, split-K
    (1, 128, 128, 'm64', 1, False),
    (7, 200, 272, 'm64', 1, True),          # N tail 72, K tail 16
    (64, 256, 512, 'm64', 2, False),
    (65, 130, 400, 'm128', 1, False),       # M tail: second warpgroup holds 1 row
    (130, 300, 1040, 'm128', 3, True),      # M tile 2 (2 rows), N tail 44, K tail 16, 3 splits
    (3, 576, 384, None, None, False),       # kv_a's N = 4.5 x 128; the planned split
]


@pytest.mark.parametrize('m,n,k,variant,splits,bf16', CASES)
def test_simulated_kernel_matches_reference(m, n, k, variant, splits, bf16):
    a, a_scale, b, b_scale = F.random_case(m, n, k, seed=m * 7 + n + k)
    p = F.plan(m, n, k, variant=variant, splits=splits, sms=8)
    got = F.simulate(a, a_scale, b, b_scale, p, bf16=bf16)
    ref, bound = F.reference(a, a_scale, b, b_scale, splits=p.splits, bf16=bf16)
    got = F.bf16_value(got) if bf16 else got
    assert got.shape == (m, n)
    assert np.all(np.abs(got - ref) <= bound)
    if not bf16:   # exact block sums: only f32 rounding is left
        assert np.abs(got - ref).max() <= 1e-5 * np.abs(ref).max()


@pytest.mark.parametrize('mutation', [
    {'k_step': 16},                                                        # the no-swizzle bf16 step
    {'k_step': 0},
    {'desc': lambda addr: F.sw128_desc(addr) & ~(3 << 62)},                # swizzle mode 0
    {'desc': lambda addr: (F.sw128_desc(addr) & ~(0x3FFF << 32)) | (8 << 32)},   # SBO 128, not 1024
])
def test_mutated_descriptors_fail(mutation):
    m, n, k = 64, 128, 256
    a, a_scale, b, b_scale = F.random_case(m, n, k, seed=3)
    p = F.plan(m, n, k, splits=1)
    ref, bound = F.reference(a, a_scale, b, b_scale)
    got = F.simulate(a, a_scale, b, b_scale, p, **mutation)
    assert np.mean(np.abs(got - ref) > bound) > 0.5


def test_swizzle_is_a_permutation_within_each_atom():
    offsets = np.arange(4096)
    sw = F.swizzle128(offsets)
    assert sorted(sw) == list(offsets)
    assert np.all(sw // 1024 == offsets // 1024) and np.all(sw // 128 == offsets // 128)


def test_fragment_layout_covers_the_tile_once():
    row, col = F.fragment_index()
    assert sorted(zip(row.ravel(), col.ravel())) == [(r, c) for r in range(64) for c in range(128)]


def test_descriptor_fields():
    desc = F.sw128_desc(66560)
    assert desc & 0x3FFF == 66560 >> 4
    assert (desc >> 16) & 0x3FFF == 1 and (desc >> 32) & 0x3FFF == 64
    assert (desc >> 49) & 7 == 0 and desc >> 62 == 1


# --- the numerical contract ------------------------------------------------------------------------------

def truncate(x, bits):
    """Keep `bits` significant bits of each value (toward zero): the reduced-precision adder."""
    x = np.asarray(x, np.float64)
    e = np.floor(np.log2(np.where(x == 0, 1, np.abs(x))))
    q = 2.0 ** (e - (bits - 1))
    return np.trunc(x / q) * q


def emulate_hopper(a_codes, a_scale, b_codes, b_scale, bits=14):
    """A pessimistic model of the tensor core: each k32 step's sum and each accumulation keep
    `bits` significant bits (truncated); the promotion in f32 as the kernel does it."""
    a, b = F.e4m3_values(a_codes), F.e4m3_values(b_codes)
    m, k = a.shape
    n = b.shape[0]
    blocks = -(-k // 128)
    acc = np.zeros((m, n), np.float32)
    for kb in range(blocks):
        part = np.zeros((m, n))
        for step in range(4):
            ks = slice(kb * 128 + 32 * step, min(k, kb * 128 + 32 * step + 32))
            part = truncate(part + truncate(a[:, ks] @ b[:, ks].T, bits), bits)
        s = (a_scale[:, kb:kb + 1] * np.repeat(b_scale[:, kb], 128)[None, :n]).astype(np.float32)
        acc = acc + part.astype(np.float32) * s
    return acc


def test_bound_covers_a_14_bit_accumulator():
    m, n, k = 16, 256, 2048
    a, a_scale, b, b_scale = F.random_case(m, n, k, seed=11)
    ref, bound = F.reference(a, a_scale, b, b_scale)
    err = np.abs(emulate_hopper(a, a_scale, b, b_scale, bits=14) - ref)
    assert np.all(err <= bound)
    # and the bound is not vacuous: a 9-bit accumulator breaks it
    assert np.any(np.abs(emulate_hopper(a, a_scale, b, b_scale, bits=9) - ref) > bound)


def test_bf16_rounding_is_nearest_even():
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, 3.0e-3], np.float32)
    assert list(F.bf16_round(x)) == [0x3F80, 0x3F80, 0x3F82, 0xC020, 0x3B45]


# --- plan and the host wrapper ---------------------------------------------------------------------------

@pytest.mark.parametrize('m,n,k', [(1, 2048, 6144), (1, 576, 6144), (7, 6144, 2048), (64, 12288, 6144),
                                   (128, 6144, 12288), (256, 2048, 6144), (65, 130, 1040)])
def test_plan_covers_k_once_and_fills_the_gpu(m, n, k):
    p = F.plan(m, n, k)
    assert p.variant.name == ('m64' if m <= 64 else 'm128')
    assert (p.splits - 1) * p.split_blocks < p.blocks_k <= p.splits * p.split_blocks   # no empty split
    slots = F.SM90_SMS * p.variant.ctas_per_sm
    assert p.grid >= min(slots, p.tiles_m * p.tiles_n * max(1, p.blocks_k // 2)) * 0.4


def test_plan_refuses():
    with pytest.raises(ValueError, match='multiple of 16'):
        F.plan(1, 128, 100)
    with pytest.raises(ValueError):
        F.plan(0, 128, 128)
    with pytest.raises(ValueError):
        F.plan(1, 128, 256, splits=3)


KERNEL_PARAMS = [('maps', 'u8', False), ('a_scale', 'f32', False), ('b_scale', 'f32', False), ('out', 'f32', True),
                 ('out_bf16', 'u16', True)]
SCALARS = ['rows_m', 'rows_n', 'cols', 'splits', 'split_blocks', 'bf16_out']


def manifest(entry, tensors, scalars, shared):
    params = [{'name': n, 'kind': 'memref', 'element': e, 'rank': 1, 'mut': mut} for n, e, mut in tensors]
    params += [{'name': n, 'kind': 'scalar', 'type': 'i32'} for n in scalars]
    return {'schema': 'glm-vx-spawn-v2', 'entry': entry, 'params': params, 'dynamic_shared_bytes': shared}


@pytest.fixture
def fake(tmp_path):
    shared = {'m64': 98368, 'm128': 131136}
    for name in F.VARIANTS:
        (tmp_path / f'fp8_gemm_{name}.ptx').write_text('// fake\n')
        (tmp_path / f'fp8_gemm_{name}.json').write_text(json.dumps(
            manifest(f'glm_vx_fp8_gemm_{name}', KERNEL_PARAMS, SCALARS, shared[name])))
    (tmp_path / 'fp8_gemm_reduce.ptx').write_text('// fake\n')
    (tmp_path / 'fp8_gemm_reduce.json').write_text(json.dumps(manifest(
        'glm_vx_fp8_gemm_reduce', [('partial', 'f32', False), ('out', 'f32', True), ('out_bf16', 'u16', True)],
        ['count', 'splits', 'bf16_out'], 0)))
    library = TensorMapLibrary(capability=(9, 0))
    launches = []

    def on_launch(args):
        params = args[9]
        launches.append({'kernel': library.kernel_names[value(args[0])], 'grid': value(args[1]),
                         'block': value(args[4]), 'shared': value(args[7]), 'params': params,
                         'scalar': lambda i: C.cast(params[i], C.POINTER(C.c_int32))[0],
                         'pointer': lambda i: C.cast(params[i], C.POINTER(C.c_uint64))[0]})
    library.on_launch = on_launch
    ctx = CUDAContext(_driver=CUDADriver(_library=library))
    yield ctx, library, launches, tmp_path
    ctx.close()


def tensors(ctx, m, n, k, out_dtype):
    blocks = -(-k // 128)
    return (ctx.tensor((m * k,), 'uint8'), ctx.tensor((m * blocks,), 'float32'), ctx.tensor((n * k,), 'uint8'),
            ctx.tensor((-(-n // 128) * blocks,), 'float32'), ctx.tensor((m * n,), out_dtype))


def test_wrapper_launches_one_kernel_and_uploads_both_maps(fake):
    ctx, library, launches, build = fake
    gemm = F.FP8GEMM(ctx, build)
    a, sa, b, sb, out = tensors(ctx, 128, 6144, 12288, 'uint16')
    p = gemm(a, sa, b, sb, out, m=128, n=6144, k=12288, splits=1)
    assert [l['kernel'] for l in launches] == ['glm_vx_fp8_gemm_m128']
    l = launches[0]
    assert (l['grid'], l['block'], l['shared']) == (p.grid, 288, 131136) == (48, 288, 131136)
    # params: 5 tensors x 5 words, then rows_m, rows_n, cols, splits, split_blocks, bf16_out
    assert [l['scalar'](25 + i) for i in range(6)] == [128, 6144, 12288, 1, 96, 1]
    assert l['pointer'](20) == out.pointer                      # out_bf16's aligned pointer
    maps_ptr = l['pointer'](0)
    allocation, offset = library.storage(maps_ptr, 256)
    raw = bytes(allocation[offset:offset + 256])
    assert maps_ptr % 256 == 0
    assert decode_map(raw[:128]) == {'address': a.pointer, 'dims': [12288, 128], 'stride': 12288, 'box': [128, 128],
                                     'swizzle': 3, 'l2': 3}
    assert decode_map(raw[128:]) == {'address': b.pointer, 'dims': [12288, 6144], 'stride': 12288,
                                     'box': [128, 128], 'swizzle': 3, 'l2': 3}
    # the maps are cached: a second call with the same buffers encodes nothing
    encoded = len(library.encoded)
    gemm(a, sa, b, sb, out, m=128, n=6144, k=12288, splits=1)
    assert len(library.encoded) == encoded


def test_wrapper_split_k_runs_the_reduction(fake):
    ctx, library, launches, build = fake
    gemm = F.FP8GEMM(ctx, build)
    a, sa, b, sb, out = tensors(ctx, 1, 2048, 6144, 'float32')
    p = gemm(a, sa, b, sb, out, m=1, n=2048, k=6144)
    assert p.variant.name == 'm64' and p.splits > 1
    main, reduce = launches
    assert (main['kernel'], main['block'], main['shared']) == ('glm_vx_fp8_gemm_m64', 160, 98368)
    assert main['scalar'](28) == p.splits and main['scalar'](29) == p.split_blocks
    workspace = main['pointer'](15)
    assert workspace != out.pointer
    assert reduce['kernel'] == 'glm_vx_fp8_gemm_reduce' and reduce['grid'] == (2048 + 255) // 256
    assert reduce['pointer'](0) == workspace and reduce['pointer'](5) == out.pointer
    assert [reduce['scalar'](15 + i) for i in range(3)] == [2048, p.splits, 0]
    # the A map's box is 64 rows for the m64 variant
    allocation, offset = library.storage(main['pointer'](0), 256)
    assert decode_map(bytes(allocation[offset:offset + 128]))['box'] == [128, 64]


@pytest.mark.parametrize('change,error', [
    ({'a': ('uint16', 64 * 512)}, TypeError),
    ({'a': ('uint8', 64 * 511)}, ValueError),
    ({'a_scale': ('float32', 64 * 3)}, ValueError),
    ({'b_scale': ('float32', 3)}, ValueError),
    ({'out': ('float16', 64 * 256)}, TypeError),
    ({'out': ('float32', 64 * 255)}, ValueError),
])
def test_wrapper_checks_arguments(fake, change, error):
    ctx, library, launches, build = fake
    m, n, k = 64, 256, 512
    shapes = {'a': ('uint8', m * k), 'a_scale': ('float32', m * 4), 'b': ('uint8', n * k),
              'b_scale': ('float32', 2 * 4), 'out': ('float32', m * n)}
    shapes.update(change)
    args = {name: ctx.tensor((size,), dtype) for name, (dtype, size) in shapes.items()}
    with pytest.raises(error):
        F.FP8GEMM(ctx, build)(args['a'], args['a_scale'], args['b'], args['b_scale'], args['out'], m=m, n=n, k=k)
    assert launches == []


def test_wrapper_refuses_unaligned_a(fake):
    ctx, library, launches, build = fake
    m, n, k = 2, 128, 128
    base = ctx.tensor((m * k + 16,), 'uint8')
    a = base.view((m * k,), offset=8)
    _, sa, b, sb, out = tensors(ctx, m, n, k, 'float32')
    with pytest.raises(ValueError, match='16-byte'):
        F.FP8GEMM(ctx, build)(a, sa, b, sb, out, m=m, n=n, k=k)


# --- generated sources --------------------------------------------------------------------------------------

def test_generated_sources_are_current():
    result = subprocess.run([sys.executable, str(SOURCES / 'gen_fp8_gemm.py'), '--check'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', list(F.VARIANTS))
def test_vx_constants_match_the_host(name):
    v = F.VARIANTS[name]
    text = (SOURCES / f'fp8_gemm_{name}.vx').read_text()
    assert f'block({v.threads})' in text
    assert f'mbar_arrive_expect_tx(full, {v.a_stage + v.b_stage})' in text
    assert f'for s in 0..{v.stages}' in text
    assert f'Tensor<i32, [{v.stages * v.a_stage // 4}], Memory::SMEM, aligned(1024)>' in text
    assert f'tma_load_2d(a_base + {v.a_stage} * (s as i64), a_map, k, m0, full)' in text
    sys.path.insert(0, str(SOURCES))
    try:
        import gen_fp8_gemm
        assert gen_fp8_gemm.VARIANTS[name] == {'consumers': v.consumers, 'stages': v.stages}
    finally:
        sys.path.remove(str(SOURCES))


def test_built_manifests_match_the_fake_ones():
    """If gpu/spawn/fp8_gemm/build.sh has run, the real manifests have the parameters the wrapper
    passes (the fake tests above assume them)."""
    if not (F.BUILD / 'fp8_gemm_m64.json').exists():
        pytest.skip('gpu/spawn/fp8_gemm/build.sh has not run')
    for name in F.VARIANTS:
        real = json.loads((F.BUILD / f'fp8_gemm_{name}.json').read_text())
        assert [p['name'] for p in real['params']] == [n for n, _, _ in KERNEL_PARAMS] + SCALARS
        assert real['dynamic_shared_bytes'] > 0


class SimulatedGEMM:
    """FP8GEMM's call signature, computed by F.simulate on a fake driver's memory: lets the
    hardware test's harness (sentinels, timing, the bound, the JSON record) run without a GPU."""
    def __call__(self, a, a_scale, b, b_scale, out, *, m, n, k, splits=None, variant=None, stream=None):
        p = F.plan(m, n, k, splits=splits, variant=variant)
        bf16 = F.check_arguments(a, a_scale, b, b_scale, out, m, n, k)
        get = lambda t, dt: np.frombuffer(t.download(), dt)
        c = F.simulate(get(a, np.uint8).reshape(m, k), get(a_scale, np.float32), get(b, np.uint8).reshape(n, k),
                       get(b_scale, np.float32), p, bf16=bf16)
        out.upload(np.ascontiguousarray(c).tobytes())
        return p


def test_hardware_harness_runs_on_a_simulated_gpu(tmp_path, monkeypatch):
    from gpu import test_fp8_gemm_hardware as H
    ctx = CUDAContext(_driver=CUDADriver(_library=TensorMapLibrary(capability=(9, 0))))
    try:
        monkeypatch.setattr(H, 'RECORDS', [])
        for bf16, timed in ((True, True), (False, False)):
            record = H.run_case((ctx, SimulatedGEMM()), 65, 200, 272, bf16=bf16, timed=timed, seed=1)
            assert record['max_err_over_bound'] <= 1 and record['variant'] == 'm128'
        assert 'tflops' in H.RECORDS[0] and 'tflops' not in H.RECORDS[1]
    finally:
        ctx.close()
