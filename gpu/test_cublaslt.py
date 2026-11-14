"""gpu/cublaslt.py's plumbing without a GPU: a fake libcublasLt records what Fp8BlockMatmul asks for.
The library's arithmetic and which orientation it accepts are gpu/test_cublaslt_hardware.py's."""
import ctypes as C
from contextlib import contextmanager

import numpy as np
import pytest

from gpu import cublaslt as L


class _Fn:
    def __init__(self, lib, name):
        self.lib, self.name = lib, name

    def __call__(self, *args):
        return self.lib.call(self.name, args)


class FakeLib:
    """Records layouts, attributes and the matmul call; handles are distinct integers."""
    def __init__(self, algos=1):
        self.algos, self.next, self.layouts, self.attrs, self.matmuls, self.destroyed = algos, 100, {}, {}, [], []

    def __getattr__(self, name):
        if name.startswith('cublasLt'):
            return _Fn(self, name)
        raise AttributeError(name)

    def _handle(self, out):
        self.next += 1
        C.cast(out, C.POINTER(C.c_void_p))[0] = self.next
        return self.next

    def call(self, name, args):
        if name in ('cublasLtCreate', 'cublasLtMatmulDescCreate', 'cublasLtMatmulPreferenceCreate'):
            self._handle(args[0])
        elif name == 'cublasLtMatrixLayoutCreate':
            self.layouts[self._handle(args[0])] = tuple(int(a) for a in args[1:])
        elif name == 'cublasLtMatmulDescSetAttribute':
            pointer = args[2]._obj if hasattr(args[2], '_obj') else args[2]
            self.attrs[args[1]] = pointer.value
        elif name == 'cublasLtMatmulAlgoGetHeuristic':
            # cuBLASLt 13.10 refuses VEC128 / BLK128x128 scale modes without scale pointers (INVALID_VALUE).
            assert self.attrs.get(L.DESC_A_SCALE_POINTER) and self.attrs.get(L.DESC_B_SCALE_POINTER)
            C.cast(args[9], C.POINTER(C.c_int))[0] = self.algos
        elif name == 'cublasLtMatmul':
            self.matmuls.append(dict(a=args[3].value, a_layout=self.layouts[args[4].value], b=args[5].value,
                                     b_layout=self.layouts[args[6].value], d=args[10].value,
                                     d_layout=self.layouts[args[11].value], scales=(self.attrs[L.DESC_A_SCALE_POINTER],
                                                                              self.attrs[L.DESC_B_SCALE_POINTER])))
        elif name.endswith('Destroy'):
            self.destroyed.append(name)
        return 0


class FakeTensor:
    def __init__(self, shape, dtype, pointer):
        self.shape, self.dtype, self.pointer = tuple(shape), dtype, pointer


class FakeContext:
    def __init__(self):
        self.next = 0x1000

    @contextmanager
    def _activate(self):
        yield

    def tensor(self, shape, dtype='float32'):
        self.next += 0x1000
        return FakeTensor(shape, dtype, self.next)


def _operands(ctx, m, n, k, out_dtype='float32'):
    m = -(-m // 4) * 4  # cuBLASLt's token count: x and out are padded to it
    return dict(out=ctx.tensor((m, n), out_dtype), x_codes=ctx.tensor((m, k), 'uint8'),
                x_scales=ctx.tensor((k // 128, m)), w_codes=ctx.tensor((n, k), 'uint8'),
                w_scales=ctx.tensor((-(-n // 128), -(-(k // 128) // 4) * 4)))


@pytest.mark.parametrize('orientation', ['weights_a', 'weights_b'])
def test_layouts_scales_and_pointers_follow_the_orientation(orientation):
    lib, ctx = FakeLib(), FakeContext()
    lt = L.CublasLt(ctx, lib=lib)
    m, n, k = 1, 576, 6144
    mm = L.Fp8BlockMatmul(lt, m, n, k, orientation=orientation)
    ops = _operands(ctx, m, n, k)
    mm(**ops)
    call, = lib.matmuls
    weights_a = orientation == 'weights_a'
    assert lib.attrs[L.DESC_TRANSA] == L.CUBLAS_OP_T and lib.attrs[L.DESC_TRANSB] == L.CUBLAS_OP_N
    w_mode, x_mode = L.SCALE_BLK128x128_32F, L.SCALE_VEC128_32F
    assert (lib.attrs[L.DESC_A_SCALE_MODE], lib.attrs[L.DESC_B_SCALE_MODE]) == ((w_mode, x_mode) if weights_a else (x_mode, w_mode))
    w, x = (ops['w_codes'].pointer, ops['x_codes'].pointer)
    assert (call['a'], call['b']) == ((w, x) if weights_a else (x, w))
    ws, xs = ops['w_scales'].pointer, ops['x_scales'].pointer
    assert call['scales'] == ((ws, xs) if weights_a else (xs, ws))
    # (type, rows, cols, ld), column major: op T on a k x rows operand is a row-major [rows, k] matrix.
    # One token is multiplied as 4 (padded_m): cuBLASLt refuses other token counts.
    e4m3, mp = L.CUDA_R_8F_E4M3, mm.padded_m
    assert mp == 4
    if weights_a:
        assert (call['a_layout'], call['b_layout'], call['d_layout']) == ((e4m3, k, n, k), (e4m3, k, mp, k), (L.CUDA_R_32F, n, mp, n))
    else:
        assert (call['a_layout'], call['b_layout'], call['d_layout']) == ((e4m3, k, mp, k), (e4m3, k, n, k), (L.CUDA_R_32F, mp, n, mp))
    assert call['d'] == ops['out'].pointer
    mm.close()
    assert lib.destroyed.count('cublasLtMatrixLayoutDestroy') == 3


def test_refuses_wrong_sizes_and_a_missing_algorithm():
    lib, ctx = FakeLib(), FakeContext()
    lt = L.CublasLt(ctx, lib=lib)
    mm = L.Fp8BlockMatmul(lt, 1, 256, 384)
    ops = _operands(ctx, 1, 256, 384)
    ops['w_scales'] = ctx.tensor((2, 3))  # unpadded: k/128 = 3 must be padded to 4
    with pytest.raises(ValueError, match='w_scales'):
        mm(**ops)
    with pytest.raises(ValueError, match='multiple of 128'):
        L.Fp8BlockMatmul(lt, 1, 256, 200)
    with pytest.raises(L.CublasLtUnavailable, match='no cuBLASLt algorithm'):
        L.Fp8BlockMatmul(L.CublasLt(ctx, lib=FakeLib(algos=0)), 1, 256, 384)


def test_pack_weight_scales_pads_each_row_block_to_four():
    packed = L.pack_weight_scales(np.arange(6, dtype='f4').reshape(2, 3))
    np.testing.assert_array_equal(packed, [[0, 1, 2, 0], [3, 4, 5, 0]])
    assert L.pack_weight_scales(np.ones((1, 8), 'f4')).shape == (1, 8)


def test_loader_prefers_the_wheels_and_skips_libraries_before_12_9(monkeypatch, tmp_path):
    for sub, name in (('cu13/lib', 'libcublasLt.so.13'), ('cublas/lib', 'libcublasLt.so.12')):
        (tmp_path / sub).mkdir(parents=True)
        (tmp_path / sub / name).touch()
    monkeypatch.delenv('VX_CUBLASLT_LIB', raising=False)
    monkeypatch.setattr(L, '_wheel_dirs', lambda: iter([str(tmp_path / 'cu13/lib'), str(tmp_path / 'cublas/lib')]))
    names = list(L._library_names())
    assert names[:2] == [str(tmp_path / 'cu13/lib/libcublasLt.so.13'), str(tmp_path / 'cublas/lib/libcublasLt.so.12')]
    assert names[2:] == ['libcublasLt.so.13', 'libcublasLt.so.12', 'libcublasLt.so']

    versions = {names[0]: None, names[1]: 120800, 'libcublasLt.so.13': 130600}  # None: fails to load
    opened = []

    class FakeCDLL:
        def __init__(self, name):
            if versions.get(name) is None:
                raise OSError('not found')
            opened.append(name)
            self.cublasLtGetVersion = lambda: versions[name]
            self.cublasLtGetStatusName = self.cublasLtGetVersion  # restype set on it, unused

        def __getattr__(self, attr):
            fn = lambda *a: 0
            self.__dict__[attr] = fn
            return fn

    monkeypatch.setattr(L.C, 'CDLL', FakeCDLL)
    L.load_library()
    assert opened == [names[1], 'libcublasLt.so.13']  # 12.8 opened and skipped, 13.6 taken
    versions['libcublasLt.so.13'] = 120800
    with pytest.raises(L.CublasLtUnavailable, match='120800 < 120900'):
        L.load_library()
