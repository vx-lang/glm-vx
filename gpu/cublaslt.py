"""cuBLASLt's block-scaled FP8 matmul for GLM's FP8 Linears (#29): the library path SGLang's Hopper
kernels are measured against, on glm-vx's own context and streams.

The checkpoint's FP8 Linears store e4m3 weight codes with one float32 scale per 128x128 block
(``weight_scale_inv``). cuBLASLt 12.9+ takes exactly that (CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F)
against activations quantized to e4m3 per 128 columns with float32 scales (VEC128_32F), which
gpu/fp8_quant.vx makes. That is SGLang's w8a8 block-FP8 scheme, so the logits are close to the
float32-activation kernels', not equal (the maintainer's call, #29: SGLang quantizes too).

``orientation`` picks which cuBLASLt operand the weights are:

* ``'weights_b'`` (cuBLASLt's LtBlk128x128Fp8Matmul sample): A = x (op T, VEC128 scales),
  B = W (op N, 128x128 scales), D [m, n] column major with ld m.
* ``'weights_a'``: A = W (op T, 128x128 scales), B = x (op N, VEC128 scales), D [n, m] with ld n,
  the layout to try when a one-token D's 4-byte leading dimension is refused.

Either way ``out`` holds, for each token, its n outputs contiguously when m == 1. Which orientation the
library accepts, and how fast, is a GPU question: gpu/test_cublaslt_hardware.py and
``validation/matvec_profile.py --cublaslt-orientation`` answer it. Not a performance claim.

Hopper's block scales need cuBLASLt 12.9 or later; load_library skips older ones (CUDA 12.8 images
ship 12.8). Where the system's is older, the pip wheel works: ``pip install nvidia-cublas`` (CUDA 13,
driver 580+) or ``nvidia-cublas-cu12`` (12.9), found under the ``nvidia`` package as torch keeps it.
"""
import ctypes as C
import os

import numpy as np

# cublasLt.h, cublas_api.h and library_types.h (CUDA 13.3, cuBLAS 13.6).
CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUBLAS_COMPUTE_32F = 68
CUDA_R_32F, CUDA_R_16BF, CUDA_R_8F_E4M3 = 0, 14, 28
DESC_TRANSA, DESC_TRANSB = 3, 4
DESC_A_SCALE_POINTER, DESC_B_SCALE_POINTER = 17, 18
DESC_A_SCALE_MODE, DESC_B_SCALE_MODE = 31, 32
SCALE_VEC128_32F, SCALE_BLK128x128_32F = 4, 5
PREF_MAX_WORKSPACE_BYTES = 1
_OUT_TYPES = {'float32': CUDA_R_32F, 'uint16': CUDA_R_16BF}  # uint16 tensors hold bfloat16 bits

_P, _U64, _I64, _SIZE = C.c_void_p, C.c_uint64, C.c_int64, C.c_size_t


class CublasLtUnavailable(RuntimeError):
    """No libcublasLt, or it refused the problem (no algorithm for these shapes and types)."""


class _Algo(C.Structure):
    _fields_ = [('data', C.c_uint64 * 8)]


class _Heuristic(C.Structure):
    _fields_ = [('algo', _Algo), ('workspaceSize', _SIZE), ('state', C.c_int), ('wavesCount', C.c_float),
                ('reserved', C.c_int * 4)]


MIN_VERSION = 120900  # cublasLtGetVersion(): 12.9.0, the first with Hopper's VEC128 / BLK128x128 scales


def _wheel_dirs():
    """The nvidia namespace package's lib directories: cu13/lib (CUDA 13 wheels), cublas/lib (12.x)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec('nvidia')
    except (ImportError, ValueError):
        return
    for root in (spec.submodule_search_locations or []) if spec else []:
        yield os.path.join(root, 'cu13', 'lib')
        yield os.path.join(root, 'cublas', 'lib')


def _library_names():
    if os.environ.get('VX_CUBLASLT_LIB'):
        yield os.environ['VX_CUBLASLT_LIB']
    for directory in _wheel_dirs():  # before the system's: an image's CUDA may predate 12.9
        for name in ('libcublasLt.so.13', 'libcublasLt.so.12'):
            if os.path.exists(os.path.join(directory, name)):
                yield os.path.join(directory, name)
    yield from ('libcublasLt.so.13', 'libcublasLt.so.12', 'libcublasLt.so')


def load_library():
    """The first libcublasLt of at least MIN_VERSION: VX_CUBLASLT_LIB, the pip wheels, the system's."""
    errors = []
    for name in _library_names():
        try:
            lib = C.CDLL(name)
        except OSError as error:
            errors.append(f'{name}: {error}')
            continue
        lib.cublasLtGetStatusName.restype = C.c_char_p
        lib.cublasLtGetVersion.restype = _SIZE
        version = int(lib.cublasLtGetVersion())
        if version < MIN_VERSION:
            errors.append(f'{name}: version {version} < {MIN_VERSION} (no Hopper block scales)')
            continue
        lib.cublasLtMatrixLayoutCreate.argtypes = [C.POINTER(_P), C.c_int, _U64, _U64, _I64]
        lib.cublasLtMatmulDescCreate.argtypes = [C.POINTER(_P), C.c_int, C.c_int]
        lib.cublasLtMatmulDescSetAttribute.argtypes = [_P, C.c_int, _P, _SIZE]
        lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [_P, C.c_int, _P, _SIZE]
        lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [_P, _P, _P, _P, _P, _P, _P, C.c_int,
                                                       C.POINTER(_Heuristic), C.POINTER(C.c_int)]
        lib.cublasLtMatmul.argtypes = [_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, C.POINTER(_Algo), _P,
                                       _SIZE, _P]
        return lib
    raise CublasLtUnavailable('no libcublasLt: ' + '; '.join(errors))


def pack_weight_scales(scales):
    """[ceil(n/128), ceil(k/128)] float32 block scales (the checkpoint's weight_scale_inv) in the layout
    BLK128x128_32F takes: each 128-row block's k-block scales contiguous, padded with zeros to a
    multiple of 4 (cuBLASLt's sample sizes the tensor roundup(ceil(k/128), 4) * ceil(n/128))."""
    scales = np.asarray(scales, np.float32)
    if scales.ndim != 2:
        raise ValueError('weight scales must be [ceil(n/128), ceil(k/128)]')
    blocks, kb = scales.shape
    packed = np.zeros((blocks, -(-kb // 4) * 4), np.float32)
    packed[:, :kb] = scales
    return packed


class CublasLt:
    """A cuBLASLt handle on a gpu.runtime CUDAContext (its context is made current around each call)."""
    def __init__(self, context, lib=None):
        self.context = context
        self.lib = lib or load_library()
        self.handle = _P()
        with context._activate():
            self._check(self.lib.cublasLtCreate(C.byref(self.handle)), 'cublasLtCreate')

    @property
    def version(self):
        return int(self.lib.cublasLtGetVersion())

    def _check(self, status, what):
        if status:
            name = self.lib.cublasLtGetStatusName(status).decode()
            raise CublasLtUnavailable(f'{what}: {name} ({status})')

    def close(self):
        if self.handle:
            with self.context._activate():
                self.lib.cublasLtDestroy(self.handle)
            self.handle = _P()


class Fp8BlockMatmul:
    """out = dequant(x) @ dequant(W)^T for m tokens: x [mp, k] e4m3 codes with VEC128 scales
    [k/128, mp] (gpu/fp8_quant.vx's layout, scale_rows mp), W [n, k] e4m3 codes with
    pack_weight_scales' 128x128 block scales. float32 accumulation; out float32, or bfloat16 bits in a
    uint16 tensor. k must be a multiple of 128 (cuBLASLt's block size), and in practice n and k of 16.

    mp (``padded_m``) is m rounded up to a multiple of 4: cuBLASLt 13.10 on an H100 refuses every
    other token count (CUBLAS_STATUS_NOT_SUPPORTED at m = 1, 2, 3, 5). Rows m..mp-1 of x are
    multiplied too and their outputs are not used, so they need only be readable. Token i's n
    outputs are out[i*n:(i+1)*n] for 'weights_a' (D column major [n, mp]); 'weights_b' interleaves them
    (D [mp, n], ld mp). On the H100 a matmul costs about the same at 4 tokens as at 1 would: the
    weights are the traffic. Its float32 sums differ from float64 ones by ~3e-4 of the largest
    output (Hopper's FP8 tensor-core accumulation), as SGLang's kernels do."""
    def __init__(self, lt, m, n, k, *, out_dtype='float32', orientation='weights_a', workspace_bytes=32 << 20):
        if k % 128:
            raise ValueError('k must be a multiple of 128')
        if out_dtype not in _OUT_TYPES:
            raise ValueError(f'out_dtype must be one of {tuple(_OUT_TYPES)}')
        if orientation not in ('weights_a', 'weights_b'):
            raise ValueError("orientation must be 'weights_a' or 'weights_b'")
        self.lt, self.m, self.n, self.k = lt, m, n, k
        self.padded_m = mp = -(-m // 4) * 4
        self.out_dtype, self.orientation = out_dtype, orientation
        lib, check = lt.lib, lt._check
        self._handles = []
        self.desc = self._make(lib.cublasLtMatmulDescCreate, 'cublasLtMatmulDescCreate', lib.cublasLtMatmulDescDestroy,
                               CUBLAS_COMPUTE_32F, CUDA_R_32F)
        weights_a = orientation == 'weights_a'
        self._set(DESC_TRANSA, C.c_int(CUBLAS_OP_T))
        self._set(DESC_TRANSB, C.c_int(CUBLAS_OP_N))
        self._set(DESC_A_SCALE_MODE, C.c_int(SCALE_BLK128x128_32F if weights_a else SCALE_VEC128_32F))
        self._set(DESC_B_SCALE_MODE, C.c_int(SCALE_VEC128_32F if weights_a else SCALE_BLK128x128_32F))
        layout = lambda dtype, rows, cols, ld: self._make(lib.cublasLtMatrixLayoutCreate, 'cublasLtMatrixLayoutCreate',
                                                          lib.cublasLtMatrixLayoutDestroy, dtype, rows, cols, ld)
        out_type = _OUT_TYPES[out_dtype]
        if weights_a:  # D[n, m] = W^T(k x n)^T x(k x m)
            self.a = layout(CUDA_R_8F_E4M3, k, n, k)
            self.b = layout(CUDA_R_8F_E4M3, k, mp, k)
            self.c = layout(out_type, n, mp, n)
        else:  # D[m, n] = x^T(k x m)^T W(k x n)
            self.a = layout(CUDA_R_8F_E4M3, k, mp, k)
            self.b = layout(CUDA_R_8F_E4M3, k, n, k)
            self.c = layout(out_type, mp, n, mp)
        self.workspace_bytes = workspace_bytes
        self.workspace = lt.context.tensor((max(workspace_bytes, 1),), 'uint8')
        # The heuristic refuses VEC128 / BLK128x128 modes without scale pointers ("requires a custom
        # pointer"); any device address stands in until __call__ sets the real ones.
        self._set(DESC_A_SCALE_POINTER, _P(self.workspace.pointer))
        self._set(DESC_B_SCALE_POINTER, _P(self.workspace.pointer))
        pref =self._make(lib.cublasLtMatmulPreferenceCreate, 'cublasLtMatmulPreferenceCreate',
                          lib.cublasLtMatmulPreferenceDestroy)
        size = _U64(workspace_bytes)
        check(lib.cublasLtMatmulPreferenceSetAttribute(pref, PREF_MAX_WORKSPACE_BYTES, C.byref(size), C.sizeof(size)),
              'cublasLtMatmulPreferenceSetAttribute')
        self.heuristic, found = _Heuristic(), C.c_int(0)
        with lt.context._activate():
            check(lib.cublasLtMatmulAlgoGetHeuristic(lt.handle, self.desc, self.a, self.b, self.c, self.c, pref, 1,
                                                     C.byref(self.heuristic), C.byref(found)),
                  f'cublasLtMatmulAlgoGetHeuristic (m={m}, n={n}, k={k}, {orientation}, {out_dtype})')
        if found.value < 1:
            raise CublasLtUnavailable(f'no cuBLASLt algorithm for m={m}, n={n}, k={k}, {orientation}, {out_dtype}')
        self._alpha, self._beta = C.c_float(1.0), C.c_float(0.0)

    def _make(self, create, what, destroy, *args):
        handle = _P()
        self.lt._check(create(C.byref(handle), *args), what)
        self._handles.append((destroy, handle))
        return handle

    def _set(self, attribute, value):
        self.lt._check(self.lt.lib.cublasLtMatmulDescSetAttribute(self.desc, attribute, C.byref(value), C.sizeof(value)),
                       f'cublasLtMatmulDescSetAttribute({attribute})')

    def __call__(self, out, x_codes, x_scales, w_codes, w_scales, stream=None):
        m, n, k = self.padded_m, self.n, self.k
        padded_kb = -(-(k // 128) // 4) * 4
        expect = {'out': (out, m * n, self.out_dtype), 'x_codes': (x_codes, m * k, 'uint8'),
                  'x_scales': (x_scales, k // 128 * m, 'float32'), 'w_codes': (w_codes, n * k, 'uint8'),
                  'w_scales': (w_scales, -(-n // 128) * padded_kb, 'float32')}
        for name, (tensor, size, dtype) in expect.items():
            if tensor.dtype != dtype or int(np.prod(tensor.shape)) != size:
                raise ValueError(f'{name}: want {size} {dtype}, got {tensor.shape} {tensor.dtype}')
        weights_a = self.orientation == 'weights_a'
        a, b = (w_codes, x_codes) if weights_a else (x_codes, w_codes)
        a_scale, b_scale = (w_scales, x_scales) if weights_a else (x_scales, w_scales)
        self._set(DESC_A_SCALE_POINTER, _P(a_scale.pointer))
        self._set(DESC_B_SCALE_POINTER, _P(b_scale.pointer))
        lib = self.lt.lib
        with self.lt.context._activate():
            self.lt._check(lib.cublasLtMatmul(self.lt.handle, self.desc, C.byref(self._alpha), _P(a.pointer), self.a,
                                              _P(b.pointer), self.b, C.byref(self._beta), _P(out.pointer), self.c,
                                              _P(out.pointer), self.c, C.byref(self.heuristic.algo),
                                              _P(self.workspace.pointer), self.workspace_bytes,
                                              _P(stream.handle.value if stream is not None else None)),
                           'cublasLtMatmul')
        return out

    def close(self):
        while self._handles:
            destroy, handle = self._handles.pop()
            destroy(handle)
