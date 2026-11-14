"""Activations to e4m3 FP8 per 128 columns (gpu/fp8_quant.vx), for cuBLASLt's block-scaled FP8 matmul
(gpu/cublaslt.py, #29), and the NumPy model of both.

quant_groups is SGLang's per-token-group quantization with group 128 and float32 scales: a group's
scale is max(|x|, 1e-10) / 448 and its codes are x / scale rounded to the nearest e4m3fn value, ties
to the even code, clamped to +-448. glm_vx_gpu_fp8_quant_groups gives its codes and scales bit for bit
(gpu/test_fp8_quant.py).
"""
import ctypes as C

import numpy as np

GROUP = 128


def e4m3_values():
    """The value of every e4m3fn code 0..255 (NaN for 0x7f and 0xff)."""
    codes = np.arange(256)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    magnitude = np.where(exponent == 0, mantissa / 512.0, (8 + mantissa) * 2.0 ** (exponent - 7) / 8.0)
    magnitude = np.where((exponent == 15) & (mantissa == 7), np.nan, magnitude)
    return np.where(codes & 128, -magnitude, magnitude)


def e4m3_encode(v):
    """The nearest e4m3fn code to each float32 v, ties to the even code, |v| clamped to 448; NaN -> 0x7f.
    A negative v keeps its sign bit, -0.0 does not (the kernel tests v < 0)."""
    v = np.asarray(v, np.float32)
    table = e4m3_values()[:127]  # codes 0..126: the nonnegative finite values, increasing
    a = np.minimum(np.abs(v).astype(np.float64), 448.0)
    hi = np.clip(np.searchsorted(table, a), 0, 126)
    lo = np.maximum(hi - 1, 0)
    below, above = a - table[lo], table[hi] - a
    code = np.where(below < above, lo, np.where(above < below, hi, np.where(lo % 2 == 0, lo, hi)))
    code = np.where(v < 0, code + 128, code)
    return np.where(np.isnan(v), 127, code).astype(np.uint8)


def quant_groups(x):
    """(codes [rows, cols] uint8, scales [cols / 128, rows] float32) of x [rows, cols], cols % 128 == 0."""
    x = np.asarray(x, np.float32)
    rows, cols = x.shape
    if cols % GROUP:
        raise ValueError('cols must be a multiple of 128')
    g = x.reshape(rows, cols // GROUP, GROUP)
    scales = (np.maximum(np.abs(g).max(-1), np.float32(1e-10)) / np.float32(448)).astype(np.float32)
    q = (g / scales[..., None]).astype(np.float32)
    return e4m3_encode(q).reshape(rows, cols), np.ascontiguousarray(scales.T)


def dequant_groups(codes, scales):
    """float64 [rows, cols] values of quant_groups' output."""
    rows, cols = codes.shape
    values = e4m3_values()[codes].reshape(rows, cols // GROUP, GROUP)
    return (values * np.asarray(scales, np.float64).T[..., None]).reshape(rows, cols)


def launch(kernel, codes, scales, x, rows, cols, stream=None):
    """glm_vx_gpu_fp8_quant_groups: a warp a (row, group), 128 threads a block."""
    warps = rows * cols // GROUP
    kernel.launch((-(-warps // 4),), (128,), [codes, scales, x, C.c_int32(rows), C.c_int32(cols)], stream=stream)
