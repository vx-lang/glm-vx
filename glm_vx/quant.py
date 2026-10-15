"""Reference math for quantized weight matmuls (M5): INT4 group quantization and FP8 block matvec.

CPU oracles for the future GPU kernels, plus the packing format they must read.

INT4 (symmetric, per-row groups along the input dimension):
  w[r, g*G:(g+1)*G] ~ scale[r, g] * code, code in [-8, 7], scale = amax/7 (fp16-rounded).
  Codes are stored two per byte, low nibble first (element 2i in the low nibble), as
  two's-complement 4-bit values; an odd trailing element is padded with a zero nibble.

FP8 block matvec: `fp8_matvec_blocked` accumulates each 128-wide block's partial dot product
in float64 *before* applying that block's scale, the order a fused kernel uses, so it is a
check on the kernel's structure and not just on dequantize-then-multiply.
"""
import math

import numpy as np

from .checkpoint import FP8_E4M3FN_TABLE

INT4_MIN, INT4_MAX = -8, 7


def int4_quantize(w, group=128):
    """(packed uint8 [rows, ceil(cols/2)], scales float16 [rows, ceil(cols/group)])."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or group < 1:
        raise ValueError('int4_quantize expects a matrix and a positive group size')
    rows, cols = w.shape
    groups = math.ceil(cols / group)
    scales = np.ones((rows, groups), dtype=np.float16)
    codes = np.zeros((rows, cols), dtype=np.int8)
    for g in range(groups):
        tile = w[:, g * group:(g + 1) * group]
        amax = np.abs(tile).max(axis=1)
        scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float16)
        scales[:, g] = scale
        s = scale.astype(np.float32)[:, None]
        codes[:, g * group:(g + 1) * group] = np.clip(np.rint(tile / s), INT4_MIN, INT4_MAX).astype(np.int8)
    nibbles = (codes & 0xF).astype(np.uint8)
    if cols % 2:
        nibbles = np.concatenate([nibbles, np.zeros((rows, 1), np.uint8)], axis=1)
    packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    return packed, scales


def int4_unpack(packed, cols):
    """Signed codes int8 [rows, cols] from the packed nibbles."""
    packed = np.asarray(packed, dtype=np.uint8)
    nibbles = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    nibbles[:, 0::2] = packed & 0xF
    nibbles[:, 1::2] = packed >> 4
    nibbles = nibbles[:, :cols].astype(np.int8)
    return np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.int8)


def int4_dequantize(packed, scales, cols, group=128):
    codes = int4_unpack(packed, cols).astype(np.float32)
    scale = np.repeat(np.asarray(scales, dtype=np.float32), group, axis=1)[:, :cols]
    return codes * scale


def int4_matvec(packed, scales, x, group=128):
    """float64 reference y = dequant(W) @ x, accumulating per group then applying its scale."""
    x = np.asarray(x, dtype=np.float64)
    cols = x.shape[0]
    codes = int4_unpack(packed, cols).astype(np.float64)
    y = np.zeros(codes.shape[0])
    for g in range(math.ceil(cols / group)):
        sl = slice(g * group, (g + 1) * group)
        y += (codes[:, sl] @ x[sl]) * np.asarray(scales[:, g], dtype=np.float64)
    return y


def fp8_matvec_blocked(codes, scales, x, block=(128, 128)):
    """float64 reference y = dequant(W) @ x, summing each block's raw-code dot product before its scale."""
    codes = np.asarray(codes, dtype=np.uint8)
    values = FP8_E4M3FN_TABLE[codes].astype(np.float64)
    x = np.asarray(x, dtype=np.float64)
    br, bc = block
    rows, cols = codes.shape
    if x.shape != (cols,) or scales.shape != (math.ceil(rows / br), math.ceil(cols / bc)):
        raise ValueError('shape mismatch between codes, scales and x')
    y = np.zeros(rows)
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            r, c = slice(i * br, (i + 1) * br), slice(j * bc, (j + 1) * bc)
            y[r] += (values[r, c] @ x[c]) * float(scales[i, j])
    return y


def quantize_weights(weights, fmt, unquantized=('embed_tokens', 'lm_head', '.mlp.gate.', 'norm',
                                                'e_score_correction_bias', 'weights_proj')):
    """Round-trip a weights dict through `fmt` ('fp8' or 'int4'); returns dequantized float32 tensors.

    Used to measure the model-level effect of quantization with the same exemptions as the
    official FP8 checkpoint. Matrices only; everything in `unquantized` is passed through.
    """
    from validation.write_checkpoint import fp8_dequantize_reference, fp8_quantize
    out = {}
    for name, w in weights.items():
        w = np.asarray(w, dtype=np.float32)
        if w.ndim != 2 or not name.endswith('.weight') or any(u in name for u in unquantized):
            out[name] = w
        elif fmt == 'fp8':
            out[name] = fp8_dequantize_reference(*fp8_quantize(w))
        elif fmt == 'int4':
            packed, scales = int4_quantize(w)
            out[name] = int4_dequantize(packed, scales, w.shape[1])
        else:
            raise ValueError(f'unknown format {fmt!r}')
    return out
