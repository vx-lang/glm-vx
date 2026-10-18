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


def bf16_to_f32(bits):
    """float32 values of raw bfloat16 bits (uint16); exact for every encoding."""
    bits = np.asarray(bits)
    if bits.dtype != np.uint16:
        raise ValueError('bfloat16 bits must be uint16')
    return (bits.astype(np.uint32) << 16).view(np.float32)


# compressed-tensors "pack-quantized" INT4 (llm-compressor W4A16; the format of published GLM INT4
# checkpoints). Per Linear: weight_packed int32 [N, ceil(K/8)], weight_scale [N, ceil(K/G)] in the
# model dtype (bfloat16 for GLM), weight_shape int64 [2] = (N, K). Codes are symmetric in [-8, 7],
# stored as code + 8; element k of a row sits at bits 4*(k % 8) of word k // 8 (dense little-endian
# packing), rows padded to a multiple of 32 elements with stored value 0 before truncation.
# Dequantization is code * scale. The reference decompressor rounds that product to the scale dtype;
# these oracles and the kernels keep the exact float32 product.

def compressed_int4_unpack(weight_packed, cols):
    """Signed codes int8 [rows, cols] from compressed-tensors int32 words (bit shifts, no byte view)."""
    words = np.asarray(weight_packed)
    if words.dtype != np.int32 or words.ndim != 2 or words.shape[1] != math.ceil(cols / 8):
        raise ValueError('weight_packed must be int32 [rows, ceil(cols/8)]')
    u = words.astype(np.int64) & 0xFFFFFFFF
    nibbles = np.stack([(u >> (4 * j)) & 0xF for j in range(8)], axis=2).reshape(words.shape[0], -1)
    return (nibbles[:, :cols] - 8).astype(np.int8)


def compressed_int4_pack(codes):
    """compressed-tensors int32 words from signed codes int8 [rows, cols] (the encoder, for tests)."""
    codes = np.asarray(codes)
    if codes.dtype != np.int8 or codes.ndim != 2 or codes.min(initial=0) < INT4_MIN or codes.max(initial=0) > INT4_MAX:
        raise ValueError('codes must be int8 [rows, cols] in [-8, 7]')
    rows, cols = codes.shape
    padded = math.ceil(cols / 32) * 32
    stored = np.zeros((rows, padded), np.uint32)
    stored[:, :cols] = (codes.astype(np.int32) + 8).astype(np.uint32)
    words = np.zeros((rows, padded // 8), np.uint32)
    for j in range(8):
        words |= stored[:, j::8] << np.uint32(4 * j)
    return words[:, :math.ceil(cols / 8)].view(np.int32)


def compressed_int4_to_packed(weight_packed, cols):
    """Repack compressed-tensors words into the int4.vx layout, uint8 [rows, ceil(cols/2)].

    Same nibble order byte for byte (little-endian words); only the code encoding differs:
    offset-8 storage is two's complement with the top nibble bit flipped, so XOR 0x88. A pad nibble
    past `cols` is cleared to the int4.vx convention (zero).
    """
    words = np.ascontiguousarray(weight_packed)
    if words.dtype != np.int32 or words.ndim != 2 or words.shape[1] != math.ceil(cols / 8):
        raise ValueError('weight_packed must be int32 [rows, ceil(cols/8)]')
    raw = words.astype('<i4', copy=False).view(np.uint8)[:, :(cols + 1) // 2] ^ np.uint8(0x88)
    if cols % 2:
        raw[:, -1] &= np.uint8(0x0F)
    return np.ascontiguousarray(raw)


def compressed_int4_dequantize(packed, scale_bits, scale_format, cols, group=128):
    """Exact float32 weights from the int4.vx layout: code * scale (a 4-bit code times a bf16 or
    fp16 scale is representable in float32, so nothing rounds)."""
    bits = np.asarray(scale_bits, dtype=np.uint16)
    scales = bf16_to_f32(bits) if scale_format == 'bfloat16' else bits.view(np.float16).astype(np.float32)
    return int4_unpack(packed, cols).astype(np.float32) * np.repeat(scales, group, axis=1)[:, :cols]


def compressed_int4_encode(w, group=128):
    """A test encoder: (weight_packed int32, bf16 scale bits) for float32 w [N, K], scale =
    bf16(amax / 7) per group (1 for an all-zero group). Not claimed to reproduce llm-compressor's
    observer; real checkpoints are read, never re-encoded."""
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    groups = math.ceil(cols / group)
    bits = np.zeros((rows, groups), np.uint16)
    codes = np.zeros((rows, cols), np.int8)
    for g in range(groups):
        tile = w[:, g * group:(g + 1) * group]
        amax = np.abs(tile).max(axis=1)
        scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
        bits[:, g] = ((scale.view(np.uint32) + 0x7FFF + ((scale.view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
        s = bf16_to_f32(bits[:, g])[:, None]
        codes[:, g * group:(g + 1) * group] = np.clip(np.rint(tile / s), INT4_MIN, INT4_MAX).astype(np.int8)
    return compressed_int4_pack(codes), bits


class CompressedInt4Weights:
    """Weight source with some Linears in compressed-tensors INT4, the in-memory twin of an INT4
    SafetensorsCheckpoint. ``linears`` maps a Linear prefix to (weight_packed int32 [N, ceil(K/8)],
    scale bits uint16 [N, ceil(K/128)], (N, K)) with bf16 scales; ``base`` (a dict or a callable)
    supplies every other tensor. ``prefix.weight`` reads back as the exact float32 dequantization,
    which is what the NumPy reference model computes with."""

    def __init__(self, base, linears):
        self.base, self.linears = base, dict(linears)
        self._dense = {}

    def is_compressed_int4(self, prefix):
        return prefix in self.linears

    def compressed_int4(self, prefix):
        words, bits, (n, k) = self.linears[prefix]
        return compressed_int4_to_packed(words, k), np.asarray(bits, np.uint16), 'bfloat16', (n, k)

    def __call__(self, name):
        prefix = name[:-len('.weight')] if name.endswith('.weight') else None
        if prefix in self.linears:
            if name not in self._dense:
                packed, bits, fmt, (_, k) = self.compressed_int4(prefix)
                dense = compressed_int4_dequantize(packed, bits, fmt, k)
                dense.flags.writeable = False
                self._dense[name] = dense
            return self._dense[name]
        return self.base(name) if callable(self.base) else self.base[name]

    __getitem__ = __call__


def compressed_int4_scheme(quantization_config):
    """Validate a config.json ``quantization_config`` as the one INT4 scheme the kernels implement.

    Accepts compressed-tensors, format pack-quantized, every config group INT4 symmetric
    group-128 weight-only (no activation quantization). Returns (targets, ignore) so the caller
    can tell quantized Linears from the ones left in the model dtype. Raises ValueError otherwise.
    """
    q = quantization_config
    if not isinstance(q, dict) or q.get('quant_method') != 'compressed-tensors':
        raise ValueError('quant_method must be compressed-tensors')
    if q.get('format') != 'pack-quantized' or q.get('quantization_status', 'compressed') != 'compressed':
        raise ValueError('only compressed pack-quantized checkpoints are supported')
    if q.get('kv_cache_scheme') is not None:
        raise ValueError('quantized KV cache is not supported')
    groups = q.get('config_groups')
    if not isinstance(groups, dict) or not groups:
        raise ValueError('config_groups missing')
    targets = []
    for name, group in groups.items():
        w = group.get('weights') or {}
        expected = {'num_bits': 4, 'type': 'int', 'symmetric': True, 'strategy': 'group', 'group_size': 128, 'dynamic': False}
        bad = {key: w.get(key) for key, value in expected.items() if w.get(key) != value}
        if bad or group.get('format', 'pack-quantized') != 'pack-quantized':
            raise ValueError(f'config group {name}: unsupported weight scheme {bad or group.get("format")}')
        if group.get('input_activations') is not None or group.get('output_activations') is not None:
            raise ValueError(f'config group {name}: activation quantization is not supported')
        targets += list(group.get('targets') or [])
    return tuple(targets), tuple(q.get('ignore') or ())


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
