"""Write a Hugging Face-style checkpoint directory (config.json + safetensors shards +
index) in F32, BF16 or block-scaled FP8, from in-memory weights.

Test/validation utility: it lets the production loader (`glm_vx.checkpoint`) be
exercised end to end on the formats the real GLM-5.x checkpoint uses, and gives a
directory that Transformers' `from_pretrained` can read for `hf_reference.py`.
Tensor names follow the official checkpoint (per-expert `experts.E.*_proj.weight`,
FP8 scales in `<name>.weight_scale_inv`, F32, one scale per 128x128 block).

FP8 encoding: scale = block amax / 448 (E4M3FN max), codes = nearest representable
finite E4M3FN value of w/scale. Tensors the official config leaves unquantized
(embeddings, lm_head, routers, norms, biases, indexer weights_proj) are written BF16.
"""
import json
import math
import struct
from pathlib import Path

import numpy as np

from glm_vx.checkpoint import FP8_E4M3FN_TABLE

UNQUANTIZED = ('embed_tokens', 'lm_head', '.mlp.gate.', 'norm', 'e_score_correction_bias', 'weights_proj')
FP8_MAX = 448.0

_finite = np.isfinite(FP8_E4M3FN_TABLE)
_codes = np.nonzero(_finite)[0].astype(np.uint8)
_values = FP8_E4M3FN_TABLE[_codes].astype(np.float64)  # float64 distances: nearest-value ties as before
_order = np.argsort(_values, kind='stable')
_sorted_values, _sorted_codes = _values[_order], _codes[_order]


def to_bf16_bits(x):
    """float32 -> bfloat16 bit pattern (uint16), round to nearest even; NaN stays NaN."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, dtype=np.float32))
    rounded[nan] = 0x7FC0
    return rounded


def from_bf16_bits(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def fp8_quantize(w, block=(128, 128)):
    """(codes uint8 [r,c], scales f32 [ceil(r/br), ceil(c/bc)])."""
    w = np.asarray(w, dtype=np.float32)
    br, bc = block
    rows, cols = w.shape
    grid = (math.ceil(rows / br), math.ceil(cols / bc))
    scales = np.ones(grid, dtype=np.float32)
    codes = np.zeros(w.shape, dtype=np.uint8)
    for i in range(grid[0]):
        for j in range(grid[1]):
            tile = w[i * br:(i + 1) * br, j * bc:(j + 1) * bc]
            amax = float(np.abs(tile).max())
            scale = np.float32(amax / FP8_MAX) if amax > 0 else np.float32(1.0)
            scales[i, j] = scale
            normalized = np.clip(tile / scale, -FP8_MAX, FP8_MAX)
            pos = np.clip(np.searchsorted(_sorted_values, normalized), 1, len(_sorted_values) - 1)
            lo, hi = _sorted_values[pos - 1], _sorted_values[pos]
            pick = np.where(np.abs(normalized - lo) <= np.abs(hi - normalized), pos - 1, pos)
            codes[i * br:(i + 1) * br, j * bc:(j + 1) * bc] = _sorted_codes[pick]
    return codes, scales


def fp8_dequantize_reference(codes, scales, block=(128, 128)):
    """Independent (not the loader's) decode: table lookup times the block's scale."""
    out = FP8_E4M3FN_TABLE[codes].astype(np.float32)
    br, bc = block
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            out[i * br:(i + 1) * br, j * bc:(j + 1) * bc] *= scales[i, j]
    return out


def write_safetensors(path, entries):
    """entries: ordered {name: (dtype_str, shape, raw_bytes)}."""
    header, offset = {}, 0
    for name, (dtype, shape, raw) in entries.items():
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * ((-len(encoded)) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
        for _, _, raw in entries.values():
            f.write(raw)


def export_checkpoint(config, weights, outdir, fmt='bf16', shards=1, block=(128, 128)):
    """Write the directory; return {name: float32 array the loader must reproduce}."""
    if fmt not in ('f32', 'bf16', 'fp8'):
        raise ValueError(f'unknown format {fmt!r}')
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries, expected = {}, {}
    for name, w in weights.items():
        w = np.asarray(w, dtype=np.float32)
        quantize = fmt == 'fp8' and w.ndim == 2 and name.endswith('.weight') and not any(u in name for u in UNQUANTIZED)
        if quantize:
            codes, scales = fp8_quantize(w, block)
            entries[name] = ('F8_E4M3', w.shape, codes.tobytes())
            entries[name[:-len('weight')] + 'weight_scale_inv'] = ('F32', scales.shape, scales.astype('<f4').tobytes())
            expected[name] = fp8_dequantize_reference(codes, scales, block)
        elif fmt == 'f32':
            entries[name] = ('F32', w.shape, w.astype('<f4').tobytes())
            expected[name] = w
        else:
            bits = to_bf16_bits(w)
            entries[name] = ('BF16', w.shape, bits.astype('<u2').tobytes())
            expected[name] = from_bf16_bits(bits)
    names = list(entries)
    per = math.ceil(len(names) / shards)
    weight_map, total = {}, 0
    for s in range(shards):
        chunk = names[s * per:(s + 1) * per]
        if not chunk:
            continue
        fname = 'model.safetensors' if shards == 1 else f'model-{s + 1:05d}-of-{shards:05d}.safetensors'
        write_safetensors(outdir / fname, {n: entries[n] for n in chunk})
        for n in chunk:
            weight_map[n] = fname
            total += len(entries[n][2])
    if shards > 1:
        (outdir / 'model.safetensors.index.json').write_text(
            json.dumps({'metadata': {'total_size': total}, 'weight_map': weight_map}, indent=1))
    cfg = dict(config)
    if fmt == 'fp8':
        cfg['quantization_config'] = {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8',
                                      'weight_block_size': list(block)}
    (outdir / 'config.json').write_text(json.dumps(cfg, indent=1))
    return expected
