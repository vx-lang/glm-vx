"""Check glm_vx's compressed-tensors INT4 loader against the compressed-tensors library itself.

Needs torch and compressed-tensors (not project dependencies; use a scratch venv). For every
pack-quantized Linear in a (partial) checkpoint directory, compares:
  * codes: compressed_tensors.unpack_from_int32 vs int4_unpack of our int4.vx repack;
  * weights: PackedQuantizationCompressor.decompress (rounds code*scale to bf16) vs our exact
    float32 code*scale rounded to bf16 the same way: must be bitwise equal;
  * the size of that rounding: max |our float32 - their bf16| relative to max |weight|.

  python -m validation.compressed_int4_oracle CHECKPOINT_DIR [--output report.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization import QuantizationScheme

from glm_vx.checkpoint import SafetensorsCheckpoint
from glm_vx.quant import bf16_to_f32, compressed_int4_scheme, int4_unpack


def bf16_bits_round(values):
    """Round float32 to bfloat16 (nearest even) the way torch does, returning uint16 bits."""
    return torch.from_numpy(np.ascontiguousarray(values, np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.checkpoint)
    config = json.loads((root / 'config.json').read_text())['quantization_config']
    compressed_int4_scheme(config)
    group = next(iter(config['config_groups'].values()))
    scheme = QuantizationScheme(targets=group['targets'], weights=group['weights'])
    report = {'checkpoint': str(root), 'linears': {}}
    ok = True
    with SafetensorsCheckpoint(root) as ckpt:
        prefixes = sorted(n[:-len('.weight_packed')] for n in ckpt.names if n.endswith('.weight_packed'))
        for prefix in prefixes:
            packed, bits, scale_format, (n, k) = ckpt.compressed_int4(prefix)
            assert scale_format == 'bfloat16'
            words = torch.from_numpy(ckpt.raw(prefix + '.weight_packed'))
            state = {'weight_packed': words,
                     'weight_scale': torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16),
                     'weight_shape': torch.tensor([n, k])}
            theirs_codes = unpack_from_int32(words, 4, torch.Size([n, k])).numpy()
            theirs = PackedQuantizationCompressor.decompress(state, scheme)['weight']
            theirs_bits = theirs.view(torch.int16).numpy().view(np.uint16)
            codes = int4_unpack(packed, k)
            ours = codes.astype(np.float32) * np.repeat(bf16_to_f32(bits), 128, axis=1)[:, :k]
            ours_bits = bf16_bits_round(ours)
            rounding = np.abs(ours - bf16_to_f32(theirs_bits)).max() / max(np.abs(ours).max(), 1e-30)
            row = {'shape': [n, k], 'codes_equal': bool(np.array_equal(codes, theirs_codes)),
                   'bf16_weights_bitwise_equal': bool(np.array_equal(ours_bits, theirs_bits)),
                   'their_bf16_rounding_rel_to_max': float(rounding),
                   'codes_histogram': np.bincount(codes.reshape(-1) + 8, minlength=16).tolist()}
            ok &= row['codes_equal'] and row['bf16_weights_bitwise_equal']
            report['linears'][prefix] = row
            print(prefix, row['shape'], 'codes', row['codes_equal'], 'bf16', row['bf16_weights_bitwise_equal'],
                  f'rounding {rounding:.2e}')
    report['ok'] = bool(ok)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + '\n')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
