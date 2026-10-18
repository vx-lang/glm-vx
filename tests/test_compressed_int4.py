"""compressed-tensors pack-quantized INT4 (published GLM W4A16 checkpoints): packing, repack to
the int4.vx layout, config validation and the safetensors loader. No GPU, no torch."""
import copy
import json

import numpy as np
import pytest

from glm_vx.checkpoint import CheckpointError, SafetensorsCheckpoint
from glm_vx.quant import (bf16_to_f32, compressed_int4_pack, compressed_int4_scheme, compressed_int4_to_packed,
                          compressed_int4_unpack, int4_unpack)
from validation.write_checkpoint import write_safetensors

# compressed_tensors 0.19.0 pack_to_int32(codes, 4) for these rows (40 columns: one full 32-element
# block plus a padded partial one).
LIBRARY_CODES = [[(i * 5) % 16 - 8 for i in range(40)], [7 - (i % 16) for i in range(40)]]
LIBRARY_WORDS = [[1049950800, -1239649576, 1049950800, -1239649576, 1049950800],
                 [-1985229329, 19088743, -1985229329, 19088743, -1985229329]]

# The weights group of the lowbitcoffee/GLM-5.2-W4A16 config (glm_moe_dsa), ignore list shortened.
GLM_CONFIG = {
    'quant_method': 'compressed-tensors', 'format': 'pack-quantized', 'quantization_status': 'compressed',
    'kv_cache_scheme': None, 'ignore': ['model.layers.0.self_attn.indexer.wq_b', 'lm_head'],
    'config_groups': {'group_0': {
        'format': 'pack-quantized', 'input_activations': None, 'output_activations': None, 'targets': ['Linear'],
        'weights': {'actorder': 'static', 'block_structure': None, 'dynamic': False, 'group_size': 128,
                    'num_bits': 4, 'observer': 'memoryless_minmax', 'observer_kwargs': {}, 'scale_dtype': None,
                    'strategy': 'group', 'symmetric': True, 'type': 'int', 'zp_dtype': None}}}}


def test_packing_matches_the_library_vector():
    codes = np.array(LIBRARY_CODES, np.int8)
    words = np.array(LIBRARY_WORDS, np.int32)
    assert np.array_equal(compressed_int4_pack(codes), words)
    assert np.array_equal(compressed_int4_unpack(words, 40), codes)


def test_offset_encoding_and_nibble_order():
    # stored = code + 8 at bits 4*j: codes -8..-1 are 0x76543210, 0..7 are 0xFEDCBA98.
    words = compressed_int4_pack(np.arange(-8, 8, dtype=np.int8)[None])
    assert words.view(np.uint32).tolist() == [[0x76543210, 0xFEDCBA98]]


@pytest.mark.parametrize('cols', [1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 300])
def test_repack_is_the_int4_vx_layout(cols):
    rng = np.random.default_rng(cols)
    codes = rng.integers(-8, 8, (3, cols)).astype(np.int8)
    packed = compressed_int4_to_packed(compressed_int4_pack(codes), cols)
    assert packed.dtype == np.uint8 and packed.shape == (3, (cols + 1) // 2)
    assert np.array_equal(int4_unpack(packed, cols), codes)
    nibbles = (codes & 0xF).astype(np.uint8)
    if cols % 2:
        nibbles = np.concatenate([nibbles, np.zeros((3, 1), np.uint8)], axis=1)  # zero pad nibble
    assert np.array_equal(packed, nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))


def test_pack_and_repack_reject_bad_input():
    with pytest.raises(ValueError):
        compressed_int4_pack(np.array([[8]], np.int8))
    with pytest.raises(ValueError):
        compressed_int4_to_packed(np.zeros((2, 2), np.int32), 17)  # needs ceil(17/8) = 3 words
    with pytest.raises(ValueError):
        compressed_int4_unpack(np.zeros((2, 2), np.uint32), 16)


def test_scheme_accepts_the_glm_config():
    targets, ignore = compressed_int4_scheme(GLM_CONFIG)
    assert targets == ('Linear',) and 'lm_head' in ignore


@pytest.mark.parametrize('path,value', [
    (('quant_method',), 'awq'), (('format',), 'int-quantized'), (('kv_cache_scheme',), {'num_bits': 8}),
    (('config_groups', 'group_0', 'weights', 'symmetric'), False),
    (('config_groups', 'group_0', 'weights', 'group_size'), 64),
    (('config_groups', 'group_0', 'weights', 'num_bits'), 8),
    (('config_groups', 'group_0', 'weights', 'strategy'), 'channel'),
    (('config_groups', 'group_0', 'weights', 'dynamic'), True),
    (('config_groups', 'group_0', 'input_activations'), {'num_bits': 8}),
])
def test_scheme_rejects_what_the_kernels_do_not_implement(path, value):
    config = copy.deepcopy(GLM_CONFIG)
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        compressed_int4_scheme(config)


def bf16_bits(values):
    return (np.asarray(values, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def linear_entries(prefix, n, k, seed, scale_dtype='BF16'):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-8, 8, (n, k)).astype(np.int8)
    if scale_dtype == 'BF16':
        scales = bf16_bits(rng.uniform(1e-3, 0.1, (n, (k + 127) // 128)))
    else:
        scales = rng.uniform(1e-3, 0.1, (n, (k + 127) // 128)).astype(np.float16).view(np.uint16)
    words = compressed_int4_pack(codes)
    entries = {f'{prefix}.weight_packed': ('I32', words.shape, words.astype('<i4').tobytes()),
               f'{prefix}.weight_scale': (scale_dtype, scales.shape, scales.astype('<u2').tobytes()),
               f'{prefix}.weight_shape': ('I64', (2,), np.array([n, k], '<i8').tobytes())}
    return entries, codes, scales


def write(tmp_path, entries):
    write_safetensors(tmp_path / 'model.safetensors', entries)
    return SafetensorsCheckpoint(tmp_path)


@pytest.mark.parametrize('scale_dtype,fmt', [('BF16', 'bfloat16'), ('F16', 'float16')])
def test_loader_returns_the_int4_vx_layout(tmp_path, scale_dtype, fmt):
    entries, codes, scales = linear_entries('model.layers.3.mlp.experts.7.down_proj', 6, 257, 1, scale_dtype)
    with write(tmp_path, entries) as ckpt:
        packed, bits, scale_format, shape = ckpt.compressed_int4('model.layers.3.mlp.experts.7.down_proj')
        assert shape == (6, 257) and scale_format == fmt
        assert np.array_equal(int4_unpack(packed, 257), codes)
        assert bits.dtype == np.uint16 and np.array_equal(bits, scales)
        assert np.array_equal(ckpt.raw('model.layers.3.mlp.experts.7.down_proj.weight_packed', rows=slice(2, 4)),
                              compressed_int4_pack(codes)[2:4])
        with pytest.raises(CheckpointError, match='raw'):
            ckpt.tensor('model.layers.3.mlp.experts.7.down_proj.weight_packed')
        if scale_dtype == 'BF16':
            assert np.array_equal(ckpt.tensor('model.layers.3.mlp.experts.7.down_proj.weight_scale'), bf16_to_f32(scales))


def test_loader_rejects_inconsistent_or_unsupported_tensors(tmp_path):
    prefix = 'model.layers.0.self_attn.o_proj'
    base, _, _ = linear_entries(prefix, 4, 130, 2)

    def case(name, mutate):
        entries = dict(base)
        mutate(entries)
        directory = tmp_path / name
        directory.mkdir()
        with write(directory, entries) as ckpt, pytest.raises(CheckpointError):
            ckpt.compressed_int4(prefix)

    case('shape', lambda e: e.__setitem__(f'{prefix}.weight_shape', ('I64', (2,), np.array([4, 140], '<i8').tobytes())))
    case('scale_shape', lambda e: e.__setitem__(f'{prefix}.weight_scale', ('BF16', (4, 1), b'\0' * 8)))
    case('scale_dtype', lambda e: e.__setitem__(f'{prefix}.weight_scale', ('F32', (4, 2), b'\0' * 32)))
    nan = bf16_bits(np.full((4, 2), np.nan))
    case('nan_scale', lambda e: e.__setitem__(f'{prefix}.weight_scale', ('BF16', (4, 2), nan.tobytes())))
    case('zero_point', lambda e: e.__setitem__(f'{prefix}.weight_zero_point', ('I32', (1, 2), b'\0' * 8)))
    case('g_idx', lambda e: e.__setitem__(f'{prefix}.weight_g_idx', ('I32', (130,), b'\0' * 520)))
    case('dense_too', lambda e: e.__setitem__(f'{prefix}.weight', ('BF16', (4, 130), b'\0' * 1040)))
    case('missing_scale', lambda e: e.pop(f'{prefix}.weight_scale'))


def test_glm_config_file_round_trip(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'quantization_config': GLM_CONFIG}))
    assert compressed_int4_scheme(json.loads((tmp_path / 'config.json').read_text())['quantization_config'])
