"""validation/real_layers.py without the network: what `fetch` keeps and the config it writes."""
import json
from pathlib import Path

import pytest

from glm_vx.config import GLMConfig
from validation import real_layers

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'metadata/config.json').read_text())
INDEX = json.loads((ROOT / 'metadata/model.safetensors.index.json').read_text())['weight_map']


@pytest.mark.parametrize('layers', [1, 3, 4, 7])
def test_the_official_index_has_every_tensor_of_the_cut_model(layers):
    config = real_layers.cut_config(CONFIG, layers)
    expected = GLMConfig.from_dict(config).tensor_shapes()
    names = {n for n in INDEX if real_layers.kept(n, layers)}
    assert set(expected) <= names
    # The rest are the FP8 scales of the kept Linears, nothing from a later layer or the MTP layer.
    extra = names - set(expected)
    assert all(n.endswith('.weight_scale_inv') and n.removesuffix('_scale_inv') in expected for n in extra)


def test_the_cut_config_keeps_the_real_layer_pattern_and_no_mtp():
    config = real_layers.cut_config(CONFIG, 4)
    assert config['num_hidden_layers'] == 4 and config['num_nextn_predict_layers'] == 0
    assert config['indexer_types'] == ['full', 'full', 'full', 'shared']
    assert config['mlp_layer_types'] == ['dense', 'dense', 'dense', 'sparse']
    skipped = config['quantization_config']['modules_to_not_convert']
    assert 'lm_head' in skipped and 'model.layers.3.mlp.gate' in skipped
    assert not any(m.startswith(('model.layers.4.', 'model.layers.78.')) for m in skipped)
    assert real_layers.official_config(config).keys().isdisjoint({'quantization_config', 'architectures', 'dtype'})


def test_kept_names():
    assert real_layers.kept('model.embed_tokens.weight', 1) and real_layers.kept('lm_head.weight', 1)
    assert real_layers.kept('model.layers.0.self_attn.o_proj.weight_scale_inv', 1)
    assert not real_layers.kept('model.layers.1.self_attn.o_proj.weight', 1)
    assert not real_layers.kept('model.layers.10.self_attn.o_proj.weight', 2)


def test_coalesce_joins_near_spans_in_offset_order():
    groups = real_layers.coalesce([(10, 20, 'a'), (0, 5, 'b'), (100, 200, 'c'), (25, 30, 'd')], gap=5)
    assert [(s, e, [n for *_, n in m]) for s, e, m in groups] == [(0, 30, ['b', 'a', 'd']), (100, 200, ['c'])]
    assert len(real_layers.coalesce([(0, 1, 'a'), (100, 101, 'b')], gap=0)) == 2


def test_torch_decodes_fp8_with_its_block_scale(tmp_path):
    torch = pytest.importorskip('torch')
    import numpy as np
    from glm_vx.checkpoint import Checkpoint, decode_e4m3fn
    codes = np.arange(130 * 3, dtype=np.uint8).reshape(130, 3)
    codes[codes == 0x7f] = 0  # no NaN code
    scales = np.array([[2.0], [0.5]], np.float32)
    header = {'w.weight': {'dtype': 'F8_E4M3', 'shape': [130, 3], 'data_offsets': [0, codes.nbytes]},
              'w.weight_scale_inv': {'dtype': 'F32', 'shape': [2, 1], 'data_offsets': [codes.nbytes, codes.nbytes + 8]}}
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    (tmp_path / 'model.safetensors').write_bytes(len(encoded).to_bytes(8, 'little') + encoded + codes.tobytes() + scales.tobytes())
    checkpoint = Checkpoint(tmp_path)
    got = real_layers.torch_weight(checkpoint, 'w.weight').numpy()
    want = decode_e4m3fn(codes) * np.repeat(scales, 128, 0)[:130]
    assert got.shape == (130, 3)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, checkpoint.tensor('w.weight'))
