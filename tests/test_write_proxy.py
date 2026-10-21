"""validation/write_proxy.py: the single-GPU GLM-5.3 proxy has the official layout and glm-vx loads it."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from glm_vx.backend import NumpyBackend
from glm_vx.checkpoint import FP8_E4M3FN_TABLE, Checkpoint
from glm_vx.config import GLMConfig
from glm_vx.model import GlmMoeDsaModel, tiny_weights
from validation import write_proxy as wp
from validation.write_checkpoint import from_bf16_bits

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = json.loads((ROOT / 'metadata/model.safetensors.index.json').read_text())['weight_map']
SAMPLE = json.loads((ROOT / 'metadata/sample-shard-header.json').read_text())['header']
REAL = json.loads((ROOT / 'model-config.json').read_text())
TOKENS = [3, 17, 5, 299, 9]

# Real config, small widths (160 is not a multiple of 128: partial scale blocks), 7 layers so the
# real indexer pattern reaches its 4th full layer (6).
SMALL = dict(REAL, hidden_size=160, vocab_size=300, num_attention_heads=4, num_key_value_heads=4,
             q_lora_rank=96, kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=16, qk_head_dim=48,
             head_dim=16, v_head_dim=32, index_n_heads=4, index_head_dim=32, index_topk=4,
             intermediate_size=288, moe_intermediate_size=48, n_routed_experts=8, num_experts_per_tok=2,
             eos_token_id=[0], pad_token_id=0)


def layer_of(name):
    return int(name.split('.')[2]) if name.startswith('model.layers.') else None


def official_subset(layers):
    return {n for n in OFFICIAL if layer_of(n) is None or layer_of(n) < layers}


def test_default_proxy_has_the_official_names_dtypes_and_shapes():
    config = wp.proxy_config(9)
    specs = wp.tensor_specs(config)
    names = [s[0] for s in specs]
    assert len(names) == len(set(names)) and set(names) == official_subset(9)
    checked = 0
    for name, dtype, shape, _, _ in specs:
        if name in SAMPLE:
            assert (dtype, list(shape)) == (SAMPLE[name]['dtype'], SAMPLE[name]['shape']), name
            checked += 1
    assert checked > 30
    g = GLMConfig.from_dict(config)
    assert set(g.tensor_shapes()) == {s[0] for s in specs if s[3] != 'scale'}
    assert all(tuple(g.tensor_shapes()[s[0]]) == s[2] for s in specs if s[3] != 'scale')
    assert g.indexer_types == ('full',) * 3 + ('shared',) * 3 + ('full', 'shared', 'shared')
    assert g.mlp_layer_types == ('dense',) * 3 + ('sparse',) * 6
    assert config['num_nextn_predict_layers'] == 0 and config['index_topk'] == 2048
    # the docstring's byte arithmetic: weights fit the 72.7 GB static pool with ~8 GB for KV
    total = sum(wp.nbytes(s) for s in specs)
    assert 64.2e9 < total < 64.3e9
    assert sum(wp.nbytes(s) for s in wp.tensor_specs(wp.proxy_config(10))) > 0.85 * 81559 * 2**20


def test_fp8_and_bf16_follow_the_official_quantization():
    config = wp.proxy_config(9)
    skipped = config['quantization_config']['modules_to_not_convert']
    official = REAL['quantization_config']['modules_to_not_convert']
    assert set(skipped) == {m for m in official if layer_of(m) is None or layer_of(m) < 9}
    for name, dtype, shape, _, _ in wp.tensor_specs(config):
        module = name.rsplit('.', 1)[0]
        if name.endswith('e_score_correction_bias'):
            assert dtype == 'F32'
        elif name.endswith('weight_scale_inv'):
            assert dtype == 'F32'
        elif len(shape) == 1:
            assert dtype == 'BF16' and (module in skipped or module.removesuffix('.bias') in skipped), name
        elif dtype == 'BF16':
            # official spelling for the indexer head weights is `indexers_proj` (no such tensor)
            assert module in skipped or module.replace('indexer.weights_proj', 'indexers_proj') in skipped, name
        else:
            assert dtype == 'F8_E4M3' and module not in skipped, name


def test_mtp_layer_uses_the_official_layer_78_names():
    config = wp.proxy_config(9, mtp=True)
    names = {s[0] for s in wp.tensor_specs(config, mtp=True)}
    assert {n for n in names if layer_of(n) == 9} == {n.replace('layers.78.', 'layers.9.') for n in OFFICIAL if layer_of(n) == 78}
    assert config['num_nextn_predict_layers'] == 1 and len(config['indexer_types']) == 9
    assert 'model.layers.9.eh_proj' in config['quantization_config']['modules_to_not_convert']
    full = wp.proxy_config(78, mtp=True)
    assert full['quantization_config']['modules_to_not_convert'] == REAL['quantization_config']['modules_to_not_convert']
    assert {s[0] for s in wp.tensor_specs(full, mtp=True)} == set(OFFICIAL)


def test_fp8_table_is_finite_normal_and_hits_448():
    values = FP8_E4M3FN_TABLE[wp.FP8_LUT].astype(np.float64)
    assert np.isfinite(values).all() and np.all((wp.FP8_LUT & 0x7F) != 0x7F)
    assert values.max() == 448 and values.min() == -448 and abs(values.mean()) < 1e-9
    pair = wp.FP8_LUT16.view(np.uint8).reshape(-1, 2)
    i = np.arange(65536)
    assert np.array_equal(pair[:, 0], wp.FP8_LUT[i & 255]) and np.array_equal(pair[:, 1], wp.FP8_LUT[i >> 8])


def reference_weights(config, seed):
    """Decode the generator's bytes independently of glm_vx.checkpoint (repeat-expanded scales)."""
    specs = wp.tensor_specs(config)
    data = {s[0]: b''.join(wp.tensor_chunk(s, seed, k) for k in range(wp.chunk_count(s))) for s in specs}
    out = {}
    for name, dtype, shape, init, _ in specs:
        if init == 'scale':
            continue
        if dtype == 'F8_E4M3':
            scale = np.frombuffer(data[name + '_scale_inv'], '<f4').reshape(-(-shape[0] // 128), -(-shape[1] // 128))
            full = np.repeat(np.repeat(scale, 128, 0), 128, 1)[:shape[0], :shape[1]]
            out[name] = FP8_E4M3FN_TABLE[np.frombuffer(data[name], np.uint8).reshape(shape)] * full
        elif dtype == 'BF16':
            out[name] = from_bf16_bits(np.frombuffer(data[name], '<u2')).reshape(shape)
        else:
            out[name] = np.frombuffer(data[name], '<f4').reshape(shape).copy()
    return out


@pytest.fixture(scope='module')
def small(tmp_path_factory):
    path = tmp_path_factory.mktemp('proxy') / 'a'
    return path, wp.write_proxy(path, layers=7, seed=3, base=SMALL, shard_bytes=200_000)


def test_small_proxy_loads_with_the_official_layout(small):
    path, manifest = small
    config = json.loads((path / 'config.json').read_text())
    assert manifest['config'] == config and (manifest['dense_layers'], manifest['moe_layers']) == (3, 4)
    g = GLMConfig.from_dict(config)
    specs = wp.tensor_specs(config)
    shapes, _, _ = tiny_weights(config, parameter_limit=10**9, shapes_only=True)  # glm-vx's own names
    assert set(shapes) == {s[0] for s in specs if s[3] != 'scale'}
    with Checkpoint(path, block_size=g.weight_block_size) as ck:
        ck.validate_manifest(g.tensor_shapes(), inspect_shards=True)
        assert len(set(ck.weight_map.values())) > 3 and set(ck.names) == {s[0] for s in specs}
        for name, dtype, shape, init, fan_in in specs:
            info = ck.info(name)
            assert (info.dtype, info.shape) == (dtype, shape), name
            if dtype == 'F8_E4M3':
                raw = ck.raw(name)
                assert np.all((raw & 0x7F) != 0x7F), name  # no NaN code
                assert ck.info(name + '_scale_inv').shape == (-(-shape[0] // 128), -(-shape[1] // 128))
                assert ck.is_block_fp8(name.removesuffix('.weight'))
        w = ck.tensor('model.layers.0.mlp.down_proj.weight')
        assert abs(w.std() * np.sqrt(w.shape[1]) - 1) < .1
        assert ck.is_bf16('model.layers.3.mlp.gate') and ck.is_bf16('model.layers.0.self_attn.indexer.weights_proj')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    assert index['metadata']['total_size'] == manifest['tensor_bytes'] == sum(wp.nbytes(s) for s in specs)
    for name, entry in manifest['files'].items():
        data = (path / name).read_bytes()
        assert entry == {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}, name
    assert json.loads((path / 'proxy.json').read_text()) == manifest
    assert manifest['tensors'] == [[s[0], s[1], list(s[2])] for s in specs] and manifest['format'] == 'fp8'


def test_bf16_proxy_is_the_fp8_proxy_dequantized(small, tmp_path):
    path, fp8 = small
    manifest = wp.write_proxy(tmp_path / 'bf16', layers=7, seed=3, base=SMALL, fmt='bf16')
    config = json.loads((tmp_path / 'bf16/config.json').read_text())
    assert 'quantization_config' not in config and GLMConfig.from_dict(config).quant_method is None
    assert {n for n, _, _ in manifest['tensors']} == {n for n, _, _ in fp8['tensors'] if not n.endswith('_scale_inv')}
    assert {d for _, d, _ in manifest['tensors']} == {'BF16', 'F32'}
    with Checkpoint(path) as a, Checkpoint(tmp_path / 'bf16') as b:
        for name in ('model.layers.0.self_attn.o_proj.weight', 'model.layers.4.mlp.experts.5.down_proj.weight'):
            assert b.is_bf16(name.removesuffix('.weight'))
            np.testing.assert_array_equal(b.tensor(name), from_bf16_bits(wp._bf16_bits(a.tensor(name))))
        want = GlmMoeDsaModel(json.loads((path / 'config.json').read_text()), a, NumpyBackend()).prefill(TOKENS)
        got = GlmMoeDsaModel(config, b, NumpyBackend()).prefill(TOKENS)
    np.testing.assert_allclose(got, want, atol=.05 * np.abs(want).max())


def test_tokenizer_files_are_copied_and_hashed(tmp_path):
    snapshot = tmp_path / 'snap'
    snapshot.mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json',
                 'config.json', 'model-00001-of-00002.safetensors', 'README.md'):
        (snapshot / name).write_text(name)
    manifest = wp.write_proxy(tmp_path / 'p', layers=1, base=SMALL, tokenizer=snapshot)
    copied = {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'}
    assert copied <= set(manifest['files']) and 'README.md' not in manifest['files']
    assert (tmp_path / 'p/tokenizer.json').read_text() == 'tokenizer.json'
    assert json.loads((tmp_path / 'p/config.json').read_text())['num_hidden_layers'] == 1  # not the snapshot's
    (snapshot / 'tokenizer.json').unlink()
    with pytest.raises(FileNotFoundError):
        wp.write_proxy(tmp_path / 'q', layers=1, base=SMALL, tokenizer=snapshot)
    assert not (tmp_path / 'q').exists()


def test_numpy_model_runs_on_the_proxy_and_matches_an_independent_decode(small):
    path, _ = small
    config = json.loads((path / 'config.json').read_text())
    with Checkpoint(path) as ck:
        got = GlmMoeDsaModel(config, ck, NumpyBackend()).prefill(TOKENS)
    want = GlmMoeDsaModel(config, reference_weights(config, 3), NumpyBackend()).prefill(TOKENS)
    assert np.isfinite(got).all() and got.std() > .1
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_output_depends_only_on_the_seed(small, tmp_path, monkeypatch):
    _, manifest = small
    again = wp.write_proxy(tmp_path / 'b', layers=7, seed=3, base=SMALL, shard_bytes=200_000, workers=3)
    assert again == manifest
    other = wp.write_proxy(tmp_path / 'c', layers=7, seed=4, base=SMALL, shard_bytes=200_000)
    shards = [n for n in manifest['files'] if n.endswith('.safetensors')]
    assert all(other['files'][n] != manifest['files'][n] for n in shards)
    # many chunks per tensor: the bounded window writes them in order whatever the thread count
    monkeypatch.setattr(wp, 'CHUNK', 1000)
    one = wp.write_proxy(tmp_path / 'd', layers=4, seed=3, base=SMALL, workers=1)
    four = wp.write_proxy(tmp_path / 'e', layers=4, seed=3, base=SMALL, workers=4)
    assert one == four
    wp.write_proxy(tmp_path / 'f', layers=4, seed=3, base=SMALL, workers=3, fmt='bf16')  # chunks start mid-row
    with Checkpoint(tmp_path / 'd') as a, Checkpoint(tmp_path / 'f') as b:
        for name in ('model.layers.0.self_attn.o_proj.weight', 'model.layers.0.mlp.down_proj.weight'):
            np.testing.assert_array_equal(b.tensor(name), from_bf16_bits(wp._bf16_bits(a.tensor(name))))
    with pytest.raises(FileExistsError):
        wp.write_proxy(tmp_path / 'e', layers=4, seed=3, base=SMALL)
