import json
from pathlib import Path

import pytest

from glm_vx.admission import (GIB, Machine, analyze, kv_bytes_per_token, model_shapes, tensor_bytes,
                              weight_budget)
from glm_vx.tiny import tiny_config

ROOT = Path(__file__).resolve().parent.parent
REAL = json.loads((ROOT / 'model-config.json').read_text())


def test_fp8_total_matches_official_checkpoint_size():
    """Our exact-shape accounting vs. the published checkpoint (which also holds the MTP layer)."""
    official = json.loads((ROOT / 'metadata/model.safetensors.index.json').read_text())['metadata']['total_size']
    _, per_layer_experts, _, total = weight_budget(REAL, 'fp8')
    shapes = model_shapes(REAL)
    moe_layer_rest = sum(tensor_bytes(n, s, 'fp8') for n, s in shapes.items()
                         if n.startswith('model.layers.3.') and '.experts.' not in n)
    mtp = per_layer_experts + moe_layer_rest + 6144 * 12288 + 6144 * 2 * 3  # extra MoE layer + eh_proj + 3 norms
    assert abs(total + mtp - official) / official < 0.002


def test_kv_bytes_per_token_matches_mla_layout():
    # 78 layers x (512 latent + 64 rope) + 128 index dims on each 'full' indexer layer.
    full = sum(1 for t in REAL['indexer_types'] if t == 'full')
    assert kv_bytes_per_token(REAL, 'bf16') == (78 * 576 + full * 128) * 2
    assert kv_bytes_per_token(REAL, 'f32') == 2 * kv_bytes_per_token(REAL, 'bf16')
    assert kv_bytes_per_token(REAL, 'fp8') * 2 == kv_bytes_per_token(REAL, 'bf16')


def test_formats_scale_as_expected():
    bf16, fp8, int4 = (weight_budget(REAL, f)[3] for f in ('bf16', 'fp8', 'int4'))
    assert bf16 > fp8 > int4
    assert 1.9 < bf16 / fp8 < 2.0   # embeddings/norms stay bf16, so just under 2x
    assert 1.8 < fp8 / int4 < 2.0
    assert tensor_bytes('model.layers.0.mlp.gate.weight', (256, 6144), 'fp8') == 256 * 6144 * 2  # router stays bf16
    assert tensor_bytes('x.q_proj.weight', (256, 6144), 'fp8') == 256 * 6144 + 2 * 48 * 4  # 128x128 block scales


def test_tiny_config_bf16_bytes_equal_two_per_parameter():
    cfg = tiny_config()
    n = sum(__import__('math').prod(s) for s in model_shapes(cfg).values())
    core, per_layer, moe, total = weight_budget(cfg, 'bf16')
    assert total == pytest.approx(2 * n + 2 * sum(  # e_score_correction_bias is fp32: 2 more bytes each
        __import__('math').prod(s) for k, s in model_shapes(cfg).items() if k.endswith('e_score_correction_bias')))


def test_a100_80g_with_big_host_is_hybrid_with_a_binding_split():
    v = analyze(REAL, Machine(80, 2000, vcpus=128), 131072)
    assert v.verdict == 'hybrid'
    assert 0 < v.expert_layers_on_gpu < v.moe_layers
    assert v.expert_layers_on_gpu + v.expert_layers_on_host == v.moe_layers == 75
    assert v.gpu_used_gib <= 80 * 0.95 - 2 + 1e-6      # never exceeds the usable GPU budget
    assert v.max_context_all_on_gpu == 0                # 745 GB of weights cannot all be resident
    assert v.max_context > 131072


def test_no_fit_reasons_are_specific():
    small_host = analyze(REAL, Machine(80, 400), 8192)
    assert small_host.verdict == 'no_fit' and 'host RAM' in small_host.reason
    tiny_gpu = analyze(REAL, Machine(8, 2000), 8192)
    assert tiny_gpu.verdict == 'no_fit' and 'GPU cannot hold' in tiny_gpu.reason
    assert tiny_gpu.max_context == 0


def test_longer_context_never_adds_gpu_expert_layers():
    layers = [analyze(REAL, Machine(80, 2000), c).expert_layers_on_gpu for c in (1024, 65536, 262144, 524288)]
    assert layers == sorted(layers, reverse=True)
    assert layers[0] > layers[-1]


def test_context_beyond_max_context_is_rejected():
    v = analyze(REAL, Machine(80, 2000), 8192)
    over = analyze(REAL, Machine(80, 2000), v.max_context + 1)
    assert over.verdict == 'no_fit'
    at = analyze(REAL, Machine(80, 2000), v.max_context)
    assert at.verdict != 'no_fit' and at.expert_layers_on_gpu == 0


def test_everything_on_gpu_when_it_fits():
    v = analyze(tiny_config(), Machine(8, 1), 128, weight_format='bf16')  # host smaller than its reserve: unused, so irrelevant
    assert v.verdict == 'fits_gpu' and v.expert_layers_on_host == 0 and v.host_used_gib == 0


def test_int4_fits_more_experts_than_fp8():
    fp8 = analyze(REAL, Machine(80, 2000), 8192, 'fp8')
    int4 = analyze(REAL, Machine(80, 2000), 8192, 'int4')
    assert int4.expert_layers_on_gpu > fp8.expert_layers_on_gpu
    assert int4.host_used_gib < fp8.host_used_gib
