"""ResidentGLM graph=True on a real GPU (#25, #26): the decode step captured once and replayed every
token, against the per-launch path, bit for bit, also with fused_attention (mla_decode_at, #27).
The long runs select up to 700 tokens, so the replayed graph runs 1, 2 and 3 MLA splits from the
step buffer. Needs a GPU, gpu/build (gpu/build.sh) and the gpu/spawn kernels (gpu/spawn/build.sh)."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import SPAWN, TOKENS, build, expert_models


@pytest.fixture(scope='module')
def ctx():
    from gpu.runtime import CUDAContext, CUDAUnavailable
    if not (SPAWN / 'moe_gate_up.ptx').exists():
        pytest.skip('gpu/spawn expert kernels not built (gpu/spawn/build.sh)')
    try:
        context = CUDAContext(minimum_compute_capability=(8, 0))
    except CUDAUnavailable as exc:
        pytest.skip('real GPU unavailable: ' + str(exc))
    yield context
    context.close()


def both(ctx, model, tokens, max_tokens, **kwargs):
    graph = ResidentGLM(model, ctx=ctx, max_tokens=max_tokens, device_experts=True, graph=True, **kwargs)
    plain = ResidentGLM(model, ctx=ctx, max_tokens=max_tokens, device_experts=True, **kwargs)
    return graph.prefill(tokens), plain.prefill(tokens), graph


@pytest.mark.parametrize('kind', ['float32', 'fused', 'int4', 'fp8', 'bf16'])
def test_graph_replay_equals_the_per_launch_path_on_the_gpu(ctx, kind):
    got, want, graph = both(ctx, expert_models(kind), TOKENS, 32)
    np.testing.assert_array_equal(got, want)
    assert graph.replays == len(TOKENS) - 2  # a short and a long step graph (#43)


def test_graph_replay_through_several_mla_splits(ctx):
    steps = 700
    model = build(seed=79, index_topk=1024, max_position_embeddings=1024)
    tokens = np.random.default_rng(79).integers(0, model.config['vocab_size'], size=steps).tolist()
    got, want, graph = both(ctx, model, tokens, steps + 8)
    np.testing.assert_array_equal(got, want)
    assert graph.replays == steps - 1


@pytest.mark.skipif(not (SPAWN / 'mla_decode_at.ptx').exists(), reason='gpu/spawn mla_decode_at not built')
def test_graph_replay_with_fused_attention_through_several_splits(ctx):
    """mla_decode_at in the graph (#27) against mla_decode per step, 700 steps through 1 to 3 splits
    of splits_max 16, and the arrival counts back at zero after the last replay."""
    steps = 700
    model = build(seed=89, index_topk=1024, max_position_embeddings=1024)
    tokens = np.random.default_rng(89).integers(0, model.config['vocab_size'], size=steps).tolist()
    got, want, graph = both(ctx, model, tokens, steps + 8, fused_attention=True)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(np.frombuffer(graph.arrivals.download(), np.int32), 0)
    assert graph.replays == steps - 1



@pytest.mark.skipif(not (SPAWN / 'mla_flash_at.ptx').exists(), reason='gpu/spawn mla_flash_at not built')
def test_mla_flash_matches_mla_chunks_through_its_splits(ctx):
    """mla_kernel='flash' (#33) against 'chunks' under replay: the same scores, the softmax folded a
    token at a time, so close with the same argmax. GLM-5.3's MLA shape (16 heads, rank 512, rope 64)
    on a small model; 600 steps select up to 600 tokens, so the steps run 1 to 64 splits of 8."""
    steps = 600
    model = build(seed=97, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, qk_rope_head_dim=64,
                  qk_head_dim=72, index_head_dim=64, index_topk=1024, max_position_embeddings=1024)
    tokens = np.random.default_rng(97).integers(0, model.config['vocab_size'], size=steps).tolist()
    out = {}
    for kernel in ('flash', 'chunks'):
        resident = ResidentGLM(model, ctx=ctx, max_tokens=steps + 8, device_experts=True, graph=True,
                               fused_attention=True, mla_kernel=kernel)
        try:
            out[kernel] = resident.prefill(tokens)
            assert resident.replays == steps - 1
            if kernel == 'flash':
                assert resident.launches.get('mla_flash_merge_at', 0) > 0 and 'mla_chunks_at' not in resident.launches
        finally:
            resident.close()
    np.testing.assert_allclose(out['flash'], out['chunks'], rtol=1e-4, atol=1e-4)
    assert (out['flash'].argmax(-1) == out['chunks'].argmax(-1)).all()


@pytest.mark.skipif(not (SPAWN / 'mla_tc_at.ptx').exists(), reason='gpu/spawn mla_tc_at not built')
def test_mla_tc_matches_mla_flash_to_bf16_rounding(ctx):
    """mla_kernel='tc' (#33): bf16 caches and queries, the products on the tensor cores, against 'flash'
    (f32) under replay; 600 steps run 1 to 19 splits of 32. The logits agree to bf16 rounding."""
    steps = 600
    model = build(seed=101, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, qk_rope_head_dim=64,
                  qk_head_dim=72, index_head_dim=64, index_topk=1024, max_position_embeddings=1024)
    tokens = np.random.default_rng(101).integers(0, model.config['vocab_size'], size=steps).tolist()
    out = {}
    for kernel in ('tc', 'flash'):
        resident = ResidentGLM(model, ctx=ctx, max_tokens=steps + 8, device_experts=True, graph=True,
                               fused_attention=True, mla_kernel=kernel)
        try:
            out[kernel] = resident.prefill(tokens)
            assert resident.replays == steps - 1
            if kernel == 'tc':
                # fuse_glue (the default) writes both bf16 rows with kv_rows_bf16_at, not store_row_bf16_at.
                assert resident.launches.get('mla_tc_at', 0) > 0 and resident.launches.get('kv_rows_bf16_at', 0) > 0
        finally:
            resident.close()
    scale = np.abs(out['flash']).max()
    np.testing.assert_allclose(out['tc'], out['flash'], rtol=0, atol=2e-2 * scale)
    assert (out['tc'].argmax(-1) == out['flash'].argmax(-1)).mean() > 0.95


@pytest.mark.skipif(not (SPAWN / 'kv_rows_bf16_at.ptx').exists(), reason='gpu/spawn kv_rows_bf16_at not built')
@pytest.mark.parametrize('kind', ['float32', 'fp8'])
def test_fused_glue_gives_the_same_bits_on_the_gpu(ctx, kind):
    """fuse_glue (#39) under replay with mla_kernel='tc': residual_rmsnorm8, rmsnorm_pair, index_key_at
    and kv_rows_bf16_at give the bits of the launches they replace, past index_topk too (fp8: the cvt
    kernels' stacked q_a, kv_a and wk)."""
    steps = 40
    if kind == 'fp8':
        from gpu.test_resident import fp8_model
        model = fp8_model(103, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, qk_rope_head_dim=64,
                          qk_head_dim=72, index_head_dim=64, index_topk=16, hidden_size=144)[0]
        options = dict(fp8_kernels='cvt')
    else:
        model = build(seed=103, num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, qk_rope_head_dim=64,
                      qk_head_dim=72, index_head_dim=64, index_topk=16)
        options = {}
    tokens = np.random.default_rng(103).integers(0, model.config['vocab_size'], size=steps).tolist()
    out, launches = {}, {}
    for fuse in (False, True):
        resident = ResidentGLM(model, ctx=ctx, max_tokens=steps + 8, device_experts=True, graph=True, fused_attention=True,
                               mla_kernel='tc', norm_block=1024, fuse_glue=fuse, **options)
        try:
            out[fuse], launches[fuse] = resident.prefill(tokens), dict(resident.launches)
        finally:
            resident.close()
    np.testing.assert_array_equal(out[True], out[False])
    assert launches[True]['kv_rows_bf16_at'] > 0 and 'store_row_bf16_at' not in launches[True]
    assert launches[True]['index_key_at'] > 0 and 'layernorm' not in launches[True]
    assert launches[True]['rmsnorm_pair'] > 0 and 'residual_rmsnorm' not in launches[True]


@pytest.mark.skipif(not (SPAWN / 'index_scores_tc_at.ptx').exists(), reason='gpu/spawn index_scores_tc_at not built')
@pytest.mark.parametrize('kernel,name', [('tc', 'index_scores_tc_at'), ('mma', 'index_mma/index_scores_mma_at'),
                                         ('fp8', 'index_mma/index_scores_fp8_at')])
def test_index_kernel_tc_matches_simt_to_bf16_rounding(ctx, kernel, name):
    """index_kernel='tc', 'mma' (bf16) and 'fp8' (#32): the indexer's q and key rows rounded, the scores on the
    tensor cores, against 'simt' (f32) under replay at GLM-5.3's indexer shape (32 heads, dim 128); 600 steps, index_topk 128, so
    472 of them select by score. A score within rounding of the cut can change one selected token,
    so the logits agree closely, not bit for bit, with the same argmax almost everywhere."""
    steps = 600
    model = build(seed=107, index_n_heads=32, index_head_dim=128, index_topk=128, max_position_embeddings=1024)
    tokens = np.random.default_rng(107).integers(0, model.config['vocab_size'], size=steps).tolist()
    out = {}
    for option in (kernel, 'simt'):
        resident = ResidentGLM(model, ctx=ctx, max_tokens=steps + 8, device_experts=True, graph=True,
                               index_kernel=option)
        try:
            out[option] = resident.prefill(tokens)
            # Two step graphs (#43), within index_topk and past it, each run once before its capture.
            assert resident.replays == steps - 2
            assert (resident.launches.get(name, 0) > 0) == (option == kernel)
            if option == 'fp8':  # every cached key row quantized as it is stored, within index_topk too
                assert resident.launches['index_mma/index_key_fp8_at'] == resident.launches['index_key_at']
        finally:
            resident.close()
    scale = np.abs(out['simt']).max()
    far = (np.abs(out[kernel] - out['simt']) > 2e-2 * scale).any(-1)
    # FP8 rounds the scores far more than bf16, so a step's selection can lose a token at the cut (H200:
    # 1 step of 600, evidence/h200-20261117-w); bf16 keeps every step within the tolerance here.
    assert far.sum() <= (steps // 100 if kernel == 'fp8' else 0), np.flatnonzero(far)
    assert (out[kernel].argmax(-1) == out['simt'].argmax(-1)).mean() > 0.95
