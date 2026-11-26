"""ResidentGLM.prefill_chunk (#63, item 5b) under the CPU SIMT emulator: a prompt in one pass leaves the
caches forward() leaves one token at a time, and gives the same last logits and every later step; bit for
bit with prefill_gemm='rows', close with cuBLASLt's quantized activations."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import TOKENS, build, fp8_model
from gpu.test_resident_many import device_rows
from gpu.testing.context import Context

PROMPT = TOKENS[:10]  # past index_topk = 4: the later tokens' selections are sparse
NEXT = TOKENS[10]


def run(model, chunks, **options):
    """(token at a time, prefill_chunk over `chunks`) on two residents: the prompt's last logits, the next
    token's logits, the caches and the launches."""
    results = []
    for chunked in (False, True):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, **options)
        try:
            if chunked:
                for chunk in chunks[:-1]:
                    assert resident.prefill_chunk(chunk) is None
                last = resident.prefill_chunk(chunks[-1], logits=True)
            else:
                prompt = [t for chunk in chunks for t in chunk]
                for t in prompt[:-1]:
                    resident.forward(t, logits=False)
                last = resident.forward(prompt[-1])
            after = resident.forward(NEXT)
            results.append((last, after, device_rows(resident, len(PROMPT) + 1), dict(resident.launches)))
        finally:
            resident.close()
    return results


def assert_same(results):
    (last, after, caches, _), (chunk_last, chunk_after, chunk_caches, _) = results
    np.testing.assert_array_equal(chunk_last.view(np.uint32), last.view(np.uint32))
    np.testing.assert_array_equal(chunk_after.view(np.uint32), after.view(np.uint32))
    assert caches.keys() == chunk_caches.keys()
    for key in caches:
        np.testing.assert_array_equal(chunk_caches[key].view(np.uint32), caches[key].view(np.uint32), err_msg=str(key))


@pytest.mark.parametrize('chunks', [[PROMPT], [PROMPT[:3], PROMPT[3:]]], ids=['one', 'two'])
@pytest.mark.parametrize('fuse_glue', [False, True])
def test_float32_prompt_in_one_pass(chunks, fuse_glue):
    results = run(build(seed=3), chunks, fuse_glue=fuse_glue)
    assert_same(results)
    assert results[1][3].get('rmsnorm_rows', 0) > 0


def test_graph_resident_takes_each_tokens_step_inputs():
    # graph=True's attention kernels read the step buffer: prefill_chunk points it at each token's.
    assert_same(run(build(seed=4), [PROMPT[:6], PROMPT[6:]], graph=True, fused_attention=True, device_experts=True))


@pytest.mark.parametrize('device_experts', [False, True])
def test_fp8_prompt_in_one_pass(device_experts):
    model, _ = fp8_model(31, hidden_size=144, moe_intermediate_size=144)
    assert_same(run(model, [PROMPT], device_experts=device_experts))


def test_cublaslt_rows_are_near_the_token_at_a_time_logits():
    # prefill_gemm='cublaslt': the FP8 Linears of 128-multiple width over the rows, inputs quantized to e4m3.
    model, _ = fp8_model(29, hidden_size=128, moe_intermediate_size=128)
    (last, after, caches, _), (chunk_last, chunk_after, chunk_caches, launches) = run(
        model, [PROMPT], prefill_gemm='cublaslt', device_experts=True)
    scale = float(np.abs(last).max())
    np.testing.assert_allclose(chunk_last, last, rtol=0.1, atol=0.05 * scale)
    np.testing.assert_allclose(chunk_after, after, rtol=0.1, atol=0.05 * scale)
    assert np.abs(chunk_last - last).max() > 0  # the activations were quantized
    assert launches.get('cublaslt_fp8_rows', 0) > 0 and launches.get('fp8_quant_groups', 0) >= launches['cublaslt_fp8_rows']


def test_limits():
    resident = ResidentGLM(build(seed=3), ctx=Context(), max_tokens=8)
    try:
        with pytest.raises(ValueError, match='prefill_chunk takes 1 to 8'):
            resident.prefill_chunk([])
        with pytest.raises(ValueError, match='prefill_chunk takes'):
            resident.prefill_chunk([1] * 9)
        resident.prefill_chunk([1] * 5)
        with pytest.raises(ValueError, match='max_tokens'):
            resident.prefill_chunk([1] * 4)
        with pytest.raises(ValueError, match="prefill_gemm must be"):
            ResidentGLM(build(seed=3), ctx=Context(), max_tokens=8, prefill_gemm='x')
    finally:
        resident.close()


@pytest.mark.parametrize('prefill_sum', ['rows', 'nccl'])
def test_two_ranks_prompt_in_one_pass(prefill_sum):
    # Under tensor parallelism: 'rows' sums each row as forward() does (its bits); 'nccl' one all-reduce over
    # the rows (here the emulator's rank-order sum; on GPUs NCCL's, the same on every rank). Both ranks agree.
    from gpu.testing.collective import LocalJob, run_ranks
    from gpu.tp import TensorParallel
    model = build(seed=3)
    results = {}
    for chunked in (False, True):
        job = LocalJob(2)

        def rank_main(rank):
            resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_sum=prefill_sum,
                                   tp=TensorParallel(rank, 2, job.member(rank)))
            try:
                if chunked:
                    last = resident.prefill_chunk(PROMPT, logits=True)
                else:
                    for t in PROMPT[:-1]:
                        resident.forward(t, logits=False)
                    last = resident.forward(PROMPT[-1])
                return last, resident.forward(NEXT), dict(resident.launches)
            finally:
                resident.close()
        results[chunked] = run_ranks(job, rank_main)
    for chunked, ranks in results.items():
        for r in (0, 1):
            np.testing.assert_array_equal(ranks[1][r].view(np.uint32), ranks[0][r].view(np.uint32))
    (last, after, _), (chunk_last, chunk_after, launches) = results[False][0], results[True][0]
    if prefill_sum == 'rows':
        np.testing.assert_array_equal(chunk_last.view(np.uint32), last.view(np.uint32))
        np.testing.assert_array_equal(chunk_after.view(np.uint32), after.view(np.uint32))
    else:
        np.testing.assert_allclose(chunk_last, last, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(chunk_after, after, rtol=1e-5, atol=1e-5)
        assert launches['tp_all_reduce_rows'] == 2 * model.n_layers


def test_sequence_model_prefills_in_chunks_of_prefill_rows():
    # The scheduler's entry point: prefill_tokens exists only with prefill_rows, and takes the tokens in
    # prefill_chunk calls of up to prefill_rows; then forward gives prefill_token's logits.
    from gpu.resident import ResidentSequenceModel
    model = build(seed=3)
    outputs = []
    for rows in (0, 4):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_rows=rows)
        try:
            sequence = ResidentSequenceModel(model, resident)
            cache = sequence.new_cache()
            if rows:
                sequence.prefill_tokens(PROMPT[:-1], cache)
            else:
                assert not hasattr(sequence, 'prefill_tokens')
                for t in PROMPT[:-1]:
                    sequence.prefill_token(t, cache)
            outputs.append((sequence.forward(PROMPT[-1], cache), dict(resident.launches)))
        finally:
            resident.close()
    np.testing.assert_array_equal(outputs[1][0].view(np.uint32), outputs[0][0].view(np.uint32))
    # 9 tokens: chunks of 4, 4 and 1, each with two RMSNorms over its rows a layer (input, q_a)
    assert outputs[1][1].get('rmsnorm_rows', 0) == 2 * 3 * model.n_layers


def silu(x):
    return x / (1.0 + np.exp(-x))


def bf16_values(x):
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16).astype(np.uint32).view(np.float32)


@pytest.mark.parametrize('fold', [False, True])
def test_grouped_experts_are_the_quantized_experts_of_each_token(fold):
    # prefill_moe='grouped' on one MoE layer against NumPy: each token's experts and weights from forward()'s
    # router (_route), x and the activations quantized to e4m3 per 128 columns (fp8_quant.quant_groups), each
    # (token, expert) pair's gate, up, SwiGLU and down on the dequantized FP8 weights, added in ascending expert id,
    # the shared expert last (fold_shared). The grouped GEMM is the test context's stand-in.
    from gpu import fp8_quant
    from gpu.test_resident import fp8_bf16_router_model
    model = fp8_bf16_router_model(37, hidden_size=128, moe_intermediate_size=128)
    layer = next(i for i in range(model.n_layers) if model.mlp_type(i) != 'dense')
    prefix = f'model.layers.{layer}.mlp'
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_moe='grouped', device_experts=True,
                           fuse_router=True, fold_shared=fold, fp8_kernels='cvt' if fold else 'hand')
    try:
        count, hidden = 6, resident.hidden
        x = np.random.default_rng(3).normal(size=(count, hidden)).astype(np.float32)
        rows = resident._prefill_rows('norm', hidden)
        rows.view((count * hidden,)).upload(x.tobytes())
        got = np.frombuffer(resident._prefill_moe_grouped(prefix, rows, count).view((count * hidden,)).download(),
                            np.float32).reshape(count, hidden)
        k, experts = int(model.config['num_experts_per_tok']), int(model.config['n_routed_experts'])
        routes = []
        for i in range(count):
            resident._route(prefix, resident._prefill_row(rows, i))
            routes.append((np.frombuffer(resident.route_ids.download(), np.int32)[:k].copy(),
                           np.frombuffer(resident.route_weights.download(), np.float32)[:k].copy()))
        assert resident.launches['grouped_moe'] == 1
    finally:
        resident.close()

    def dense(name):
        codes, scales, _, (n, kk) = model.weights.block_fp8(name)
        return fp8_quant.e4m3_values()[codes] * np.repeat(np.repeat(scales, 128, 0), 128, 1)[:n, :kk]

    def quantized(v):
        codes, scales = fp8_quant.quant_groups(v[None].astype(np.float32))
        return fp8_quant.dequant_groups(codes, scales)[0]

    want = np.zeros((count, hidden))
    for t in range(count):
        ids, weights = routes[t]
        pairs = sorted(zip(ids.tolist(), weights.tolist()))
        names = [(f'{prefix}.experts.{e}', w) for e, w in pairs]
        if fold:
            names.append((f'{prefix}.shared_experts', 1.0))
        xq = quantized(x[t])
        for name, w in names:
            g, u = dense(name + '.gate_proj') @ xq, dense(name + '.up_proj') @ xq
            act = (silu(g.astype(np.float32)) * u.astype(np.float32)).astype(np.float32)
            y = (dense(name + '.down_proj') @ quantized(act)).astype(np.float32)  # down's C in bf16
            want[t] += w * bf16_values(y)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4 * np.abs(want).max())


def test_cublaslt_router_is_the_bf16_rows_times_the_gate():
    # prefill_gemm='cublaslt': the router's logits over the rows from cuBLASLt's bfloat16 matmul (the test context's
    # float64 stand-in), x rounded to bfloat16 times the BF16 gate as SGLang's linear_bf16_fp32; router_topk_rows
    # then picks each token's experts, here the ones the token-at-a-time router (float32 x) picks. _prefill_linear
    # takes the same path for any BF16 Linear (the indexer's weights_proj in the checkpoint).
    from gpu.test_resident import fp8_bf16_router_model
    model = fp8_bf16_router_model(37, hidden_size=128, moe_intermediate_size=128)
    layer = next(i for i in range(model.n_layers) if model.mlp_type(i) != 'dense')
    prefix = f'model.layers.{layer}.mlp'
    resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_gemm='cublaslt', prefill_moe='grouped',
                           device_experts=True, fuse_router=True)
    try:
        count, hidden = 6, resident.hidden
        k, experts = int(model.config['num_experts_per_tok']), int(model.config['n_routed_experts'])
        x = np.random.default_rng(3).normal(size=(count, hidden)).astype(np.float32)
        rows = resident._prefill_rows('norm', hidden)
        rows.view((count * hidden,)).upload(x.tobytes())
        ids, _ = resident._prefill_route(prefix, rows, count, k, k)
        logits = np.frombuffer(resident._prefill_rows('router_logits', experts).download(), np.float32)
        got_ids = np.frombuffer(ids.download(), np.int32).reshape(-1, k)[:count]
        out = resident._prefill_rows('gate_out', experts)
        resident._prefill_linear(prefix + '.gate', rows, out, count)
        linear = np.frombuffer(out.download(), np.float32).reshape(-1, experts)[:count]
        routes = []
        for i in range(count):
            resident._route(prefix, resident._prefill_row(rows, i))
            routes.append(np.sort(np.frombuffer(resident.route_ids.download(), np.int32)[:k]))
        assert resident.launches['cublaslt_bf16_rows'] == 2 and 'router_logits_rows' not in resident.launches
    finally:
        resident.close()
    bits, _ = model.weights.bf16(prefix + '.gate')
    want = bf16_values(x).astype(np.float64) @ (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64).T
    np.testing.assert_allclose(logits.reshape(-1, experts)[:count], want, rtol=1e-6, atol=1e-6 * np.abs(want).max())
    np.testing.assert_array_equal(linear, logits.reshape(-1, experts)[:count])
    for i in range(count):
        np.testing.assert_array_equal(np.sort(got_ids[i]), routes[i])


def test_grouped_experts_in_the_chunk_stay_near_the_token_at_a_time_logits():
    # prefill_chunk with prefill_moe='grouped' end to end (the shared expert added after, not folded): e4m3 activations
    # in every routed expert move this tiny model's logits by up to ~9% of the largest (the layer itself is pinned
    # by the test above).
    from gpu.test_resident import fp8_bf16_router_model
    model = fp8_bf16_router_model(37, hidden_size=128, moe_intermediate_size=128)
    (last, after, _, _), (chunk_last, chunk_after, _, launches) = run(
        model, [PROMPT[:7], PROMPT[7:]], prefill_moe='grouped', device_experts=True, fuse_router=True)
    scale = float(np.abs(last).max())
    np.testing.assert_allclose(chunk_last, last, atol=0.15 * scale)
    np.testing.assert_allclose(chunk_after, after, atol=0.15 * scale)
    assert launches['grouped_moe'] > 0 and launches.get('router_logits_rows', 0) == launches['grouped_moe']


@pytest.mark.parametrize('graph', [False, True])
def test_mha_attention_is_the_absorbed_attention(graph):
    # prefill_attention='mha' (index_topk 16: the prompt is dense): kv_b_proj expands each cached latent row, one
    # causal mha_prefill launch; the same attention as forward()'s absorbed MLA in float32, summed in another order.
    # Layer 0's cache rows are forward()'s bits (kv_rows_prefill: its norm and rope).
    model = build(seed=3, index_topk=16)
    options = dict(graph=True, fused_attention=True, device_experts=True) if graph else {}
    (last, after, caches, _), (chunk_last, chunk_after, chunk_caches, launches) = run(
        model, [PROMPT[:4], PROMPT[4:]], prefill_attention='mha', **options)
    np.testing.assert_allclose(chunk_last, last, rtol=1e-4, atol=1e-4 * float(np.abs(last).max()))
    np.testing.assert_allclose(chunk_after, after, rtol=1e-4, atol=1e-4 * float(np.abs(after).max()))
    for key in caches:  # layer 0's rows are forward()'s bits (its inputs are the same); later layers' inputs differ
        if key[1] == 0 and not graph:
            np.testing.assert_array_equal(chunk_caches[key].view(np.uint32), caches[key].view(np.uint32), err_msg=str(key))
        else:
            np.testing.assert_allclose(chunk_caches[key], caches[key], rtol=1e-4, atol=1e-5, err_msg=str(key))
    assert launches.get('mha_prefill', 0) == 2 * model.n_layers



# GLM-5.3's head sizes on the tiny model: mha_prefill_tc takes nope 192, rope 64 and value 256 only.
TC_HEADS = dict(qk_nope_head_dim=192, qk_rope_head_dim=64, qk_head_dim=256, v_head_dim=256, index_head_dim=128)


def test_mha_tc_attention_is_the_mha_attention_to_bf16_rounding():
    # prefill_attention='mha_tc': each dense mha_prefill launch becomes one of mha_prefill_tc (the tensor cores: bf16
    # products, f32 sums). The prompt's last logits and the next token's agree with 'mha' (float32) to that rounding.
    model = build(seed=3, index_topk=16, **TC_HEADS)
    got = {}
    for attention in ('mha', 'mha_tc'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_attention=attention)
        try:
            assert resident.prefill_chunk(PROMPT[:4]) is None
            last = resident.prefill_chunk(PROMPT[4:], logits=True)
            got[attention] = last, resident.forward(NEXT), dict(resident.launches)
        finally:
            resident.close()
    (last, after, launches), (tc_last, tc_after, tc_launches) = got['mha'], got['mha_tc']
    np.testing.assert_allclose(tc_last, last, rtol=0, atol=5e-3 * float(np.abs(last).max()))
    np.testing.assert_allclose(tc_after, after, rtol=0, atol=5e-3 * float(np.abs(after).max()))
    assert launches.get('mha_prefill', 0) == 2 * model.n_layers and 'mha_prefill_tc' not in launches
    assert tc_launches.get('mha_prefill_tc', 0) == 2 * model.n_layers and 'mha_prefill' not in tc_launches
    with pytest.raises(ValueError, match="prefill_attention='mha_tc' needs nope 192"):
        ResidentGLM(build(seed=3), ctx=Context(), max_tokens=32, prefill_attention='mha_tc')

def test_mla_tc_past_index_topk_is_the_mha_tc_attention_to_bf16_rounding():
    # prefill_attention='mla_tc': past index_topk (4) each query attends to its selection in the absorbed form
    # (_prefill_mla_sparse: W_UK^T q_nope and W_UV latent as batched cuBLASLt bfloat16 matmuls, the test context's
    # float64 stand-in; mla_prefill_sparse_tc on the bf16 cache) instead of kv_b_proj over every key and
    # mha_prefill_sparse. The prompt's last logits and the next token's agree with 'mha_tc' to bf16 rounding.
    model = build(seed=5, num_hidden_layers=2, mlp_layer_types=['dense', 'sparse'], indexer_types=['full', 'shared'],
                  num_attention_heads=16, num_key_value_heads=16, kv_lora_rank=512, index_topk=4, **TC_HEADS)
    options = dict(graph=True, fused_attention=True, device_experts=True, mla_kernel='tc')
    got = {}
    for attention in ('mha_tc', 'mla_tc'):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_attention=attention, **options)
        try:
            assert resident.prefill_chunk(PROMPT[:3]) is None
            last = resident.prefill_chunk(PROMPT[3:], logits=True)
            got[attention] = last, resident.forward(NEXT), dict(resident.launches)
        finally:
            resident.close()
    (last, after, launches), (mla_last, mla_after, mla_launches) = got['mha_tc'], got['mla_tc']
    np.testing.assert_allclose(mla_last, last, rtol=0, atol=1e-2 * float(np.abs(last).max()))
    np.testing.assert_allclose(mla_after, after, rtol=0, atol=1e-2 * float(np.abs(after).max()))
    assert launches.get('mha_prefill_sparse', 0) == model.n_layers and 'mla_prefill_sparse_tc' not in launches
    assert mla_launches.get('mla_prefill_sparse_tc', 0) == model.n_layers and 'mha_prefill_sparse' not in mla_launches
    assert mla_launches.get('mha_prefill_tc', 0) == model.n_layers  # the first chunk, dense


@pytest.mark.parametrize('graph', [False, True])
def test_mha_attention_past_index_topk_selects_per_query(graph):
    # index_topk 4, a 10-token prompt in chunks of 3 and 7: the second chunk's queries at positions 3.. score the keys
    # up to themselves (index_scores_rows), keep their top 4 in position order (topk_select_rows; the first, at
    # position 3, sees only 4 keys and takes them all) and attend to those (mha_prefill_sparse); shared layers reuse
    # the full layer's selection. forward()'s attention and selection a token at a time, in another order.
    model = build(seed=3)
    options = dict(graph=True, fused_attention=True, device_experts=True) if graph else {}
    (last, after, caches, _), (chunk_last, chunk_after, chunk_caches, launches) = run(
        model, [PROMPT[:3], PROMPT[3:]], prefill_attention='mha', **options)
    np.testing.assert_allclose(chunk_last, last, rtol=1e-4, atol=1e-4 * float(np.abs(last).max()))
    np.testing.assert_allclose(chunk_after, after, rtol=1e-4, atol=1e-4 * float(np.abs(after).max()))
    for key in caches:
        np.testing.assert_allclose(chunk_caches[key], caches[key], rtol=1e-4, atol=1e-5, err_msg=str(key))
    full = sum(model.indexer_type(i) == 'full' for i in range(model.n_layers))
    assert launches.get('mha_prefill', 0) == model.n_layers and launches.get('mha_prefill_sparse', 0) == model.n_layers
    assert launches.get('topk_select_rows', 0) == full and launches.get('index_scores_rows', 0) == full


def test_two_ranks_mha_prompt_past_index_topk():
    # prefill_attention='mha' under tensor parallelism: each rank's heads (q_b, kv_b_proj and o_proj split), the
    # indexer the same on every rank, the sums over the rows; both ranks agree, near two-rank forward().
    from gpu.testing.collective import LocalJob, run_ranks
    from gpu.tp import TensorParallel
    model = build(seed=3)
    results = {}
    for chunked in (False, True):
        job = LocalJob(2)

        def rank_main(rank):
            resident = ResidentGLM(model, ctx=Context(), max_tokens=32, prefill_attention='mha',
                                   tp=TensorParallel(rank, 2, job.member(rank)))
            try:
                if chunked:
                    resident.prefill_chunk(PROMPT[:3])
                    last = resident.prefill_chunk(PROMPT[3:], logits=True)
                else:
                    for t in PROMPT[:-1]:
                        resident.forward(t, logits=False)
                    last = resident.forward(PROMPT[-1])
                return last, resident.forward(NEXT), dict(resident.launches)
            finally:
                resident.close()
        results[chunked] = run_ranks(job, rank_main)
    for ranks in results.values():
        for r in (0, 1):
            np.testing.assert_array_equal(ranks[1][r].view(np.uint32), ranks[0][r].view(np.uint32))
    (last, after, _), (chunk_last, chunk_after, launches) = results[False][0], results[True][0]
    np.testing.assert_allclose(chunk_last, last, rtol=1e-4, atol=1e-4 * float(np.abs(last).max()))
    np.testing.assert_allclose(chunk_after, after, rtol=1e-4, atol=1e-4 * float(np.abs(after).max()))
    assert launches.get('mha_prefill_sparse', 0) == model.n_layers


@pytest.mark.parametrize('topk', [16, 4])
def test_every_prefill_option_together(topk):
    # z32's configuration: cuBLASLt rows (the BF16 router's too), the routed experts grouped, MHA attention. Dense
    # (index_topk 16): within a few % of the largest logit. Past index_topk 4, e4m3 activations move the indexer's
    # scores enough to change which 4 of a few keys a query takes, so the prompt's last logits move by ~30% here
    # (cuBLASLt rows alone do it too); the decode step after the prompt stays within a few %.
    from gpu.test_resident import fp8_bf16_router_model
    model = fp8_bf16_router_model(41, hidden_size=128, moe_intermediate_size=128, index_topk=topk)
    (last, after, _, _), (chunk_last, chunk_after, _, launches) = run(
        model, [PROMPT[:3], PROMPT[3:]], prefill_gemm='cublaslt', prefill_moe='grouped', prefill_attention='mha',
        device_experts=True, fuse_router=True)
    scale = float(np.abs(last).max())
    if topk == 16:
        np.testing.assert_allclose(chunk_last, last, atol=0.05 * scale)
    np.testing.assert_allclose(chunk_after, after, atol=0.05 * scale)
    names = ('cublaslt_fp8_rows', 'cublaslt_bf16_rows', 'grouped_moe', 'mha_prefill') + (('mha_prefill_sparse', 'topk_select_rows') if topk == 4 else ())
    for name in names:
        assert launches.get(name, 0) > 0, name
