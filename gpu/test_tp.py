"""Tensor parallelism (gpu/tp.py, #34 #55 #56) under the CPU emulator: n ranks in threads
(gpu/testing/collective.py), each a ResidentGLM on its slice of the model."""
import numpy as np
import pytest

from gpu.resident import ResidentGLM
from gpu.test_resident import TOKENS, build, fp8_model, needs_spawn, reference
from gpu.testing.collective import LocalJob, run_ranks
from gpu.testing.context import Context
from gpu.tp import ShardedModel, TensorParallel


def tp_prefill(model, ranks, tokens, **options):
    """Each rank's logits over tokens, and each rank's launch counts."""
    job = LocalJob(ranks)

    def rank_main(rank):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32,
                               tp=TensorParallel(rank, ranks, job.member(rank)), **options)
        try:
            return resident.prefill(tokens), dict(resident.launches)
        finally:
            resident.close()
    results = run_ranks(job, rank_main)
    return [r[0] for r in results], [r[1] for r in results]


@pytest.mark.parametrize('ranks', [2])
def test_ranks_agree_and_match_the_reference(ranks):
    model = build(seed=3)
    expected = reference(model, TOKENS)
    logits, launches = tp_prefill(model, ranks, TOKENS)
    for other in logits[1:]:
        np.testing.assert_array_equal(other.view(np.uint32), logits[0].view(np.uint32))
    np.testing.assert_allclose(logits[0], expected, rtol=2e-5, atol=2e-5)
    assert (logits[0].argmax(-1) == expected.argmax(-1)).all()
    n_layers = model.n_layers
    assert launches[0]['tp_all_gather'] == 2 * n_layers * len(TOKENS)


def test_sharded_slices():
    model = build(seed=3, num_attention_heads=4)
    shards = [ShardedModel(model, r, 2) for r in range(2)]
    for name in ('model.layers.0.self_attn.q_b_proj.weight', 'model.layers.0.mlp.gate_proj.weight', 'lm_head.weight'):
        np.testing.assert_array_equal(np.concatenate([s._weight(name) for s in shards]), model._weight(name))
    for name in ('model.layers.0.self_attn.o_proj.weight', 'model.layers.0.mlp.down_proj.weight'):
        np.testing.assert_array_equal(np.concatenate([s._weight(name) for s in shards], axis=1), model._weight(name))
    for name in ('model.layers.0.self_attn.q_a_proj.weight', 'model.layers.0.self_attn.indexer.wq_b.weight'):
        assert shards[1]._weight(name) is model._weight(name) or np.array_equal(shards[1]._weight(name), model._weight(name))
    assert shards[0].heads == 2 and shards[0].config['num_attention_heads'] == 2
    with pytest.raises(ValueError, match='split'):
        ShardedModel(model, 0, 3)


def test_device_job_shares_one_context_and_gives_the_local_bits():
    # validation/tp_decode_check's one-GPU reference: the ranks as threads on one context, the gather a
    # device copy (gpu/testing/collective.DeviceJob); here on the emulator's context.
    from gpu.testing.collective import DeviceJob
    model = build(seed=3)
    want, _ = tp_prefill(model, 2, TOKENS[:5])
    context, job = Context(), DeviceJob(2)
    context.host()  # made once here: two threads making the shared host-kernel pool at once race on its environment

    def rank_main(rank):
        r = ResidentGLM(model, ctx=context, max_tokens=32, tp=TensorParallel(rank, 2, job.member(rank)))
        return r.prefill(TOKENS[:5])
    got = run_ranks(job, rank_main)
    for rank in range(2):
        np.testing.assert_array_equal(got[rank].view(np.uint32), want[rank].view(np.uint32))


def test_four_ranks():
    model = build(seed=5, num_attention_heads=4, intermediate_size=64, moe_intermediate_size=16)
    expected = reference(model, TOKENS[:6])
    logits, _ = tp_prefill(model, 4, TOKENS[:6])
    for other in logits[1:]:
        np.testing.assert_array_equal(other.view(np.uint32), logits[0].view(np.uint32))
    np.testing.assert_allclose(logits[0], expected, rtol=2e-5, atol=2e-5)


@needs_spawn
def test_graph_replay_gives_the_eager_bits_on_every_rank():
    # The cross-rank sum is recorded into each rank's step graph: replayed, the same bits as eager.
    model = build(seed=41)
    eager, _ = tp_prefill(model, 2, TOKENS, device_experts=True)
    graph, launches = tp_prefill(model, 2, TOKENS, device_experts=True, graph=True, fused_attention=True,
                                 mla_kernel='chunks')
    for rank in range(2):
        np.testing.assert_allclose(graph[rank], eager[rank], rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(graph[rank].view(np.uint32), graph[0].view(np.uint32))


def aligned_fp8_model(seed, ranks):
    # Every split Linear's slice on 128-wide FP8 scale blocks, as GLM-5.3's up to 8 ranks: 2 heads a rank
    # of nope 64 + rope 64 (q_b 256 rows a rank) and value 64 (kv_b 256 rows, o_proj 128 columns).
    return fp8_model(seed, hidden_size=128, num_attention_heads=2 * ranks, num_key_value_heads=2 * ranks,
                     qk_nope_head_dim=64, qk_rope_head_dim=64, qk_head_dim=128, v_head_dim=64, kv_lora_rank=128, index_head_dim=64,
                     intermediate_size=128 * ranks, moe_intermediate_size=128 * ranks)[0]


@needs_spawn
def test_fp8_cvt_fold_shared_graph_on_two_ranks():
    model = aligned_fp8_model(53, 2)
    options = dict(device_experts=True, fp8_kernels='cvt', fold_shared=True, graph=True, fused_attention=True,
                   mla_kernel='chunks')
    tokens = TOKENS[:6]
    logits, launches = tp_prefill(model, 2, tokens, **options)
    np.testing.assert_array_equal(logits[1].view(np.uint32), logits[0].view(np.uint32))
    one = ResidentGLM(model, ctx=Context(), max_tokens=32, **options)
    try:
        want = one.prefill(tokens)
    finally:
        one.close()
    np.testing.assert_allclose(logits[0], want, rtol=1e-5, atol=1e-5)
    assert (logits[0].argmax(-1) == want.argmax(-1)).all()


def test_fp8_slices_must_fall_on_scale_blocks():
    with pytest.raises(ValueError, match='scale blocks'):
        ShardedModel(fp8_model(3, num_attention_heads=4, num_key_value_heads=4)[0], 1, 2).weights.block_fp8(
            'model.layers.0.self_attn.q_b_proj')


# --- PeerSum's buffers with the fake driver of gpu/test_runtime.py --------------------------------

class IpcLibrary:
    """FakeLibrary with CUDA IPC: a handle holds the allocation's address; opening one maps the same
    bytes at a new address (as another process's mapping would), closing unmaps it."""

    def __init__(self):
        from gpu.test_runtime import FakeLibrary
        self.fake = FakeLibrary()
        self.opened = {}

    def __getattr__(self, name):
        import ctypes as C
        from gpu.test_runtime import FakeFunction, value, write
        fake = self.fake
        if name == 'cuIpcGetMemHandle':
            def get(handle, pointer):
                fake.calls.append((name, (pointer,)))
                handle._obj.reserved[:8] = list(int(pointer).to_bytes(8, 'little'))
                return 0
            return get
        if name == 'cuIpcOpenMemHandle_v2':
            def open_(out, handle, flags):
                source = int.from_bytes(bytes(handle.reserved[:8]), 'little')
                address = fake.next_address
                fake.next_address += 1 << 20
                fake.allocations[address] = fake.allocations[source]  # the same bytes, mapped again
                self.opened[address] = source
                write(out, C.c_uint64, address)
                fake.calls.append((name, (source, flags)))
                return 0
            return open_
        if name == 'cuIpcCloseMemHandle':
            def close(pointer):
                del fake.allocations[value(pointer)]
                fake.calls.append((name, (value(pointer),)))
                return 0
            return close
        return getattr(fake, name)


def test_peer_sum_maps_every_rank_and_pads_to_eight():
    from gpu.runtime import CUDAContext, CUDADriver
    from gpu.tp import PeerSum
    library = IpcLibrary()
    context = CUDAContext(_driver=CUDADriver(_library=library))
    ranks, n = 3, 300
    handles = {}
    # Stand-ins for the other processes' allocations, whose handles the exchange returns.
    owners = []
    for r in range(ranks):
        buffer = context.allocate(2 * 300 * 4 + 512)
        handles[r] = buffer.ipc_handle()
        owners.append(buffer)
    sums = [PeerSum(context, r, ranks, n, lambda own: [handles[i] for i in range(ranks)]) for r in range(ranks)]
    peer = sums[1]
    args = peer.arguments(peer.data[1].view((n,)))
    assert 'rank' not in args and args['ranks'] == 3 and args['n'] == n
    assert peer.blocks == 2 and args['epochs'].size == 2
    # d0 and d2 are mappings of ranks 0 and 2's allocations, d1 this rank's own; d3..d7 repeat its own.
    assert library.opened[args['d0'].pointer] == owners[0].pointer
    assert library.opened[args['d2'].pointer] == owners[2].pointer
    assert args['d1'].pointer == args['mine'].pointer and args['f1'].pointer == args['my_flags'].pointer
    assert all(args[f'd{r}'].pointer == args['mine'].pointer for r in range(3, 8))
    assert args['f0'].pointer == args['d0'].pointer + 2560  # 2 x 300 floats, to 256 bytes
    with pytest.raises(ValueError, match='300 floats'):
        peer.arguments(peer.data[1])
    peer.close()
    assert sum(1 for name, _ in library.fake.calls if name == 'cuIpcCloseMemHandle') == 2
    with pytest.raises(ValueError, match='1 to 8 ranks'):
        PeerSum(context, 0, 9, n, None)


@needs_spawn
def test_greedy_picks_the_logits_argmax_on_one_and_two_ranks():
    # forward(logits='greedy'): numpy.argmax of forward()'s logits, chosen on the device: on one GPU through
    # argmax_rows; under tensor parallelism each rank's argmax_pair over its vocabulary rows, the ranks'
    # pairs gathered and the largest taken (the lower rank on a tie), under graph replay too.
    model = aligned_fp8_model(53, 2)
    options = dict(device_experts=True, fp8_kernels='cvt', fold_shared=True, graph=True, fused_attention=True,
                   mla_kernel='chunks')
    tokens = TOKENS[:6]
    one = ResidentGLM(model, ctx=Context(), max_tokens=32, **options)
    twin = ResidentGLM(model, ctx=Context(), max_tokens=32, **options)
    try:
        want = [int(np.argmax(one.forward(t))) for t in tokens]
        assert [twin.forward(t, logits='greedy') for t in tokens] == want
    finally:
        one.close()
        twin.close()
    job = LocalJob(2)

    def rank_main(rank):
        resident = ResidentGLM(model, ctx=Context(), max_tokens=32, tp=TensorParallel(rank, 2, job.member(rank)), **options)
        try:
            got = [resident.forward(t, logits='greedy') for t in tokens]
            return got, resident.replays
        finally:
            resident.close()
    results = run_ranks(job, rank_main)
    assert all(r[0] == want for r in results), (results, want)
    assert all(r[1] > 0 for r in results)


@needs_spawn
def test_argmax_pair_takes_the_first_largest():
    # argmax_pair: the largest value and offset + its index, the first on a tie, as float32 bits.
    from gpu.test_resident import SPAWN
    context = Context()
    kernel = context.spawn_kernel(SPAWN / 'argmax_pair.ptx')
    rng = np.random.default_rng(5)
    for dim, offset in ((19360, 3 * 19360), (5, 0), (2049, 7)):
        x = rng.normal(size=dim).astype(np.float32)
        x[dim // 3] = x[dim - 1] = x.max() + 1  # a tie: the first wins
        out = context.tensor((2,))
        kernel.launch(out=out, x=context.tensor((dim,), 'float32', x.tobytes()), dim=dim, offset=offset, threads=1024)
        pair = np.frombuffer(out.download(), dtype=np.float32)
        assert pair[0] == x.max() and int(pair[1:].view(np.int32)[0]) == offset + int(np.argmax(x))
