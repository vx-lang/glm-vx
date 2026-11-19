"""The prefill rows kernels (#63, item 5b) on the CPU through Vx's own runtime (gpu/spawn/host.py), against NumPy
and against the decode kernels they repeat a row at a time: fp8_quant_rows (fp8_quant_groups' codes and scales),
router_logits_rows + router_topk_rows (router_bf16's bits a token), moe_combine_rows.

Skipped unless gpu/spawn/build.sh made the libraries (VX_DISPATCH_LIB set)."""
import numpy as np
import pytest

from gpu import fp8_quant
from gpu.test_spawn_vx_host import F, bits_equal, run  # noqa: F401  (run: the module's fixture)


@pytest.mark.parametrize('rows,cols,gather', [(5, 384, False), (7, 6144, True), (1, 128, True), (40, 4224, False)])
def test_fp8_quant_rows_is_quant_groups_row_major(run, rows, cols, gather):
    rng = np.random.default_rng(rows * cols)
    source = rows + 3 if gather else rows
    x = (rng.normal(size=(source, cols)) * rng.choice([1e-3, 1.0, 300.0], size=(source, 1))).astype(F)
    x[0, :128] = 0.0  # an all-zero group: the 1e-10 floor
    x[-1, 5] = np.nan
    x[-1, 7] = -1e6  # past 448 times the scale's other values: clamped
    src = rng.permutation(source)[:rows].astype(np.int32) if gather else np.zeros(1, np.int32)
    got = run('fp8_quant_rows', codes=np.zeros((rows, cols), np.uint8), scales=np.zeros((rows, cols // 128), F), x=x,
              src=src, rows=rows, cols=cols, gather=int(gather))
    picked = x[src] if gather else x
    # The hand kernel's max skips a NaN (vxk_quant_max), NumPy's would give the group a NaN scale: the NaN alone
    # gets the NaN code.
    codes, scales = fp8_quant.quant_groups(np.where(np.isnan(picked), F(0), picked))  # scales [cols / 128, rows]
    codes[np.isnan(picked)] = 127
    np.testing.assert_array_equal(got['codes'], codes)
    bits_equal(got['scales'], scales.T)


def test_moe_combine_rows_sums_in_the_given_order(run):
    rng = np.random.default_rng(7)
    tokens, cols, kk, rows = 6, 300, 9, 60
    y = rng.normal(size=(rows, cols)).astype(F)
    slots = rng.integers(0, rows, size=(tokens, kk)).astype(np.int32)
    weights = rng.random(size=(tokens, kk)).astype(F)
    got = run('moe_combine_rows', out=np.zeros((tokens, cols), F), y=y, slots=slots, weights=weights, tokens=tokens,
              cols=cols, kk=kk)
    want = np.zeros((tokens, cols), F)
    for j in range(kk):
        want = (want + weights[:, j:j + 1] * y[slots[:, j]]).astype(F)
    bits_equal(got['out'], want)


@pytest.mark.parametrize('experts,cols,k,tokens', [(256, 6144, 8, 3), (33, 520, 8, 4), (7, 136, 3, 2)])
def test_router_rows_are_router_bf16_a_token(run, experts, cols, k, tokens):
    rng = np.random.default_rng(experts + cols)
    w = (rng.normal(size=(experts, cols)).astype(F).view(np.uint32) >> 16).astype(np.uint16)
    x = rng.normal(size=(tokens, cols)).astype(F)
    x[-1] = x[0]  # two tokens with one row: the same choice
    bias = (rng.normal(size=experts) * 0.1).astype(F)
    gate = w.view(np.uint32).reshape(-1)
    logits = run('router_logits_rows', logits=np.zeros((tokens, experts), F), gate=gate, x=x, tokens=tokens,
                 experts=experts, cols=cols)['logits']
    stride = k + 1
    got = run('router_topk_rows', indices=np.full((tokens, stride), -7, np.int32), weights=np.full((tokens, stride), 3.0, F),
              logits=logits, bias=bias, tokens=tokens, experts=experts, k=k, scale=2.5)
    for t in range(tokens):
        one = run('router_bf16', indices=np.full(stride, -7, np.int32), weights=np.full(stride, 3.0, F),
                  logits=np.zeros(experts, F), gate=gate, x=x[t], bias=bias, experts=experts, cols=cols, k=k, scale=2.5,
                  arrivals=np.zeros(1, np.uint32))
        bits_equal(logits[t], one['logits'])
        np.testing.assert_array_equal(got['indices'][t], one['indices'])
        bits_equal(got['weights'][t], one['weights'])
    assert (got['indices'][:, k] == -7).all() and (got['weights'][:, k] == 3.0).all()
