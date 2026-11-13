"""validation/layer_gate.py on the H100 reports of Oct 26 (evidence/h100-20261026)."""
import copy
import json
from pathlib import Path

import pytest

from validation import layer_gate

EVIDENCE = Path(__file__).resolve().parents[1] / 'evidence' / 'h100-20261026'


def report(stem):
    return json.loads((EVIDENCE / f'{stem}.json').read_text())


def target(summary, issue, what):
    return next(t for t in summary['targets'] if t['issue'] == issue and t['what'] == what)


def test_bytes_per_weight_counts_the_scales_the_device_holds():
    assert layer_gate.bytes_per_weight('fp8_matvec', 256, 384) == 256 * 384 + 4 * 2 * 3
    assert layer_gate.bytes_per_weight('fp8_matvec4', 129, 128) == 129 * 128 + 4 * 2 * 1
    assert layer_gate.bytes_per_weight('int4_matvec4', 2, 256) == 256 + 2 * 2 * 2
    assert layer_gate.bytes_per_weight('bf16_matvec', 3, 5) == 30
    assert layer_gate.bytes_per_weight('matvec', 3, 5) == 60
    assert layer_gate.bytes_per_weight('moe_down', 3, 5) == 60


def test_a_run_against_itself_has_no_regressions_and_the_targets_of_issue_29():
    run = report('layer-events-fp8')
    text, summary = layer_gate.gate({'run.json': run}, EVIDENCE)
    assert summary['regressions'] == []
    assert summary['compared'][0]['us'] == pytest.approx(summary['compared'][0]['ref_us'])
    o_proj = target(summary, '#30', 'self_attn.o_proj')
    # #29: o_proj takes 224 us, against 30 us at peak (13%).
    assert o_proj['us'] == pytest.approx(224.3, abs=0.1)
    assert o_proj['us_at_peak'] == pytest.approx(30.0, abs=0.1)
    assert round(o_proj['value']) == 13 and not o_proj['met']
    # #29: the experts take 964 us at k = 8, against 90 us at peak.
    experts = target(summary, '#31', 'mlp.experts')
    assert experts['layer'] == 3 and experts['us'] == pytest.approx(963.7, abs=0.1)
    assert experts['us_at_peak'] == pytest.approx(90.1, abs=0.1)
    topk = target(summary, '#32', 'topk at 64 tokens')
    assert topk['value'] == pytest.approx(331.0, abs=0.1) and not topk['met']
    assert '**No regressions.**' in text


def test_the_superseded_reports_are_regressions():
    # h100-20261026/README.md: layer-profile-* timed host gaps, layer-events-* are the rerun.
    _, summary = layer_gate.gate({'old.json': report('layer-profile-f32')}, EVIDENCE)
    assert summary['regressions']
    assert {r['reference'] for r in summary['regressions']} == {'baseline layer-events-f32.json'}


def x4_run(hand, faster=(), slower=(), factor=2.0):
    """The hand fp8 run as if from fp8_kernels='x4': other kernel names, some groups faster or slower."""
    run = copy.deepcopy(hand)
    run['fp8_kernels'] = 'x4'
    for launch in run['kernels']:
        launch['kernel'] = launch['kernel'].replace('fp8_matvec', 'fp8_matvec4').replace('fp8_moe_', 'fp8_moe4_')
        if launch['what'] in faster:
            launch['us'] /= factor
        if launch['what'] in slower:
            launch['us'] *= factor
    return run


def test_other_kernels_line_up_by_what_they_compute_and_meet_against_the_same_run():
    hand = report('layer-events-fp8')
    run = x4_run(hand, faster=('self_attn.o_proj',), slower=('self_attn.q_b_proj',))
    text, summary = layer_gate.gate({'hand.json': hand, 'x4.json': run}, EVIDENCE)
    references = {c['reference'] for c in summary['compared'] if c['run'] == 'fp8-x4'}
    assert references == {'baseline layer-events-fp8.json', 'this run, fp8 hand kernels'}
    slow = {(r['layer'], r['what']) for r in summary['regressions'] if r['run'] == 'fp8-x4'}
    assert slow == {(layer, 'self_attn.q_b_proj') for layer in range(4)}
    o_proj = next(t for t in summary['targets'] if t['run'] == 'fp8-x4' and t['what'] == 'self_attn.o_proj')
    assert o_proj['value'] == pytest.approx(2 * target(summary, '#30', 'self_attn.o_proj')['value'])
    assert 'fp8_matvec4 (was fp8_matvec)' in text


def test_small_slowdowns_are_noise():
    hand = report('layer-events-fp8')
    run = x4_run(hand, slower=('mlp.swiglu',), factor=1.3)  # 6 us -> 8 us: under the 2 us floor
    _, summary = layer_gate.gate({'x4.json': run}, EVIDENCE)
    assert summary['regressions'] == []
    _, summary = layer_gate.gate({'x4.json': run}, EVIDENCE, floor_us=1.0)
    assert {r['what'] for r in summary['regressions']} == {'mlp.swiglu'}


def test_the_command_line_keeps_evidence_and_exits_one_on_a_regression(tmp_path):
    out = tmp_path / 'gate.md'
    assert layer_gate.main([str(EVIDENCE / 'layer-events-int4.json'), '--baseline', str(EVIDENCE),
                            '--output', str(out)]) == 0
    assert json.loads(out.with_suffix('.json').read_text())['regressions'] == []
    with pytest.raises(SystemExit):
        layer_gate.main([str(EVIDENCE / 'layer-events-int4.json'), '--output', str(out)])
    assert layer_gate.main([str(EVIDENCE / 'layer-profile-int4.json'), '--baseline', str(EVIDENCE),
                            '--output', str(tmp_path / 'old.md')]) == 1
