"""validation/layer_profile's matching of ncu's rows to the step's launch record (no GPU)."""
import pytest

from validation.layer_profile import report, with_library_kernels


def row(name, ns=1000.0):
    return {'Kernel Name': name, 'gpu__time_duration.sum': ns, 'launch__grid_size': '1', 'launch__block_size': '32',
            'launch__registers_per_thread': '32', 'dram__bytes_read.sum': 1e6, 'dram__bytes_write.sum': 0,
            'dram__throughput.avg.pct_of_peak_sustained_elapsed': 1, 'sm__throughput.avg.pct_of_peak_sustained_elapsed': 1}


LAUNCHES = [['rmsnorm', 0, 'layer.rmsnorm'], ['fp8_quant_groups', 0, 'self_attn.o_proj'], ['add', 0, 'layer.add']]


def test_library_kernels_take_the_projection_of_the_launch_before_them():
    # fp8_kernels='cublaslt' (#29): cuBLASLt's matmul and its split-K reduction run after our quantizer,
    # and the launch record has neither.
    rows = [row('glm_vx_gpu_rmsnorm'), row('glm_vx_gpu_fp8_quant_groups'),
            row('nvjet_qqsss_128x128_splitK_TNN'), row('void cublasLt::splitKreduce_kernel<32, 16, int>(Params)'),
            row('glm_vx_gpu_add')]
    got = with_library_kernels(LAUNCHES, rows)
    assert got == [LAUNCHES[0], LAUNCHES[1], ['nvjet_qqsss_128x128_splitK_TNN', 0, 'self_attn.o_proj'],
                   ['splitKreduce_kernel', 0, 'self_attn.o_proj'], LAUNCHES[2]]
    rep = report(dict(launches=LAUNCHES), rows, 1000.0)
    assert rep['groups']['layer 0']['kernels'] == 5


def test_a_missing_or_extra_launch_of_ours_still_fails_the_count():
    with pytest.raises(SystemExit, match='ncu profiled 2 kernels, the step launched 3'):
        report(dict(launches=LAUNCHES), [row('glm_vx_gpu_rmsnorm'), row('glm_vx_gpu_fp8_quant_groups')], 1000.0)
    rows = [row('glm_vx_gpu_rmsnorm'), row('glm_vx_gpu_fp8_quant_groups'), row('glm_vx_gpu_add'), row('glm_vx_gpu_add')]
    with pytest.raises(SystemExit, match='ncu profiled 4 kernels, the step launched 3'):
        report(dict(launches=LAUNCHES), rows, 1000.0)
