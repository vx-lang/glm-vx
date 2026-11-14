"""gpu/fp8_quant.vx under the CPU SIMT emulator (NOT GPU validation): codes and scales bit for bit
against gpu/fp8_quant.py's model, SGLang's group-128 e4m3 quantization."""
import numpy as np
import pytest

from gpu import fp8_quant
from gpu.test_simt import simt  # noqa: F401  (the module's emulator fixture)


def _run(simt, x):
    rows, cols = x.shape
    codes, scales = np.zeros((rows, cols), np.uint8), np.zeros((cols // 128, rows), np.float32)
    simt.run('glm_vx_gpu_fp8_quant_groups', codes, scales, np.ascontiguousarray(x), rows, cols,
             grid=-(-rows * cols // 128 // 4), block=128)
    return codes, scales


def test_every_code_midpoint_and_neighbour_rounds_to_the_nearest_even_code(simt):
    # Each group holds 448, so its scale is 1 and x / scale is x: every e4m3 value, every midpoint
    # between neighbours (the ties) and the float32 values on either side of each, both signs.
    values = fp8_quant.e4m3_values()[:127]
    points = np.concatenate([values, (values[:-1] + values[1:]) / 2]).astype(np.float32)
    points = np.concatenate([points, np.nextafter(points, np.float32(np.inf)), np.nextafter(points, np.float32(-np.inf))])
    points = points[points <= 448]  # 448's upper neighbour would raise its group's scale
    points = np.concatenate([points, -points]).astype(np.float32)
    points = np.concatenate([points, np.zeros(-len(points) % 127, np.float32)]).reshape(-1, 127)
    x = np.concatenate([np.full((len(points), 1), 448, np.float32), points], 1).reshape(1, -1)
    codes, scales = _run(simt, x)
    want_codes, want_scales = fp8_quant.quant_groups(x)
    assert (scales == 1).all() and (want_scales == 1).all()
    np.testing.assert_array_equal(codes, want_codes)


@pytest.mark.parametrize('rows,cols,seed', [(1, 6144, 1), (4, 2048, 2), (3, 128, 3)])
def test_random_activations_match_the_model_bitwise(simt, rows, cols, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, cols)) * rng.choice([1e-6, 1e-2, 1, 30], size=(rows, cols))).astype(np.float32)
    x[0, :128] = 0  # an all-zero group: scale 1e-10 / 448, codes 0
    codes, scales = _run(simt, x)
    want_codes, want_scales = fp8_quant.quant_groups(x)
    np.testing.assert_array_equal(scales.view(np.uint32), want_scales.view(np.uint32))
    np.testing.assert_array_equal(codes, want_codes)
    error = np.abs(fp8_quant.dequant_groups(codes, scales) - x)
    assert (error <= np.repeat(scales.T, 128, 1) * 16).all()  # half of e4m3's widest step, 32 at 448


def test_widths_that_are_not_whole_groups_write_nothing(simt):
    codes, scales = _run(simt, np.ones((1, 200), np.float32)[:, :128])  # a whole group runs
    assert (codes == 126).all()
    out = np.full((1, 200), 7, np.uint8)
    simt.run('glm_vx_gpu_fp8_quant_groups', out, np.zeros((2, 1), np.float32), np.ones((1, 200), np.float32), 1, 200,
             grid=1, block=128)
    assert (out == 7).all()
