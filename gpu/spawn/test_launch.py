"""launch.py without a GPU: launch sizes from the manifest, evaluated as Vx does."""
import json

import pytest

from gpu.spawn.launch import SpawnKernel, launch_size


@pytest.mark.parametrize('expression,values,expected', [
    ('((rows - 1) / 4) + 1', {'rows': 1}, 1),
    ('((rows - 1) / 4) + 1', {'rows': 0}, 1),        # truncation toward zero: -1 / 4 is 0
    ('((rows - 1) / 4) + 1', {'rows': 9}, 3),
    ('((n - 1) / 256) + 1', {'n': 257}, 2),
    ('heads * splits', {'heads': 64, 'splits': 3}, 192),
    ('((rank + 31) / 32) * 32', {'rank': 520}, 544),
    ('(0 - 7) % 2', {}, -1),                          # the remainder has the dividend's sign
    ('7 % (0 - 2)', {}, 1),
])
def test_computed_sizes_follow_vx_i32_arithmetic(expression, values, expected):
    assert launch_size({'computed': expression}, values) == expected


@pytest.mark.parametrize('expression,values', [
    ('rows * rows', {'rows': 1 << 16}),   # overflows i32
    ('rows / (rows - rows)', {'rows': 3}),
    ('max(rows, 1)', {'rows': 3}),
    ('-rows', {'rows': 3}),
])
def test_refuses_what_vx_would_not_compute(expression, values):
    with pytest.raises(ValueError):
        launch_size({'computed': expression}, values)


def test_numbers_and_parameters():
    assert launch_size(256, {}) == 256
    assert launch_size({'param': 'rows'}, {'rows': 7}) == 7


def kernel(launch):
    k = SpawnKernel.__new__(SpawnKernel)
    k.manifest = {'entry': 'k', 'launch': launch}
    return k


def test_shape_from_manifest():
    k = kernel({'grid': [{'computed': '((rows - 1) / 4) + 1'}], 'block': 128})
    assert k.shape({'rows': 4097}) == ((1025,), 128)
    with pytest.raises(TypeError):
        kernel(None).shape({'rows': 1})
