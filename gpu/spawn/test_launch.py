"""launch.py without a GPU: launch sizes from the manifest, evaluated as Vx does."""
import json

import pytest

from gpu.spawn.launch import SpawnKernel, descriptor_words, launch_size


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


class Shaped:
    def __init__(self, *shape):
        self.shape, self.size = shape, 1
        for d in shape:
            self.size *= d


def test_descriptor_words_are_row_major():
    assert descriptor_words({'name': 'x', 'rank': 1}, Shaped(4, 5)) == [0, 20, 1]  # rank 1: the flat view
    assert descriptor_words({'name': 'x', 'rank': 2}, Shaped(4, 5)) == [0, 4, 5, 5, 1]
    with pytest.raises(TypeError, match='rank 2'):
        descriptor_words({'name': 'x', 'rank': 2}, Shaped(20))


def kernel(launch):
    k = SpawnKernel.__new__(SpawnKernel)
    k.manifest = {'entry': 'k', 'launch': launch}
    return k


def test_shape_from_manifest():
    k = kernel({'grid': [{'computed': '((rows - 1) / 4) + 1'}], 'block': 128})
    assert k.shape({'rows': 4097}) == ((1025,), 128)
    with pytest.raises(TypeError):
        kernel(None).shape({'rows': 1})


class FakeKernel:
    def __init__(self):
        self.allowed, self.launched = [], []

    def allow_dynamic_shared(self, nbytes):
        self.allowed.append(nbytes)

    def launch(self, grid, block, args, *, shared_bytes=0, stream=None):
        self.launched.append((grid, block, shared_bytes))


class FakeContext:
    def __init__(self, kernel):
        self.kernel_ = kernel

    def load_ptx(self, path):
        return self

    def kernel(self, name):
        return self.kernel_


@pytest.mark.parametrize('nbytes', [0, 160 * 1024])
def test_dynamic_shared_bytes_from_the_manifest(tmp_path, nbytes):
    """Vx #1984 (PR #1995): shared tensors over 48 KiB are dynamic; the launch opts in and passes the size."""
    manifest = {'entry': 'k', 'params': [{'name': 'n', 'kind': 'scalar', 'type': 'i32'}],
                'launch': {'grid': [1], 'block': 128}}
    if nbytes:
        manifest['dynamic_shared_bytes'] = nbytes
    (tmp_path / 'k.json').write_text(json.dumps(manifest))
    fake = FakeKernel()
    SpawnKernel(FakeContext(fake), tmp_path / 'k.ptx').launch(n=3)
    assert fake.allowed == ([nbytes] if nbytes else [])
    assert fake.launched == [((1,), 128, nbytes)]
