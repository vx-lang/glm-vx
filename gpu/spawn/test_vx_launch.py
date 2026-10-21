"""vx_launch.py without a GPU: a fake kernel library and the fake CUDA driver of
gpu/test_runtime.py. Checks how VxKernel hands our stream to Vx (vx_cuda_set_stream, Vx PR
#2016) and gives it back; gpu/test_spawn_vx_graph.py runs it for real."""
import ctypes as C
import json

import pytest

from gpu.runtime import CUDAContext, CUDADriver
from gpu.spawn import vx_launch
from gpu.spawn.vx_launch import VxKernel, VxStreamUnsupported
from gpu.test_runtime import FakeLibrary

MANIFEST = {'entry': 'glm_vx_spawn_add', 'library': 'libadd.so', 'dispatch_lib': '/opt/vx/lib/libvx_cuda_dispatch.so',
            'params': [{'name': 'out', 'kind': 'memref', 'element': 'f32', 'rank': 1, 'mut': True},
                       {'name': 'a', 'kind': 'memref', 'element': 'f32', 'rank': 1, 'mut': False},
                       {'name': 'n', 'kind': 'scalar', 'type': 'i32'}]}


class FakeFunction:
    def __init__(self, log, name, fail=False):
        self.log, self.name, self.fail = log, name, fail

    def __call__(self, *args):
        self.log.append((self.name, args))
        if self.fail:
            raise OSError('the pub fn failed')


class FakeKernelLibrary:
    """lib<kernel>.so: the pub fn and, with PR #2016, vx_cuda_set_stream from its dispatch library."""
    def __init__(self, *, has_set_stream=True, entry_fails=False, current_stream=None):
        self.log = []
        self.glm_vx_spawn_add = FakeFunction(self.log, 'entry', entry_fails)
        if has_set_stream:
            self.vx_cuda_set_stream = FakeFunction(self.log, 'set_stream')
        if current_stream is not None:  # a Vx with vx_cuda_get_stream
            self.vx_cuda_get_stream = lambda: current_stream


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def make(**library_options):
        (tmp_path / 'add.json').write_text(json.dumps(MANIFEST))
        library = FakeKernelLibrary(**library_options)
        monkeypatch.setattr(vx_launch.C, 'CDLL', lambda path: library)
        driver = FakeLibrary()
        context = CUDAContext(primary=True, _driver=CUDADriver(_library=driver))
        return VxKernel(tmp_path / 'add.ptx'), library, context, driver
    return make


def stream_value(arg):
    return arg.value if isinstance(arg, C.c_void_p) else arg


def test_without_a_stream_vx_is_not_asked(setup):
    kernel, library, context, _ = setup()
    x = context.tensor((4,))
    kernel(out=x, a=x, n=4)
    assert [name for name, _ in library.log] == ['entry']


def test_a_stream_is_set_for_the_call_then_null(setup):
    kernel, library, context, driver = setup()
    x, stream = context.tensor((4,)), context.stream()
    stack = []
    kernel.function = lambda *args: (library.log.append(('entry', args)), stack.append(list(driver.stack)))
    kernel(out=x, a=x, n=4, stream=stream)
    assert [(name, stream_value(args[0])) if name == 'set_stream' else name for name, args in library.log] == \
        [('set_stream', stream.handle.value), 'entry', ('set_stream', None)]
    assert stack == [[777, context.handle.value]]  # the primary context is current for the call
    assert driver.stack == [777]


def test_the_stream_is_given_back_when_the_call_fails(setup):
    kernel, library, context, _ = setup(entry_fails=True)
    x, stream = context.tensor((4,)), context.stream()
    with pytest.raises(OSError):
        kernel(out=x, a=x, n=4, stream=stream)
    assert [name for name, _ in library.log] == ['set_stream', 'entry', 'set_stream']
    assert library.log[-1][1] == (None,)


def test_the_previous_stream_is_put_back(setup):
    kernel, library, context, _ = setup(current_stream=1234)
    x, stream = context.tensor((4,)), context.stream()
    kernel(out=x, a=x, n=4, stream=stream)
    assert [(name, stream_value(args[0])) if name == 'set_stream' else name for name, args in library.log] == \
        [('set_stream', stream.handle.value), 'entry', ('set_stream', 1234)]


def test_an_old_vx_fails_only_when_a_stream_is_asked_for(setup):
    kernel, library, context, _ = setup(has_set_stream=False)
    x, stream = context.tensor((4,)), context.stream()
    kernel(out=x, a=x, n=4)
    with pytest.raises(VxStreamUnsupported, match=r'PR #2016 \(Vx#1986\)') as caught:
        kernel(out=x, a=x, n=4, stream=stream)
    assert '/opt/vx/lib/libvx_cuda_dispatch.so' in str(caught.value)
    assert [name for name, _ in library.log] == ['entry']  # nothing was launched without the stream


def test_stream_checks(setup):
    kernel, library, context, _ = setup()
    x = context.tensor((4,))
    with pytest.raises(TypeError):
        kernel(out=x, a=x, n=4, stream=object())
    with CUDAContext(primary=True, _driver=context.driver) as other:
        with pytest.raises(ValueError, match='different CUDA context'):
            kernel(out=x, a=x, n=4, stream=other.stream())
    closed = context.stream()
    closed.close()
    with pytest.raises(RuntimeError):
        kernel(out=x, a=x, n=4, stream=closed)
    assert library.log == []
