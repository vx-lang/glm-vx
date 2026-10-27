"""gpu/collective.py without a GPU: a fake Vx dispatch library and the fake CUDA driver of
gpu/test_runtime.py. Checks the calls Collective makes (Vx PR #2306), the stream around each,
and the argument checks."""
import ctypes as C

import pytest

from gpu.collective import Collective, CollectiveUnsupported
from gpu.runtime import CUDAContext, CUDADriver
from gpu.spawn.test_vx_launch import FakeFunction, stream_value
from gpu.test_runtime import FakeLibrary

NAMES = ['vx_collective_init', 'vx_collective_finish', 'vx_collective_all_reduce_sum_f32',
         'vx_collective_all_reduce_sum_bf16', 'vx_collective_all_gather_f32', 'vx_collective_all_gather_bf16']


class FakeDispatch:
    def __init__(self, *, collectives=True):
        self.log = []
        self.vx_cuda_set_stream = FakeFunction(self.log, 'set_stream')
        if collectives:
            for name in NAMES:
                setattr(self, name, FakeFunction(self.log, name.removeprefix('vx_collective_')))


@pytest.fixture
def setup():
    def make(**options):
        library = FakeDispatch(**options)
        context = CUDAContext(primary=True, _driver=CUDADriver(_library=FakeLibrary()))
        return library, context
    return make


def names(library):
    return [name for name, _ in library.log]


def test_init_then_finish(setup):
    library, context = setup()
    with Collective(library, context, 1, 2) as collective:
        assert collective.rank == 1 and collective.ranks == 2
    assert library.log == [('init', (1, 2)), ('finish', ())]


def test_all_reduce_is_in_place_on_the_stream(setup):
    library, context = setup()
    x, stream = context.tensor((3, 4)), context.stream()
    collective = Collective(library, context, 0, 2)
    assert collective.all_reduce_sum(x, stream=stream) is x
    assert names(library) == ['init', 'set_stream', 'all_reduce_sum_f32', 'set_stream']
    assert stream_value(library.log[1][1][0]) == stream.handle.value
    assert library.log[2][1] == (x.pointer, x.pointer, 12)
    assert library.log[3][1] == (None,)


def test_bf16_and_all_gather(setup):
    library, context = setup()
    x, out, stream = context.tensor((4,), 'uint16'), context.tensor((8,), 'uint16'), context.stream()
    Collective(library, context, 0, 2).all_gather(x, out, stream=stream)
    assert library.log[2] == ('all_gather_bf16', (x.pointer, out.pointer, 4))


def test_wrong_arguments_are_refused_before_vx(setup):
    library, context = setup()
    stream = context.stream()
    collective = Collective(library, context, 0, 2)
    with pytest.raises(ValueError):
        collective.all_gather(context.tensor((4,)), context.tensor((4,)), stream=stream)
    with pytest.raises(TypeError):
        collective.all_reduce_sum(context.tensor((4,), 'int32'), stream=stream)
    with pytest.raises(TypeError):
        collective.all_reduce_sum(context.tensor((4,)), out=context.tensor((4,), 'uint16'), stream=stream)
    with pytest.raises(TypeError):
        collective.all_reduce_sum(context.tensor((4,)), stream=None)
    assert names(library) == ['init']
    with pytest.raises(ValueError):
        Collective(library, context, 2, 2)


def test_a_vx_without_collectives(setup):
    library, context = setup(collectives=False)
    with pytest.raises(CollectiveUnsupported):
        Collective(library, context, 0, 1)


def test_after_finish_calls_are_refused(setup):
    library, context = setup()
    collective = Collective(library, context, 0, 1)
    collective.finish()
    collective.finish()
    with pytest.raises(RuntimeError):
        collective.all_reduce_sum(context.tensor((4,)), stream=context.stream())
    assert names(library) == ['init', 'finish']
