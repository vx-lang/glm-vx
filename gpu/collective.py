"""All-reduce and all-gather across GPUs through Vx's collectives (#34; Vx PR #2306, abi.md
section 3.7). Every cross-GPU call of the model goes through Collective, so a Vx-native
all-reduce can replace NCCL here alone.

A job runs one process per GPU (CUDA_VISIBLE_DEVICES=<rank>), each one rank. The functions come
from Vx's dispatch library, looked up in a kernel library's dependencies as VxKernel looks up
vx_cuda_set_stream, so they share the stream that the kernels' copy of the library sets. Each
call goes on the stream given, so a CUDA graph can record it; the Collective must be made (it
calls vx_collective_init) before the recording. With more than one rank, every rank needs the
same VX_NCCL_ID_FILE, removed before the ranks start. Vx stops the process on any error
("[Vx CUDA] FATAL:").

Tensors are float32 (the _f32 functions) or uint16 holding bf16 (_bf16), in device memory of the
primary context.
"""
import ctypes as C

from gpu.runtime import DeviceTensor, Stream
from gpu.spawn.vx_launch import on_stream

_SUFFIX = {'float32': 'f32', 'uint16': 'bf16'}


class CollectiveUnsupported(RuntimeError):
    """The Vx dispatch library has no vx_collective_* functions (a Vx build without #2306)."""


class Collective:
    def __init__(self, library, context, rank, ranks):
        """library: a ctypes handle of a kernel library (VxKernel.library) or of the dispatch
        library itself. context: the primary CUDAContext of this rank's GPU."""
        if not getattr(context, 'primary', False):
            raise ValueError('the collectives need the primary context: CUDAContext(primary=True)')
        if not 0 <= rank < ranks:
            raise ValueError(f'rank {rank} is not in 0..{ranks}')
        self.library, self.context, self.rank, self.ranks = library, context, rank, ranks
        self._functions = {}
        self._set_stream = self._function('vx_cuda_set_stream', [C.c_void_p], None)
        get_stream = getattr(library, 'vx_cuda_get_stream', None)
        if get_stream is not None:
            get_stream.argtypes, get_stream.restype = [], C.c_void_p
        self._get_stream = get_stream
        init = self._function('vx_collective_init', [C.c_int32, C.c_int32], None)
        with context._activate():
            init(rank, ranks)
        self._open = True

    def _function(self, name, argtypes, restype):
        try:
            function = getattr(self.library, name)
        except AttributeError:
            raise CollectiveUnsupported(f'the Vx dispatch library has no {name}: build vxc from Vx with '
                                        'PR #2306 (Vx #2305) and rebuild the kernel libraries') from None
        function.argtypes, function.restype = argtypes, restype
        return function

    def _call(self, operation, source, out, count, stream):
        if not self._open:
            raise RuntimeError('the collective is finished')
        for name, tensor in (('in', source), ('out', out)):
            if not isinstance(tensor, DeviceTensor):
                raise TypeError(f'{name} must be a DeviceTensor')
            if tensor.context is not self.context:
                raise ValueError(f'{name} belongs to another CUDA context')
        if str(source.dtype) != str(out.dtype) or str(source.dtype) not in _SUFFIX:
            raise TypeError(f'in and out must both be float32 or both uint16 (bf16), not {source.dtype} and {out.dtype}')
        if not isinstance(stream, Stream) or stream.context is not self.context:
            raise TypeError("stream must be a gpu.runtime.Stream of the collective's context")
        name = f'vx_collective_{operation}_{_SUFFIX[str(source.dtype)]}'
        if name not in self._functions:
            self._functions[name] = self._function(name, [C.c_void_p, C.c_void_p, C.c_int64], None)
        with self.context._activate(), on_stream(self._set_stream, self._get_stream, stream):
            self._functions[name](source.pointer, out.pointer, count)
        return out

    def all_reduce_sum(self, tensor, *, stream, out=None):
        """out (tensor itself by default) becomes the sum of tensor over the ranks."""
        out = tensor if out is None else out
        if out.size != tensor.size:
            raise ValueError(f'out has {out.size} elements, in has {tensor.size}')
        return self._call('all_reduce_sum', tensor, out, tensor.size, stream)

    def all_gather(self, tensor, out, *, stream):
        """out becomes the elements of tensor on rank 0, then rank 1, and so on."""
        if out.size != tensor.size * self.ranks:
            raise ValueError(f'out has {out.size} elements, {self.ranks} ranks of {tensor.size} need '
                             f'{tensor.size * self.ranks}')
        return self._call('all_gather', tensor, out, tensor.size, stream)

    def finish(self):
        if self._open:
            finish = self._function('vx_collective_finish', [], None)
            with self.context._activate():
                finish()
            self._open = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
