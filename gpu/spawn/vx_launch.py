"""Launch a gpu/spawn kernel through Vx's own runtime: call the kernel's `pub fn` in the shared
library build.py --library made, and Vx's CUDA backend launches it with the shape that the
region's `across` loops give (Vx #252).

The tensors must live in the device's primary context (CUDAContext(primary=True)), the one
Vx's runtime uses, and are passed with no copy: a tensor is its memref descriptor (pointer,
aligned pointer, offset 0, sizes, row-major strides; launch.descriptor_words), and a scalar is passed
by value, in the order of the function's parameters. The call runs with that context current,
since Vx stops on any other (Vx abi.md section 3.6).

Streams. With stream=None, Vx launches on the legacy default stream. A launch does not wait
for the GPU (Vx #25); a read of a result through the same context does, since it is a blocking
copy on the legacy stream. Our launches with stream=None are ordered with Vx's; a Stream from
context.stream() is non-blocking by default and is not, so pass nonblocking=False or
synchronize at the boundary.

With stream=<Stream>, the call sets that stream for this thread with Vx's
`vx_cuda_set_stream` (Vx #1986; on main as #2100, was PR #2068), calls the function, and puts
back the stream that was set before (`vx_cuda_get_stream`; null, the legacy default stream, with
a Vx that lacks it) before returning, even on an error. Vx's launch is then ordered on our stream,
not on the legacy one, with our own launches on it; and a stream can capture Vx's launches into
a CUDA graph (Stream.capture()). Call each kernel once before a capture: its first call loads
the kernel image. While the stream captures, Vx stops the program with a message at anything a
graph cannot hold (a wait, a host copy, a free that needs a wait, VX_CUDA_SYNC=1,
VX_TIME_KERNEL). The symbol is looked up in the kernel library's own dependencies, so it is
the copy of the dispatch library that launches this kernel; a Vx build without #2100 (or the
old stack's #2068) lacks it, which is an error only when a stream is asked for.

VX_CUDA_SYNC=1 makes Vx wait after each launch (on our stream when one is set).
"""
import ctypes as C
import json
from pathlib import Path

from gpu.runtime import DeviceTensor, Stream
from gpu.spawn.launch import descriptor_words

_SCALAR = {'i32': C.c_int32, 'i64': C.c_int64, 'f32': C.c_float}
_ELEMENT = {'f32': 'float32', 'u16': 'uint16', 'u8': 'uint8', 'i32': 'int32'}


def _descriptor(rank):
    return [C.c_void_p, C.c_void_p] + [C.c_int64] * (1 + 2 * rank)


class VxStreamUnsupported(RuntimeError):
    """The Vx dispatch library has no vx_cuda_set_stream (a Vx build without #2100 or #2068)."""


class VxKernel:
    def __init__(self, ptx_path):
        ptx_path = Path(ptx_path)
        self.manifest = json.loads(ptx_path.with_suffix('.json').read_text())
        if 'library' not in self.manifest:
            raise FileNotFoundError(f'{ptx_path} was built without --library (set VX_DISPATCH_LIB for build.sh)')
        self.params = self.manifest['params']
        self.library = C.CDLL(str(ptx_path.with_name(self.manifest['library'])))
        self.function = getattr(self.library, self.manifest['entry'])
        self.function.restype = None
        self.function.argtypes = [t for p in self.params
                                  for t in (_descriptor(p.get('rank', 1)) if p['kind'] == 'memref' else [_SCALAR[p['type']]])]
        self._set_stream = None
        self._get_stream = False  # not looked up yet; None when Vx lacks vx_cuda_get_stream

    @property
    def set_stream(self):
        """Vx's `void vx_cuda_set_stream(CUstream)`, from the dispatch library this kernel's
        library links (dlsym on its handle searches its dependencies)."""
        if self._set_stream is None:
            try:
                function = self.library.vx_cuda_set_stream
            except AttributeError:
                raise VxStreamUnsupported(
                    f'{self.manifest["library"]}: its Vx dispatch library '
                    f'({self.manifest.get("dispatch_lib", "libvx_cuda_dispatch.so")}) has no vx_cuda_set_stream, '
                    'so Vx cannot launch on our stream. Build vxc from Vx main (#2100, Vx#1986) and rebuild '
                    'the libraries; stream=None still launches on the legacy default stream.') from None
            function.argtypes, function.restype = [C.c_void_p], None
            self._set_stream = function
        return self._set_stream

    @property
    def get_stream(self):
        """Vx's `CUstream vx_cuda_get_stream(void)`, or None for a Vx without it."""
        if self._get_stream is False:
            function = getattr(self.library, 'vx_cuda_get_stream', None)
            if function is not None:
                function.argtypes, function.restype = [], C.c_void_p
            self._get_stream = function
        return self._get_stream

    def __call__(self, *, stream=None, **values):
        missing = {p['name'] for p in self.params} ^ set(values)
        if missing:
            raise TypeError(f'{self.manifest["entry"]}: arguments do not match parameters: {sorted(missing)}')
        args, context = [], None
        for p in self.params:
            value = values[p['name']]
            if p['kind'] == 'scalar':
                args.append(value)
                continue
            if not isinstance(value, DeviceTensor):
                raise TypeError(f'{p["name"]} must be a DeviceTensor')
            if str(value.dtype) != _ELEMENT[p['element']]:
                raise TypeError(f'{p["name"]} is {value.dtype}, expected {_ELEMENT[p["element"]]}')
            if not getattr(value.context, 'primary', False):
                raise ValueError(f'{p["name"]} must live in the primary context: CUDAContext(primary=True)')
            context = value.context
            args += [value.pointer, value.pointer, *descriptor_words(p, value)]
        if stream is not None:
            if not isinstance(stream, Stream):
                raise TypeError('stream must be a gpu.runtime.Stream or None')
            if context is not None and stream.context is not context:
                raise ValueError('stream belongs to a different CUDA context than the tensors')
            context = stream.context
            set_stream = self.set_stream
        if context is None:
            self.function(*args)
            return self
        with context._activate():
            if stream is None:
                self.function(*args)
                return self
            stream._ensure_open()
            get_stream = self.get_stream
            previous = get_stream() if get_stream is not None else None
            set_stream(stream.handle)
            try:
                self.function(*args)
            finally:
                set_stream(previous)
        return self
