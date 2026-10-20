"""Launch a gpu/spawn kernel through Vx's own runtime: call the kernel's `pub fn` in the shared
library build.py --library made, and Vx's CUDA backend launches it with the shape of its
`spawn on(..) grid([..]) block(..)` region (Vx #252), on the legacy default stream.

The tensors must live in the device's primary context (CUDAContext(primary=True)), the one
Vx's runtime uses, and are passed with no copy: a rank-1 tensor is the five words of its
memref descriptor (pointer, aligned pointer, offset 0, size, stride 1), and a scalar is passed
by value, in the order of the function's parameters. The call runs with that context current,
since Vx stops on any other (Vx abi.md section 3.6).

A launch does not wait for the GPU (Vx #25); a read of a result through the same context does,
since it is a blocking copy on the legacy stream. Our launches with stream=None are ordered
with Vx's; a Stream from context.stream() is non-blocking by default and is not, so pass
nonblocking=False or synchronize at the boundary. VX_CUDA_SYNC=1 makes Vx wait after each
launch.
"""
import ctypes as C
import json
from pathlib import Path

from gpu.runtime import DeviceTensor

_SCALAR = {'i32': C.c_int32, 'i64': C.c_int64, 'f32': C.c_float}
_ELEMENT = {'f32': 'float32', 'u16': 'uint16', 'u8': 'uint8', 'i32': 'int32'}
_DESCRIPTOR = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_int64]


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
                                  for t in (_DESCRIPTOR if p['kind'] == 'memref' else [_SCALAR[p['type']]])]

    def __call__(self, **values):
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
            args += [value.pointer, value.pointer, 0, value.size, 1]
        if context is None:
            self.function(*args)
        else:
            with context._activate():
                self.function(*args)
        return self
