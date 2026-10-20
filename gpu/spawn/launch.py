"""Launch kernels built by gpu/spawn/build.py with the existing CUDA runtime (gpu/runtime.py).

The manifest (.json beside the .ptx) gives the parameter order vxc chose. A rank-1 tensor
argument expands to its memref descriptor: base and aligned pointer (both the tensor's device
pointer), offset 0, size, stride 1. Launch dimensions are ours; Vx cannot set them yet (#252).
"""
import ctypes as C
import json
from pathlib import Path

from gpu.runtime import DeviceTensor

_SCALAR = {'i32': C.c_int32, 'i64': C.c_int64, 'f32': C.c_float}
_ELEMENT = {'f32': 'float32', 'u16': 'uint16', 'u8': 'uint8', 'i32': 'int32'}


class SpawnKernel:
    def __init__(self, context, ptx_path):
        ptx_path = Path(ptx_path)
        self.manifest = json.loads(ptx_path.with_suffix('.json').read_text())
        self.params = self.manifest['params']
        self.module = context.load_ptx(ptx_path)
        self.kernel = self.module.kernel(self.manifest['entry'])

    def arguments(self, values):
        missing = {p['name'] for p in self.params} ^ set(values)
        if missing:
            raise TypeError(f'{self.manifest["entry"]}: arguments do not match parameters: {sorted(missing)}')
        args = []
        for p in self.params:
            value = values[p['name']]
            if p['kind'] == 'scalar':
                args.append(_SCALAR[p['type']](value))
                continue
            if not isinstance(value, DeviceTensor):
                raise TypeError(f'{p["name"]} must be a DeviceTensor')
            if str(value.dtype) != _ELEMENT[p['element']]:
                raise TypeError(f'{p["name"]} is {value.dtype}, expected {_ELEMENT[p["element"]]}')
            args += [value, value, C.c_uint64(0), C.c_uint64(value.size), C.c_uint64(1)]
        return args

    def launch(self, grid, block, *, stream=None, **values):
        self.kernel.launch(grid, block, self.arguments(values), stream=stream)
        return self
