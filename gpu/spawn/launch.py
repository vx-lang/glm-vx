"""Launch kernels built by gpu/spawn/build.py with the existing CUDA runtime (gpu/runtime.py).

The manifest (.json beside the .ptx) gives the entry name and the parameters, which vxc takes
in the order the function declares them. Arguments are passed by name. A tensor argument
expands to its memref descriptor: base and aligned pointer (both the tensor's device pointer),
offset 0, then its sizes and row-major strides (rank 1: size, stride 1). A kernel with code after
a group of blocks takes `arrivals` too: an int32 DeviceTensor of the manifest's `elements` counts,
zeroed once before its first launch (each group's last block zeroes its count again, so launches
and graph replays reuse it; two launches in flight at once need two). Without grid and
block, a launch takes the shape that the kernel's `across` loops give in the manifest (Vx #252),
which is the shape Vx's own launch uses; each computed size is evaluated as Vx does, in i32 with
division and remainder truncating toward zero.
"""
import ast
import ctypes as C
import json
from pathlib import Path

from gpu.runtime import DeviceTensor

_SCALAR = {'i32': C.c_int32, 'i64': C.c_int64, 'f32': C.c_float}
_ELEMENT = {'f32': 'float32', 'u32': 'uint32', 'u16': 'uint16', 'u8': 'uint8', 'i32': 'int32', 'i64': 'int64'}


_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1),
        ast.Mod: lambda a, b: a - abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1) * b}


def launch_size(size, values):
    """A manifest launch size: a number, {"param": name} or {"computed": "(rows + 3) / 4"}."""
    if isinstance(size, int):
        return size
    if 'param' in size:
        return int(values[size['param']])
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            result = node.value
        elif isinstance(node, ast.Name):
            result = int(values[node.id])
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = evaluate(node.right)
            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise ValueError(f'launch size {size["computed"]!r} divides by zero')
            result = _OPS[type(node.op)](evaluate(node.left), right)
        else:
            raise ValueError(f'unsupported launch size {size["computed"]!r}')
        if not -(1 << 31) <= result < (1 << 31):
            raise ValueError(f'launch size {size["computed"]!r} overflows i32')
        return result
    return evaluate(ast.parse(size['computed'], mode='eval').body)


def descriptor_words(param, value):
    """Offset, sizes and strides of the memref descriptor for a DeviceTensor (row-major, offset 0).
    A rank-1 parameter takes any tensor as its flat view; a rank-2 one needs a rank-2 tensor."""
    rank = param.get('rank', 1)
    if rank == 1:
        return [0, value.size, 1]
    if len(value.shape) != rank:
        raise TypeError(f'{param["name"]} must have rank {rank}, not shape {tuple(value.shape)}')
    strides = [1] * rank
    for axis in range(rank - 2, -1, -1):
        strides[axis] = strides[axis + 1] * value.shape[axis + 1]
    return [0, *value.shape, *strides]


def check_aligned(entry, param, pointer):
    """A tensor whose PTX pointer is `.align N` (the machine file's HBM alignment, Vx #2375) may be
    read N bytes at a time, so a view that starts off that boundary would fault or read wrong."""
    align = param.get('align', 1)
    if pointer % align:
        raise ValueError(f'{entry}: {param["name"]} starts at {pointer:#x}, not on the {align}-byte boundary the kernel loads from')


class SpawnKernel:
    def __init__(self, context, ptx_path):
        ptx_path = Path(ptx_path)
        self.manifest = json.loads(ptx_path.with_suffix('.json').read_text())
        self.params = self.manifest['params']
        self.module = context.load_ptx(ptx_path)
        self.kernel = self.module.kernel(self.manifest['entry'])
        # Shared tensors over 48 KiB are dynamic shared memory (Vx #1984, PR #1995): opt in once.
        self.shared_bytes = self.manifest.get('dynamic_shared_bytes', 0)
        if self.shared_bytes:
            self.kernel.allow_dynamic_shared(self.shared_bytes)

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
            if p['kind'] == 'arrivals' and value.size < launch_size(p['elements'], values):
                raise ValueError(f'{p["name"]} needs {launch_size(p["elements"], values)} counts, got {value.size}')
            check_aligned(self.manifest['entry'], p, value.pointer)
            args += [value, value] + [C.c_uint64(w) for w in descriptor_words(p, value)]
        return args

    def shape(self, values):
        """(grid, block) from the manifest's launch shape for these arguments."""
        launch = self.manifest.get('launch')
        if not launch:
            raise TypeError(f'{self.manifest["entry"]} has no launch shape; pass grid and block')
        return (tuple(launch_size(s, values) for s in launch['grid']), launch_size(launch['block'], values))

    def launch(self, grid=None, block=None, *, stream=None, **values):
        if (grid is None) != (block is None):
            raise TypeError('pass both grid and block, or neither')
        if grid is None:
            grid, block = self.shape(values)
        elif 'launch' in self.manifest and block != self.manifest['launch']['block']:
            # Vx lays a block out as its manifest says (lanes, subgroups per block): another block
            # size leaves some of a block's work undone, or does it twice.
            raise ValueError(f'{self.manifest["entry"]}: block {block}, but the kernel is built for '
                             f'{self.manifest["launch"]["block"]} threads')
        self.kernel.launch(grid, block, self.arguments(values), shared_bytes=self.shared_bytes, stream=stream)
        return self
