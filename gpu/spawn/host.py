"""Run a gpu/spawn kernel library on the CPU, with NumPy arrays: no GPU needed.

Without each `across`, a Vx program means the same (Vx #252), so with no CUDA device Vx's
runtime runs a region's loops in order on the host: every block, every lane loop pass by pass
(Vx's across_loops_run_in_order_on_the_cpu.vx). The race proof makes that order's results the
GPU's, except where the two targets' arithmetic differs (exp, and contraction to FMA by ptxas).

The library must not see a GPU, or it launches on it with host pointers; so each call runs in a
worker process started with CUDA_VISIBLE_DEVICES empty, one for every library (Vx main 1bda2cb7
on, #2134: each library runs its own kernel, though all of them name it vx_npu_kernel_0). A
tensor is a C-contiguous NumPy array (a rank-1 parameter takes any array as its flat view), a
scalar a Python number; the call returns the mutable tensors by name.
"""
import concurrent.futures
import ctypes as C
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

_SCALAR = {'i32': C.c_int32, 'i64': C.c_int64, 'f32': C.c_float}
_ELEMENT = {'f32': 'float32', 'u32': 'uint32', 'u16': 'uint16', 'u8': 'uint8', 'i32': 'int32', 'i64': 'int64'}
_libraries = {}


def _call_in_worker(ptx_path, values):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('host kernels run only with CUDA_VISIBLE_DEVICES empty')
    ptx_path = Path(ptx_path)
    manifest = json.loads(ptx_path.with_suffix('.json').read_text())
    if 'library' not in manifest:
        raise FileNotFoundError(f'{ptx_path} was built without --library (set VX_DISPATCH_LIB for build.sh)')
    params = manifest['params']
    missing = {p['name'] for p in params} ^ set(values)
    if missing:
        raise TypeError(f'{manifest["entry"]}: arguments do not match parameters: {sorted(missing)}')
    # The arrival count of a group's code (Vx #2188) is taken, as launch.py takes it, but it is not
    # an argument of the library's function: Vx's host side makes its own.
    params = [p for p in params if p['kind'] != 'arrivals']
    if ptx_path not in _libraries:
        _libraries[ptx_path] = C.CDLL(str(ptx_path.with_name(manifest['library'])))
    function = getattr(_libraries[ptx_path], manifest['entry'])
    function.restype = None
    args, keep, outs = [], [], {}
    for p in params:
        value = values[p['name']]
        if p['kind'] == 'scalar':
            args.append(_SCALAR[p['type']](value))
            continue
        array = np.ascontiguousarray(value)
        if str(array.dtype) != _ELEMENT[p['element']]:
            raise TypeError(f'{p["name"]} is {array.dtype}, expected {_ELEMENT[p["element"]]}')
        rank = p.get('rank', 1)
        if rank == 1:
            array = array.reshape(-1)
        elif array.ndim != rank:
            raise TypeError(f'{p["name"]} must have rank {rank}, not shape {array.shape}')
        keep.append(array)
        pointer = C.c_void_p(array.ctypes.data)
        args += [pointer, pointer, C.c_int64(0), *(C.c_int64(s) for s in array.shape),
                 *(C.c_int64(s // array.itemsize) for s in array.strides)]
        if p.get('mut'):
            outs[p['name']] = array
    function(*args)
    return outs


class Host:
    """A worker process that runs kernel libraries on the CPU; close() stops it."""

    def __init__(self):
        self._worker = None

    def _pool(self):
        if self._worker is None:
            saved = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            try:
                pool = concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn'))
                pool.submit(os.getpid).result()  # start the worker while the variable is empty
            finally:
                if saved is None:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = saved
            self._worker = pool
        return self._worker

    def __call__(self, ptx_path, **values):
        ptx_path = str(Path(ptx_path).resolve())
        return self._pool().submit(_call_in_worker, ptx_path, values).result()

    def close(self):
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
