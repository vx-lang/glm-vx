"""Host wrappers for the INT4 and batched/grouped FP8 weight kernels (int4.vx, fp8_batched.vx).

Kept out of runtime.py so that file stays untouched; it reuses runtime.py's validation helpers.
Construct with ``QuantKernels.from_ptx(ctx, int4_ptx, fp8_batched_ptx)`` (either may be None).

INT4 layout (glm_vx/quant.py): packed uint8 [rows, ceil(cols/2)], two symmetric 4-bit codes per
byte (element 2i in the low nibble), float16 scales [rows, ceil(cols/128)].
All activations/outputs are float32. Grouped ops take stacked expert tensors of identical shape and
an int32 ``expert_ids[M]``; an id outside [0, E) yields a zero output row.
"""
import ctypes as C

from .runtime import DeviceTensor, _I32, _i32, _integer, _separate, _INT_MAX

_BLOCK = 128
_WARPS = _BLOCK // 32
_NULL = C.c_uint64(0)


class QuantKernels:
    def __init__(self, context, *, int4_module=None, fp8_module=None):
        self.context = context
        self._int4 = self._fp8 = self._int4_vec = None
        if int4_module is not None:
            self._int4 = int4_module.kernel("glm_vx_gpu_int4_matmul")
            self._int4_vec = int4_module.kernel("glm_vx_gpu_int4_matvec")
        if fp8_module is not None:
            self._fp8 = fp8_module.kernel("glm_vx_gpu_fp8_matmul")

    @classmethod
    def from_ptx(cls, context, int4_ptx=None, fp8_batched_ptx=None):
        return cls(context,
                   int4_module=None if int4_ptx is None else context.load_ptx(int4_ptx),
                   fp8_module=None if fp8_batched_ptx is None else context.load_ptx(fp8_batched_ptx))

    def _check(self, tensor, name, dtype, ndim=None, shape=None):
        if not isinstance(tensor, DeviceTensor):
            raise TypeError(f"{name} must be a DeviceTensor")
        tensor.allocation._ensure_open()
        if tensor.context is not self.context:
            raise ValueError(f"{name} belongs to a different CUDA context")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if (ndim is not None and len(tensor.shape) != ndim) or (shape is not None and tensor.shape != tuple(shape)):
            raise ValueError(f"{name} has incompatible shape {tensor.shape}")
        _integer(tensor.size, f"{name} element count", maximum=(1 << 63) - 1)
        for dim in tensor.shape:
            _integer(dim, f"{name} dimension", maximum=_INT_MAX)
        return tensor

    def _launch(self, kernel, tasks, args, stream):
        grid = (tasks + _WARPS - 1) // _WARPS
        _integer(grid * _BLOCK, "launch stride", minimum=1, maximum=(1 << 62))
        kernel.launch(grid, _BLOCK, args, stream=stream)

    def _prepare(self, out, ids, x, n, experts):
        self._check(x, "x", "float32", ndim=2)
        m, k = x.shape
        _i32(m, "M", minimum=1)
        _i32(n, "N", minimum=1)
        _i32(k, "K")
        if experts:
            self._check(ids, "expert_ids", "int32", shape=(m,))
            ids_arg = ids
        else:
            if ids is not None:
                raise ValueError("expert_ids requires stacked [E, ...] weights")
            ids_arg = _NULL
        if out is None:
            out = self.context.tensor((m, n))
        else:
            self._check(out, "out", "float32", shape=(m, n))
        return m, k, ids_arg, out

    def int4_matvec(self, packed, scales, x, *, out=None, stream=None):
        """Single vector: packed [N, ceil(K/2)] uint8, scales [N, ceil(K/128)] float16, x [K] float32."""
        if self._int4_vec is None:
            raise RuntimeError("INT4 module not loaded")
        self._check(packed, "packed", "uint8", ndim=2)
        self._check(x, "x", "float32", ndim=1)
        n, row_bytes = packed.shape
        k = x.shape[0]
        _i32(n, "N", minimum=1)
        _i32(k, "K")
        if row_bytes != (k + 1) // 2:
            raise ValueError("packed width must be ceil(K/2)")
        self._check(scales, "scales", "float16", shape=(n, (k + 127) // 128))
        out = self.context.tensor((n,)) if out is None else self._check(out, "out", "float32", shape=(n,))
        _separate(out, [packed, scales, x])
        self._launch(self._int4_vec, n, [out, packed, scales, x, _I32(n), _I32(k)], stream)
        return out

    def int4_matmul(self, packed, scales, x, *, expert_ids=None, out=None, stream=None):
        """out [M, N] = x [M, K] @ W^T. Dense: packed [N, ceil(K/2)], scales [N, G]. Grouped
        (expert_ids int32 [M] given): packed [E, N, ceil(K/2)], scales [E, N, G]."""
        if self._int4 is None:
            raise RuntimeError("INT4 module not loaded")
        grouped = expert_ids is not None
        self._check(packed, "packed", "uint8", ndim=3 if grouped else 2)
        experts = packed.shape[0] if grouped else 0
        n, row_bytes = packed.shape[-2:]
        _i32(experts, "E")
        m, k, ids_arg, out = self._prepare(out, expert_ids, x, n, experts)
        if row_bytes != (k + 1) // 2:
            raise ValueError("packed width must be ceil(K/2)")
        sshape = ((experts,) if grouped else ()) + (n, (k + 127) // 128)
        self._check(scales, "scales", "float16", shape=sshape)
        _separate(out, [packed, scales, x] + ([expert_ids] if grouped else []))
        self._launch(self._int4, m * n,
                     [out, packed, scales, x, ids_arg, _I32(m), _I32(n), _I32(k), _I32(experts)], stream)
        return out

    def fp8_matmul(self, weight, scales, x, *, expert_ids=None, out=None, stream=None):
        """out [M, N] = x [M, K] @ W^T with E4M3FN weights. Dense: weight [N, K] uint8, scales
        float32 [ceil(N/128), ceil(K/128)]. Grouped: weight [E, N, K], scales [E, ...]."""
        if self._fp8 is None:
            raise RuntimeError("batched FP8 module not loaded")
        grouped = expert_ids is not None
        self._check(weight, "weight", "uint8", ndim=3 if grouped else 2)
        experts = weight.shape[0] if grouped else 0
        n, wk = weight.shape[-2:]
        _i32(experts, "E")
        m, k, ids_arg, out = self._prepare(out, expert_ids, x, n, experts)
        if wk != k:
            raise ValueError("weight columns must equal K")
        sshape = ((experts,) if grouped else ()) + ((n + 127) // 128, (k + 127) // 128)
        self._check(scales, "scales", "float32", shape=sshape)
        _separate(out, [weight, scales, x] + ([expert_ids] if grouped else []))
        self._launch(self._fp8, m * n,
                     [out, weight, scales, x, ids_arg, _I32(m), _I32(n), _I32(k), _I32(experts)], stream)
        return out
