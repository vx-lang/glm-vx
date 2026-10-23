"""Host wrapper for the device stable top-k (topk.vx); contract in TOPK.md."""
import ctypes as C

from .runtime import DeviceTensor, _I32, _integer, _separate, _INT_MAX

SCRATCH_BASE = 66824
BLOCK = 256


def scratch_ints(tokens, k):
    """int32 elements of scratch the kernel needs for this (tokens, k)."""
    return SCRATCH_BASE + 2 * max(min(k, tokens), 0)


def spawn_buffers(tokens, k):
    """(shape, dtype) of gpu/spawn/topk.vx's scratch, order_a and order_b: a histogram row per
    lane, then a row per candidate of a lane's positions; and the sort's two buffers of the power
    of two at or above min(k, tokens)."""
    sort = 1 << (min(k, tokens) - 1).bit_length()
    return [((256 + -(-tokens // 256), 256), 'int32'), ((sort,), 'int64'), ((sort,), 'int64')]


class TopK:
    def __init__(self, module):
        self.context = module.context
        self._kernel = module.kernel("glm_vx_gpu_topk_stable")

    @classmethod
    def from_ptx(cls, context, ptx):
        return cls(context.load_ptx(ptx))

    def _check(self, tensor, name, dtype, minimum=1):
        if not isinstance(tensor, DeviceTensor):
            raise TypeError(f"{name} must be a DeviceTensor")
        tensor.allocation._ensure_open()
        if tensor.context is not self.context:
            raise ValueError(f"{name} belongs to a different CUDA context")
        if tensor.dtype != dtype or len(tensor.shape) != 1:
            raise ValueError(f"{name} must be a 1-D {dtype} tensor")
        if tensor.size < minimum:
            raise ValueError(f"{name} needs at least {minimum} elements")
        return tensor

    def topk_stable(self, scores, k, *, out=None, scratch=None, stream=None):
        """ids[:min(k, tokens)] = np.argsort(-scores, kind='stable')[:min(k, tokens)] on the device.
        scores float32 [tokens]; out int32 [>= min(k, tokens)] (allocated [min(k,tokens)] if None);
        scratch int32 [>= scratch_ints(tokens, k)] (allocated if None; pass one to avoid allocation)."""
        self._check(scores, "scores", "float32")
        tokens = scores.shape[0]
        _integer(tokens, "tokens", minimum=1, maximum=_INT_MAX)
        _integer(k, "k", minimum=1, maximum=_INT_MAX)
        count = min(k, tokens)
        out = self.context.tensor((count,), "int32") if out is None else self._check(out, "out", "int32", count)
        scratch = (self.context.tensor((scratch_ints(tokens, k),), "int32") if scratch is None
                   else self._check(scratch, "scratch", "int32", scratch_ints(tokens, k)))
        _separate(out, [scores, scratch])
        _separate(scratch, [scores])
        self._kernel.launch(1, BLOCK, [out, scores, scratch, _I32(tokens), _I32(k)], stream=stream)
        return out
