"""Test-only: n ranks of a tensor-parallel job in one process, one thread a rank (#56).

LocalJob(n).member(rank) has gpu/collective.Collective's all_gather (and all_reduce_sum): each rank's call (run on its
stream, or recorded into the graph that stream captures, as the CPU emulator's launches are) waits
for every rank's, then copies the ranks' tensors, in rank order, into its out. run_ranks runs one
function a rank in threads and returns their results; if one raises, the others' waits are broken
so the test fails instead of hanging.
"""
import threading

import numpy as np

from gpu.testing.context import _run


class LocalJob:
    def __init__(self, ranks, timeout=600):
        self.ranks = ranks
        self.barrier = threading.Barrier(ranks, timeout=timeout)
        self.slots = [None] * ranks
        self.calls = [0] * ranks

    def member(self, rank):
        return LocalCollective(self, rank)


class LocalCollective:
    def __init__(self, job, rank):
        if not 0 <= rank < job.ranks:
            raise ValueError(f'rank {rank} is not in 0..{job.ranks}')
        self.job, self.rank, self.ranks = job, rank, job.ranks

    def all_gather(self, tensor, out, *, stream):
        if out.data.size != tensor.data.size * self.ranks:
            raise ValueError(f'out has {out.data.size} elements, {self.ranks} ranks of {tensor.data.size} need '
                             f'{tensor.data.size * self.ranks}')
        if str(tensor.dtype) != str(out.dtype):
            raise TypeError(f'in is {tensor.dtype}, out {out.dtype}')
        job = self.job

        def work():
            job.slots[self.rank] = tensor.data.reshape(-1).copy()
            job.calls[self.rank] += 1
            job.barrier.wait()  # every rank's tensor is in
            out.data.reshape(-1)[...] = np.concatenate(job.slots)
            job.barrier.wait()  # every rank has read them: the slots may be reused
        _run(stream, work)
        return out

    def all_reduce_sum(self, tensor, *, stream, out=None):
        """out (tensor itself by default) becomes the sum over the ranks, in rank order ((p0 + p1) + p2) ..
        in float32: NCCL's order is its own, but its result too is the same on every rank."""
        out = tensor if out is None else out
        if out.data.size != tensor.data.size:
            raise ValueError(f'out has {out.data.size} elements, in has {tensor.data.size}')
        job = self.job

        def work():
            job.slots[self.rank] = tensor.data.reshape(-1).copy()
            job.calls[self.rank] += 1
            job.barrier.wait()
            total = job.slots[0].copy()
            for part in job.slots[1:]:
                total = total + part
            out.data.reshape(-1)[...] = total
            job.barrier.wait()
        _run(stream, work)
        return out


def run_ranks(job, function):
    """function(rank) on job.ranks threads; their results by rank. Re-raises the first error."""
    results, errors = [None] * job.ranks, [None] * job.ranks

    def body(rank):
        try:
            results[rank] = function(rank)
        except BaseException as error:
            errors[rank] = error
            job.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,), name=f'tp-rank-{rank}') for rank in range(job.ranks)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # A broken barrier is the others' symptom of the one rank that raised: report that one.
    first = next((e for e in errors if e is not None and not isinstance(e, threading.BrokenBarrierError)), None)
    first = first or next((e for e in errors if e is not None), None)
    if first is not None:
        raise first
    return results


class DeviceJob(LocalJob):
    """LocalJob for ranks that are threads sharing one CUDAContext on a real GPU (gpu.runtime tensors):
    all_gather synchronizes the rank's stream, waits for every rank, then copies each rank's tensor
    into its out with device copies. Eager only: a stream that captures a graph cannot synchronize.
    The one-GPU reference of validation/tp_decode_check.py."""

    def member(self, rank):
        return DeviceCollective(self, rank)


class DeviceCollective(LocalCollective):
    def all_gather(self, tensor, out, *, stream):
        size = lambda t: int(np.prod(t.shape))
        if size(out) != size(tensor) * self.ranks:
            raise ValueError(f'out has {size(out)} elements, {self.ranks} ranks of {size(tensor)} need '
                             f'{size(tensor) * self.ranks}')
        job = self.job
        stream.synchronize()
        job.slots[self.rank] = tensor
        job.calls[self.rank] += 1
        job.barrier.wait()
        for r, source in enumerate(job.slots):
            out.view((size(source),), offset=r * source.nbytes).copy_from(source, stream)
        stream.synchronize()
        job.barrier.wait()
        return out
