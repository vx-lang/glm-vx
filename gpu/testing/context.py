"""Test-only runtime-shaped host wrapper for the same compiled GPU Vx kernels.

Never imported by the production runtime or serving code. Deliberately retains
'cpu-simt' in identity; this is not a mocked GPU performance result.
"""
import contextlib
import numpy as np
from gpu.test_simt import Simt
class Tensor:
    ctx=None
    def __init__(self,shape,dtype='float32',data=None):
        # gpu/runtime's DeviceTensor takes a dtype by name only: refuse np.uint32 and the like here too.
        if not isinstance(dtype,str):raise ValueError(f'unsupported dtype {dtype!r}: pass its name, as gpu/runtime takes it')
        self.shape=tuple(shape);self.dtype=dtype;self._closed=False;self.parent=None
        self.data=np.zeros(shape,dtype=dtype) if data is None else np.frombuffer(data,dtype=dtype).reshape(shape).copy()
    @property
    def closed(self):return self._closed or (self.parent is not None and self.parent.closed)
    @property
    def nbytes(self):return self.data.nbytes
    def view(self,shape,dtype=None,offset=0):
        assert not self.closed
        view=Tensor.__new__(Tensor);view.shape=tuple(shape);view.dtype=dtype or self.dtype;view._closed=False;view.parent=self;view.ctx=self.ctx
        size=int(np.prod(shape))*np.dtype(view.dtype).itemsize
        view.data=self.data.view(np.uint8).reshape(-1)[offset:offset+size].view(view.dtype).reshape(shape)
        return view
    def _no_capture(self,what):
        if self.ctx is not None and self.ctx.capturing:raise RuntimeError(f'cannot {what} while a stream captures a CUDA graph')
    def download(self):
        assert not self.closed
        self._no_capture('download')
        return self.data.tobytes()
    def upload(self,data):
        assert not self.closed
        self._no_capture('upload')
        raw=np.frombuffer(bytes(memoryview(data)),dtype=self.data.dtype)
        assert raw.size==self.data.size,'upload size mismatch'
        self.data[...]=raw.reshape(self.data.shape)
        return self
    def copy_from(self,source,stream=None):
        assert not self.closed and not source.closed and source.nbytes==self.nbytes
        def work():self.data.view(np.uint8).reshape(-1)[...]=source.data.view(np.uint8).reshape(-1)
        _run(stream,work)
        return self
    def download_async(self,host,*,stream,offset=0):
        """The bytes into HostBuffer host at offset, in stream order (recorded into a capture)."""
        assert not self.closed and offset+self.nbytes<=host.nbytes
        def work():host.data[offset:offset+self.nbytes]=self.data.tobytes()
        _run(stream,work)
        return self
    def close(self):self._closed=True
class HostBuffer:
    """gpu/runtime.HostBuffer's read() over a bytearray."""
    def __init__(self,nbytes):self.nbytes=nbytes;self.data=bytearray(nbytes)
    def read(self,offset=0,nbytes=None):
        nbytes=self.nbytes-offset if nbytes is None else nbytes
        assert offset+nbytes<=self.nbytes
        return bytes(self.data[offset:offset+nbytes])
class Event:
    """Work here runs when it is launched, so an event is complete when recorded."""
    def record(self,stream=None):return self
    def synchronize(self):pass
    def query(self):return True
class Graph:
    """A capture's launches, run again in order by launch(), on the tensors' contents then."""
    def __init__(self):self.work=[]
    def launch(self,stream=None):
        for run in self.work:run()
        return self
    def close(self):self.work=[]
class Stream:
    def __init__(self,ctx=None):self.ctx,self.graph=ctx,None
    def synchronize(self):
        if self.graph is not None:raise RuntimeError('cannot synchronize a stream while it captures a CUDA graph')
    @contextlib.contextmanager
    def capture(self,mode='thread_local'):
        if self.graph is not None:raise RuntimeError('stream is already capturing a CUDA graph')
        graph=self.graph=Graph();self.ctx.capturing+=1
        try:yield graph
        finally:self.graph=None;self.ctx.capturing-=1
    def run(self,work):
        """Run work now, or record it into the graph this stream is capturing."""
        if self.graph is None:work()
        else:self.graph.work.append(work)
def _run(stream,work):
    if isinstance(stream,Stream):stream.run(work)
    else:work()
class Kernel:
    # gpu/testing/simt.cpp starts an OS thread per lane per block, so its cost is threads launched:
    # matvec_rows (gpu/resident.py) runs a warp a row here, the hand matvec's thread count.
    rows_block=32
    def __init__(self,ctx,name):self.ctx,self.name=ctx,name
    def launch(self,grid,block,args,stream=None):
        unpacked=[]
        for a in args:
            if isinstance(a,Tensor):assert not a.closed;unpacked.append(a.data)
            else:unpacked.append(a.value)
        _run(stream,lambda:self.ctx.simt.run(self.name,*unpacked,grid=grid[0],block=block[0]))
class Module:
    def __init__(self,ctx):self.ctx=ctx
    def kernel(self,name):
        getattr(self.ctx.simt.lib,name)
        return Kernel(self.ctx,name)
def _offset(tensor):
    root=tensor
    while root.parent is not None:root=root.parent
    return tensor.data.ctypes.data-root.data.ctypes.data
class SpawnKernel:
    """A gpu/spawn kernel library run by Vx's host path (gpu/spawn/host.py), on these tensors."""
    def __init__(self,ctx,path):
        import json
        from pathlib import Path
        self.ctx,self.path=ctx,str(path)
        manifest=json.loads(Path(path).with_suffix('.json').read_text())
        self.entry,self.params=manifest['entry'],[p for p in manifest['params'] if p['kind']!='scalar']
        self.mutable=[p['name'] for p in manifest['params'] if p.get('mut')]
    def launch(self,grid=None,block=None,*,stream=None,**values):
        from gpu.spawn.launch import check_aligned
        # A view's offset from its root tensor stands in for its device address (allocations align to 256).
        for p in self.params:
            v=values.get(p['name'])
            if isinstance(v,Tensor):check_aligned(self.entry,p,_offset(v))
        arrays={}
        for name,v in values.items():
            if isinstance(v,Tensor):assert not v.closed;arrays[name]=v.data
            else:arrays[name]=v
        def work():
            outs=self.ctx.host()(self.path,**arrays)
            for name in self.mutable:values[name].data[...]=outs[name].reshape(values[name].data.shape)
        _run(stream,work)
class Fp8BlockMatmul:
    """gpu/cublaslt.Fp8BlockMatmul's contract ('weights_a': x and out padded to padded_m tokens, token i's
    outputs at out[i*n:(i+1)*n]; padded weight scales) as float64 NumPy on these tensors, rounded to
    float32 once: cuBLASLt's own sum order is the GPU tests' to check."""
    def __init__(self,ctx,m,n,k):self.ctx,self.m,self.n,self.k,self.padded_m=ctx,m,n,k,-(-m//4)*4
    def __call__(self,out,x_codes,x_scales,w_codes,w_scales,stream=None):
        from gpu import fp8_quant
        mp,n,k=self.padded_m,self.n,self.k
        assert out.data.size==mp*n and x_codes.data.size==mp*k and x_scales.data.size==k//128*mp
        def work():
            x=fp8_quant.dequant_groups(x_codes.data.reshape(mp,k),x_scales.data.reshape(k//128,mp))
            blocks=w_scales.data.reshape(-(-n//128),-1)[:,:k//128].astype(np.float64)
            w=fp8_quant.e4m3_values()[w_codes.data.reshape(n,k)]*np.repeat(np.repeat(blocks,128,0),128,1)[:n,:k]
            out.data[...]=(x@w.T).astype(np.float32).reshape(out.data.shape)
        _run(stream,work)
    def close(self):pass
class Fp8GroupedGemm:
    """gpu/fp8_gemm.FP8GroupedGEMM's contract (ResidentGLM._grouped_gemm): for each expert e, the grouped rows
    offsets[e] <= r < offsets[e + 1] of a (e4m3 codes, row-major per-128 scales) times expert e's codes with their
    128x128 block scales, as float64 NumPy rounded to float32 once (then to bf16 for a uint16 out); the wgmma kernel's
    own order is the GPU tests'."""
    def __call__(self,a,a_scale,b,b_scale,out,offsets,*,rows,n,k,experts,stream=None):
        from gpu import fp8_quant
        def work():
            e4m3=fp8_quant.e4m3_values()
            x=e4m3[a.data.reshape(-1)[:rows*k].reshape(rows,k)]*np.repeat(a_scale.data.reshape(-1)[:rows*(k//128)].reshape(rows,k//128),128,1)
            codes=b.data.reshape(-1)[:experts*n*k].reshape(experts,n,k)
            blocks=b_scale.data.reshape(-1)[:experts*(-(-n//128))*(-(-k//128))].reshape(experts,-(-n//128),-(-k//128)).astype(np.float64)
            o=out.data.reshape(-1)
            for e in range(experts):
                lo,hi=int(offsets[e]),int(offsets[e+1])
                if hi>lo:
                    w=e4m3[codes[e]]*np.repeat(np.repeat(blocks[e],128,0),128,1)[:n,:k]
                    v=(x[lo:hi]@w.T).astype(np.float32).reshape(-1)
                    if o.dtype==np.uint16:  # bf16 C: nearest even
                        bits=v.view(np.uint32).astype(np.uint64); v=((bits+0x7FFF+((bits>>16)&1))>>16).astype(np.uint16)
                    o[lo*n:hi*n]=v
        _run(stream,work)
class Context:
    name='cpu-simt-test-only'
    def fp8_block_matmul(self,m,n,k):return Fp8BlockMatmul(self,m,n,k)
    def fp8_grouped_gemm(self):return Fp8GroupedGemm()
    def __init__(self,*args,**kwargs):self.simt=Simt();self.tensors=[];self._host=None;self.capturing=0
    def host(self):
        if self._host is None:
            from gpu.spawn.host import Host
            self._host=Host()
        return self._host
    def spawn_kernel(self,path):return SpawnKernel(self,path)
    def load_ptx(self,path):return Module(self)
    def stream(self):return Stream(self)
    def host_buffer(self,nbytes):return HostBuffer(nbytes)
    def event(self,*,timing=True):return Event()
    def tensor(self,shape,dtype='float32',data=None):
        if self.capturing:raise RuntimeError('cannot allocate while a stream captures a CUDA graph')
        t=Tensor(shape,dtype,data);t.ctx=self;self.tensors.append(t);return t
    def close(self):
        for t in self.tensors:t.close()
        if self._host is not None:self._host.close();self._host=None
