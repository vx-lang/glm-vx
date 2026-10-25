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
    def close(self):self._closed=True
class Graph:
    """A capture's launches, run again in order by launch(), on the tensors' contents then."""
    def __init__(self):self.work=[]
    def launch(self,stream=None):
        for run in self.work:run()
        return self
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
class SpawnKernel:
    """A gpu/spawn kernel library run by Vx's host path (gpu/spawn/host.py), on these tensors."""
    def __init__(self,ctx,path):
        import json
        from pathlib import Path
        self.ctx,self.path=ctx,str(path)
        manifest=json.loads(Path(path).with_suffix('.json').read_text())
        self.mutable=[p['name'] for p in manifest['params'] if p.get('mut')]
    def launch(self,grid=None,block=None,*,stream=None,**values):
        arrays={}
        for name,v in values.items():
            if isinstance(v,Tensor):assert not v.closed;arrays[name]=v.data
            else:arrays[name]=v
        def work():
            outs=self.ctx.host()(self.path,**arrays)
            for name in self.mutable:values[name].data[...]=outs[name].reshape(values[name].data.shape)
        _run(stream,work)
class Context:
    name='cpu-simt-test-only'
    def __init__(self,*args,**kwargs):self.simt=Simt();self.tensors=[];self._host=None;self.capturing=0
    def host(self):
        if self._host is None:
            from gpu.spawn.host import Host
            self._host=Host()
        return self._host
    def spawn_kernel(self,path):return SpawnKernel(self,path)
    def load_ptx(self,path):return Module(self)
    def stream(self):return Stream(self)
    def tensor(self,shape,dtype='float32',data=None):
        if self.capturing:raise RuntimeError('cannot allocate while a stream captures a CUDA graph')
        t=Tensor(shape,dtype,data);t.ctx=self;self.tensors.append(t);return t
    def close(self):
        for t in self.tensors:t.close()
        if self._host is not None:self._host.close();self._host=None
