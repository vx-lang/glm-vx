import time
import numpy as np
import pytest
from glm_vx.scheduler import Scheduler,BusyError
class Model:
    config={'vocab_size':8,'max_position_embeddings':32,'eos_token_id':[]}
    def new_cache(self):return []
    def forward(self,t,c):
        c.append(t); out=np.zeros(8,dtype=np.float32); out[(sum(c)+1)%8]=10; return out

def collect(req):
    tokens=[]
    while True:
        e=req.events.get(timeout=5)
        if e['type']=='done':return tokens,e
        tokens.append(e['token_id'])

def test_concurrent_isolation_and_release():
    s=Scheduler(Model(),prefill_chunk=1)
    try:
        a=s.submit([1,2,3],4);b=s.submit([3,2,1],5)
        aa,ea=collect(a);bb,eb=collect(b)
        assert aa==collect(s.submit([1,2,3],4))[0]
        assert bb==collect(s.submit([3,2,1],5))[0]
        assert ea['reason']==eb['reason']=='length'
        assert s.status()['reserved_tokens']==0
    finally:s.close()

def test_admission_validation():
    s=Scheduler(Model(),token_budget=5)
    try:
        with pytest.raises(BusyError):s.submit([1,2],4)
        for tokens,n in [([],1),([True],1),([8],1),([1],0),([1],True)]:
            with pytest.raises(ValueError):s.submit(tokens,n)
        for temp in [float('nan'),-1,float('inf')]:
            with pytest.raises(ValueError):s.submit([1],1,temp)
        assert s.status()['reserved_tokens']==0
    finally:s.close()

def test_cancel_error_releases():
    class Slow(Model):
        def forward(self,t,c):time.sleep(.01);return super().forward(t,c)
    s=Scheduler(Slow(),prefill_chunk=1)
    try:
        r=s.submit([1]*10,10);r.cancelled.set()
        assert collect(r)[1]['reason']=='cancelled'
        assert s.status()['reserved_tokens']==0
    finally:s.close()
    class Bad(Model):
        def forward(self,t,c):raise RuntimeError('test failure')
    s=Scheduler(Bad())
    try:
        assert collect(s.submit([1],1))[1]['reason']=='error'
        assert s.status()['reserved_tokens']==0
    finally:s.close()

def test_shutdown_does_not_claim_worker_stopped():
    import threading
    entered,release=threading.Event(),threading.Event()
    class Blocked(Model):
        def forward(self,t,c):entered.set();release.wait(2);return super().forward(t,c)
    s=Scheduler(Blocked());s.submit([1],3)
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError,match='resources must remain open'):s.close(timeout=.01)
        assert s.thread.is_alive()
    finally:
        release.set();s.close()
    assert not s.thread.is_alive()
    assert s.status()['reserved_tokens']==0

class GreedyModel(Model):
    """Model with forward_greedy: the token its forward's logits argmax to, counted."""
    def __init__(self):self.greedy_calls=self.forward_calls=0
    def forward(self,t,c):
        self.forward_calls+=1; return Model.forward(self,t,c)
    def forward_greedy(self,t,c):
        self.greedy_calls+=1; return int(np.argmax(Model.forward(self,t,c)))

def test_greedy_requests_take_the_device_token():
    # temperature 0 with a model that picks on the device: forward_greedy for every forward (this model has no
    # prefill_token, so the 3 prompt tokens too, then 3 decode steps), no logits back, the same tokens as from
    # the logits; a sampled request still gets logits (3 prompt tokens and 2 decode steps for 3 tokens).
    plain,greedy=Model(),GreedyModel()
    s,g=Scheduler(plain,prefill_chunk=1),Scheduler(greedy,prefill_chunk=1)
    try:
        want=collect(s.submit([1,2,3],4))[0]
        assert collect(g.submit([1,2,3],4))[0]==want
        assert greedy.greedy_calls==6 and greedy.forward_calls==0
        collect(g.submit([1,2,3],3,temperature=0.5,seed=1))
        assert greedy.forward_calls==5 and greedy.greedy_calls==6
    finally:s.close();g.close()

class AheadModel(GreedyModel):
    """forward_greedy that would launch the next step before returning (greedy_ahead): the flags it was given."""
    greedy_ahead=True
    def __init__(self):super().__init__();self.lasts=[]
    def forward_greedy(self,t,c,*,last=False):
        self.lasts.append(last); return super().forward_greedy(t,c)

def test_greedy_ahead_is_told_when_no_step_follows():
    # last=True where no step starts from the returned token: the prompt's tokens before its last (this model
    # has no prefill_token) and the request's last token; the same tokens as without.
    plain,ahead=Model(),AheadModel()
    s,a=Scheduler(plain,prefill_chunk=1),Scheduler(ahead,prefill_chunk=1)
    try:
        assert collect(a.submit([1,2,3],4))[0]==collect(s.submit([1,2,3],4))[0]
        assert ahead.lasts==[True,True,False,False,False,True]
    finally:s.close();a.close()
