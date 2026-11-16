"""Same compiled Vx GPU kernels under test-only CPU lanes, NOT GPU validation."""
import ctypes as C
from pathlib import Path
import numpy as np
import pytest

LIB=Path(__file__).parent/'build/libsimt.so'

class Simt:
    def __init__(self):
        from gpu.testing.fingerprint import validate
        validate(Path(__file__).resolve().parents[1])
        self.lib=C.CDLL(str(LIB))
        self.entry=C.CFUNCTYPE(None,C.c_void_p)
        self.lib.simt_launch.argtypes=[self.entry,C.c_void_p,C.c_int32,C.c_int32]
        self.lib.simt_launch.restype=C.c_int32
    def run(self,name,*args,grid=1,block=32):
        values=[];types=[]
        for arg in args:
            if isinstance(arg,np.ndarray):
                assert arg.flags.c_contiguous
                values.append(C.c_void_p(arg.ctypes.data));types.append(C.c_void_p)
            elif isinstance(arg,float):values.append(C.c_float(arg));types.append(C.c_float)
            else:values.append(C.c_int32(arg));types.append(C.c_int32)
        f=getattr(self.lib,name);f.argtypes=types;f.restype=None
        errors=[]
        @self.entry
        def entry(_):
            try:f(*values)
            except BaseException as e:errors.append(e)
        assert self.lib.simt_launch(entry,None,grid,block)==0
        assert not errors

@pytest.fixture(scope='module')
def simt():return Simt()

@pytest.mark.parametrize('shape',[(7,131),(3,6144)])
def test_matvec_coalesced_tail(simt,shape):
    rng=np.random.default_rng(4);w=rng.normal(size=shape).astype('f4');x=rng.normal(size=shape[1]).astype('f4');out=np.zeros(shape[0],'f4')
    simt.run('glm_vx_gpu_matvec',out,w,x,*shape,grid=(shape[0]+1)//2,block=64)
    np.testing.assert_allclose(out,w.astype('f8')@x.astype('f8'),rtol=1e-5,atol=3e-5)

def test_residual_rmsnorm_multiwarp(simt):
    rng=np.random.default_rng(2);x=rng.normal(size=(2,67)).astype('f4');r=rng.normal(size=x.shape).astype('f4');w=rng.normal(size=67).astype('f4')
    out=np.empty_like(x);res=np.empty_like(x)
    simt.run('glm_vx_gpu_residual_rmsnorm',out,res,x,r,w,2,67,1e-5,grid=2,block=64)
    np.testing.assert_array_equal(res,x+r)
    expected=(x+r)/np.sqrt(np.mean((x+r)**2,axis=1,keepdims=True)+1e-5)*w
    np.testing.assert_allclose(out,expected,rtol=2e-6,atol=2e-6)

# --- Glue fusions (#39, ResidentGLM(fuse_glue=True)): each gives the bits of the launches it replaces.

@pytest.mark.parametrize('dim,block',[(67,32),(333,64),(512,64),(6144,1024),(2048,256)])
def test_residual_rmsnorm8_is_residual_rmsnorm_bit_for_bit(simt,dim,block):
    rng=np.random.default_rng(dim);x=rng.normal(size=dim).astype('f4');r=rng.normal(size=dim).astype('f4');w=rng.normal(size=dim).astype('f4')
    want=np.full(dim,np.nan,'f4');want_res=want.copy();got=want.copy();got_res=want.copy()
    simt.run('glm_vx_gpu_residual_rmsnorm',want,want_res,x,r,w,1,dim,1e-6,grid=1,block=block)
    simt.run('glm_vx_gpu_residual_rmsnorm8',got,got_res,x,r,w,dim,1e-6,grid=1,block=block)
    np.testing.assert_array_equal(got_res,want_res)
    np.testing.assert_array_equal(got,want)
    # add then rmsnorm (the layer boundary it replaces): the same bits.
    added=np.full(dim,np.nan,'f4');normed=added.copy()
    simt.run('glm_vx_gpu_add',added,x,r,dim,grid=(dim+255)//256,block=256)
    simt.run('glm_vx_gpu_rmsnorm',normed,added,w,dim,1e-6,grid=1,block=block)
    np.testing.assert_array_equal(got_res,added)
    np.testing.assert_array_equal(got,normed)

def test_residual_rmsnorm8_writes_nothing_past_eight_a_thread(simt):
    x=np.ones(257,'f4');out=np.full(257,np.nan,'f4');res=out.copy()
    simt.run('glm_vx_gpu_residual_rmsnorm8',out,res,x,x,x,257,1e-6,grid=1,block=32)
    assert np.isnan(out).all() and np.isnan(res).all()

@pytest.mark.parametrize('dims,block',[((2048,512),1024),((67,333),64),((1536,512),256)])
def test_rmsnorm_pair_is_two_rmsnorms_bit_for_bit(simt,dims,block):
    rng=np.random.default_rng(sum(dims));xa,xb,wa,wb=(rng.normal(size=d).astype('f4') for d in dims+dims)
    want_a=np.empty(dims[0],'f4');want_b=np.empty(dims[1],'f4');got_a=np.full(dims[0],np.nan,'f4');got_b=np.full(dims[1],np.nan,'f4')
    simt.run('glm_vx_gpu_rmsnorm',want_a,xa,wa,dims[0],1e-6,grid=1,block=block)
    simt.run('glm_vx_gpu_rmsnorm',want_b,xb,wb,dims[1],1e-6,grid=1,block=block)
    simt.run('glm_vx_gpu_rmsnorm_pair',got_a,xa,wa,dims[0],got_b,xb,wb,dims[1],1e-6,grid=2,block=block)
    np.testing.assert_array_equal(got_a,want_a)
    np.testing.assert_array_equal(got_b,want_b)

@pytest.mark.parametrize('dim,rot',[(128,64),(64,16),(40,0),(300,64)])
def test_index_key_at_is_layernorm_rope_and_store_row_bit_for_bit(simt,dim,rot):
    rng=np.random.default_rng(dim+rot);x,w,b=(rng.normal(size=dim).astype('f4') for _ in range(3))
    angle=rng.normal(size=max(rot//2,1)).astype('f4');co=np.cos(angle);si=np.sin(angle)
    step=np.array([3,4,4,1],np.int32)
    normed=np.empty(dim,'f4');row=np.empty(dim,'f4')
    simt.run('glm_vx_gpu_layernorm',normed,x,w,b,1,dim,1e-6,grid=1,block=256)
    simt.run('glm_vx_gpu_copy',row,normed,dim,grid=1,block=256)
    if rot:
        simt.run('glm_vx_gpu_rope_strided',row,normed,co,si,1,rot,rot,rot,grid=1,block=256)
    want=np.full((6,dim),np.nan,'f4');got=want.copy()
    simt.run('glm_vx_gpu_store_row',want,row,step,dim,grid=(dim+255)//256,block=256)
    simt.run('glm_vx_gpu_index_key_at',got,x,w,b,co,si,step,dim,rot,1e-6,grid=1,block=256)
    np.testing.assert_array_equal(got,want)
    assert np.isnan(got[[0,1,2,4,5]]).all()

def test_rope_and_swiglu_tail(simt):
    rng=np.random.default_rng(23);x=rng.normal(size=(3,8)).astype('f4');angle=np.array([.2,.3,.5,.7],dtype='f4');co=np.cos(angle);si=np.sin(angle);out=np.empty_like(x)
    simt.run('glm_vx_gpu_rope_glm',out,x,co,si,3,8,block=32)
    expected=np.concatenate((x[:,::2]*co-x[:,1::2]*si,x[:,1::2]*co+x[:,::2]*si),axis=1)
    np.testing.assert_allclose(out,expected,rtol=1e-6,atol=1e-6)
    gate=rng.normal(size=103).astype('f4');up=rng.normal(size=103).astype('f4');out=np.empty_like(gate)
    simt.run('glm_vx_gpu_swiglu',out,gate,up,103,grid=2,block=32)
    np.testing.assert_allclose(out,gate/(1+np.exp(-gate))*up,rtol=2e-6,atol=1e-6)

def test_fp8_coalesced_scale_boundary(simt):
    from glm_vx.checkpoint import FP8_E4M3FN_TABLE
    rng=np.random.default_rng(7);rows,cols=3,257
    w=rng.integers(0,255,(rows,cols),dtype=np.uint8);w[w==127]=126;w[w==255]=254
    scales=np.array([[.01,.02,.03]],dtype='f4');x=rng.normal(size=cols).astype('f4');out=np.empty(rows,'f4')
    simt.run('glm_vx_gpu_fp8_matvec',out,w,scales,x,rows,cols,grid=2,block=64)
    decoded=FP8_E4M3FN_TABLE[w]*scales[0,np.arange(cols)//128]
    np.testing.assert_allclose(out,decoded.astype('f8')@x.astype('f8'),rtol=2e-5,atol=3e-5)

@pytest.mark.parametrize('count,splits',[(5,3),(0,3),(5,8)])
def test_split_online_mla_matches_dense_selected_oracle(simt,count,splits):
    rng=np.random.default_rng(23);heads,rank,rope,tokens=2,48,4,7
    q=rng.normal(size=(heads,rank)).astype('f4');qr=rng.normal(size=(heads,rope)).astype('f4')
    keys=rng.normal(size=(tokens,rank)).astype('f4');kr=rng.normal(size=(tokens,rope)).astype('f4')
    selected=np.array([0,-1,2,6,99],dtype='i4')[:count].copy()
    numerator=np.empty((heads*splits,rank),'f4');stats=np.empty((heads*splits,2),'f4');out=np.empty((heads,rank),'f4');scale=.25
    simt.run('glm_vx_gpu_mla_partial',numerator,stats,q,qr,keys,kr,selected,heads,rank,rope,tokens,count,0,splits,scale,grid=heads*splits,block=64)
    simt.run('glm_vx_gpu_mla_merge',out,numerator,stats,heads,rank,splits,grid=heads,block=64)
    ids=selected[(selected>=0)&(selected<tokens)]
    expected=np.zeros_like(out)
    if len(ids):
        scores=(q.astype('f8')@keys[ids].astype('f8').T+qr.astype('f8')@kr[ids].astype('f8').T)*scale
        probs=np.exp(scores-scores.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
        expected=probs@keys[ids]
    np.testing.assert_allclose(out,expected,rtol=2e-5,atol=3e-6)


@pytest.mark.parametrize('heads,tokens,dim,block', [
    (1, 1, 1, 32), (3, 5, 31, 64), (5, 7, 32, 128),
    (3, 5, 33, 128), (4, 3, 131, 64), (64, 1, 128, 32),
    # eight heads a group (#29): GLM-5.3's 32 x 128, and groups ending past the heads
    (32, 9, 128, 128), (9, 4, 128, 64), (17, 3, 40, 96)])
def test_index_scores_all_heads_and_warp_tails(simt, heads, tokens, dim, block):
    from gpu.testing.index_reference import inputs, ordered_f32, warp_order_f32, reference_f64
    q, keys, weights = args = inputs(heads, tokens, dim)
    scale = float(np.float32(dim**-.5))
    # Sentinel guards catch writes by tail warps outside the output extent.
    guarded = np.full(tokens + 2, -98765., dtype='f4')
    out = guarded[1:-1]
    simt.run('glm_vx_gpu_index_scores', out, *args, tokens, heads, dim, scale,
             grid=(tokens + block//32 - 1)//(block//32) + 1, block=block)
    np.testing.assert_array_equal(out, warp_order_f32(*args, scale))
    np.testing.assert_allclose(out, ordered_f32(*args, scale), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(out, reference_f64(*args, scale), rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(guarded[[0, -1]], [-98765., -98765.])


@pytest.mark.parametrize('tokens,heads,dim,block', [
    (0, 3, 33, 32), (-1, 3, 33, 32), (1, 0, 33, 32),
    (1, 3, 0, 32), (1, 3, -1, 32)])
def test_index_scores_invalid_scalar_geometry_does_not_write(simt, tokens, heads, dim, block):
    out = np.array([-7654.], 'f4')
    source = np.ones((3, 33), 'f4')
    simt.run('glm_vx_gpu_index_scores', out, source, source, source,
             tokens, heads, dim, 1., block=block)
    np.testing.assert_array_equal(out, [-7654.])

# The hand matvec's order, bit for bit (gpu/test_spawn_vx_host.py's model): lane l adds columns
# l, l + 32, .. in turn, then vxk_warp_sum's tree. Its loop loads eight steps a batch (#29), so
# widths around the batch's 256 columns, and the router's and the indexer's shapes.
@pytest.mark.parametrize('shape',[(7,131),(5,256),(3,257),(2,511),(4,31),(256,6144),(32,6144)])
def test_matvec_is_its_lane_order_bitwise(simt,shape):
    from gpu.test_spawn_vx_host import strided_sums,warp_tree
    rng=np.random.default_rng(shape[0]*shape[1]);w=rng.normal(size=shape).astype('f4');x=rng.normal(size=shape[1]).astype('f4')
    out=np.zeros(shape[0],'f4')
    simt.run('glm_vx_gpu_matvec',out,w,x,*shape,grid=(shape[0]+3)//4,block=128)
    np.testing.assert_array_equal(out.view(np.uint32),warp_tree(strided_sums(w*x,32)).view(np.uint32))

# matvec_rows (gpu/matvec_rows.vx, #29) stages a row's products in shared memory and adds them in
# the hand matvec's lane order: the same bits as glm_vx_gpu_matvec, for the router's and the
# indexer's shapes, rows shorter than a lane, a staging batch or the summing batch, and 8192
# columns, its limit. One block a row; the emulator runs blocks of up to 256.
@pytest.mark.parametrize('shape,block',[((256,6144),256),((32,6144),256),((7,131),64),((3,257),32),
                                        ((2,2049),96),((4,31),32),((3,1),32),((2,8192),256),((3,0),64)])
def test_matvec_rows_is_the_hand_matvec_bitwise(simt,shape,block):
    from gpu.test_spawn_vx_host import strided_sums,warp_tree
    rng=np.random.default_rng(shape[0]*shape[1]+block);w=rng.normal(size=shape).astype('f4');x=rng.normal(size=shape[1]).astype('f4')
    w[0,:3]=-0.0
    guarded=np.full(shape[0]+2,-5.,'f4');out=guarded[1:-1]
    simt.run('glm_vx_gpu_matvec_rows',out,w,x,*shape,grid=shape[0]+1,block=block)
    hand=np.zeros(shape[0],'f4')
    simt.run('glm_vx_gpu_matvec',hand,w,x,*shape,grid=(shape[0]+3)//4,block=128)
    np.testing.assert_array_equal(out.view(np.uint32),hand.view(np.uint32))
    if shape[1]:np.testing.assert_array_equal(out.view(np.uint32),warp_tree(strided_sums(w*x,32)).view(np.uint32))
    np.testing.assert_array_equal(guarded[[0,-1]],[-5.,-5.])

def test_matvec_rows_refuses_more_columns_than_shared_memory(simt):
    out=np.full(1,-5.,'f4');w=np.ones((1,8193),'f4')
    simt.run('glm_vx_gpu_matvec_rows',out,w,w[0],1,8193,block=32)
    np.testing.assert_array_equal(out,[-5.])

# The hand rmsnorm's order, bit for bit (gpu/test_spawn_vx_host.py's rms_reference): thread t adds
# d = t, t + block, .. in turn, then vxk_block_sum. Its loops take eight steps a batch (#29), so
# rows shorter than a batch, ending inside one, and GLM-5.3's 6144.
@pytest.mark.parametrize('rows,dim,block',[(1,6144,256),(3,131,64),(2,333,96),(1,2049,256),(4,2048,128),(2,7,32)])  # the emulator runs blocks of up to 256
def test_rmsnorm_rows_is_its_order_bitwise(simt,rows,dim,block):
    from gpu.test_spawn_vx_host import rms_reference
    rng=np.random.default_rng(rows*dim+block);x=rng.normal(size=(rows,dim)).astype('f4');w=rng.normal(size=dim).astype('f4')
    out=np.zeros_like(x)
    simt.run('glm_vx_gpu_rmsnorm_rows',out,x,w,rows,dim,1e-5,grid=rows,block=block)
    np.testing.assert_array_equal(out.view(np.uint32),rms_reference(x,w,1e-5,block).view(np.uint32))
    inplace=x.copy()
    simt.run('glm_vx_gpu_rmsnorm_rows',inplace,inplace,w,rows,dim,1e-5,grid=rows,block=block)
    np.testing.assert_array_equal(inplace.view(np.uint32),out.view(np.uint32))

# The router's selection rule, lane by lane: lane l keeps the first largest score among its free
# experts l, l + 32, .., then vxk_warp_argmax's shuffle-down tree keeps the larger score and, on a
# tie, the lower id. Its scores live in registers since #29; the ids are this model's exactly,
# ties, NaN and infinities included, and the weights the normalized sigmoids.
def router_model(logits, bias, k, scale):
    with np.errstate(over='ignore', invalid='ignore'):
        sig = (np.float32(1) / (np.float32(1) + np.exp(-logits))).astype('f4')
        sig[logits < 0] = (np.exp(logits) / (np.float32(1) + np.exp(logits)))[logits < 0]
    score = (sig + bias).astype('f4'); experts = len(logits); taken = set(); ids = []
    for slot in range(k):
        lanes = []
        for lane in range(32):
            index, best = -1, np.float32(0)
            for e in range(lane, experts, 32):
                if e not in taken and (index < 0 or score[e] > best): index, best = e, score[e]
            lanes.append((best, index))
        for delta in (16, 8, 4, 2, 1):
            lanes = [(o if (o := lanes[l + delta if l + delta < 32 else l])[1] >= 0 and (
                lanes[l][1] < 0 or o[0] > lanes[l][0] or (o[0] == lanes[l][0] and o[1] < lanes[l][1]))
                else lanes[l]) for l in range(32)]
        ids.append(lanes[0][1]); taken.add(lanes[0][1])
    weights = sig[ids].astype('f8')
    return np.array(ids, 'i4'), (weights / (weights.sum() + 1e-20) * scale).astype('f4')

@pytest.mark.parametrize('experts,k,mode', [(256, 8, 'normal'), (256, 8, 'ties'), (160, 8, 'bias'),
                                            (33, 5, 'nan'), (7, 7, 'inf'), (1, 1, 'normal'), (32, 8, 'ties')])
def test_router_is_its_selection_rule(simt, experts, k, mode):
    rng = np.random.default_rng(experts * 10 + k)
    logits = rng.normal(size=experts).astype('f4'); bias = (rng.normal(size=experts) * .1).astype('f4')
    if mode == 'ties': logits = rng.integers(-2, 3, size=experts).astype('f4'); bias[:] = 0
    if mode == 'bias': logits[:] = .5; bias = rng.integers(0, 2, size=experts).astype('f4')
    if mode == 'nan': logits[rng.random(experts) < .25] = np.nan
    if mode == 'inf': logits[[0, 3, 5]] = [np.inf, -np.inf, np.inf]
    indices = np.full(8, -7, 'i4'); weights = np.full(8, -7, 'f4')
    simt.run('glm_vx_gpu_router', indices, weights, logits, bias, experts, k, 2.5, block=32)
    ids, expected = router_model(logits, bias, k, 2.5)
    np.testing.assert_array_equal(indices[:k], ids)
    np.testing.assert_allclose(weights[:k], expected, rtol=2e-6, equal_nan=True)
    assert (indices[k:] == -7).all() and (weights[k:] == -7).all()
