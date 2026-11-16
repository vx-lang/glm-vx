"""One decode step of ResidentGLM at GLM-5.3 width (one dense layer, FP8 Linears, bf16 lm_head), graph
replay, after CONTEXT tokens, with index_topk 2048; the step between cuProfilerStart/Stop for nsys.
  nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node python step_profile.py KERNEL CONTEXT [LONG_TOPK: merge (default) or select]"""
import sys, time
import numpy as np
from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel
from gpu.resident import ResidentGLM
from validation.gpu_vs_official import full_geometry, random_weights
from validation.resident_compare import fp8_source
kernel, context = sys.argv[1], int(sys.argv[2])
long_topk = sys.argv[3] if len(sys.argv) > 3 else "merge"
config = full_geometry(1, topk=2048)
config['max_position_embeddings'] = max(1024, context + 64)
weights = random_weights(config, 1)
model = GlmMoeDsaModel(config, fp8_source(weights, bf16_head=True), NumpyBackend())
tokens = [int(t) for t in np.random.default_rng(1).integers(0, config['vocab_size'], context + 2)]
r = ResidentGLM(model, max_tokens=context + 8, device_experts=True, graph=True, fused_attention=True, mla_kernel=kernel,
                fp8_kernels='cvt', matvec_kernels='x4', norm_block=1024, long_topk=long_topk)
t0 = time.time()
for t in tokens[:context]:
    r.forward(t, logits=False)
r.ctx.synchronize()
print(f"{kernel} {long_topk}: {context} steps in {time.time() - t0:.1f} s, replays {r.replays}", flush=True)
ctx = r.ctx
for _ in range(2):
    with ctx._activate():
        ctx.driver.call('cuProfilerStart')
    r.forward(tokens[context], logits=False)
    ctx.synchronize()
    with ctx._activate():
        ctx.driver.call('cuProfilerStop')
r.close()
