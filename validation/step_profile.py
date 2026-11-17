"""One warm decode step of ResidentGLM at GLM-5.3's width under graph replay, after --context tokens,
between cuProfilerStart and cuProfilerStop (twice), for nsys's kernel durations:

  nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat --cuda-graph-trace=node \\
      python -m validation.step_profile --context 3001

FP8 Linears through the cvt kernels, matvec_kernels='x4', norm_block 1024, a bf16 lm_head, index_topk
2048, random weights. One dense layer by default (as evidence/h200-20261116-m to -q); --layers 2
--moe gives a dense layer then a MoE one (16 routed experts), so a layer boundary and a MoE output are
in the step for fuse_glue's fusions there. --long-topk, --no-fuse-glue and --index-kernel select
ResidentGLM's options, for an A/B in one build (#32, #39)."""
import argparse
import time

import numpy as np

from glm_vx.backend import NumpyBackend
from glm_vx.model import GlmMoeDsaModel
from gpu.resident import ResidentGLM
from validation.gpu_vs_official import full_geometry, random_weights
from validation.resident_compare import fp8_source


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--context', type=int, default=1000)
    p.add_argument('--mla-kernel', choices=['decode', 'chunks', 'heads', 'flash', 'tc'], default='tc')
    p.add_argument('--long-topk', choices=['position', 'merge', 'select', 'warp', 'ilp', 'ilp_hist'], default='position')
    p.add_argument('--fuse-glue', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--index-kernel', choices=['simt', 'tc', 'mma', 'fp8'], default='mma')
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--moe', action='store_true', help='layers past the first are MoE (16 routed experts)')
    p.add_argument('--repeats', type=int, default=2)
    args = p.parse_args()
    config = full_geometry(args.layers, topk=2048)
    config['max_position_embeddings'] = max(1024, args.context + 64)
    n = args.layers
    config['indexer_types'] = ['full'] * n
    config['mlp_layer_types'] = ['dense' if i == 0 or not args.moe else 'sparse' for i in range(n)]
    config['n_routed_experts'] = 16  # GLM-5.3's 256 would be 9.7G random parameters; a step reads 8 of them
    weights = random_weights(config, 1)
    model = GlmMoeDsaModel(config, fp8_source(weights, bf16_head=True), NumpyBackend())
    tokens = [int(t) for t in np.random.default_rng(1).integers(0, config['vocab_size'], args.context + args.repeats)]
    r = ResidentGLM(model, max_tokens=args.context + 8, device_experts=True, graph=True, fused_attention=True,
                    mla_kernel=args.mla_kernel, fp8_kernels='cvt', matvec_kernels='x4', norm_block=1024,
                    long_topk=args.long_topk, fuse_glue=args.fuse_glue, index_kernel=args.index_kernel)
    t0 = time.time()
    for t in tokens[:args.context]:
        r.forward(t, logits=False)
    r.ctx.synchronize()
    print(f'{args}: {args.context} steps in {time.time() - t0:.1f} s, replays {r.replays}', flush=True)
    ctx = r.ctx
    for t in tokens[args.context:]:
        with ctx._activate():
            ctx.driver.call('cuProfilerStart')
        r.forward(t, logits=False)
        ctx.synchronize()
        with ctx._activate():
            ctx.driver.call('cuProfilerStop')
    print('launches:', dict(sorted(r.launches.items())), flush=True)
    r.close()


if __name__ == '__main__':
    main()
