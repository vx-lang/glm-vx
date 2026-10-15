"""Admission verdict: does a model fit a machine, at what context, with what split?

Given a model config and a machine (GPU memory, host RAM), report fit / no-fit, the
largest context that fits, and the placement chosen, as the issue's M4 asks of
`vxc`. This is the Python reference for that computation; the numbers come from
the model's exact tensor shapes (the same enumeration the engine loads), not from
a parameter-count estimate. Pure arithmetic, no GPU needed.

Placement policy (hybrid CPU+GPU MoE):
  * GPU-resident "core": embeddings, lm_head, attention, indexers, dense MLPs,
    shared experts, routers, norms, plus the KV cache and a workspace reserve.
  * Routed experts fill the GPU's remaining memory whole layers at a time (a layer's
    experts are either all on GPU or all in host RAM); the rest live in host RAM.
  * Verdict `fits_gpu` (all on GPU), `hybrid` (some in host), or `no_fit`, with the
    binding reason (core+KV exceeds the GPU, or host RAM cannot hold the remainder).

Assumptions, all parameters of the call and none measured here: weight formats
(`bf16`, `fp8` = 1 byte + one fp32 scale per 128x128 block, `int4` = 0.5 byte + one
fp16 scale per 128-element group), KV in bf16/fp8/f32, and the unquantized set
(embeddings, lm_head, routers, norms, biases stay bf16/fp32 as in the official
FP8 checkpoint's `modules_to_not_convert`).
"""
import argparse
import json
import math
from dataclasses import dataclass, field

GIB = 1024 ** 3

WEIGHT_BYTES = {
    # bytes per element for quantized 2-D matrices; scales added separately below.
    'bf16': 2.0, 'f32': 4.0, 'fp8': 1.0, 'int4': 0.5,
}
KV_BYTES = {'bf16': 2, 'fp8': 1, 'f32': 4}

# Tensors kept in bf16 even in a quantized checkpoint (matches the official config's
# modules_to_not_convert families). Names are matched as substrings.
UNQUANTIZED = ('embed_tokens', 'lm_head', '.mlp.gate.', 'norm', 'e_score_correction_bias', 'weights_proj')


@dataclass
class Machine:
    gpu_gib: float
    host_gib: float
    gpu_reserve_gib: float = 2.0   # CUDA context, kernels, activation workspace
    host_reserve_gib: float = 16.0  # OS, Python, pinned staging
    gpu_fraction: float = 0.95      # fraction of nominal GPU memory usable
    vcpus: int = 0                  # recorded in the report; offload speed depends on it

    @property
    def gpu_bytes(self):
        return self.gpu_gib * GIB * self.gpu_fraction - self.gpu_reserve_gib * GIB

    @property
    def host_bytes(self):
        return self.host_gib * GIB - self.host_reserve_gib * GIB


@dataclass
class Verdict:
    verdict: str                      # fits_gpu | hybrid | no_fit
    reason: str
    context: int
    max_context: int                  # largest context this machine can hold (all routed experts in host RAM)
    max_context_all_on_gpu: int       # largest context with every expert on the GPU (0 if impossible)
    weight_format: str
    kv_dtype: str
    gpu_core_gib: float
    kv_gib: float
    expert_layer_gib: float
    moe_layers: int
    expert_layers_on_gpu: int
    expert_layers_on_host: int
    gpu_used_gib: float
    host_used_gib: float
    detail: dict = field(default_factory=dict)


def tensor_bytes(name, shape, weight_format):
    n = math.prod(shape)
    if weight_format == 'bf16' or len(shape) != 2 or any(u in name for u in UNQUANTIZED):
        return n * (4 if name.endswith('e_score_correction_bias') else 2)
    if weight_format == 'fp8':
        blocks = math.ceil(shape[0] / 128) * math.ceil(shape[1] / 128)
        return n * 1.0 + blocks * 4
    if weight_format == 'int4':
        return n * 0.5 + math.ceil(n / 128) * 2
    if weight_format == 'f32':
        return n * 4
    raise ValueError(f'unknown weight format {weight_format!r}')


def model_shapes(config):
    from .model import tiny_weights
    c = {k: v for k, v in config.items() if k not in ('architectures', 'quantization_config', 'dtype', 'transformers_version')}
    shapes, _, _ = tiny_weights(c, 0, parameter_limit=10 ** 13, shapes_only=True)
    return shapes


def weight_budget(config, weight_format):
    """(core_bytes, bytes_per_moe_layer_of_routed_experts, moe_layers, total_bytes)."""
    core, expert_layers, total = 0.0, {}, 0.0
    for name, shape in model_shapes(config).items():
        b = tensor_bytes(name, shape, weight_format)
        total += b
        if '.mlp.experts.' in name:
            layer = int(name.split('.layers.')[1].split('.')[0])
            expert_layers[layer] = expert_layers.get(layer, 0.0) + b
        else:
            core += b
    moe_layers = len(expert_layers)
    per_layer = max(expert_layers.values()) if expert_layers else 0.0
    return core, per_layer, moe_layers, total


def kv_bytes_per_token(config, kv_dtype):
    """MLA latent + rope key per layer, plus indexer keys on 'full' indexer layers."""
    layers = int(config['num_hidden_layers'])
    types = config.get('indexer_types') or ['full'] * layers
    full = sum(1 for t in types[:layers] if t == 'full')
    per = (layers * (int(config['kv_lora_rank']) + int(config['qk_rope_head_dim']))
           + full * int(config['index_head_dim']))
    return per * KV_BYTES[kv_dtype]


def analyze(config, machine, context, weight_format='fp8', kv_dtype='bf16'):
    core, per_layer, moe_layers, total = weight_budget(config, weight_format)
    kv_per_token = kv_bytes_per_token(config, kv_dtype)
    kv = kv_per_token * context
    gpu_free_for_experts = machine.gpu_bytes - core - kv
    all_experts = per_layer * moe_layers

    def max_context(with_experts_on_gpu):
        room = machine.gpu_bytes - core - (all_experts if with_experts_on_gpu else 0)
        return max(0, int(room // kv_per_token))

    max_ctx = max_context(False)
    # With experts on the host, host RAM must also hold what is not on the GPU.
    layers_on_gpu = max(0, min(moe_layers, int(gpu_free_for_experts // per_layer))) if per_layer else 0
    layers_on_host = moe_layers - layers_on_gpu
    host_need = layers_on_host * per_layer

    if gpu_free_for_experts < 0:
        verdict, reason = 'no_fit', (f'GPU cannot hold the non-expert weights plus KV for {context} tokens '
                                     f'({(core + kv) / GIB:.1f} GiB needed, {machine.gpu_bytes / GIB:.1f} GiB usable); '
                                     f'max context with all experts on host is {max_ctx}')
        layers_on_gpu, layers_on_host, host_need = 0, moe_layers, all_experts
    elif host_need > 0 and host_need > machine.host_bytes:
        verdict, reason = 'no_fit', (f'host RAM cannot hold {layers_on_host} expert layers '
                                     f'({host_need / GIB:.1f} GiB needed, {machine.host_bytes / GIB:.1f} GiB usable)')
    elif layers_on_host == 0:
        verdict, reason = 'fits_gpu', 'all weights and KV fit in GPU memory'
    else:
        verdict, reason = 'hybrid', (f'{layers_on_gpu}/{moe_layers} expert layers on GPU, {layers_on_host} in host RAM '
                                     f'({host_need / GIB:.1f} GiB)')
    return Verdict(
        verdict=verdict, reason=reason, context=context, max_context=max_ctx,
        max_context_all_on_gpu=max_context(True) if machine.gpu_bytes - core - all_experts > 0 else 0,
        weight_format=weight_format, kv_dtype=kv_dtype, gpu_core_gib=core / GIB, kv_gib=kv / GIB,
        expert_layer_gib=per_layer / GIB, moe_layers=moe_layers, expert_layers_on_gpu=layers_on_gpu,
        expert_layers_on_host=layers_on_host,
        gpu_used_gib=(core + kv + layers_on_gpu * per_layer) / GIB, host_used_gib=host_need / GIB,
        detail=dict(total_weights_gib=total / GIB, kv_bytes_per_token=kv_per_token,
                    machine=dict(gpu_gib=machine.gpu_gib, host_gib=machine.host_gib, vcpus=machine.vcpus,
                                 gpu_usable_gib=machine.gpu_bytes / GIB, host_usable_gib=machine.host_bytes / GIB)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='model-config.json')
    p.add_argument('--gpu-gib', type=float, required=True)
    p.add_argument('--host-gib', type=float, required=True)
    p.add_argument('--vcpus', type=int, default=0)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--weights', choices=['bf16', 'fp8', 'int4', 'f32'], default='fp8')
    p.add_argument('--kv', choices=sorted(KV_BYTES), default='bf16')
    p.add_argument('--json', action='store_true')
    args = p.parse_args()
    config = json.load(open(args.config))
    v = analyze(config, Machine(args.gpu_gib, args.host_gib, vcpus=args.vcpus), args.context, args.weights, args.kv)
    if args.json:
        print(json.dumps(v.__dict__, indent=2))
        return 0 if v.verdict != 'no_fit' else 1
    print(f'{v.verdict.upper()}: {v.reason}')
    print(f'  weights {v.weight_format}, KV {v.kv_dtype}, context {v.context}')
    print(f'  GPU: core {v.gpu_core_gib:.1f} GiB + KV {v.kv_gib:.1f} GiB + {v.expert_layers_on_gpu}/{v.moe_layers} '
          f'expert layers x {v.expert_layer_gib:.2f} GiB = {v.gpu_used_gib:.1f} GiB')
    print(f'  host: {v.expert_layers_on_host} expert layers = {v.host_used_gib:.1f} GiB')
    print(f'  max context (experts on host): {v.max_context}; with every expert on GPU: {v.max_context_all_on_gpu}')
    return 0 if v.verdict != 'no_fit' else 1


if __name__ == '__main__':
    raise SystemExit(main())
