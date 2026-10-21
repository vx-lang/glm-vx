"""Write a single-GPU GLM-5.3 proxy checkpoint that SGLang (--tp 1) and glm-vx both load (issue #5).

  python -m validation.write_proxy --output /data/glm53-proxy --tokenizer-dir GLM53_SNAPSHOT \\
      [--format fp8|bf16] [--layers N] [--seed 0] [--mtp]
  python -m sglang.launch_server --model-path /data/glm53-proxy --tp 1 --mem-fraction-static 0.85
  python -m glm_vx.server --checkpoint /data/glm53-proxy --backend vx-gpu   # or gpu.resident.ResidentGLM

The real GLM-5.3 config (model-config.json) with num_hidden_layers cut: the first 3 layers dense
(first_k_dense_replace), the real `indexer_types` / `mlp_layer_types` prefix (full indexer at layers
0-2 and every 4th from 6), every width unchanged. Random weights in the official layout: tensor
names, dtypes, shapes and FP8 scale grids are those of zai-org/GLM-5.3's index and shard headers
(metadata/model.safetensors.index.json, sample-shard-header.json; tests check them), which is also
what SGLang's DeepseekV2 loader reads. FP8 (E4M3, F32 `weight_scale_inv` per 128x128 block): every
attention/indexer/MLP/expert Linear. BF16: embeddings, lm_head, norms, indexer k_norm and
weights_proj, router `mlp.gate`, MTP eh_proj. F32: `e_score_correction_bias` (the official index's
total_size only adds up with an F32 bias and BF16 router/eh_proj; SGLang keeps it F32 too).
`quantization_config.modules_to_not_convert` is the official list restricted to the kept layers,
including its `self_attn.indexers_proj` entries (the tensor is `indexer.weights_proj`; SGLang builds
weights_proj unquantized regardless). No MTP layer unless --mtp (speculation stays off on random
weights); --mtp appends the official layer-78 tensors as layer N and sets num_nextn_predict_layers=1.
--format bf16 writes those FP8 Linears as BF16 (the FP8 proxy's dequantized values for the same seed,
rounded) with no quantization_config, since SGLang always follows quantization_config; it defaults to
5 layers = 3 dense + 2 MoE (45.7 GB: a BF16 MoE layer is 19.7 GB; 6 layers, 65.4 GB, also fits but
leaves ~6 GB). --tokenizer-dir copies the snapshot's tokenizer/chat files (SGLang and bench_serving
need them). proxy.json also lists every tensor (name, dtype, shape, scales included): SGLang does
not report a missing or misnamed tensor, it leaves it uninitialized, so check against this list.

Values: each 16M-element chunk of a tensor has its own generator seeded by (seed, sha256(name),
chunk), so output is deterministic and independent of write order and thread count. FP8 codes are random bytes mapped through a 256-entry table of
finite E4M3FN codes at normal quantiles (no per-tile loop, no NaN code); each block's scale is the
table's std scaled to 1/sqrt(fan_in) with a +-25% per-block jitter (a wrong scale index shows up).
BF16 matrices use a 65536-entry table of bf16 normal quantiles the same way. Tensors stream to
~5 GB shards, a few chunks generated ahead on threads (~1 GB/s, <1 GB RSS), hashed as written;
proxy.json records every file's sha256, the config, layer/parameter counts and bytes.

Default --layers 9 = 3 dense + 6 MoE. Bytes (FP8 = 1 B/param, scales F32 per 128x128 block):
  embed_tokens + lm_head  2 x 154880 x 6144 x 2 B (BF16)                      = 3.806 GB
  attention, every layer  q_a 2048x6144 + q_b 16384x2048 + kv_a 576x6144
                          + kv_b 28672x512 + o 6144x16384 = 164.9 M FP8 + 40 KB scales = 0.165 GB
  indexer, full layers    wq_b 4096x2048 + wk 128x6144 FP8 + weights_proj 32x6144 BF16  = 0.0096 GB
  dense MLP (layers 0-2)  3 x 12288x6144 FP8                                  = 0.227 GB
  MoE MLP                 256 experts x 3 x 2048x6144 FP8 = 9.664 GB, + shared expert 0.038 GB,
                          + scales 2.4 MB, + router 256x6144 BF16 3.1 MB      = 9.707 GB
  => dense layer (all have a full indexer) 0.401 GB, MoE layer 9.872 GB (+0.0096 at layer 6)
  9 layers: 3.806 + 3 x 0.401 + 6 x 9.872 + 0.0096 + norms = 64.25 GB (59.8 GiB) of weights.
H100 80 GB: 81559 MiB = 85.5 GB; --mem-fraction-static 0.85 gives a 72.7 GB static pool (weights +
KV), so 9 layers leave ~8 GB for the KV cache after ~0.5 GB of runtime overhead. DSA KV is
(512 + 64) latent x 2 B + 128 indexer key + scale ~ 1.3 KB per token per layer in BF16 (~0.8 KB
with SGLang's FP8 KV), ~12 KB/token for 9 layers: ~650k tokens, i.e. 32k context x 20 requests.
A 7th MoE layer (74.1 GB) does not fit the 72.7 GB pool. MTP (--mtp) adds 10.0 GB: use --layers 8.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

from glm_vx.checkpoint import FP8_E4M3FN_TABLE

ROOT = Path(__file__).resolve().parent.parent
BLOCK = 128
FP8_MAX = 448.0
SHARD_BYTES = 5 * 1000**3
CHUNK = 16 * 1024**2  # elements per generated chunk; part of the format (each chunk has its own seed)
DTYPE_BYTES = {'F8_E4M3': 1, 'BF16': 2, 'F32': 4}
TOKENIZER_FILES = ('tokenizer*', 'special_tokens_map.json', 'added_tokens.json', 'chat_template*',
                   'generation_config.json', 'vocab*', 'merges.txt', '*.model', '*.tiktoken')
DEFAULT_LAYERS = {'fp8': 9, 'bf16': 5}


def _normal_quantiles(n):
    inv = statistics.NormalDist().inv_cdf
    return np.array([inv((i + .5) / n) for i in range(n)])


def _bf16_bits(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype('<u2')


def _fp8_lut():
    """256 finite E4M3FN codes at normal quantiles, the extreme quantile at 448 like a block's amax."""
    finite = np.nonzero(np.isfinite(FP8_E4M3FN_TABLE))[0]
    values = FP8_E4M3FN_TABLE[finite].astype(np.float64)
    z = _normal_quantiles(256)
    codes = finite[np.abs(values[None, :] - (z * FP8_MAX / np.abs(z).max())[:, None]).argmin(1)].astype(np.uint8)
    return codes, float(FP8_E4M3FN_TABLE[codes].astype(np.float64).std())


FP8_LUT, FP8_LUT_STD = _fp8_lut()
_pairs = np.arange(65536)  # one lookup per random uint16 yields two codes: twice the throughput
FP8_LUT16 = (FP8_LUT[_pairs & 255].astype('<u2') | (FP8_LUT[_pairs >> 8].astype('<u2') << 8)).astype('<u2')
NORMAL_LUT = _normal_quantiles(65536).astype(np.float32)  # scaled per tensor, then rounded to bf16


# ---------------------------------------------------------------- layout
def proxy_config(layers, mtp=False, base=None, fmt='fp8'):
    """The official config cut to `layers`, keeping the real per-layer patterns and quantization
    (none for fmt='bf16': SGLang always obeys quantization_config, so BF16 needs its own directory)."""
    if fmt not in ('fp8', 'bf16'):
        raise ValueError(f'unknown format {fmt!r}')
    real = json.loads(json.dumps(base) if base is not None else (ROOT / 'model-config.json').read_text())
    full = int(real['num_hidden_layers'])
    if not 1 <= layers <= full:
        raise ValueError(f'layers must be in 1..{full}')
    q = real['quantization_config']
    kept = []
    for name in q.get('modules_to_not_convert', []):
        parts = name.split('.')
        if parts[:2] != ['model', 'layers'] or int(parts[2]) < layers:
            kept.append(name)
        elif mtp and int(parts[2]) == full:  # the MTP layer moves from index `full` to `layers`
            kept.append('.'.join(parts[:2] + [str(layers)] + parts[3:]))
    real.update(num_hidden_layers=layers, indexer_types=real['indexer_types'][:layers],
                mlp_layer_types=real['mlp_layer_types'][:layers], num_nextn_predict_layers=int(bool(mtp)),
                quantization_config=dict(q, modules_to_not_convert=kept))
    if fmt == 'bf16':
        del real['quantization_config']
    return real


def tensor_specs(config, mtp=False):
    """[(name, dtype, shape, init, fan_in)] in write order, init one of fp8/scale/fp8bf16/bf16/norm/bias/
    f32bias. Names and dtypes follow the official checkpoint; an FP8 weight is followed by its
    weight_scale_inv. Without a quantization_config those Linears are BF16 ('fp8bf16'): the same
    seed's FP8 values times their scales, rounded to bf16, so the two formats are the same model."""
    c = config
    h, heads, ql, kl = c['hidden_size'], c['num_attention_heads'], c['q_lora_rank'], c['kv_lora_rank']
    nope, rope, v = c['qk_nope_head_dim'], c['qk_rope_head_dim'], c['v_head_dim']
    quantized = 'quantization_config' in c
    specs = []

    def fp8(name, rows, cols):
        if not quantized:
            specs.append((name + '.weight', 'BF16', (rows, cols), 'fp8bf16', cols))
            return
        specs.append((name + '.weight', 'F8_E4M3', (rows, cols), 'fp8', cols))
        specs.append((name + '.weight_scale_inv', 'F32', (-(-rows // BLOCK), -(-cols // BLOCK)), 'scale', cols))

    def bf16(name, rows, cols):
        specs.append((name + '.weight', 'BF16', (rows, cols), 'bf16', cols))

    def norm(name, width, bias=False):
        specs.append((name + '.weight', 'BF16', (width,), 'norm', width))
        if bias:
            specs.append((name + '.bias', 'BF16', (width,), 'bias', width))

    def mlp(name, width):
        fp8(name + '.gate_proj', width, h)
        fp8(name + '.up_proj', width, h)
        fp8(name + '.down_proj', h, width)

    def layer(i, indexer, dense):
        p = f'model.layers.{i}'
        norm(p + '.input_layernorm', h)
        norm(p + '.post_attention_layernorm', h)
        a = p + '.self_attn'
        fp8(a + '.q_a_proj', ql, h)
        norm(a + '.q_a_layernorm', ql)
        fp8(a + '.q_b_proj', heads * (nope + rope), ql)
        fp8(a + '.kv_a_proj_with_mqa', kl + rope, h)
        norm(a + '.kv_a_layernorm', kl)
        fp8(a + '.kv_b_proj', heads * (nope + v), kl)
        fp8(a + '.o_proj', h, heads * v)
        if indexer:
            fp8(a + '.indexer.wq_b', c['index_n_heads'] * c['index_head_dim'], ql)
            fp8(a + '.indexer.wk', c['index_head_dim'], h)
            norm(a + '.indexer.k_norm', c['index_head_dim'], bias=True)
            bf16(a + '.indexer.weights_proj', c['index_n_heads'], h)
        if dense:
            mlp(p + '.mlp', c['intermediate_size'])
            return
        experts = c['n_routed_experts']
        bf16(p + '.mlp.gate', experts, h)
        specs.append((p + '.mlp.gate.e_score_correction_bias', 'F32', (experts,), 'f32bias', experts))
        for e in range(experts):
            mlp(p + f'.mlp.experts.{e}', c['moe_intermediate_size'])
        mlp(p + '.mlp.shared_experts', c['moe_intermediate_size'] * c['n_shared_experts'])

    bf16('model.embed_tokens', c['vocab_size'], h)
    n = c['num_hidden_layers']
    for i in range(n):
        layer(i, c['indexer_types'][i] == 'full', c['mlp_layer_types'][i] == 'dense')
    if mtp:  # the official MTP layer: a full-indexer MoE decoder layer plus NextN projection and norms
        layer(n, True, False)
        p = f'model.layers.{n}'
        bf16(p + '.eh_proj', h, 2 * h)
        norm(p + '.enorm', h)
        norm(p + '.hnorm', h)
        norm(p + '.shared_head.norm', h)
    norm('model.norm', h)
    bf16('lm_head', c['vocab_size'], h)
    return specs


def nbytes(spec):
    return math.prod(spec[2]) * DTYPE_BYTES[spec[1]]


# ---------------------------------------------------------------- values
def chunk_count(spec):
    return max(1, -(-math.prod(spec[2]) // CHUNK)) if spec[3] in ('fp8', 'fp8bf16', 'bf16') else 1


def tensor_chunk(spec, seed, k):
    """Bytes of chunk k (CHUNK elements) of the tensor. Every chunk has its own generator seeded by
    (seed, sha256(name), k): output does not depend on write order or thread count."""
    name, dtype, shape, init, fan_in = spec
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
    bits = np.random.PCG64(np.random.SeedSequence([seed, key, k]))
    count = min(CHUNK, math.prod(shape) - k * CHUNK)
    if init == 'fp8':  # random byte pairs -> two quantile codes; the block scale sets the std
        raw = bits.random_raw(-(-count // 8)).view('<u2')
        return FP8_LUT16[raw].view(np.uint8)[:count].tobytes()
    if init == 'fp8bf16':  # the FP8 proxy's value for this element, dequantized, in bf16
        rows, cols = shape
        codes = np.frombuffer(tensor_chunk((name, 'F8_E4M3', shape, 'fp8', fan_in), seed, k), np.uint8)
        grid = (-(-rows // BLOCK), -(-cols // BLOCK))
        scales = np.frombuffer(tensor_chunk((name + '_scale_inv', 'F32', grid, 'scale', fan_in), seed, 0),
                               '<f4').reshape(grid)
        # pad the chunk out to whole rows so the scales broadcast as a [rows, cols] grid
        first, lead = divmod(k * CHUNK, cols)
        n = -(-(lead + count) // cols)
        values = np.zeros(n * cols, np.float32)
        values[lead:lead + count] = FP8_E4M3FN_TABLE[codes]
        rows_scale = np.repeat(scales[(first + np.arange(n)) // BLOCK], BLOCK, axis=1)[:, :cols]
        return _bf16_bits((values.reshape(n, cols) * rows_scale).reshape(-1)[lead:lead + count]).tobytes()
    if init == 'bf16':  # random 16-bit index -> N(0, 1/fan_in) quantile in bf16
        lut = _bf16_bits(NORMAL_LUT * np.float32(1 / math.sqrt(fan_in)))
        return lut[bits.random_raw(-(-count // 4)).view('<u2')[:count]].tobytes()
    rng = np.random.Generator(bits)
    if init == 'scale':  # code std x scale = 1/sqrt(fan_in), +-25% per block
        return (rng.uniform(.75, 1.25, shape) / (math.sqrt(fan_in) * FP8_LUT_STD)).astype('<f4').tobytes()
    if init == 'norm':
        return _bf16_bits(rng.uniform(.7, 1.3, shape)).tobytes()
    if init == 'bias':
        return _bf16_bits(rng.normal(0, .1, shape)).tobytes()
    if init == 'f32bias':
        return rng.normal(0, .1, shape).astype('<f4').tobytes()
    raise ValueError(f'unknown init {init!r}')


# ---------------------------------------------------------------- writing
def plan_shards(specs, shard_bytes=SHARD_BYTES):
    shards, current, size = [], [], 0
    for spec in specs:
        # an FP8 weight and its scale stay in one shard
        if current and size + nbytes(spec) > shard_bytes and spec[3] != 'scale':
            shards.append(current)
            current, size = [], 0
        current.append(spec)
        size += nbytes(spec)
    if current:
        shards.append(current)
    return shards


def _write_shard(path, specs, seed, pool, window):
    header, offset = {'__metadata__': {'format': 'pt'}}, 0
    for spec in specs:
        header[spec[0]] = {'dtype': spec[1], 'shape': list(spec[2]), 'data_offsets': [offset, offset + nbytes(spec)]}
        offset += nbytes(spec)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    digest = hashlib.sha256()
    jobs = iter([(spec, k) for spec in specs for k in range(chunk_count(spec))])
    pending = collections.deque()
    with open(path, 'wb') as f:
        for part in (struct.pack('<Q', len(encoded)), encoded):
            f.write(part)
            digest.update(part)
        # Chunks are generated `window` ahead on worker threads (numpy and hashlib release the
        # GIL) and written in order: memory stays at a few chunks, not a tensor.
        for job in itertools.islice(jobs, window):
            pending.append(pool.submit(tensor_chunk, job[0], seed, job[1]))
        while pending:
            chunk = pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(tensor_chunk, job[0], seed, job[1]))
            f.write(chunk)
            digest.update(chunk)
    if path.stat().st_size != 8 + len(encoded) + offset:
        raise AssertionError(f'{path.name}: size does not match its header')
    return digest.hexdigest(), path.stat().st_size


def _file_entry(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_proxy(output, layers=None, seed=0, mtp=False, base=None, shard_bytes=SHARD_BYTES, tokenizer=None, log=None,
                workers=None, fmt='fp8'):
    """Write the proxy directory and return its manifest (also written as proxy.json)."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; preserve evidence')
    layers = layers or DEFAULT_LAYERS[fmt]
    config = proxy_config(layers, mtp, base, fmt)
    if tokenizer:
        tokenizer_files(tokenizer)  # fail before writing gigabytes
    output.mkdir(parents=True, exist_ok=True)
    specs = tensor_specs(config, mtp)
    shards = plan_shards(specs, shard_bytes)
    files, weight_map = {}, {}
    start = time.monotonic()
    workers = workers or min(8, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, chunk in enumerate(shards):
            name = f'model-{i + 1:05d}-of-{len(shards):05d}.safetensors'
            sha, size = _write_shard(output / name, chunk, seed, pool, 2 * workers)
            files[name] = {'bytes': size, 'sha256': sha}
            weight_map.update((spec[0], name) for spec in chunk)
            if log:
                log(f'{name}: {size / 1e9:.2f} GB, {time.monotonic() - start:.0f} s')
    tensor_bytes = sum(nbytes(s) for s in specs)
    (output / 'model.safetensors.index.json').write_text(
        json.dumps({'metadata': {'total_size': tensor_bytes}, 'weight_map': weight_map}, indent=2) + '\n')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ('model.safetensors.index.json', 'config.json'):
        files[name] = _file_entry(output / name)
    for source in tokenizer_files(tokenizer) if tokenizer else ():
        shutil.copyfile(source, output / source.name)
        files[source.name] = _file_entry(output / source.name)
    types = config['mlp_layer_types']
    manifest = dict(
        generator='validation/write_proxy.py', format=fmt, seed=seed, layers=layers,
        dense_layers=types.count('dense'), moe_layers=types.count('sparse'),
        full_indexer_layers=config['indexer_types'].count('full'), mtp=bool(mtp),
        parameters=sum(math.prod(s[2]) for s in specs if s[3] != 'scale'), tensor_count=len(specs),
        tensor_bytes=tensor_bytes, file_bytes=sum(f['bytes'] for f in files.values()),
        files=dict(sorted(files.items())), config=config,
        # SGLang never reports a missing or misnamed tensor (it stays uninitialized): check against this
        tensors=[[s[0], s[1], list(s[2])] for s in specs])
    (output / 'proxy.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def tokenizer_files(directory):
    """The tokenizer/chat files a GLM-5.3 snapshot ships (not its config, index or weights)."""
    found = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and any(p.match(pattern) for pattern in TOKENIZER_FILES))
    if not any(p.name == 'tokenizer.json' for p in found):
        raise FileNotFoundError(f'no tokenizer.json in {directory}')
    return found


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--format', choices=['fp8', 'bf16'], default='fp8')
    p.add_argument('--layers', type=int, help='decoder layers kept (default 9 = 3 dense + 6 MoE for fp8, '
                   '5 = 3 dense + 2 MoE for bf16)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mtp', action='store_true', help='append the MTP (NextN) layer; off by default')
    p.add_argument('--shard-gb', type=float, default=SHARD_BYTES / 1e9)
    p.add_argument('--tokenizer-dir', type=Path, help='GLM-5.3 snapshot whose tokenizer/chat files are copied in')
    p.add_argument('--workers', type=int, help='generator threads (default min(8, cpus)); output is the same')
    args = p.parse_args(argv)
    manifest = write_proxy(args.output, args.layers, args.seed, args.mtp, shard_bytes=int(args.shard_gb * 1e9),
                           tokenizer=args.tokenizer_dir, log=lambda m: print(m, file=sys.stderr, flush=True),
                           workers=args.workers, fmt=args.format)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('config', 'tensors')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
