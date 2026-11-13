"""The trained GLM-5.3 weights, cut to their first layers, against the official implementation (#35).

  python -m validation.real_layers fetch --output DIR [--layers 3]
  python -m validation.real_layers compare --checkpoint DIR --output RUN [--backend numpy-reference|vx]

`fetch` copies the first --layers decoder layers of zai-org/GLM-5.3 at the revision pinned in
metadata/sources.json, with the embeddings, the final norm and lm_head, out of the official shards
by HTTP range requests (validation/fetch_tensors.py does single tensors): byte for byte, FP8 codes
and their F32 128x128 scales as published. It writes them as a checkpoint glm-vx loads
(glm_vx.checkpoint.Checkpoint): the shards it read from under their own names, holding only the
kept tensors, model.safetensors.index.json, config.json cut to the kept layers (indexer_types,
mlp_layer_types, quantization_config.modules_to_not_convert; no MTP layer), tokenizer.json, and
fetch.json with each file's sha256. The 3 dense layers (0-2) are 5.2 GB; layer 3, the first MoE
layer, adds 9.7 GB of FP8 and needs about 40 GB in float32 for the official model.

`compare` runs the official GlmMoeDsaForCausalLM (Transformers at the pinned revision, eager, float32,
CPU) and glm-vx's model on the same prompts, and checks what validation/gpu_vs_official.py checks on
random weights: logits within atol/rtol at every position, the same top-1 at every position, and the
same greedy tokens. The official model's weights are decoded by torch (float8_e4m3fn codes times their
block scale, bfloat16 widened), not by glm-vx's decoder, so a wrong decode or scale index fails here.
The prompts are short, so the indexer keeps every token (index_topk is 2048): this checks trained
values through every Linear, norm, RoPE and the MLA attention, not the sparse selection.

This is not a speed claim and not a quality claim: a model cut to its first layers is not GLM-5.3, so
its tokens mean nothing. It shows that glm-vx computes what the official code computes on the
trained weights, whose scales and outliers random weights do not have.
"""
import argparse
import hashlib
import json
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ['The capital of France is Paris. The capital of Germany is',
           'def fibonacci(n):\n    if n < 2:\n        return n\n    return']


def pinned():
    sources = json.loads((ROOT / 'metadata/sources.json').read_text())
    return sources['model'], sources['revision']


def shard_url(name):
    model, revision = pinned()
    return f'https://huggingface.co/{model}/resolve/{revision}/{name}'


def _range(url, start, end, tries=5):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != end - start:
                raise RuntimeError(f'range {start}-{end} of {url} returned {len(data)} bytes')
            return data
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def kept(name, layers):
    if name in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'):
        return True
    m = re.match(r'model\.layers\.(\d+)\.', name)
    return m is not None and int(m.group(1)) < layers


def cut_config(config, layers):
    """The real config with only the first `layers` decoder layers and no MTP layer."""
    out = dict(config, num_hidden_layers=layers, num_nextn_predict_layers=0,
               indexer_types=config['indexer_types'][:layers], mlp_layer_types=config['mlp_layer_types'][:layers])
    q = out.get('quantization_config')
    if q and q.get('modules_to_not_convert'):
        def keep(module):
            m = re.match(r'model\.layers\.(\d+)\.', module)
            return m is None or int(m.group(1)) < layers
        out['quantization_config'] = dict(q, modules_to_not_convert=[m for m in q['modules_to_not_convert'] if keep(m)])
    return out


def coalesce(spans, gap):
    """[(start, end, name)] sorted by start -> [(start, end, [(start, end, name)])], joining spans
    less than `gap` bytes apart into one request."""
    groups = []
    for span in sorted(spans):
        if groups and span[0] - groups[-1][1] <= gap:
            groups[-1][1] = max(groups[-1][1], span[1])
            groups[-1][2].append(span)
        else:
            groups.append([span[0], span[1], [span]])
    return groups


def fetch(output, layers, gap=4 << 20, log=print):
    output = Path(output)
    if output.exists():
        raise SystemExit(f'{output} exists; preserve it or remove it first')
    index = json.loads((ROOT / 'metadata/model.safetensors.index.json').read_text())
    weight_map = {n: s for n, s in index['weight_map'].items() if kept(n, layers)}
    config = cut_config(json.loads((ROOT / 'metadata/config.json').read_text()), layers)
    from glm_vx.config import GLMConfig
    expected = GLMConfig.from_dict(config).tensor_shapes()
    missing = sorted(set(expected) - set(weight_map))
    if missing:
        raise SystemExit(f'the official index lacks {len(missing)} tensors, e.g. {missing[:3]}')
    output.mkdir(parents=True)
    shards = sorted(set(weight_map.values()))
    files, total, started = {}, 0, time.time()
    for number, shard in enumerate(shards, 1):
        url = shard_url(shard)
        length = struct.unpack('<Q', _range(url, 0, 8))[0]
        header = json.loads(_range(url, 8, 8 + length))
        base = 8 + length
        names = sorted(n for n, s in weight_map.items() if s == shard)
        spans = [(base + header[n]['data_offsets'][0], base + header[n]['data_offsets'][1], n) for n in names]
        data = {}
        for start, end, members in coalesce(spans, gap):
            block = _range(url, start, end)
            for s, e, n in members:
                data[n] = block[s - start:e - start]
        out_header, cursor = {}, 0
        for n in names:
            size = len(data[n])
            out_header[n] = {'dtype': header[n]['dtype'], 'shape': header[n]['shape'], 'data_offsets': [cursor, cursor + size]}
            cursor += size
        out_header['__metadata__'] = {'source': url}
        encoded = json.dumps(out_header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        digest = hashlib.sha256()
        with open(output / shard, 'wb') as f:
            for piece in [struct.pack('<Q', len(encoded)), encoded] + [data[n] for n in names]:
                f.write(piece)
                digest.update(piece)
        files[shard] = dict(sha256=digest.hexdigest(), bytes=8 + len(encoded) + cursor, tensors=len(names), source=url)
        total += cursor
        log(f'[{number}/{len(shards)}] {shard}: {len(names)} tensors, {cursor / 1e9:.2f} GB '
            f'({total / 1e9:.2f} GB in {time.time() - started:.0f} s)')
    tokenizer = urllib.request.urlopen(shard_url('tokenizer.json'), timeout=120).read()
    (output / 'tokenizer.json').write_bytes(tokenizer)
    files['tokenizer.json'] = dict(sha256=hashlib.sha256(tokenizer).hexdigest(), bytes=len(tokenizer),
                                   source=shard_url('tokenizer.json'))
    (output / 'model.safetensors.index.json').write_text(json.dumps(
        dict(metadata=dict(total_size=total), weight_map=dict(sorted(weight_map.items()))), indent=1) + '\n')
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model, revision = pinned()
    (output / 'fetch.json').write_text(json.dumps(dict(model=model, revision=revision, layers=layers, tensor_bytes=total,
                                                       files=files), indent=2) + '\n')
    return total


def official_config(config):
    out = {k: v for k, v in config.items()
           if k not in ('architectures', 'quantization_config', 'dtype', 'transformers_version', 'use_cache')}
    return dict(out, attention_dropout=0.0)


def torch_weight(checkpoint, name):
    """The float32 tensor `name` decoded by torch from the checkpoint's raw bytes."""
    import torch
    info = checkpoint.info(name)
    raw = np.ascontiguousarray(checkpoint.raw(name))
    if info.dtype in ('F8_E4M3', 'F8_E4M3FN'):
        codes = torch.from_numpy(raw.view(np.uint8).reshape(info.shape)).view(torch.float8_e4m3fn).float()
        scale_name = name[:-len('weight')] + 'weight_scale_inv'
        scales = torch.from_numpy(np.ascontiguousarray(checkpoint.raw(scale_name)).view(np.float32)
                                  .reshape(checkpoint.info(scale_name).shape).copy())
        rows, cols = info.shape
        block = checkpoint.block_size
        full = scales.repeat_interleave(block[0], 0)[:rows].repeat_interleave(block[1], 1)[:, :cols]
        return codes * full
    if info.dtype == 'BF16':
        return torch.from_numpy(raw.view(np.uint16).reshape(info.shape).astype(np.int16)).view(torch.bfloat16).float()
    if info.dtype == 'F32':
        return torch.from_numpy(raw.view(np.float32).reshape(info.shape).copy())
    raise SystemExit(f'{name}: dtype {info.dtype} not handled')


def official_model(config, checkpoint):
    import torch
    from transformers.models.glm_moe_dsa import configuration_glm_moe_dsa as configuration
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as official
    cfg = configuration.GlmMoeDsaConfig(**official_config(config))
    cfg._attn_implementation = 'eager'
    with torch.device('meta'):
        model = official.GlmMoeDsaForCausalLM(cfg)
    state = {}
    for name in model.state_dict():
        if name.endswith('.experts.gate_up_proj'):
            p = name.removesuffix('.experts.gate_up_proj')
            state[name] = torch.stack([torch.cat([torch_weight(checkpoint, f'{p}.experts.{e}.{k}_proj.weight')
                                                  for k in ('gate', 'up')]) for e in range(config['n_routed_experts'])])
        elif name.endswith('.experts.down_proj'):
            p = name.removesuffix('.experts.down_proj')
            state[name] = torch.stack([torch_weight(checkpoint, f'{p}.experts.{e}.down_proj.weight')
                                       for e in range(config['n_routed_experts'])])
        else:
            state[name] = torch_weight(checkpoint, name)
    model.load_state_dict(state, strict=True, assign=True)
    # Built on the meta device to skip the random init; buffers outside the state dict (the RoPE
    # inv_freq) are not loaded, so their modules are built again on the CPU from the config.
    persistent = set(model.state_dict())
    for path in sorted({n.rsplit('.', 1)[0] for n, _ in model.named_buffers() if n not in persistent}):
        parent, _, leaf = path.rpartition('.')
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, leaf, type(owner.get_submodule(leaf))(config=cfg))
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise SystemExit(f'official model tensors without data: {left[:5]}')
    return model.float().eval()


def compare(args):
    import torch
    import transformers
    from tokenizers import Tokenizer
    from glm_vx.checkpoint import Checkpoint
    from glm_vx.config import GLMConfig
    from glm_vx.model import GlmMoeDsaModel
    from validation.gpu_vs_official import candidate_run, official_run
    if args.output.exists():
        raise SystemExit('output exists; preserve evidence')
    config = json.loads((args.checkpoint / 'config.json').read_text())
    validated = GLMConfig.from_dict(config)
    checkpoint = Checkpoint(args.checkpoint, block_size=validated.weight_block_size)
    checkpoint.validate_manifest(validated.tensor_shapes(), inspect_shards=True)
    if args.backend == 'vx':
        from kernels.backend import VxBackend
        backend = VxBackend()
    else:
        from glm_vx.backend import NumpyBackend
        backend = NumpyBackend()
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    torch.set_num_threads(args.threads)
    started = time.time()
    reference = official_model(config, checkpoint)
    print(f'official model loaded in {time.time() - started:.0f} s', file=sys.stderr)
    candidate = GlmMoeDsaModel(config, checkpoint, backend)
    reports, arrays = [], {}
    for number, prompt in enumerate(args.prompts):
        tokens = tokenizer.encode(prompt).ids
        ref_logits, ref_ids = official_run(reference, tokens, args.new)
        got_logits, got_ids = candidate_run(candidate, tokens, args.new)
        diff = np.abs(ref_logits - got_logits)
        top1 = bool(np.array_equal(ref_logits.argmax(-1), got_logits.argmax(-1)))
        first_mismatch = next((i for i, (a, b) in enumerate(zip(ref_ids, got_ids)) if a != b), len(ref_ids))
        ok = bool(np.allclose(got_logits, ref_logits, rtol=args.rtol, atol=args.atol)) and top1 and first_mismatch == len(ref_ids)
        reports.append(dict(prompt=prompt, tokens=len(tokens), max_abs_diff=float(diff.max()),
                            mean_abs_diff=float(diff.mean()), max_abs_logit=float(np.abs(ref_logits).max()),
                            top1_match_all_positions=top1, greedy_tokens=len(ref_ids),
                            greedy_first_mismatch=first_mismatch, ok=ok))
        arrays.update({f'prompt{number}.ref_logits': ref_logits, f'prompt{number}.cand_logits': got_logits,
                       f'prompt{number}.ref_ids': ref_ids, f'prompt{number}.cand_ids': got_ids})
        print(json.dumps({k: v for k, v in reports[-1].items() if k != 'prompt'}))
    args.output.mkdir(parents=True)
    np.savez(args.output / 'arrays.npz', **arrays)
    fetched = json.loads((args.checkpoint / 'fetch.json').read_text()) if (args.checkpoint / 'fetch.json').exists() else {}
    failed = not all(r['ok'] for r in reports)
    report = dict(status='fail' if failed else 'pass', backend=args.backend, layers=config['num_hidden_layers'],
                  model=fetched.get('model'), revision=fetched.get('revision'), atol=args.atol, rtol=args.rtol,
                  torch=torch.__version__, transformers=transformers.__version__,
                  library=str(getattr(backend, 'library_path', '')) or None,
                  scope=f"trained GLM-5.3 weights, first {config['num_hidden_layers']} layers + embeddings + head, "
                        'vs official eager F32 CPU; short prompts (the indexer keeps every token)',
                  seconds=round(time.time() - started), reports=reports)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'])))
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fetch', help='copy the first layers of the pinned checkpoint')
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--layers', type=int, default=3)
    c = sub.add_parser('compare', help='glm-vx against the official model on the fetched layers')
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--backend', choices=['numpy-reference', 'vx'], default='numpy-reference')
    c.add_argument('--prompts', nargs='+', default=PROMPTS)
    c.add_argument('--new', type=int, default=8)
    c.add_argument('--atol', type=float, default=2e-3)
    c.add_argument('--rtol', type=float, default=2e-3)
    c.add_argument('--threads', type=int, default=8)
    args = p.parse_args(argv)
    if args.command == 'fetch':
        total = fetch(args.output, args.layers)
        print(f'{total / 1e9:.2f} GB of tensors -> {args.output}')
        return 0
    return compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
