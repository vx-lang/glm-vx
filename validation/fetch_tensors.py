"""Copy a few tensors out of a remote safetensors shard by HTTP range requests.

Writes a valid single-file ``model.safetensors`` holding only the named tensors (or every
tensor whose name starts with a given prefix), byte for byte, so real checkpoint data can be
tested without downloading a 400 GB model. Hugging Face ``resolve`` URLs support ranges.

  python -m validation.fetch_tensors URL OUT_DIR NAME_OR_PREFIX... [--max-bytes N]
"""
import argparse
import json
import struct
import urllib.request
from pathlib import Path


def _range(url, start, end):
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
    with urllib.request.urlopen(request) as response:
        data = response.read()
    if len(data) != end - start:
        raise RuntimeError(f'range {start}-{end} returned {len(data)} bytes (server ignored Range?)')
    return data


def fetch(url, out_dir, selectors, max_bytes=256 << 20):
    length = struct.unpack('<Q', _range(url, 0, 8))[0]
    header = json.loads(_range(url, 8, 8 + length))
    base = 8 + length
    header.pop('__metadata__', None)
    names = sorted(n for n in header if any(n == s or n.startswith(s) for s in selectors))
    if not names:
        raise SystemExit('no tensor matches ' + ' '.join(selectors))
    total = sum(header[n]['data_offsets'][1] - header[n]['data_offsets'][0] for n in names)
    if total > max_bytes:
        raise SystemExit(f'{len(names)} tensors are {total} bytes, over --max-bytes {max_bytes}')
    out_header, chunks, cursor = {}, [], 0
    for name in names:
        start, end = header[name]['data_offsets']
        chunks.append(_range(url, base + start, base + end))
        out_header[name] = {'dtype': header[name]['dtype'], 'shape': header[name]['shape'],
                            'data_offsets': [cursor, cursor + end - start]}
        cursor += end - start
    out_header['__metadata__'] = {'source': url}
    encoded = json.dumps(out_header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'model.safetensors', 'wb') as f:
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
        for chunk in chunks:
            f.write(chunk)
    return names, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('url')
    parser.add_argument('out_dir')
    parser.add_argument('selectors', nargs='+')
    parser.add_argument('--max-bytes', type=int, default=256 << 20)
    args = parser.parse_args()
    names, total = fetch(args.url, args.out_dir, args.selectors, args.max_bytes)
    print(f'{len(names)} tensors, {total} bytes -> {args.out_dir}/model.safetensors')


if __name__ == '__main__':
    main()
