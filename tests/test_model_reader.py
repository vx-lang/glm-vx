"""glm-vx-reader (reader/, #68) against glm_vx/checkpoint.py's own reader.

The Vx reader reads config.json, model.safetensors.index.json and each shard's header through a
`Folder`, and writes a table (docs/MODEL-TABLE.md). For a folder that both accept, the table must
give what Python's reader gives: config.json as json.loads reads it, the same weight map, the same
TensorInfo for every tensor and the same bytes. A folder that Python refuses, the Vx reader must
refuse too. The Vx reader is stricter in the ways reader/src/model.vx lists, and the cases below
check each one.

FRONT_PRELOAD runs the reader under AddressSanitizer (see tests/front_server.py), and
MODEL_READER_FUZZ sets how many mutated headers to try (1000 by default).
"""
import json
import math
import os
import random
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from glm_vx.checkpoint import CheckpointError, SafetensorsCheckpoint, SafetensorsFile
from glm_vx.config import GLMConfig
from glm_vx.model_table import open_model, parse_table
from test_checkpoint import write_bad, write_safe

ROOT = Path(__file__).resolve().parents[1]
READER = ROOT / 'reader' / 'build' / 'glm-vx-reader'
VXC = os.environ.get('VXC') or shutil.which('vxc')
ENV = {**os.environ, **({'LD_PRELOAD': os.environ['FRONT_PRELOAD'], 'ASAN_OPTIONS': 'detect_leaks=0'}
                        if os.environ.get('FRONT_PRELOAD') else {})}
pytestmark = pytest.mark.skipif(not READER.exists(), reason='needs reader/build.sh')
CONFIG = (ROOT / 'metadata' / 'config.json').read_text()  # GLM-5.3's


def run(root):
    """(returncode, table or None, stderr) of the reader on `root`."""
    out = subprocess.run([str(READER), str(root)], capture_output=True, timeout=600, env=ENV)
    return out.returncode, parse_table(out.stdout) if out.returncode == 0 else None, out.stderr.decode()


def python_reads(root):
    """Whether glm_vx/checkpoint.py reads the folder: its index, every shard and every tensor named."""
    try:
        with SafetensorsCheckpoint(root) as c:
            for name in c.weight_map:
                c.info(name)
        json.loads((Path(root) / 'config.json').read_text())
        return True
    except (CheckpointError, ValueError, OSError):
        return False


def same_as_python(root):
    code, table, err = run(root)
    assert code == 0, err
    assert table.config == json.loads((Path(root) / 'config.json').read_text())
    config, ours = open_model(root, READER)
    with ours, SafetensorsCheckpoint(root) as theirs:
        assert ours.weight_map == theirs.weight_map
        for name in theirs.weight_map:
            assert ours.info(name) == theirs.info(name), name
            a, b = ours.raw(name), theirs.raw(name)  # copies: a view would hold the mapping open
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    return table


def folder(tmp_path, shards, config=CONFIG, index=True):
    """A model folder: `shards` maps a file name to write_safe's entries."""
    (tmp_path / 'config.json').write_text(config)
    weight_map = {}
    for name, entries in shards.items():
        write_safe(tmp_path / name, entries)
        weight_map.update({e[0]: name for e in entries})
    if index:
        (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': 0},
                                                                          'weight_map': weight_map}))
    return tmp_path


DTYPES = {'F32': '<f4', 'F16': '<f2', 'BF16': '<u2', 'F8_E4M3': 'u1', 'F8_E4M3FN': 'u1', 'I32': '<i4', 'I64': '<i8'}


def entries(rng, prefix, n):
    out = []
    for i in range(n):
        dtype = rng.choice(sorted(DTYPES))
        shape = [rng.choice([0, 1, 2, 3, 5, 8]) for _ in range(rng.randint(0, 3))]
        data = rng.randbytes(math.prod(shape) * np.dtype(DTYPES[dtype]).itemsize)
        out.append((f'{prefix}.{i}.weight', dtype, shape, data))
    return out


def test_random_folders_read_alike(tmp_path):
    rng = random.Random(68)
    for k in range(20):
        root = tmp_path / str(k)
        root.mkdir()
        shards = {f'model-{s:05d}-of-00003.safetensors': entries(rng, f'layer{s}', rng.randint(1, 12)) for s in range(3)}
        same_as_python(folder(root, shards))


def test_one_file_without_an_index(tmp_path):
    rng = random.Random(1)
    table = same_as_python(folder(tmp_path, {'model.safetensors': entries(rng, 'x', 7)}, index=False))
    assert [s.name for s in table.shards] == ['model.safetensors']


ESCAPED = [chr(0xE9), chr(0x6570), chr(0x1F600), chr(0), chr(0x2028), chr(0x7F), chr(0x10FFFF)]


def test_the_config_reads_as_json_loads_reads_it(tmp_path):
    rng = random.Random(2)
    shards = {'model.safetensors': entries(rng, 'x', 2)}
    extras = [  # JSON text after GLM-5.3's own fields
        '"numbers": [0, -0, 1, -1, 9223372036854775807, -9223372036854775808, 0.0, -0.0, 1e-5, 2.5, 8e6, 1e400, -1e400,'
        ' 5e-324, 1.7976931348623157e308, 0.30000000000000004, 123456789.123456789, 1e22, 1e23, 1E2, 1.0, 0e0, -12.5e+3]',
        r'"texts": ["", "é", "数", "😀", "\u0000", "\"\\/\b\f\n\r\t", " ", "é😀"]',
        '"escaped": ' + json.dumps(ESCAPED, ensure_ascii=True),  # each one escaped, the emoji as a surrogate pair
        '"nested": [[[{"a": [null, true, false, {}, []]}]]]',
    ]
    tables = []
    for i, extra in enumerate(extras):
        root = tmp_path / str(i)
        root.mkdir()
        config = CONFIG[:CONFIG.rindex('}')] + ', ' + extra + '}'
        tables.append(same_as_python(folder(root, shards, config=config)))
        assert tables[-1].config == json.loads(config)
        key = extra[1:extra.index('"', 1)]
        assert GLMConfig.from_dict({k: v for k, v in tables[-1].config.items() if k != key}).num_hidden_layers == 78
    # -0 is an int, 1E2 and 1.0 are floats, 1e400 is inf, -0.0 keeps its sign, as in json.loads.
    numbers = tables[0].config['numbers']
    assert type(numbers[1]) is int and [type(v) for v in numbers[-4:]] == [float] * 4
    assert math.isinf(numbers[11]) and math.copysign(1, numbers[7]) == -1
    assert tables[1].config['texts'][3] == chr(0x1F600) and tables[2].config['escaped'] == ESCAPED


def test_python_refusals_are_refused(tmp_path):
    """test_checkpoint.py's malformed headers, each as the one shard of an indexed folder."""
    base = {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
    cases = [({'x': dict(base, dtype='I64')}, b'\0' * 4), ({'x': dict(base, shape=[-1])}, b'\0' * 4),
             ({'x': dict(base, shape=[True])}, b'\0' * 4), ({'x': dict(base, shape=[2])}, b'\0' * 4),
             ({'x': dict(base, shape=[1.0])}, b'\0' * 4), ({'x': dict(base, shape=[1] * 33)}, b'\0' * 4),
             ({'x': dict(base, data_offsets=[-1, 3])}, b'\0' * 4), ({'x': dict(base, data_offsets=[0, 8])}, b'\0' * 4),
             ({'x': dict(base, data_offsets=[1, 5])}, b'\0' * 5), ({'x': dict(base, data_offsets=[0])}, b'\0' * 4),
             ({'x': dict(base, data_offsets=[4, 0])}, b'\0' * 4), ({'x': base, 'y': base}, b'\0' * 4),
             ({'x': base}, b'\0' * 5), ({'__metadata__': {'a': 5}, 'x': base}, b'\0' * 4),
             ({'x': dict(base, extra=1)}, b'\0' * 4), ({'x': {'dtype': 'F32', 'shape': [1]}}, b'\0' * 4),
             ({'': base}, b'\0' * 4), ({'x': dict(base, dtype='F64')}, b'\0' * 4), ({'x': dict(base, dtype=1)}, b'\0' * 4),
             ({'x': dict(base, shape=[2**62, 4], data_offsets=[0, 4])}, b'\0' * 4),
             (b'{"x":{},"x":{}}', b''), (b'{"x":NaN}', b''), (b'[]', b''), (b'{"x": 1} junk', b''), (b' {}', b'')]
    for i, (header, payload) in enumerate(cases):
        root = tmp_path / str(i)
        root.mkdir()
        (root / 'config.json').write_text(CONFIG)
        write_bad(root / 'model.safetensors', header, payload)
        assert not python_reads(root), header
        code, _, err = run(root)
        assert code == 1 and err.startswith('glm-vx-reader: '), (header, code, err)
    for i, raw in enumerate((b'', b'a' * 7, struct.pack('<Q', 1000000000), struct.pack('<Q', 10) + b'{}',
                             struct.pack('<Q', 1), struct.pack('<Q', 2**64 - 1) + b'{}')):
        root = tmp_path / f'raw{i}'
        root.mkdir()
        (root / 'config.json').write_text(CONFIG)
        (root / 'model.safetensors').write_bytes(raw)
        assert not python_reads(root) and run(root)[0] == 1, raw


def test_an_index_python_refuses_is_refused(tmp_path):
    rng = random.Random(3)
    good = entries(rng, 'x', 2)
    names = [e[0] for e in good]
    for i, index in enumerate([
            {'weight_map': {}}, {'weight_map': []}, {}, {'weight_map': {names[0]: 'model.safetensors'}, 'other': 1},
            {'metadata': [], 'weight_map': {names[0]: 'model.safetensors'}},
            {'metadata': {'total_size': -1}, 'weight_map': {names[0]: 'model.safetensors'}},
            {'metadata': {'total_size': 1.5}, 'weight_map': {names[0]: 'model.safetensors'}},
            {'metadata': {'total_size': True}, 'weight_map': {names[0]: 'model.safetensors'}},
            {'weight_map': {names[0]: 5}}, {'weight_map': {'': 'model.safetensors'}},
            *({'weight_map': {names[0]: shard}} for shard in ('../outside.safetensors', '/a.safetensors', 'a\\b.safetensors',
                                                               'a/b.safetensors', 'a.bin', '', '.safetensor')),
            '{"weight_map":{"x":"a.safetensors","x":"b.safetensors"}}', '[1]']):
        root = tmp_path / str(i)
        root.mkdir()
        (root / 'config.json').write_text(CONFIG)
        write_safe(root / 'model.safetensors', good)
        (root / 'model.safetensors.index.json').write_text(index if isinstance(index, str) else json.dumps(index))
        assert not python_reads(root), index
        assert run(root)[0] == 1, index
    # The index's metadata may be absent, or its total_size null.
    for i, meta in enumerate([{}, {'metadata': {}}, {'metadata': {'total_size': None, 'other': 'x'}}]):
        root = tmp_path / f'meta{i}'
        root.mkdir()
        (root / 'config.json').write_text(CONFIG)
        write_safe(root / 'model.safetensors', good)
        (root / 'model.safetensors.index.json').write_text(json.dumps({**meta, 'weight_map': {n: 'model.safetensors' for n in names}}))
        same_as_python(root)


def test_a_shard_outside_the_folder_is_refused(tmp_path):
    rng = random.Random(4)
    root, outside = tmp_path / 'model', tmp_path / 'outside.safetensors'
    root.mkdir()
    (root / 'config.json').write_text(CONFIG)
    write_safe(outside, entries(rng, 'x', 1))
    (root / 'link.safetensors').symlink_to(outside)
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x.0.weight': 'link.safetensors'}}))
    assert not python_reads(root)
    code, _, err = run(root)
    assert code == 1 and 'symbolic link' in err, err


def test_where_the_reader_is_stricter(tmp_path):
    """Folders that checkpoint.py reads and the Vx reader refuses, each named in reader/src/model.vx."""
    rng = random.Random(5)
    good = entries(rng, 'x', 2)
    cases = {}

    def case(name):
        root = tmp_path / name
        root.mkdir()
        (root / 'config.json').write_text(CONFIG)
        write_safe(root / 'model.safetensors', good)
        cases[name] = root
        return root

    # A key twice in config.json, an integer past 64 bits, NaN, and nesting past 32 levels.
    (case('config_key_twice') / 'config.json').write_text(CONFIG[:CONFIG.rindex('}')] + ', "x": 1, "x": 2}')
    (case('config_big_int') / 'config.json').write_text(CONFIG[:CONFIG.rindex('}')] + ', "x": 18446744073709551616}')
    (case('config_nan') / 'config.json').write_text(CONFIG[:CONFIG.rindex('}')] + ', "x": NaN}')
    (case('config_deep') / 'config.json').write_text(CONFIG[:CONFIG.rindex('}')] + ', "x": ' + '[' * 40 + ']' * 40 + '}')
    # A shard that is a symbolic link inside the folder.
    root = case('symlink_inside')
    (root / 'model.safetensors').rename(root / 'real.safetensors')
    (root / 'model.safetensors').symlink_to('real.safetensors')
    # A shard that the index names and that is missing, and a tensor its shard lacks.
    root = case('missing_shard')
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x.0.weight': 'model.safetensors',
                                                                                  'y': 'model-2.safetensors'}}))
    root = case('missing_tensor')
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x.0.weight': 'model.safetensors',
                                                                                  'y': 'model.safetensors'}}))
    for name, root in cases.items():
        code, _, err = run(root)
        assert code == 1 and err.startswith('glm-vx-reader: '), (name, err)
    # checkpoint.py opens shards when a tensor is read, so these two fail there only then.
    for name in ('missing_shard', 'missing_tensor'):
        with SafetensorsCheckpoint(cases[name]) as c, pytest.raises((CheckpointError, OSError)):
            c.info('y')


def mutate(rng, raw):
    raw = bytearray(raw)
    for _ in range(rng.randint(1, 4)):
        op, at = rng.random(), rng.randrange(len(raw) + 1)
        if op < 0.4 and at < len(raw):
            raw[at] = rng.choice(b'{}[]",:0123456789-+.eE \\tnulrsfa' + bytes([rng.randrange(256)]))
        elif op < 0.6 and at < len(raw):
            del raw[at:at + rng.randint(1, 8)]
        elif op < 0.8:
            raw[at:at] = bytes(rng.choice(b'{}[]",:0123456789-.eE \\') for _ in range(rng.randint(1, 4)))
        else:
            other = rng.randrange(len(raw) + 1)
            raw[at:at] = raw[min(at, other):max(at, other)][:16]
    return bytes(raw)


def test_mutated_headers_are_read_alike(tmp_path):
    """A header with random edits: the readers agree on every one, and on what it holds when both read it."""
    rng = random.Random(4157)
    good = entries(random.Random(6), 'x', 4)
    header, payload = {}, bytearray()
    for name, dtype, shape, data in good:
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    header['__metadata__'] = {'format': 'pt'}
    raw = json.dumps(header).encode()
    (tmp_path / 'config.json').write_text(CONFIG)
    path = tmp_path / 'model.safetensors'
    agreed = read = 0
    for _ in range(int(os.environ.get('MODEL_READER_FUZZ', '1000'))):
        bad = mutate(rng, raw)
        cut = rng.choice([0, 0, 0, -1, 1])
        path.write_bytes(struct.pack('<Q', len(bad)) + bad + bytes(payload)[:len(payload) + cut if cut < 0 else None]
                         + b'\0' * max(cut, 0))
        try:
            theirs = SafetensorsFile(path)
            theirs_tensors = dict(theirs.tensors)
            theirs.close()
        except CheckpointError:
            theirs_tensors = None
        code, table, err = run(tmp_path)
        assert code in (0, 1), err
        ours = None if table is None else {name: info for name, (_, info) in table.tensors.items()}
        assert (ours is None) == (theirs_tensors is None), (bad, err)
        if ours is not None:
            assert ours == theirs_tensors, bad
            read += 1
        agreed += 1
    assert read > 0 and agreed == int(os.environ.get('MODEL_READER_FUZZ', '1000'))


def test_the_reader_builds_with_no_trusted_code_and_its_audit_lists_none():
    if not VXC:
        pytest.skip('needs VXC')
    out = subprocess.run([VXC, str(ROOT / 'reader' / 'main.vx'), '--package', f'reader={ROOT / "reader" / "src"}',
                          '--package', f'front={ROOT / "front" / "src"}', '--no-trusted-code', 'reader',
                          '--no-trusted-code', 'front', '--audit'], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = [l for l in out.stdout.splitlines() if l.strip() and not l.startswith('[')]
    assert lines and lines[-1].startswith('core and std:'), lines
    for line in lines[:-1]:
        assert '/stdlib/' in line, line


def test_the_reader_gets_a_folder_and_nothing_else():
    """main.vx gives read_model only the Folder; reader/src names no other right."""
    main = '\n'.join(line.split('//')[0] for line in (ROOT / 'reader' / 'main.vx').read_text().splitlines())
    assert 'read_model(&folder)' in main and main.count('root.files') == 1 and 'Folder::open(&root.files' in main
    for path in (ROOT / 'reader' / 'src').glob('*.vx'):
        code = '\n'.join(line.split('//')[0] for line in path.read_text().splitlines())
        for right in ('Root', 'Files', 'Network', 'Environment', 'Processes', 'Stdin', 'Stdout', 'Stderr'):
            assert right not in code, (path.name, right)


@pytest.mark.parametrize('fmt', ['bf16', 'fp8'])
def test_the_server_and_the_tp_ranks_load_alike_through_the_reader(tmp_path, fmt):
    """A tiny GLM checkpoint in two shards: the logits are the same bits with and without --model-reader."""
    from glm_vx.model import tiny_weights
    from glm_vx.server import build_model, parse_args
    from glm_vx.tiny import tiny_config
    from gpu import tp_serve
    from validation.write_checkpoint import export_checkpoint
    config = tiny_config()
    rng = np.random.default_rng(11)
    weights = {name: rng.normal(0, 1 / np.sqrt(w.shape[-1]), w.shape).astype(np.float32) if 'norm' not in name
               else rng.uniform(.7, 1.3, w.shape).astype(np.float32) for name, w in tiny_weights(config, seed=5).items()}
    export_checkpoint(config, weights, tmp_path, fmt, 2)
    tokens = [3, 17, 5, 5, 9, 200, 1, 44, 8]
    logits = []
    for extra in ([], ['--model-reader', str(READER)]):
        model, backend, ck, _ = build_model(parse_args(['--checkpoint', str(tmp_path), '--backend', 'numpy-reference', *extra]))
        logits.append(model.prefill(tokens))
        ck.close()
        rank = tp_serve.load_model(tmp_path, None, READER if extra else None)
        logits.append(rank.prefill(tokens))
        rank.weights.close()
    for got in logits[1:]:
        np.testing.assert_array_equal(got, logits[0])
    assert np.all(np.isfinite(logits[0]))


def test_glm_53s_index_reads_alike(tmp_path):
    """GLM-5.3's own index (metadata/): 118,629 tensors in 141 shards, each shard a stand-in holding the
    tensors the index puts in it, as BF16 scalars."""
    import time
    index = json.loads((ROOT / 'metadata' / 'model.safetensors.index.json').read_text())
    by_shard = {}
    for name, shard in index['weight_map'].items():
        by_shard.setdefault(shard, []).append((name, 'BF16', [1], b'\0\0'))
    for shard, entries_ in by_shard.items():
        write_safe(tmp_path / shard, entries_)
    shutil.copy(ROOT / 'metadata' / 'model.safetensors.index.json', tmp_path)
    (tmp_path / 'config.json').write_text(CONFIG)
    start = time.perf_counter()
    code, table, err = run(tmp_path)
    seconds = time.perf_counter() - start
    assert code == 0, err
    assert len(table.tensors) == 118629 and len(table.shards) == 141
    with SafetensorsCheckpoint(tmp_path) as theirs:
        assert {name: table.shards[s].name for name, (s, _) in table.tensors.items()} == theirs.weight_map
        assert all(theirs.info(name) == info for name, (_, info) in table.tensors.items())
    assert seconds < 60, seconds
