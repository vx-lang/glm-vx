"""The table that glm-vx-reader writes for a model folder (#68, docs/MODEL-TABLE.md).

`read_model(root, reader)` runs the Vx reader (reader/build/glm-vx-reader) on the folder and
decodes its table with `struct`: config.json as the value json.loads gives, each shard's size and
the start of its data, and each tensor's dtype, shape and byte range. `open_checkpoint` maps the
tensors at those offsets, so the engine parses nothing that the folder holds.

    config, table = read_folder(root, reader)   # reader None: Python reads config.json, as before
    weights = open_checkpoint(root, table, block_size=GLMConfig.from_dict(config).weight_block_size)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import struct
import subprocess
from pathlib import Path

from .checkpoint import CheckpointError, SafetensorsCheckpoint, TensorInfo

MAGIC = b'glm-vx-table 1\n'
DTYPES = ('F32', 'F16', 'BF16', 'F8_E4M3', 'F8_E4M3FN', 'I32', 'I64')
MAX_DEPTH = 33  # front/src/json.vx nests at most 32 arrays and objects


@dataclass(frozen=True)
class Shard:
    name: str
    size: int
    data_start: int


@dataclass(frozen=True)
class ModelTable:
    config: dict
    shards: tuple[Shard, ...]
    tensors: dict[str, tuple[int, TensorInfo]]  # each tensor's shard index and entry, in the index's order


class _Bytes:
    def __init__(self, data: bytes):
        self.data, self.at = data, 0

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.data):
            raise CheckpointError('the model table is cut short')
        self.at += n
        return self.data[self.at - n:self.at]

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def text(self) -> str:
        try:
            return self.take(self.unpack('<I')).decode()
        except UnicodeDecodeError as e:
            raise CheckpointError('the model table holds text that is not UTF-8') from e

    def tree(self, depth: int = 0):
        if depth > MAX_DEPTH:
            raise CheckpointError('the model table nests too deep')
        tag = self.unpack('<B')
        if tag <= 2:
            return (None, False, True)[tag]
        if tag == 3:
            return self.unpack('<q')
        if tag == 4:
            return self.unpack('<d')
        if tag == 5:
            return self.text()
        if tag == 6:
            return [self.tree(depth + 1) for _ in range(self.unpack('<I'))]
        if tag == 7:
            out = {}
            for _ in range(self.unpack('<I')):
                key = self.text()
                if key in out:
                    raise CheckpointError(f'the model table has the key {key!r} twice')
                out[key] = self.tree(depth + 1)
            return out
        raise CheckpointError(f'the model table has an unknown tag {tag}')


def parse_table(data: bytes) -> ModelTable:
    b = _Bytes(data)
    if b.take(len(MAGIC)) != MAGIC:
        raise CheckpointError('not a glm-vx model table')
    config = b.tree()
    if not isinstance(config, dict):
        raise CheckpointError('the model table\'s config is not an object')
    shards = tuple(Shard(b.text(), b.unpack('<Q'), b.unpack('<Q')) for _ in range(b.unpack('<I')))
    tensors = {}
    for _ in range(b.unpack('<I')):
        name, shard, dtype, ndim = b.text(), b.unpack('<I'), b.unpack('<B'), b.unpack('<B')
        shape = tuple(b.unpack('<Q') for _ in range(ndim))
        start, end = b.unpack('<Q'), b.unpack('<Q')
        if shard >= len(shards) or dtype >= len(DTYPES) or name in tensors:
            raise CheckpointError(f'the model table has an invalid entry for {name!r}')
        tensors[name] = (shard, TensorInfo(name, DTYPES[dtype], shape, (start, end)))
    if b.at != len(data):
        raise CheckpointError('the model table has bytes after its end')
    return ModelTable(config, shards, tensors)


def read_model(root: str | Path, reader: str | Path, *, timeout: float = 600) -> ModelTable:
    """The table of the model folder `root`, from the Vx reader at `reader`."""
    out = subprocess.run([str(reader), str(root)], stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    if out.returncode != 0:
        raise CheckpointError(out.stderr.decode(errors='replace').strip() or f'{reader} exited with {out.returncode}')
    return parse_table(out.stdout)


def read_folder(root: str | Path, reader: str | Path | None = None):
    """(config.json as json.loads reads it, the table or None).

    With `reader` (reader/build/glm-vx-reader), the Vx reader reads config.json, the index and
    every shard's header, and Python parses none of them. Without it, Python reads config.json, and
    open_checkpoint lets checkpoint.py read the rest. GLMConfig.from_dict(config) checks the config
    either way.
    """
    if reader is None:
        return json.loads((Path(root) / 'config.json').read_text()), None
    table = read_model(root, reader)
    return table.config, table


def open_checkpoint(root: str | Path, table: ModelTable | None = None, **checkpoint_options) -> SafetensorsCheckpoint:
    """The checkpoint in `root`: from read_folder's table, or read by checkpoint.py without one."""
    if table is None:
        return SafetensorsCheckpoint(root, **checkpoint_options)
    return SafetensorsCheckpoint.from_table(root, table, **checkpoint_options)


def open_model(root: str | Path, reader: str | Path | None = None, **checkpoint_options):
    """(config, checkpoint) of the model folder `root`: read_folder, then open_checkpoint."""
    config, table = read_folder(root, reader)
    return config, open_checkpoint(root, table, **checkpoint_options)
