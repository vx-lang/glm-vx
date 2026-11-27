# The model folder's reader (#68)

`reader/build/glm-vx-reader MODEL_DIR` reads a safetensors model folder and writes one table on standard output.
`server.py --checkpoint MODEL_DIR --model-reader reader/build/glm-vx-reader` uses it, on every tensor-parallel rank.
The engine then parses nothing that the folder holds:
- config.json reaches it as the value `json.loads` would give;
- each tensor reaches it as a dtype, a shape and a byte range, which `glm_vx/checkpoint.py` maps.

`GLMConfig.from_dict` still checks the config, as before. Without `--model-reader`, Python reads the folder as before.

## What the reader can reach

The reader is a Vx program, built with `vxc --no-trusted-code reader --no-trusted-code front` (`reader/build.sh`). Its
`main` turns `root.files` into a `Folder` for `MODEL_DIR` and gives `read_model` only that `Folder`:
- the reader can read the files in that folder, and no others;
- a name with a `..` part, an absolute name or a symbolic link gives `NotInFolder`;
- it gets no right to the network, the environment or other programs.

`tests/test_model_reader.py` checks that the audit lists no trusted code outside Vx's `core` and `std`, and that
`reader/src` names no right but the `Folder`. The reader uses the front end's JSON parser (`front/src/json.vx`).

## What it checks

What `glm_vx/checkpoint.py` checks, and the reader refuses every folder that `checkpoint.py` refuses:
- **config.json** must be a JSON object.
- **model.safetensors.index.json** is optional. Without it, the folder has one `model.safetensors`.
  - It has a `weight_map` object, which must not be empty, and an optional `metadata` object, whose `total_size` is
    null or a nonnegative integer.
  - Each shard name is a file name ending in `.safetensors`, with no `/` or `\`.
- **Each shard's header** has a `u64` length from 2 to 100 MiB, inside the file. It begins with `{` and is a JSON object.
  - `__metadata__` maps strings to strings.
  - Each tensor has exactly `dtype`, `shape` and `data_offsets`. The dtype is F32, F16, BF16, F8_E4M3, F8_E4M3FN, I32
    or I64. The shape has at most 32 integers from 0 to 2^63-1, and its byte count matches the offsets.
  - The tensors cover the data with no gap, overlap or byte left over.

Stricter than `checkpoint.py`:
- **Every JSON file:**
  - no key twice in an object (`checkpoint.py` refuses that in the index and the headers, but not in config.json);
  - UTF-8 only, with no lone surrogate escape;
  - at most 32 levels of nesting;
  - no integer past 64 bits, and no NaN or Infinity.
- **The shards:** every shard the index names must be in the folder as a plain file, not a symbolic link (not even one
  inside the folder), with each tensor the index puts in it. `checkpoint.py` checks a shard only when it first reads a
  tensor from it, and follows a symbolic link that stays in the folder. A Hugging Face cache snapshot is made of symbolic
  links, so give the reader a folder from `hf download --local-dir`.

Between the reader and the engine, the engine checks that each shard still has the size the reader saw.

## The table

Every integer is little-endian.

1. The 15 bytes `glm-vx-table 1\n`.
2. config.json as a tree. Each value is a `u8` tag, then:
   - 0: null; 1: false; 2: true;
   - 3: an `i64`, a number with no fraction or exponent;
   - 4: an `f64`, any other number, rounded as Python's `float()` rounds it (`1e400` is inf);
   - 5: text, a `u32` length and UTF-8 bytes;
   - 6: a list, a `u32` count and the values;
   - 7: an object, a `u32` count, then each key (as text, without the tag) and its value, in the file's order.
3. The shards: a `u32` count, then each one's name (as text), `u64` file size and `u64` start of its data (8 plus the
   header's length).
4. The tensors, in the index's order: a `u32` count, then for each:
   - its name (as text);
   - `u32` shard;
   - `u8` dtype (0 F32, 1 F16, 2 BF16, 3 F8_E4M3, 4 F8_E4M3FN, 5 I32, 6 I64);
   - `u8` count of dimensions, and each as a `u64`;
   - `u64` start and `u64` end, from the start of the shard's data, as safetensors' `data_offsets`.

`glm_vx/model_table.py` decodes it with `struct`, and refuses a table that is cut short or that has bytes after its end.

## Tests

`tests/test_model_reader.py`, with `reader/build.sh` built:
- random folders, one-file folders and GLM-5.3's own index read alike in both readers: the same config, weight map,
  entries and bytes;
- every malformed header and index of `tests/test_checkpoint.py` is refused, and so are random edits of a header
  wherever `checkpoint.py` refuses them (MODEL_READER_FUZZ, 1000 by default);
- each way the reader is stricter;
- a tiny GLM checkpoint gives the same logits, to the bit, through `server.py` and `gpu/tp_serve.py` with and without
  the reader.

`FRONT_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so)` runs the reader under AddressSanitizer.
