# glm-vx-front: the HTTP front end in Vx (#68)

`glm-vx-front` reads every byte that comes from the network. The engine (`glm_vx/server.py`, Python, the GPUs) starts it and talks to it only over its standard input and output, in the fixed frames of [docs/ENGINE-LINK.md](../docs/ENGINE-LINK.md). The engine has no listener. [docs/SECURITY.md](../docs/SECURITY.md) goes through SGLang's advisories and why each cannot happen here.

```
client --HTTP--> glm-vx-front (Vx)                 --frames on stdin/stdout--> engine (Python)
                  main(root : Root, args)                                       Scheduler, GPUs
                  |- main: root.files, to read tokenizer.json before it listens
                  |- the listener: root.network, to bind once
                  |- the router: root.stdout        writes SUBMIT, CANCEL, STATUS
                  |- the reader: root.stdin         reads ACCEPTED, TOKEN, DONE, ERROR, STATUS_REPLY
                  '- each connection: a task with its TcpStream, a Sender to the router, the model name, the tokenizer
                     package `front` (src/): HTTP, JSON, the request checks, the tokenizer, the frames
```

## Build and run

```
VXC=/path/to/vxc front/build.sh       # Vx main 6322eb757 or later
python -m glm_vx.server --tiny --front front/build/glm-vx-front --port 8000
python -m glm_vx.server ... --front front/build/glm-vx-front --tokenizer /models/GLM-5.3/tokenizer.json
```

`build.sh` puts `glm-vx-front`, the test programs `json_echo`, `request_check` and `tokenize_check`, and `audit.txt` (the output of `vxc --audit`) in `front/build/`. It needs `VX_RUNTIME_DIR`, the folder of `libvx_std_core.so`, and `VX_SHIMS_DIR`, the folder of `libvx_mlir_shims.so`, when they are not next to `vxc`.

## The tokenizer

With `--tokenizer`, `server.py` gives the front end the path of the model's tokenizer.json. `main` reads it with `root.files`
before it listens, and `src/tokenizer.vx` turns each text prompt into token ids, so the engine never sees a client's text.
It implements zai-org/GLM-5.3's tokenizer and refuses a tokenizer.json with other settings:

- the added tokens first, then the GPT-4-style split as Oniguruma matches it, then byte-level BPE with `ignore_merges`;
- the merges in O(n log n), lowest rank first and then leftmost, as HF's `merge_all`;
- `std::unicode`'s letters, numbers and white space, which agree with Oniguruma's `\p{L}`, `\p{N}` and `\s` on every code point.

Loading takes about 0.8 s and 200 MB at its peak. `tests/test_front_tokenizer.py` compares the ids with HF `tokenizers` on
edge cases, every code point in seven contexts, random text and GSM8K, and checks that the engine gets SUBMIT with those ids
and no text. It needs `GLM_TOKENIZER`, the path of the 20 MB tokenizer.json, which is not in the repository.

## What the compiler checks

- **No trusted code in `front`.** The package is built with `--no-trusted-code front`. Each of these in it is an error (E5003):
  - an `unsafe` block, also one that a macro writes;
  - an `unsafe fn`, an `extern` block, an `unsafe impl` or an `extern impl`;
  - an `mlir!` block that the compiler cannot verify;
  - a `requires` of a `pub extern fn`, because a call from C does not prove it (Vx #4151).

  `tests/test_front_confinement.py` adds each one to a copy of the package and expects the error.
- **No capability made in `front`.** `Files {}`, `Network {}`, `Environment {}` and `Stdout {}` outside `std` are errors (E8014). They are tested the same way.
- **A connection's handler holds no right.**
  - `serve` gets its `TcpStream`, a `Sender<Ask>` and the model's name.
  - It cannot open a file, because `File::open` takes a `&Files`. It cannot open a connection, because `TcpStream::connect` takes a `&Network`. The tests check that both fail to type-check without one.
  - It cannot write to the engine except through the router.
- **The audit.** `vxc --audit` of the whole program lists no trusted fact outside `core` and `std`, and `main.vx` has no `unsafe` either. `tests/test_front_confinement.py` checks this.
- **Every function is checked.** With Vx #4136, a function that nothing calls is checked too.
- **The engine loads no code from data.** `tests/test_serving_path.py` reads every module that `glm_vx.server` can import. None imports
  `pickle`, `jinja2` or another deserializer, calls `eval` or `exec`, or calls `numpy.load` with `allow_pickle=True`.

## Limits

| | front end | `server.py`'s `http.server` |
|---|---|---|
| request line and headers | 16 KiB, else 431 | 64 KiB a line, else 431 or 414 |
| headers | 64, else 431 | 100, else 431 |
| body | `Content-Length` 1..1048576, digits only | the same range, and `int()` takes `+12` or ` 12 ` |
| `Transfer-Encoding` | 400 | 400 |
| a space before a header's colon | 400 | not read as a header, so the header is lost |
| two different `Content-Length`s | 400 | the first one |
| JSON | RFC 8259, UTF-8, nesting at most 32 | `json.loads`, which also takes `NaN` and lone surrogates |
| a read | 30 s, else the connection closes | 30 s for each `recv` |
| connections | 32, and the 33rd is closed | 32 (`BoundedSemaphore`) |

The replies are `server.py`'s: the same JSON fields, SSE framing and heartbeats, 400, 404, 429 and 500 cases, and error messages. `tests/test_server.py` and `tests/test_openai_compat.py` run each case against both servers. `tests/test_front_request.py` and `tests/test_front_json.py` compare the request checks and the parser with `server.py` and `json.loads`, also on random input.

`tests/test_front_fuzz.py` sends mutated HTTP requests, eight connections at a time (`FRONT_HTTP_FUZZ` cases, 300 by default).
Each must get well-formed replies or a close, and the front end must not stop.
Like `server.py`, the front end takes a client that half closes before its reply for one that has gone.

## What this does not cover

- **Memory bugs in trusted code.** That includes `std`'s `unsafe` blocks, the Vx compiler, the Python engine and the CUDA driver. `FRONT_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so)` runs the test programs and the front end under AddressSanitizer.
- **A panic.** For example, an index out of range stops the front end. That is a denial of service, not code execution:
  - every open connection is lost, and the engine cancels the requests they carried;
  - the engine then starts a new front end on the same address and shows the count as `front_end_restarts` in `/health`;
  - after 5 restarts within 60 s it stops, and so does `server.py`, with the error.
  `tests/test_front_restart.py` checks these.
- **Detokenizing.** The engine still turns the ids it generates into text with HF `tokenizers` (`glm_vx/detok.py`). That is the model's output, not a client's bytes.
