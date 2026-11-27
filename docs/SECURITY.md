# Security: glm-vx and SGLang's advisories

This page lists SGLang's published advisories and why each cannot happen in glm-vx. Each reason
is one of:
- something the Vx compiler checks, or
- a test that fails when the reason stops holding.

It covers serving with `python -m glm_vx.server --front front/build/glm-vx-front --tokenizer ... --model-reader
reader/build/glm-vx-reader` (#68).

## The boundary

```
client --HTTP--> glm-vx-front (Vx)  --fixed frames on stdin/stdout-->  engine (Python, the GPUs)
```

- **The front end reads every byte that comes from the network.** It is a Vx program (`front/`).
  - The engine starts it and has no listener of its own.
  - With `--tokenizer`, the engine gets only token ids, never a client's text.
- **The engine reads only fixed binary frames** ([ENGINE-LINK.md](ENGINE-LINK.md)). It parses them with `struct`, after a check of their length. It never uses pickle.
- **With `--model-reader`, a second Vx program reads the model folder** ([MODEL-TABLE.md](MODEL-TABLE.md)). It gets a
  right to that folder only, and checks config.json, the index and every shard's header. The engine gets a fixed-format
  table and maps each tensor at the offsets the reader checked.
- **The front end's code holds only the rights `main` gives it.** Vx's `main(root : Root, ...)` gets every right of the process, and nothing else can make one:
  - `main` reads tokenizer.json with `root.files`, before it listens;
  - the listener gets `root.network`, once, to bind;
  - the router gets `root.stdout`, and the engine reader `root.stdin`;
  - a connection's handler gets its own `TcpStream`, a channel to the router, the model's name and the tokenizer;
  - nothing gets `root.env`.
- **The compiler enforces that.** `front/` is built with `vxc --no-trusted-code front`:
  - an `unsafe` block, an `unsafe fn`, an `extern` block or impl, `unsafe impl`, an `mlir!` block the compiler cannot verify, or a `requires` of a function that C can call is an error (E5003);
  - making a `Files`, `Network`, `Environment` or `Stdout` outside `std` is an error (E8014);
  - `File::open` takes a `&Files`, and `TcpStream::connect` a `&Network`, so a handler that tries either does not type-check.

  `tests/test_front_confinement.py` adds each of these to a copy of the package and expects the error.
- **`vxc --audit`** of the whole program lists no trusted code outside Vx's `core` and `std`. That is checked in the same file.
- **Defaults.** glm-vx listens on 127.0.0.1 by default (`--host`) and has no API keys, admin routes or weight transfer. It serves 3 routes: `GET /health`, `GET /v1/models` and `POST /v1/completions`.

## SGLang's advisories

These are the 45 CVEs that NVD's keyword search for SGLang returned on 2026-11-27. Each was checked against the GitHub
Advisory Database (GHSA), which has all 45 (10 reviewed for the `sglang` package), and against CERT where CERT has a note.
Each link goes to the GHSA advisory; the number is its CVSS 3.1 score.

| SGLang advisories | Why not in glm-vx | What checks it |
|---|---|---|
| **Unpickling a peer's or a client's bytes**<br>[CVE-2026-3059](https://github.com/advisories/GHSA-rgq9-fqf5-fv58) (9.8, [CERT VU#665416](https://kb.cert.org/vuls/id/665416)): the multimodal ZMQ broker<br>[CVE-2026-3060](https://github.com/advisories/GHSA-jx93-g359-86wm) (9.8, [CERT VU#665416](https://kb.cert.org/vuls/id/665416)): encoder disaggregation<br>[CVE-2026-7301](https://github.com/advisories/GHSA-gwv6-pq6m-p3rq) (9.8, [CERT VU#326070](https://kb.cert.org/vuls/id/326070)): the multimodal scheduler's ROUTER socket on 0.0.0.0<br>[CVE-2026-14890](https://github.com/advisories/GHSA-67x7-pm2x-m2cf) (9.1, [CERT VU#326070](https://kb.cert.org/vuls/id/326070)): the expert-parallel backup's PULL socket<br>[CVE-2026-93088](https://github.com/advisories/GHSA-xv76-mrpv-26pg) (9.8): the diffusion orchestrator's ROUTER socket<br>[CVE-2026-93034](https://github.com/advisories/GHSA-w4hv-c72g-pqx6) (9.8): the ZMQ message decoder's PickleWrapper<br>[CVE-2025-10164](https://github.com/advisories/GHSA-9w53-xr52-mwgj) (7.3): `/update_weights_from_tensor`<br>[CVE-2026-86793](https://github.com/advisories/GHSA-m675-99qc-9233) (9.8): `/update_weights_from_tensor` past SafeUnpickler<br>[CVE-2026-15969](https://github.com/advisories/GHSA-359v-m36h-r94v) (9.8): `/load_lora_adapter_from_tensors` past SafeUnpickler<br>[CVE-2026-15976](https://github.com/advisories/GHSA-r344-357p-w9pp) (9.8): `torch.load` in `/update_weights_from_disk`<br>[CVE-2026-7304](https://github.com/advisories/GHSA-36m8-w8qf-g76p) (9.8, [CERT VU#326070](https://kb.cert.org/vuls/id/326070)): `dill.loads` of a custom logit processor<br>[CVE-2026-7669](https://github.com/advisories/GHSA-6m5f-673f-5vh7) (5.6): `get_tokenizer` (Transformers)<br>[CVE-2026-3989](https://github.com/advisories/GHSA-hvwj-8w5g-28rg) (7.8, [CERT VU#665416](https://kb.cert.org/vuls/id/665416)): `replay_request_dump.py` on a .pkl file | The engine has no socket: its link is fixed frames on a pipe, parsed with `struct`. No route takes a serialized object. Outside the serving path, only two tests pickle, and only their own objects. | `tests/test_serving_path.py`: no `pickle`, `dill`, `marshal`, `shelve`, `joblib`, `torch` or `transformers` on the serving path |
| **Code from a template, a filter or a debugging hook**<br>[CVE-2026-5760](https://github.com/advisories/GHSA-2wm4-697g-pfq8) (9.8, [CERT VU#915947](https://kb.cert.org/vuls/id/915947)): an unsandboxed Jinja2 chat template in `/v1/rerank`<br>[CVE-2026-111391](https://github.com/advisories/GHSA-vfgj-p7f7-r3v8) (8.1): `eval` of a filter in `/dumper/configure`<br>[CVE-2026-15971](https://github.com/advisories/GHSA-96j3-rjf9-2hxp) (9.8): the dumper subsystem | No template engine, no `eval`, no dumper. The front end reads tokenizer.json as data, and refuses one that is not GLM's. | `tests/test_serving_path.py`: no `jinja2`, `eval`, `exec`, `compile` or `__import__`; `tests/test_front_tokenizer.py` |
| **A request that reaches the network or the files**<br>[CVE-2026-15974](https://github.com/advisories/GHSA-hg26-hr8r-fxc9) (6.5): SSRF and file reads through `image_url`<br>[CVE-2026-7302](https://github.com/advisories/GHSA-qwrp-wghp-94q2) (9.1): an upload's `../` file name writes anywhere<br>[CVE-2026-112267](https://github.com/advisories/GHSA-mg89-3m78-qwqg) (6.5): SSRF through `bootstrap_host` | A handler cannot open a connection or a file. It holds no `&Network` and no `&Files`, and nothing in `front` can make one. | the compiler: `tests/test_front_confinement.py` (`connect_without_network`, `open_without_files`, `forged_network`, `forged_files`) |
| **Secrets in a reply**<br>[CVE-2026-15977](https://github.com/advisories/GHSA-vwmc-327h-6x3j) (7.5): `/server_info` returns API keys<br>[CVE-2026-105245](https://github.com/advisories/GHSA-fr3j-h92j-cm4p) (3.7): `/server_info` sends `api_key` in clear text | No keys, and no `/server_info`. `/health` is the model's and backend's names and the scheduler's counters. The front end holds no `Environment`. | the compiler (`forged_environment`); `front/main.vx` gives `root.env` to nobody |
| **Admin and weight routes with no authentication**<br>[CVE-2026-15978](https://github.com/advisories/GHSA-x6x2-r266-cx9p) (7.5): an NCCL weight broadcast to an attacker<br>[CVE-2026-112083](https://github.com/advisories/GHSA-hqw5-mw82-chh2) (5.3): the weight exporter's `group_name`<br>[CVE-2026-127222](https://github.com/advisories/GHSA-m386-99gj-pq7p) (7.5): `/load_lora_adapter_from_tensors` with no admin check<br>[CVE-2026-112082](https://github.com/advisories/GHSA-j823-x22q-54wf) (7.5): the expert-distribution recording endpoint<br>[CVE-2026-112268](https://github.com/advisories/GHSA-q6x7-8366-3wrx) (6.3): sglang-router's `/ha/*` routes | Only 3 routes: `GET /health`, `GET /v1/models`, `POST /v1/completions`. Anything else is a 404, or a 501 for another method. | `tests/test_front_http.py`; `tests/test_server.py` |
| **Prefill/decode disaggregation and Mooncake**<br>[CVE-2026-92972](https://github.com/advisories/GHSA-9hr3-qqwc-fxxh) (8.6): `PUT /route` poisons KV routing<br>[CVE-2026-93688](https://github.com/advisories/GHSA-hpqj-j2xw-q3cx) (7.5): `bootstrap_room` exhausts transfer state<br>[CVE-2026-102634](https://github.com/advisories/GHSA-mjgp-hgfr-77jq) (7.5): duplicate `bootstrap_room`<br>[CVE-2026-93838](https://github.com/advisories/GHSA-2wcg-j54r-m6wm) (5.9): ZMQ `STAGING_REQ` chunk index<br>[CVE-2026-94570](https://github.com/advisories/GHSA-gr7x-rp46-7mvw) (5.9): ZMQ `AUX_DATA` control messages<br>[CVE-2026-112269](https://github.com/advisories/GHSA-9qhw-7fv2-2hgm) (6.5): Mooncake decode registrations<br>[CVE-2026-112270](https://github.com/advisories/GHSA-q322-x26r-gm37) (6.5): Mooncake `STAGING_RSP`<br>[CVE-2026-104433](https://github.com/advisories/GHSA-ccgr-wx5m-4h7g) (7.5): Mooncake transfer engine's handshake (Mooncake's own CVE) | `server.py` has no disaggregated mode. `glm_vx.disaggregated` is a CPU demo of two processes on pipes, with no listener. | `tests/test_serving_path.py` (the walk from `glm_vx.server` does not reach it) |
| **Image, audio and video inputs**<br>[CVE-2026-109225](https://github.com/advisories/GHSA-4fxq-6r94-mchm) (7.5): audio uploads read whole<br>[CVE-2026-109226](https://github.com/advisories/GHSA-h3xf-3226-h48g) (7.5): compressed audio<br>[CVE-2026-109227](https://github.com/advisories/GHSA-6h2g-h3jf-2prp) (7.5): Qwen-VL video frames<br>[CVE-2026-109228](https://github.com/advisories/GHSA-2mhj-6r3c-c9xj) (7.5): GLM-V video limits from the request<br>[CVE-2026-109229](https://github.com/advisories/GHSA-fqvf-j3pr-77v9) (7.5): Pillow's decompression-bomb check off<br>[CVE-2026-111392](https://github.com/advisories/GHSA-fgff-m398-2gh5) (4.8): client `mm_hashes` as prefix-cache keys<br>[CVE-2026-10775](https://github.com/advisories/GHSA-jrcc-j37m-v8fg) (3.6): `data_hash` of the cache handler | Prompts are text or token ids, nothing else. The prefix cache keys on the server's own token ids. | `tests/test_front_request.py` (any other field is `400 unsupported fields`) |
| **Request fields that reach the internals**<br>[CVE-2026-10300](https://github.com/advisories/GHSA-m2jr-x4gq-5rmj) (3.7): `lora_path` reaches an assertion<br>[CVE-2026-110807](https://github.com/advisories/GHSA-9h5h-jqm8-rrcw) (4.3): `session_params`<br>[CVE-2026-110809](https://github.com/advisories/GHSA-cq8f-cv6m-8vvq) (4.3): session cleanup<br>[CVE-2026-112266](https://github.com/advisories/GHSA-8v54-r9x8-p94v) (5.9): a client's `rid` aborts other requests by prefix | The request is a fixed set of fields. Request ids are chosen by the front end's router and the engine, so a client cannot name another's request. | `tests/test_front_request.py`; `tests/test_link.py` |

## What this does not cover

- **Trusted code.** That includes `std`'s `unsafe` blocks, the Vx compiler and runtime, CPython, numpy and the CUDA driver. With `--front` and `--tokenizer`, the server does not load HF `tokenizers`: the front end tokenizes and decodes.
  - A memory bug there is not excluded.
  - The front end's tests and fuzzing also run under AddressSanitizer (`FRONT_PRELOAD`).
  - Vx issues found this way, both fixed in the Vx the front end needs (6322eb757): vx-lang/Vx#4142, a double free (fixed by vx-lang/Vx#4149), and vx-lang/Vx#4144, `--no-trusted-code` let an exported `extern fn` through (fixed by vx-lang/Vx#4151).
- **Denial of service.**
  - A panic, such as an index out of range, ends the front end. The engine cancels its requests and starts a new one (`tests/test_front_restart.py`).
  - The front end's limits:
    - 16 KiB of headers and 64 header lines;
    - a body of at most 1 MiB;
    - 32 connections;
    - a 30 s read timeout.
  - `tests/test_front_fuzz.py` sends random requests and expects no crash and no hang.
- **The model folder's weights are trusted data.**
  - With `--model-reader`, `reader/` reads config.json, the index and the shards' headers. Without it, `glm_vx/checkpoint.py` does, with no pickle or remote code.
  - The weights' bytes themselves go to the GPUs as numbers, and `tokenizer.json` is read by the front end as data.
  - The GGUF path (`--gguf`, experimental) still reads its files with gguf-py.
- **No TLS and no authentication.** Keep the default loopback address, or put a proxy in front.
- **Without `--front`**, `server.py` serves HTTP with Python's `http.server`. The checks of the engine still hold, but not the front end's.

## Checking it

```
VXC=/path/to/vxc front/build.sh && VXC=/path/to/vxc reader/build.sh
VXC=/path/to/vxc python -m pytest tests/test_front_confinement.py tests/test_serving_path.py tests/test_front_http.py \
    tests/test_front_fuzz.py tests/test_front_restart.py tests/test_model_reader.py
GLM_TOKENIZER=/path/to/GLM-5.3/tokenizer.json python -m pytest tests/test_front_tokenizer.py
FRONT_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so) python -m pytest tests/test_front_*.py tests/test_model_reader.py
```

Served latency through the front end (TTFT and TPOT on the 8×H200, under `docs/benchmarking_policy.md`) is not measured yet.
