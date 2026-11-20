# The completion API

`glm_vx.server` serves a **completion API subset** of OpenAI's, not complete chat or tool-call compatibility.
(Moved from the README on 2026-11-20, unchanged.)

- `/health` reports the backend and capacity; `/v1/models` lists the local model.
- `/v1/completions` accepts one nonempty token-ID prompt, or a string when a local `--tokenizer tokenizer.json`
  is given. Greedy sampling is the default; a finite nonnegative temperature and an optional seed give
  reproducible sampling. `ignore_eos`, `stream_options.include_usage`, and `n` / `best_of` equal to 1 are accepted
  for OpenAI benchmark clients; other unknown options are rejected.
- Each SSE token event carries its exact token ID and a `text` delta that is sent only once the decoded text is
  stable, so split UTF-8 / BPE pieces are held back rather than corrupted.
- The terminal event depends on the request. A request that names `model`, as every OpenAI client does, gets
  OpenAI framing: the terminal chunk has only the held-back text tail, `finish_reason` and `usage`, followed by a
  `choices: []` usage chunk if `include_usage` was set, and keepalives are blank lines. A request without `model`
  gets glm-vx framing: the final event repeats the complete text and `token_ids` (clients must not concatenate it
  again), and keepalives are SSE comments.

```sh
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[1,2,3],"max_tokens":8,"stream":true}'
```

The service binds loopback by default. It has no authentication or TLS; put a suitable gateway in front before
exposing it to a network.

`--backend vx-gpu-resident` serves one request at a time (`--max-sequences` is 1 until batched decode); further
requests wait in a FIFO (`--max-waiting`). See [gpu/README.md](../gpu/README.md) for its options.
