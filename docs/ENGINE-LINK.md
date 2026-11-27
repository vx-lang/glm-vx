# The engine link (#68, step 1)

The front end (Vx, step 2) reads HTTP from the network. The engine (Python, the GPUs) runs the
scheduler. They talk over the front end's standard input and output, in the frames below. The
engine starts the front end and holds the other end of both pipes, so the front end needs no file
or network right to reach the engine.

`glm_vx/link.py` is the engine's side. It parses each frame with `struct` and checks its length
before it reads a field. It never uses `pickle`, `eval` or a JSON parser on a frame.

## Frames

All integers are little-endian. A frame is a `u32` payload length `n`, then `n` bytes of payload.
The payload starts with a `u8` kind. `n` is at least 1 and at most 4 MiB.

From the front end to the engine:

| kind | name | fields after the kind |
|---|---|---|
| `0x01` | SUBMIT | `u64` id, `u32` max_tokens, `f64` temperature, `u64` seed, `u8` flags, `u32` count, then `count` `u32` token ids |
| `0x02` | SUBMIT_TEXT | the same fields, then `count` bytes of UTF-8 prompt text, which the engine tokenizes |
| `0x03` | CANCEL | `u64` id |
| `0x04` | STATUS | `u64` id |

- Flags: bit 0 is `ignore_eos`. The other bits must be 0.
- The front end picks the ids. An id stays in use from its SUBMIT until the engine's DONE or ERROR for it.
- A front end started with a tokenizer.json (`server.py --front ... --tokenizer`) tokenizes text prompts itself and sends
  only SUBMIT, so the engine gets token ids only. Without one it sends SUBMIT_TEXT, and the engine tokenizes, or refuses
  the request when it has no tokenizer either.

From the engine to the front end:

| kind | name | fields after the kind |
|---|---|---|
| `0x81` | TOKEN | `u64` id, `u32` token id, `u32` length, then that many bytes of new text |
| `0x82` | DONE | `u64` id, `u8` reason, `u32` prompt_tokens, `u32` completion_tokens, `u32` cached_prompt_tokens, four `f64` timings, `u32` tail length, `u32` text length, then the tail and the text |
| `0x83` | ERROR | `u64` id, `u8` code, `u32` length, then that many bytes of message |
| `0x84` | STATUS_REPLY | `u64` id, `u32` length, then that many bytes of JSON |
| `0x85` | ACCEPTED | `u64` id, `u64` created (Unix seconds), `u32` length, then that many bytes of the request's name |

- **ACCEPTED** comes before any TOKEN of the request, as soon as the scheduler takes it. The front end then sends a stream's headers, as `server.py` does after `Scheduler.submit`. The name (`cmpl-...`) and `created` are the reply's `id` and `created`.

- **Who decodes.** A front end started with a tokenizer.json also decodes the ids that TOKEN carries, so the engine has no
  tokenizer and sends no text: every TOKEN's text and DONE's tail and text are empty. The front end's `Detok`
  (`front/src/tokenizer.vx`) gives the same texts as below. Without a tokenizer.json, the engine sends them.
- **TOKEN's text** is what `IncrementalText.push` gives: the newly stable text, often empty.
- **DONE:**
  - The reason is 1 `length`, 2 `stop` or 3 `cancelled`.
  - The timings are `prefill_seconds`, `decode_seconds`, `prefix_seconds` and `queued_seconds`.
  - The tail is `IncrementalText.flush()`, which ends an OpenAI stream. The text is `tokenizer.decode` of all the ids, which a JSON reply and the glm-vx stream's last event carry.
- **ERROR's code:**
  - 1 busy, HTTP 429: the scheduler had no room.
  - 2 invalid, HTTP 400: the scheduler or the tokenizer refused the request.
  - 3 failed, HTTP 500: the request stopped with an error, or its client stopped reading.
- **STATUS_REPLY** holds the JSON for `/health`. The engine writes it, and the front end copies it as it is.
- The texts and the message are UTF-8.

## Errors

- **A frame that breaks these rules ends the link.** That includes:
  - a length out of range;
  - a payload cut short, or longer than its fields;
  - an unknown kind;
  - a flag bit that is not 0;
  - a SUBMIT whose id is in use.

  The engine cancels the link's requests and stops reading. It does not try to find the next frame. It kills the
  front end and starts a new one, as it does when a front end stops on its own (see `FrontEnd` in `glm_vx/link.py`).
- **A well-formed SUBMIT with a bad value is the request's error, not the link's.** Examples: a token id past the vocabulary, a negative or non-finite temperature, a prompt past the context. The engine answers with ERROR code 2.
- **A CANCEL for an id not in use is ignored**, since the request may have just finished.
