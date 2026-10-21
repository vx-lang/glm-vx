"""OpenAI /v1/completions compatibility, checked the way sglang.bench_serving uses it.

`bench_serving_completion` replays sglang 0.5.10.post1 `sglang/bench_serving.py`
`async_request_openai_completions` (the function behind `--backend vllm` and `--backend sglang-oai`):
the same default payload and the same per-line parsing, with aiohttp's line iteration replaced by
http.client.readline. Any deviation from that parser (a comment line, a chunk with `choices: []`,
a terminal chunk repeating the whole text) shows up here as a failure or a wrong count.
"""
import http.client
import json
import threading
import time

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from glm_vx.detok import IncrementalText
from glm_vx.scheduler import Scheduler
from glm_vx.server import Handler, Server
from test_scheduler import Model as ToyModel


def word_tokenizer(vocab):
    """Every token is one whitespace-separated word, so decoded text is stable after each token."""
    tok = Tokenizer(models.WordLevel({f'w{i}': i for i in range(vocab)}, unk_token='w0'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def byte_tokenizer():
    """Byte-level BPE without merges: one token per UTF-8 byte, so characters split across tokens."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    tok = Tokenizer(models.BPE({c: i for i, c in enumerate(sorted(alphabet))}, []))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def bench_serving_completion(port, model, prompt, output_len, extra_request_body=None):
    """Sync replay of sglang 0.5.10.post1 bench_serving.async_request_openai_completions."""
    extra_request_body = extra_request_body or {}
    payload = {'model': model, 'prompt': prompt, 'best_of': 1, 'max_tokens': output_len, 'stream': True}
    if 'temperature' not in extra_request_body:
        payload['temperature'] = 0.0
    if 'ignore_eos' not in extra_request_body:
        payload['ignore_eos'] = True  # not args.disable_ignore_eos
    payload.update(extra_request_body)
    output = {'generated_text': '', 'success': False, 'latency': 0.0, 'ttft': 0.0, 'itl': [],
              'text_chunks': [], 'error': '', 'output_len': 0}
    generated_text, ttft = '', 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    try:
        conn.request('POST', '/v1/completions', json.dumps(payload), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        if response.status == 200:
            while True:
                chunk_bytes = response.readline()
                if not chunk_bytes:
                    break
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = remove_prefix(chunk_bytes.decode('utf-8'), 'data: ')
                latency = time.perf_counter() - st
                if chunk == '[DONE]':
                    pass
                else:
                    data = json.loads(chunk)
                    if data['choices'][0]['text']:
                        timestamp = time.perf_counter()
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output['ttft'] = ttft
                        else:
                            output['text_chunks'].append(data['choices'][0]['text'])
                            output['itl'].append(timestamp - most_recent_timestamp)
                        most_recent_timestamp = timestamp
                        generated_text += data['choices'][0]['text']
                        output_len = (data.get('usage') or {}).get('completion_tokens', output_len)
            output.update(generated_text=generated_text, success=True, latency=latency, output_len=output_len)
        else:
            output['error'] = (response.reason or '') + ': ' + response.read().decode()
    except Exception as exc:  # bench_serving records the traceback and marks the request failed
        output['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        conn.close()
    return output


@pytest.fixture
def tiny_endpoint():
    from glm_vx.backend import NumpyBackend
    from glm_vx.model import Model, tiny_weights
    from glm_vx.tiny import tiny_config
    cfg = tiny_config()
    sched = Scheduler(Model(cfg, tiny_weights(cfg, seed=7), NumpyBackend()))
    server = Server(('127.0.0.1', 0), sched, word_tokenizer(cfg['vocab_size']))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown(); server.server_close(); sched.close(); thread.join()


def post(port, body):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
    c.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse(); status, raw = r.status, r.read(); c.close()
    return status, raw


def sse(raw):
    return [json.loads(l[6:]) for l in raw.decode().splitlines() if l.startswith('data: ') and l != 'data: [DONE]']


def test_bench_serving_parser_measures_text_stream(tiny_endpoint):
    port, name = tiny_endpoint.server_port, tiny_endpoint.model_name
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    c.request('GET', '/v1/models'); models_list = json.loads(c.getresponse().read()); c.close()
    assert models_list['data'][0]['id'] == name  # bench_serving takes the model id from here
    prompt = 'w5 w17 w200 w3'
    status, raw = post(port, {'model': name, 'prompt': prompt, 'max_tokens': 7})
    assert status == 200
    reference = json.loads(raw)
    out = bench_serving_completion(port, name, prompt, 7)
    assert out['success'], out['error']
    assert out['generated_text'] == reference['choices'][0]['text']  # deltas are not duplicated
    assert len(word_tokenizer(256).encode(out['generated_text']).ids) == 7  # bench_serving retokenizes
    assert out['output_len'] == 7 and 0 < out['ttft'] <= out['latency']
    assert len(out['itl']) == 6  # one text chunk per token after the first


def test_openai_stream_shape_usage_and_legacy_framing(tiny_endpoint):
    port, name = tiny_endpoint.server_port, tiny_endpoint.model_name
    body = {'prompt': [1, 2, 3], 'max_tokens': 5, 'stream': True}
    status, raw = post(port, {**body, 'model': name, 'ignore_eos': True, 'stream_options': {'include_usage': True}})
    assert status == 200 and raw.endswith(b'data: [DONE]\n\n')
    events = sse(raw)
    tokens, final, usage_chunk = events[:5], events[5], events[6]
    assert len(events) == 7
    assert all(len(e['choices'][0]['token_ids']) == 1 and e['choices'][0]['finish_reason'] is None for e in tokens)
    assert final['choices'][0]['finish_reason'] == 'length' and 'token_ids' not in final['choices'][0]
    assert usage_chunk['choices'] == [] and usage_chunk['usage']['completion_tokens'] == 5
    text = ''.join(e['choices'][0]['text'] for e in events[:6])
    status, legacy = post(port, body)  # no `model`: the glm-vx framing existing clients parse
    legacy = sse(legacy)
    ids = [e['choices'][0]['token_ids'][0] for e in legacy[:-1]]
    assert legacy[-1]['choices'][0]['token_ids'] == ids == [e['choices'][0]['token_ids'][0] for e in tokens]
    assert legacy[-1]['choices'][0]['text'] == text == word_tokenizer(256).decode(ids)


def test_openai_stream_heartbeat_is_a_blank_line(monkeypatch):
    release = threading.Event()

    class Slow(ToyModel):
        def forward(self, t, c):
            release.wait(3); return super().forward(t, c)
    monkeypatch.setattr(Handler, 'heartbeat_seconds', 0.2)
    sched = Scheduler(Slow())
    server = Server(('127.0.0.1', 0), sched, word_tokenizer(8))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        threading.Timer(0.8, release.set).start()
        out = bench_serving_completion(server.server_port, server.model_name, 'w1 w2', 3)
        assert out['success'], out['error']
        assert out['generated_text'] and out['output_len'] == 3
        release.clear(); threading.Timer(0.8, release.set).start()
        status, raw = post(server.server_port, {'prompt': [1], 'max_tokens': 1, 'stream': True})
        assert b': keepalive\n\n' in raw  # the glm-vx framing keeps its SSE comment
    finally:
        release.set(); server.shutdown(); server.server_close(); sched.close(); thread.join()


def test_ignore_eos_and_option_validation():
    class EOS(ToyModel):
        config = {**ToyModel.config, 'eos_token_id': [4]}
    sched = Scheduler(EOS())
    server = Server(('127.0.0.1', 0), sched)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        port = server.server_port
        stop = json.loads(post(port, {'prompt': [1, 2], 'max_tokens': 6})[1])
        assert stop['choices'][0]['finish_reason'] == 'stop' and stop['choices'][0]['token_ids'][-1] == 4
        full = json.loads(post(port, {'prompt': [1, 2], 'max_tokens': 6, 'ignore_eos': True, 'n': 1, 'best_of': 1})[1])
        assert full['choices'][0]['finish_reason'] == 'length' and full['usage']['completion_tokens'] == 6
        for bad in ({'ignore_eos': 'yes'}, {'n': 2}, {'best_of': True}, {'model': 'other'}, {'stop': 'x'},
                    {'stream_options': {'include_usage': True}},
                    {'stream': True, 'stream_options': {'continuous_usage_stats': True}},
                    {'stream': True, 'stream_options': {'include_usage': 1}}):
            status, raw = post(port, {'prompt': [1], 'max_tokens': 2, **bad})
            assert status == 400, (bad, raw)
        assert b'glm-vx-tiny-random' in post(port, {'prompt': [1], 'model': 'x'})[1]
    finally:
        server.shutdown(); server.server_close(); sched.close(); thread.join()


def test_incremental_text_never_splits_utf8():
    tok = byte_tokenizer()
    text = 'naïve 世界 🙂 ok'
    ids = tok.encode(text).ids
    assert len(ids) == len(text.encode())  # really one token per byte
    stream = IncrementalText(tok)
    deltas = [stream.push(t) for t in ids]
    assert all('�' not in d for d in deltas)
    assert ''.join(deltas) + stream.flush() == text
    assert deltas[ids.index(tok.encode('世').ids[0])] == ''  # first byte of 世 is held back
    cut = IncrementalText(tok)  # output that ends mid-character is released by flush, as decode() does
    held = ''.join(cut.push(t) for t in tok.encode('a🙂').ids[:3])
    assert held + cut.flush() == tok.decode(tok.encode('a🙂').ids[:3])
    assert IncrementalText(None).push(3) == '' and IncrementalText(None).flush() == ''
