import importlib.util
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('compare_bench', Path(__file__).parents[1] / 'benchmarks/compare.py')
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

FIRST, GAP = 0.12, 0.02


def serve(dialect, n_tokens=6, status=200, drop_done=False):
    """In-process fake of one engine's streaming format; returns (url, shutdown)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            if status != 200:
                self.send_response(status); self.end_headers(); self.wfile.write(b'boom'); return
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()

            def send(x):
                self.wfile.write(('data: ' + json.dumps(x) + '\n\n').encode()); self.wfile.flush()
            time.sleep(FIRST)
            for i in range(n_tokens):
                if i:
                    time.sleep(GAP)
                if dialect == 'glm-vx':
                    send({'choices': [{'token_ids': [i + 1], 'finish_reason': None}]})
                else:
                    send({'choices': [{'text': f' t{i}', 'finish_reason': None}]})
            if dialect == 'glm-vx':
                send({'choices': [{'token_ids': list(range(1, n_tokens + 1)), 'finish_reason': 'length'}],
                      'usage': {'completion_tokens': n_tokens}})
            elif dialect == 'vllm':
                send({'choices': [{'text': '', 'finish_reason': 'length'}]})
                send({'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': n_tokens}})
            else:
                send({'choices': [{'text': '', 'finish_reason': 'length'}],
                      'timings': {'prompt_per_second': 321.0, 'predicted_per_second': 55.5}})
            if not drop_done:
                self.wfile.write(b'data: [DONE]\n\n')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f'http://127.0.0.1:{server.server_port}', seen, lambda: (server.shutdown(), server.server_close(), thread.join())


@pytest.mark.parametrize('dialect', ['glm-vx', 'vllm', 'llamacpp'])
def test_each_dialect_is_measured_with_correct_arithmetic(dialect):
    url, seen, stop = serve(dialect)
    try:
        r = cmp.completion(dialect, url, [1, 2, 3], 6, timeout=10)
    finally:
        stop()
    assert r['ok'] and r['output_tokens'] == 6 and r['finish_reason'] == 'length'
    assert FIRST <= r['ttft_s'] < FIRST + 0.2
    assert 5 * GAP <= r['decode_s'] < 5 * GAP + 0.15
    assert r['latency_s'] >= r['ttft_s'] + r['decode_s'] - 1e-6
    body = seen[0]
    assert body['prompt'] == [1, 2, 3] and body['temperature'] == 0 and body['stream'] is True
    assert (dialect == 'vllm') == ('stream_options' in body) and (dialect == 'llamacpp') == ('return_tokens' in body)
    if dialect == 'llamacpp':
        assert r['server_timings']['prompt_per_second'] == 321.0


def test_failures_are_recorded_not_dropped():
    url, _, stop = serve('vllm', status=500)
    try:
        bad = cmp.completion('vllm', url, [1], 4, timeout=5)
    finally:
        stop()
    assert not bad['ok'] and 'HTTP 500' in bad['error']
    url, _, stop = serve('vllm', drop_done=True)
    try:
        cut = cmp.completion('vllm', url, [1], 4, timeout=5)
    finally:
        stop()
    assert not cut['ok'] and 'without [DONE]' in cut['error']
    assert not cmp.completion('vllm', 'ftp://x', [1], 4)['ok']


def test_summary_counts_failures_and_never_averages_them_in():
    rows = [{'ok': True, 'prompt_tokens': 100, 'output_tokens': 11, 'ttft_s': 0.5, 'decode_s': 1.0,
             'latency_s': 1.6, 'server_timings': {}},
            {'ok': True, 'prompt_tokens': 100, 'output_tokens': 11, 'ttft_s': 1.0, 'decode_s': 0.5,
             'latency_s': 1.6, 'server_timings': {'prompt_per_second': 90.0}},
            {'ok': False, 'prompt_tokens': 100, 'latency_s': 9, 'error': 'TimeoutError: slow'}]
    s = cmp.summarize(rows, 2.0)
    assert (s['requests'], s['successful'], s['failed']) == (3, 2, 1)
    assert s['errors'] == ['TimeoutError: slow']
    assert s['aggregate_output_tokens_per_s'] == 11  # 22 tokens / 2 s, failed request contributes nothing
    assert s['decode_tokens_per_s_client']['min'] == 10 and s['decode_tokens_per_s_client']['max'] == 20
    assert s['prefill_tokens_per_s_client']['p50'] == pytest.approx(150)  # median of 200 and 100
    assert s['prefill_tokens_per_s_server']['p50'] == 90.0
    assert cmp.summarize([], 1.0)['ttft_s']['p50'] is None


def test_engine_spec_parsing():
    e = cmp.parse_engine('mine=glm-vx@http://h:8000@pid=42@version=abc123')
    assert e == {'name': 'mine', 'kind': 'glm-vx', 'url': 'http://h:8000', 'pid': 42, 'version': 'abc123'}
    for bad in ('x', 'n=unknown@http://h', 'n=vllm', '=vllm@http://h', 'n=vllm@http://h@oops=1'):
        with pytest.raises(Exception):
            cmp.parse_engine(bad)


def test_memory_sampler_reads_host_peak_and_reports_missing_gpu():
    assert cmp.MemorySampler.host_peak_kib(os.getpid()) > 1000
    assert cmp.MemorySampler.host_peak_kib(2 ** 22 + 12345) is None
    with cmp.MemorySampler(os.getpid(), interval=0.01) as sampler:
        time.sleep(0.05)
    result = sampler.result()
    assert result['server_host_rss_peak_kib'] > 0
    if not result['gpu_sampled']:
        assert result['gpu_memory_used_mib_peak'] is None  # no GPU here: reported as unavailable, not 0


def test_main_runs_two_engines_and_refuses_to_overwrite(tmp_path, monkeypatch):
    a_url, _, stop_a = serve('glm-vx', n_tokens=4)
    b_url, _, stop_b = serve('llamacpp', n_tokens=4)
    out = tmp_path / 'cmp.json'
    argv = ['compare.py', '--engine', f'a=glm-vx@{a_url}@pid={os.getpid()}', '--engine', f'b=llamacpp@{b_url}',
            '--requests', '4', '--concurrency', '2', '--prompt-lengths', '5,9', '--tokens', '4', '--vocab-size', '50',
            '--warmup', '1', '--timeout', '10', '--model-note', 'test', '--output', str(out)]
    try:
        monkeypatch.setattr('sys.argv', argv)
        assert cmp.main() == 0
        report = json.loads(out.read_text())
        assert report['schema'] == 'glm-vx-engine-comparison-v1' and report['model_note'] == 'test'
        assert {e['engine']['name'] for e in report['engines']} == {'a', 'b'}
        for e in report['engines']:
            assert e['status'] == 'ok' and e['metrics']['successful'] == 4 and len(e['requests']) == 4
            assert set(e['by_prompt_length']) == {'5', '9'}
        assert report['engines'][0]['memory']['server_host_rss_peak_kib'] > 0
        assert 'vcpus' in report['environment']
        with pytest.raises(SystemExit):
            cmp.main()  # output exists: refuse
    finally:
        stop_a(); stop_b()
