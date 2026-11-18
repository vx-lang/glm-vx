"""GSM8K few-shot accuracy through OpenAI /v1/completions, the same client for glm-vx and SGLang.

The prompt, the questions, the labels and the scoring are those of SGLang's
``sglang.test.few_shot_gsm8k`` (0.5.22): the test set's first ``--num-shots`` items with their answers
as the shots, then "Question: ...\\nAnswer:" for each of the first ``--num-questions``; greedy; the
completion ends at the first of "Question", "Assistant:", "<|separator|>"; the prediction is the
last integer in it. That runner speaks SGLang's native API, which glm-vx does not serve, so this
one streams /v1/completions and stops on the client: it reads text chunks until a stop string
appears, then closes the connection (both servers cancel a request whose client went away) and
cuts the text at the stop string, as a server-side stop would.

    python benchmarks/gsm8k.py --base-url http://127.0.0.1:30000 --output gsm8k-sglang.json
"""
import argparse
import ast
import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA_URL = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
STOP = ('Question', 'Assistant:', '<|separator|>')
INVALID = -9999999


def one_example(lines, i, answer):
    text = 'Question: ' + lines[i]['question'] + '\nAnswer:'
    return text + ' ' + lines[i]['answer'] if answer else text


def answer_value(text):
    numbers = re.findall(r'\d+', text.replace(',', ''))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def cut(text):
    """The text before the first stop string, and whether one was found."""
    found = [text.index(s) for s in STOP if s in text]
    return (text[:min(found)], True) if found else (text, False)


def complete(base, model, prompt, max_tokens, timeout):
    """Stream one completion; returns (text, seconds to the first text, seconds in all, text chunks)."""
    url = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    body = json.dumps(dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0, stream=True))
    start = time.perf_counter()
    conn.request('POST', '/v1/completions', body, {'Content-Type': 'application/json'})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f'HTTP {response.status}: {response.read()[:500]!r}')
    text, first, chunks = '', None, 0
    try:
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(f"server error: {event['error']}")
            piece = event['choices'][0].get('text') or ''
            if piece:
                chunks += 1
                first = time.perf_counter() - start if first is None else first
                text += piece
                text, stopped = cut(text)
                if stopped:
                    break
    finally:
        conn.close()  # a stop string ends the request: the server sees the client leave
    return text, first, time.perf_counter() - start, chunks


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', help='served model name (default: the first /v1/models id)')
    p.add_argument('--data', type=Path, help=f'GSM8K test.jsonl (default: {DATA_URL}, cached in ~/.cache/glm-vx)')
    p.add_argument('--num-questions', type=int, default=200)
    p.add_argument('--num-shots', type=int, default=5)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--timeout', type=float, default=1800, help='seconds a request may take')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('output exists; preserve evidence')
    if args.data is None:  # downloaded once, then read from the cache (a later run needs no network)
        args.data = Path.home() / '.cache' / 'glm-vx' / 'gsm8k-test.jsonl'
        if not args.data.exists():
            args.data.parent.mkdir(parents=True, exist_ok=True)
            args.data.write_bytes(urllib.request.urlopen(DATA_URL, timeout=60).read())
    raw = args.data.read_text()
    lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
    model = args.model or json.load(urllib.request.urlopen(args.base_url + '/v1/models', timeout=30))['data'][0]['id']
    shots = ''.join(one_example(lines, i, True) + '\n\n' for i in range(args.num_shots))
    results, started = [], time.perf_counter()
    for i in range(min(args.num_questions, len(lines))):
        label = answer_value(lines[i]['answer'])
        text, first, seconds, chunks = complete(args.base_url, model, shots + one_example(lines, i, False), args.max_tokens, args.timeout)
        pred = answer_value(text)
        results.append(dict(i=i, label=label, pred=pred, correct=pred == label, ttft_s=first, seconds=seconds, chunks=chunks, text=text))
        done = len(results)
        print(f'{done}/{args.num_questions} accuracy {sum(r["correct"] for r in results) / done:.3f} '
              f'({seconds:.1f} s, ttft {first if first is None else round(first, 2)} s)', file=sys.stderr, flush=True)
    n = len(results)
    summary = dict(base_url=args.base_url, model=model, num_questions=n, num_shots=args.num_shots, max_tokens=args.max_tokens,
                   accuracy=sum(r['correct'] for r in results) / n, invalid=sum(r['pred'] == INVALID for r in results) / n,
                   seconds=round(time.perf_counter() - started, 1),
                   median_ttft_s=sorted(r['ttft_s'] or 0 for r in results)[n // 2], results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}))


if __name__ == '__main__':
    main()
