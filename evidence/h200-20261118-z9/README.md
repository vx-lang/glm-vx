# 8x H200: SGLang's own GSM8K runner against SGLang --tp 8, the check on benchmarks/gsm8k.py (#63), 2026-11-18

`python -m sglang.test.few_shot_gsm8k --host 127.0.0.1 --port 30000 --num-questions 200 --parallel 1` against
SGLang 0.5.22 `--tp 8` (z6's launch flags), the questions from benchmarks/gsm8k.py's cache (the same file).
z6's and z7b's runs of it passed `--host http://127.0.0.1`, which the runner prefixes with `http://` again.

**Accuracy 0.935** (187 of 200), 0 invalid, 183 s (`gsm8k-sglang-official.log`; every question's text in
`tmp_output_gsm8k.txt`).

Against benchmarks/gsm8k.py (the same prompt, questions and scoring, through /v1/completions; z6 on SGLang 0.915,
z7 on glm-vx 0.920): `gsm8k-official-vs-client.txt` (`gsm8k_official_diff.py`).

- The two runs on SGLang give the same final answer on 186 of the 200 questions and differ on 14; the first 200
  characters of the completion agree on only 45. SGLang's greedy output is not the same from run to run
  (the same server flags, both at concurrency 1; it caches the shared 5-shot prefix and computes the rest after
  it), so a question near a tie can go either way.
- So GSM8K on these 200 questions has about two points of run-to-run spread on SGLang alone: glm-vx's 0.920 and
  SGLang's 0.915 / 0.935 are the same accuracy.
