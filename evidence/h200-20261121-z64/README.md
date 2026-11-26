# Passkey at 30K tokens on z63's branch (#63)

This ran on 2026-11-21 on the same 8x H200, after z63. The prompts are about 30000 tokens long, past `index_topk`, so
the DSA indexer selects. The point is a long-context check of z63's BF16 `weights_proj`, which main computes from
float32 rows.

- **`z64.sh`:** meant to run the branch, main and SGLang in turn. It was cut to the branch's check alone to save
  budget, and stopped while the branch's server came up.
- **`z64b.sh`:** at the maintainer's request, it ran only the 30K prompts, on the branch's server
  (`passkey30.py`).

Result: 5 of 5 keys right, at depths 0.1, 0.3, 0.5, 0.7 and 0.9, about 50 s each (`ttft-passkey30.json`). Main and
SGLang weren't run.
