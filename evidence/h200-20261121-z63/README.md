# TTFT branch end to end: the router's gate and the indexer's weights_proj over the prompt's rows on cuBLASLt BF16 (#63)

This ran on 2026-11-21 on the same 8x H200, right after z61. The script is `z63.sh` and the log is `z63.log`.
- **Code:** origin/main 0589bc6 plus `ttft-bf16-gemm.patch`. Over the prompt's rows, the router's gate and the DSA
  indexer's `weights_proj` run as cuBLASLt BF16 matmuls, as SGLang does. Main uses float32 rows there.
- **Kernels:** every kernel was built with Vx main 10c6f478f.
- **Serving:** z61's glm-vx options and z61's client.
- **Rounds:** two, each on a fresh server. The third was skipped to save budget: the account balance was about $30
  at 06:35 UTC (the `.done-r3-*` markers said so).

| workload | metric | z63, round 1 | z63, round 2 | z61 main (3 rounds) |
|---|---|---|---|---|
| W1 | median TPOT, ms | 8.52 | 8.52 | 8.52 |
| W1 | median TTFT, ms | 366.5 | 363.8 | 626.0 |
| ShareGPT | median TPOT, ms | 8.30 | 8.30 | 8.30 |
| ShareGPT | median TTFT, ms | 296.9 | 300.1 | 314.2 |
| W3 | median TPOT, ms | 9.54 | 9.54 | 9.54 |
| W3 | median TTFT, ms | 54945 | 54985 | 114123 |
| GSM8K | accuracy, one run | 0.940 | | 0.925 |

The greedy check (`r*-greedy.log`) gives `tp_real`'s 32 tokens exactly. Main gives 8 (z61).

This is not a result yet: two rounds, compared with z61's rounds on the same VM earlier the same day, not alternated
with them. For the 30K passkey check of the same server, see `../h200-20261121-z64`.
