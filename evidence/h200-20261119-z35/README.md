# Everything enabled, served W1: TPOT 8.55 ms with chunked prefill, TTFT 0.62 s, E2E 9.17 s (SGLang ~9.26) (#63)

One server with every option: device_decode, moe_mma, fuse_mla_q, moe_down2, fuse_router; chunked prefill (prefill_rows
2048, cuBLASLt rows, NCCL sums, the experts grouped on the device, attention on the tensor cores: vx-7e's
`mha_prefill_tc`). The package is branch `prefill` at 3ca6fb8 plus mha-prefill-tc.

| W1 1000/1000, batch 1 | median TTFT | median TPOT | median E2E |
|---|---|---|---|
| **glm-vx, everything** (`w1.log`) | **624 ms** | **8.55 ms** (p99 ITL 8.82) | **9165 ms** |
| SGLang 0.5.22 `--tp 8` (z6, another VM) | 88.7 ms | 9.18 ms | ~9260 ms (88.7 + 999 x 9.18) |

- `greedy.log`: tp_real's prompt, coherent text; the server log has no errors.
- `mha-prefill.md` (GPU 0, an earlier step of this directory): mha_prefill_tc against mha_prefill, both checked
  against float64. 1000x1000 x 8 heads: 0.31 vs 4.7 ms; 2048x2048: 0.62 vs 18.4 ms. `hw-mha-tc.log`: 4 passed.
- `tp-real-mha-tc.*` is an out-of-memory failure: two copies of a run script ran at once after a script was
  overwritten mid-run (operator error, not the kernel).
