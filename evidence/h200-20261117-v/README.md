# H200: the indexer's scores in inline PTX, index_kernel='mma' (#32)

The same VM, toolchain and checkout as h200-20261117-u, with the files of the next commit copied over
(`v.sh`). `index_scores_mma_at` (`gpu/spawn/index_mma/gen_index_mma.py`) does the same arithmetic
as `index_scores_tc_at`, but each warp's work is one inline-PTX block in an `mlir!` macro:
- its 32 tokens' key rows and q are loaded as float4s straight from the f32 cache, and each pair is
  rounded to bf16 with one `cvt.rn.bf16x2.f32`;
- 64 `mma.sync.m16n8k16`;
- `w * relu(scale * s)` in registers, the heads added with butterfly shuffles, and the result handed
  to its lane with `shfl.idx`.

Vx writes `out[first + lane]`. The kernel uses 132 registers (sm_90), no shared memory and no spills.

- `test-spawn-hardware.txt`: 10 passed. Both kernels match float64 sums of the same bf16 values,
  1 to 32768 tokens, `max_tokens` 3001 included.
- `test-resident-graph.txt`: 13 passed. With `'mma'` the logits agree with `'simt'` under replay at
  32 heads × 128 (600 steps, index_topk 128).

Step profiles against h200-20261117-u's 'simt' (`ab-*.txt`):

| context | step, 'simt' | step, 'mma' | `index_scores_at` | `index_scores_mma_at` | SGLang `sm90_fp8_paged_mqa_logits` |
|---|---|---|---|---|---|
| 3001 | 213.6 us | 206.8 | 11.2 | **3.9** | 3.6 (1000 and 2048 tokens, h200-20261116) |
| 8192 | 230.1 | 214.5 | 21.1 | **5.1** | |
| 32768 | 312.9 | 242.7 | 74.5 | **6.7** | |
