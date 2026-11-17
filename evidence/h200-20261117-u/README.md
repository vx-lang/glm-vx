# H200: the indexer's scores on the tensor cores (#32), Vx a4045882

The same H200 VM and disk as h200-20261116-m to -t, with the toolchain restored in
`~/glm-h200/vxtc-a404`. The repo is main at 7c40a12, in its own checkout (`repo-u`). The steps are in
`u.sh`; h200-20261117-v has the inline-PTX kernel that followed.

- `test-spawn-hardware`: passes. `index_scores_tc_at` matches float64 sums of the same bf16 values at
  32 heads × 128, up to 32768 tokens.
- `test-resident-graph`: 2 tests failed (`test-resident-graph.txt`), both on outdated
  expectations, not on results:
  - `test_mla_tc_matches_mla_flash` counted `store_row_bf16_at`, which `fuse_glue` (now the
    default) replaces with `kv_rows_bf16_at`;
  - the new test expected `steps - 1` replays, but the two step graphs (#43) give `steps - 2`.

  After the fix: 12 passed (`test-resident-graph-rerun.txt`). With `index_kernel='tc'`, the logits
  agree with `'simt'` (f32).

Step profiles (`validation/step_profile.py`, one dense layer, nsys); `ab-*.txt` lists each kernel's
time, 'simt' then 'tc':

| context | step, 'simt' | step, 'tc' | `index_scores_at` | `index_scores_tc_at` |
|---|---|---|---|---|
| 3001 | 213.6 us | 215.8 | 11.2 | 11.5 |
| 8192 | 230.1 | 221.4 | 21.1 | 11.7 |
| 32768 | 312.9 | 252.2 | 74.5 | 14.2 |

`index_scores_tc_at` stays near 11.5 us whatever the context. ncu (`istc_one.py`, the kernel alone at
3001 tokens; numbers in the session notes, not saved):
- 24 blocks of 4 warps, 142 registers;
- 0.17 warps issued a cycle per scheduler, 5.95 cycles an instruction;
- about 5000 instructions a warp in series, most of them converting the f32 key cache to bf16 in
  shared memory, an element at a time.

At 32768 tokens `topk_select_at` (47.1 us) is now the step's largest indexer kernel.
