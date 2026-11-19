# vx-7e's topk_select_bitmap_at on one H200: 14.8 us at 32K tokens, SGLang's fast_topk_v2 21.4 (#63, item 6)

Branch topk-fast (c9f1bf0 / 87e9614, merged into `prefill` as ed22f2a), Vx main 8146's vxc. GPU 0 of the 8x H200;
`validation/topk_hist_ab` (`topk-bitmap.md`, every output checked, median of 20, after an L2 flush):

| tokens | bitmap | ilp_hist | topk_select_at | step's indexer before -> after (bitmap) |
|---|---|---|---|---|
| 3001 | 11.2 us | 13.9 | 16.2 | 23.4 -> 18.0 us |
| 8192 | 12.0 | 16.0 | 22.9 | 30.7 -> 19.4 |
| 32768 | 14.8 | 28.4 | 68.1 | 79.7 -> 27.7 |
| 65536 | 20.9 | 43.2 | 145.1 | 163.3 -> 41.8 |

`hw-spawn.log`: 11 passed. `hw-graph.log`: 3 passed (logits bit for bit against 'position'). The in-model 32K runs
were not run; W3 served with it is `../h200-20261119-z36`.
