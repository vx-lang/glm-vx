# H200: `long_topk='position'` and `fuse_glue` A/B (#32, #39), Vx 4f966fd6

The same H200 VM and toolchain as h200-20261116-m to -q (`~/glm-h200/vxtc-4f96`), branch topk-long at
a8bd6a1 extracted over the pod's repo (`r.sh`), one build for every run. Step profiles are nsys kernel
durations of one warm decode step under graph replay (`validation/step_profile.py`): GLM-5.3's width,
FP8 Linears through the cvt kernels, `mla_kernel='tc'`, norm_block 1024, index_topk 2048. `d1-*` is one
dense layer after CONTEXT tokens (as m to q); `moe2-*` is a dense layer then a MoE layer (16 routed
experts) at context 1000, so a layer boundary and a MoE output are in the step. `ab-*.txt` are
`abdiff.py`'s per-kernel sums, before and after.

| step (us) | `merge` | `position` | `position` + `fuse_glue` |
|---|---|---|---|
| 1 dense layer, context 1000 | 184.9 | 186.1 | **175.4** |
| 1 dense layer, 3001 tokens | 269.6 | 224.3 | **215.1** (214.7 w8) |
| 1 dense layer, 8192 tokens | 291.1 | 243.1 | **231.4** |
| dense + MoE layer, context 1000 | | 425.9 | **405.8** (**400.1** w8) |

At context 1000 no top-k runs (#43), so `merge` and `position` differ only by noise. `position` replaces
`topk_sort_at` + `topk_chunks_at` + `topk_merge_at` (57.9 us at 3001 tokens, 68.7 at 8192) with
`topk_select_at` (13.2, 19.6).

`fuse_glue` in one dense layer (ab-3001.txt): rmsnorm x3 8.4 -> `rmsnorm_pair` 2.6 + the first
layer's input rmsnorm 3.5; `residual_rmsnorm` 6.9 -> `residual_rmsnorm8` 5.4; layernorm + copy +
rope + store_row 2.1 + 1.2 + 1.4 + 1.2 -> `index_key_at` 2.4; store_row_bf16_at x2 + rope 2.4 + 1.4 ->
`kv_rows_bf16_at` 2.1.

`residual_rmsnorm8` first took 5.1 to 5.4 us: as a layer boundary's add (1.8) + rmsnorm (2.7) it gained
nothing. Loading the weights before the block sum as well (`w8.sh`, the committed version) takes it
to 3.1 to 3.3 us, the same bits (test-glue-w8.txt): ab-w8-moe2.txt, ab-w8-3001.txt. In the dense + MoE
step, fuse_glue then saves 25.8 us over two layers, about 13 a layer.

GPU tests: `test_spawn_hardware.py -k "glue or topk_select or residual_rmsnorm"` 11 passed (each fusion
bit for bit at GLM-5.3's widths), `test_spawn_vx_launch.py` 84, `test_resident_graph_hardware.py` 11
(fuse_glue on and off equal under 'tc', float32 and fp8 'cvt').
