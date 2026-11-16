# Vx 4f966fd6 on the H200: #43 on the GPU, and the MLA absorbs from kv_b_proj's FP8 codes (#42)

Same H200 VM and disk as before (Nebius, 204.12.171.122 this time). The toolchain is Vx main 4f966fd6, built
on the pod by `scripts/pod_build_bundle.sh` (~/glm-h200/vxtc-4f96).

## This directory: GPU tests and the step profile before the absorb fix

`pod_validate.sh` with every kernel built: all GPU tests pass, nothing skipped (`summary.txt`):
spawn hardware 153 (1 xfailed), spawn host 287, launch 79, resident graph 9, fused attention 1,
many 6, speculative 12. These include #45's graph tests (separate step graphs for contexts that fit in
index_topk and those that don't).

`step-profile/`: nsys kernel durations of one warm decode step (graph replay) at GLM-5.3's width, one
dense layer, FP8 Linears through the cvt kernels, index_topk 2048, after CONTEXT tokens
(`mla_step_profile.py KERNEL CONTEXT`; `trace_mla.py` lists the attention block). At context 1000 the
step has no `index_scores_at` or top-k kernel any more (#43); at 2048 and 3000 (2049 and 3001 tokens,
past index_topk) it has them, as before.

The absorbs in these traces are still float32 `matvec_heads` (11.4 + 11.8 us). The first version of #42
(`fp8_matvec_heads_cvt` on per-head layouts) required nope, value and rank to be multiples of 128, but
GLM-5.3's nope is 192 (value 256), so ResidentGLM fell back to float32: the full-width FP8 check gave the
same launches and uploaded bytes as before #42. The CPU test had used nope 128.

## ../h200-20261116-k5: the absorbs from the codes as stored

kv_b_proj is 448 rows a head (192 nope + 256 value) by 512 latent columns, in 128 x 128 scale blocks, so a
block can hold two heads. Both kernels now read the codes and scales as stored, one upload (the float32
copy is gone), each row with its own block's scale:

- query absorb, `fp8_matvec_cols_cvt`: q_latent[h] = W_k[h]^T q_nope[h], each head's nope rows summed down
  the latent columns. A block takes one head's 128 columns, a lane one word, 16 subgroups twelve rows each
  with every load first.
- value absorb, `fp8_matvec_heads_cvt`: output row (h, v) reads stored row 448 h + 192 + v; four rows a warp.

In the decode step (nsys, `tc-1000`, `tc-3000`), against SGLang's absorbs in its own decode step at
context 1000 (evidence/h200-20261116/step-profile):

| absorb | float32 matvec_heads | FP8, now | SGLang |
|---|---|---|---|
| query | 11.4 us | **5.4** (5.3 at 3000) | 5.9 |
| value | 11.8 | **6.3** (6.4) | 6.5 |

Tried on the way (`absorb_time.py`: each kernel one CUDA graph, L2 flushed, median of 50, an empty graph
about 3 us; `sweep_cols.py` / `sweep_heads.py` write the variants):

| kernel | variant | us |
|---|---|---|
| query | the transposed nope blocks through the row kernel (32768 rows of 256 columns, padded so each 128 columns keep one scale) | 15.9 in the step |
| query | column sums, 8 subgroups, 4 rows a step | 10.9 (7.7 in the step) |
| query | 16 subgroups, 4 / 6 / 12 rows a step | 11.2 / 9.6 / **8.4** |
| query | 8 subgroups 8 / 12, 12 x 16, 24 x 8, 32 x 6 | 12.1 / 10.4, 9.3, 12.5, 9.8 |
| value | 2 rows a warp (4 subgroups) | 10.5 |
| value | 4 rows a warp, 4 / 2 / 8 subgroups | **9.3** / 9.4 / 9.6 |
| value | 8 rows a warp | 10.8 (9.5 with `spawn-maxnreg` 128) |

`spawn-maxnreg` budgets didn't help either kernel (query 8.9 at 128, value 10.1 at 96).

`test-absorbs.txt`: the kernels' host and launch tests on the GPU, 17 passed.

## ../h200-20261116-k5v: full width

`resident_compare --fp8 --fp8-kernels cvt --geometry full --layers 4 --tokens 48`: logits within 9.1e-6
of float32 on the dequantized weights (largest 5.6), the same top token, 768 absorb launches, 13.86 GiB
uploaded (14.12 with the float32 kv_b_proj).
