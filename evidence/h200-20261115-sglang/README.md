# H200 (Nebius, second VM), 2026-11-15: the FMA matvec kernels, and SGLang's batch-1 decode op by op

NVIDIA H200 141 GB, driver 580.173.02, CUDA 13.0 image, ncu counters enabled; toolchain bundle
vx-toolchain-89832571. Persistent SSD `disk-a4` at /mnt/disk-a4 holds the toolchain, venvs, SGLang
0.5.22 and the 9-layer FP8 proxy (validation/write_proxy, GLM-5.3 geometry, 3 dense + 6 MoE layers,
256 experts top-8, the GLM-5.3 tokenizer). Evidence of the glm-vx runs:

* `../h200-20261115-a`: repo 5bbb5ae, builds and the FP8 matvec profile (one-row fp8_matvec_fma).
* `../h200-20261115-b`: 5bbb5ae, GPU tests, resident-fp8fma-full4, ncu layer profiles fp8fma and
  fp8lt. fp8lt's report failed on cuBLASLt's own kernels; layer-profile-fp8lt.json/.md were rebuilt
  from its saved ncu CSV with 445bcd6's layer_profile.
* `../h200-20261115-c`: 34fc5c9 (fp8_matvec_fma two rows a warp, fp8_matvec_fma_tiles), matvec
  profile and ncu layer profile.
* `../h200-20261115-proto`: the prototypes timed on the way and their ncu output.

## Correctness

* test-hardware 69 passed / 6 skipped; test-spawn-hardware 142 passed / 1 xfailed (5bbb5ae, with
  the one-row fp8_matvec_fma's device check).
* resident-fp8fma-full4 (one-row fp8_matvec_fma, 4 full-width layers, 48 tokens): max |logit diff|
  8.6e-6 against max |logit| 5.60, top-1 equal (x4's gate, 1e-4). cuBLASLt on the first H200 VM:
  3.71, top-1 different.

## FP8 dense matvec (matvec-profile, CUDA events, L2 flushed, median of 20), us

| Projection | x4 | fma (2 rows) | fma_tiles | cuBLASLt (quant + mm) |
|---|---|---|---|---|
| q_a_proj 2048 x 6144 | 19.5 | 19.1 | **14.6** | 20.3 |
| q_b_proj 16384 x 2048 | 33.5 | 23.7 | **22.3** | 24.4 |
| kv_a_proj 576 x 6144 | 16.7 | 16.4 | **12.7** | 19.9 |
| o_proj 6144 x 16384 | 118.1 | 51.4 | 57.1 | **49.6** |
| mlp.gate/up 12288 x 6144 | 68.4 | 39.9 | 39.4 | **33.6** |
| mlp.down 6144 x 12288 | 88.2 | 42.4 | 46.2 | **41.6** |
| shared_expert.down 6144 x 2048 | 20.5 | **11.6** | 12.6 | 15.6 |

ncu found the one-row fma kernel waiting on L1 (89% of stall cycles on L1TEX scoreboards, L1 74%
busy: a lane reads 64 bytes of x per 16 bytes of codes). Two rows a warp halves that; the tiles
map reads x one 512-byte run per load (L1 26% busy) but at 87 registers runs at 23% occupancy and
is issue-bound. Four rows a warp (119 registers) was slower than two.

## One decode step under ncu (each kernel alone, cold caches), us

| | hand fp8 (first VM) | fma one-row | fma 34fc5c9 | cuBLASLt |
|---|---|---|---|---|
| dense layer | 1116 | 443 | **335** | 365 |
| MoE layer | 1185 | 530 | **472** | 511 |
| head (float32 lm_head, 3.8 GB) | 854 | 838 | 838 | 838 |

## SGLang 0.5.22 batch-1 decode on the proxy (one_batch, CUDA graphs, nsys)

Median decode step 3.29 ms (304 tok/s), 2826 us of kernels in 278 launches; prefill of 1000 tokens
22 ms. One dense layer is about 210 us of the step; its kernels in order (us):

| Op | SGLang | glm-vx 34fc5c9 (ncu, cold) |
|---|---|---|
| q_a + kv_a (one DeepGEMM, activations to e4m3 first) | 1.9 + 13.1 | 12.2 + 10.6 |
| q_b | 1.6 + 15.4 | 18.0 |
| indexer: wq_b/wk, rope+hadamard+quant, logits, top-k | 10.7 + 7.1 + 3.6 + 3.5 + 1.6 (wk on a second stream) | 9.0 + 10.3, index_scores 15.6, topk_sort 8.1, rope/copy/scale ~12 |
| MLA: absorb, rope, kv store, flash decode, combine, absorb | 5.9 + 1.6 + 1.2 + 3.3 + 13.9 + 5.8 + 6.5 | matvec_heads 12.4 + 12.3, mla_partial 20.8, mla_merge 3.7 |
| o_proj | 1.4 + 33.3 | 46.2 |
| add + rmsnorm | 3.5 (fused) | 2.8 + 7.3 |
| gate + up (one GEMM) | 1.7 + 43.3 | 33.6 + 33.7 |
| swiglu | 1.8 | 2.9 |
| down | 1.3 + 27.4 | 35.4 |

An MoE layer's experts: router GEMM 2.5, router 3.0, align 2.1, fused_moe gate/up 76.6, down 34.0,
sum 16.3 (us). glm-vx: fp8_moe4_gate_up 160.9, fp8_moe4_down 91.7 for the same 201 + 100 MB.
The head: SGLang's BF16 lm_head 427 us; glm-vx's resident FP8 path keeps the lm_head float32
(3.8 GB) at 838 us, 95% of peak.

Caveat: glm-vx's numbers are ncu's, each kernel alone with cold caches; SGLang's are a warm CUDA
graph, where small kernels gain a few us. The comparison of large kernels holds.

## Where glm-vx is behind, largest first

1. Routed experts: 161 + 92 against 77 + 34 us a MoE layer (fp8_moe4_* need what fp8_matvec_fma got).
2. lm_head stored float32, not BF16: 838 against 427 us.
3. o_proj and down: 46 and 35 against DeepGEMM's 33 and 27.
4. Separate gate/up and q_a/kv_a launches (x read twice): 67 against 43, 23 against 13.
5. The indexer: index_scores 15.6 and topk_sort 8.1 against 3.6 and 1.6.
6. One-block rmsnorm (7.3 us) against fused add+rmsnorm (3.5).
