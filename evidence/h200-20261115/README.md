# H200, 2026-11-15: first ncu counters, cuBLASLt correctness at 4 layers (#29)

Nebius VM (NVIDIA H200 141 GB, driver 580.173.02, CUDA 13.0 only, glibc 2.39, 196 GB RAM), toolchain
bundle vx-toolchain-89832571, repo c180296 (copied in with `git archive`, hence REPO_REV). The VM was
**preemptible**: it was powered off twice (clean `systemd-poweroff` at 01:56:31 UTC, again near 02:11),
the second time with its disk. These files were copied off as each step finished. The SGLang profile
never ran (SGLang 0.5.22 was installed and the 9-layer proxy written, both lost with the disk).

## Setup that differed from the H100 pods

* The image has CUDA 13 only. Vx's dispatch library links `libcudart.so.12` and `libcublas.so.12`, so
  the first `build-spawn` failed with "Symbols not found: vx_plugin_dispatch_async, ...". The CUDA 12
  wheels (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12) run on the 580 driver; with their lib dirs on
  LD_LIBRARY_PATH every build passed. pod_validate.sh now does this itself when ldd reports them missing.
* The system cuBLASLt is 13.1 (130101), new enough for the block scales: no wheel was needed.
* **ncu reads the counters** (`RmProfilingAdminOnly: 0`): the first per-kernel DRAM bytes and SM
  throughput of this project. The H100 pods all refused them (ERR_NVGPUCTRPERM).
* The % of peak columns use 4814 GB/s for the H200's DRAM.

## Correctness

* `test-hardware`: 69 passed, 6 skipped (the weights_b + BF16-output cases cuBLASLt refuses), as on
  the H100.
* `resident-fp8lt-full4` (ResidentGLM fp8_kernels='cublaslt', full geometry, 4 layers, 48 tokens):
  every dense FP8 Linear of layer 0 is within 1.1e-4 to 2.3e-4 of the float64 sum of its quantized
  operands (gate 1e-3): pass. 3264 cuBLASLt launches.
* Whole-model drift against the float32-activation logits: max |diff| 3.71 against max |logit| 5.60,
  top-1 differs. At 2 layers (H100, 2026-11-14) it was 0.41 / 4.71. Quantizing the activations moves
  each projection by 2.3-3.2% (drift column of the JSON), and with random weights that compounds layer
  by layer. Whether SGLang's FP8 serving drifts as much on the same proxy is still the open question.

## One decode step, hand FP8 kernels, under ncu (layer-profile-fp8)

Each kernel alone with cold caches (not a step time). Full width, 4 layers, context 64.

| Group | Kernels | GPU us | DRAM MB | % of peak BW | us at peak BW |
|---|---|---|---|---|---|
| layer 0 (dense) | 31 | 1116 | 459 | 9 | 95 |
| layer 3 (MoE) | 26 | 1185 | 566 | 10 | 118 |
| head (float32 lm_head) | 2 | 854 | 3810 | 93 | 791 |

What the counters say:

* **fp8_matvec is instruction-bound, not memory-bound.** The large projections (o_proj 211 us,
  down 163, gate 152, up 151, q_b 68) run at 60-65% SM throughput and only 10-11% of DRAM bandwidth.
  The e4m3 decode costs more instructions than the bytes it saves.
* **The plain float32 matvec reaches 93.5% of peak** (lm_head, 3.8 GB in 847 us): the hand kernel's
  load structure can stream at full bandwidth, so what fp8_matvec needs is a cheaper decode.
* **Small projections sit at ~70 us whatever their size**: indexer.wk reads 0.8 MB in 70.7 us
  (32 blocks), kv_a 3.6 MB in 74 us (144 blocks), q_a 12.6 MB in 73 us (512 blocks). Each block walks
  all k = 6144 serially; with few blocks that is latency, not bandwidth (needs split-K or more rows in
  flight).
* The MoE layer's fp8_moe_gate_up (324 us, 13% of DRAM, 40% SM) and fp8_moe_down (157 us) behave the
  same way.
* The rest of a layer: mla_partial 20 us, index_scores 15, matvec_heads 12.6 and 12.1 (57% and 43%
  of DRAM), topk_sort 8, rmsnorms 3-7, and 2-3 us elementwise kernels.

Not reached: the x4 and cuBLASLt layer profiles, moe/matvec/topk/mla profiles, SGLang.
