# 8x H200: tensor parallelism on real GPUs (#62), 2026-11-18

Nebius 8x H200 node (NVSwitch, NV18 between every pair), glm-vx 46cefa3 then 0a42332's changes,
Vx 6a1d8f47 toolchain, spawn kernels built with Vx #3609 (sm_80 PTX). The model in every decode check is
the 4-layer GLM-5.3-width proxy of validation/tp_decode_check.py (3 dense + 1 MoE, 16 routed experts, FP8),
context 48, compared bit for bit with the same ranks emulated as threads on GPU 0.

Files here:
- top level: the first run (z4.sh). Its NCCL steps failed: ncclCommInitRank's NVLS setup (in-switch reduction)
  fails with CUDA error 401 on this VM, so later runs set NCCL_NVLS_ENABLE=0. The direct-read sum
  (allreduce-*) and single-GPU decode passed in that run.
- build/, tests/: pod_validate's build and GPU test steps. test-spawn-vx-host "FAIL" = every test skipped: the
  kernel bundle has PTX but no host libraries.
- z4b.sh, z4c.sh: the reruns. The node became unreachable (probably preempted) before their result
  directories were copied back. The numbers below are from their run logs as they printed. z4c/ holds the nsys
  logs and GPU 0's last three steps of each profile.

## z4b (NCCL_NVLS_ENABLE=0; graphs closed before collective.finish)
- decode-8 (NCCL all-gather + sum_ranks): PASS, bit_exact on all 8 ranks, 48 steps; 968.1 us/step (4 layers, ctx 48)
- decode-8-oneshot (tp_all_reduce v1): PASS, bit_exact all 8; 1520.5 us/step
- collective-8: f32 all_reduce/all_gather/graph replays PASS; bf16 all_reduce "FAILED" on every rank identically:
  the test's integers in [-64,64] summed over 8 ranks exceed 256 (bf16 exact range) -> test fixed (limit 256//ranks).
  NCCL all_reduce of 6144 floats in a graph: f32 19.9 us, bf16 19.1 us (8 ranks)
- decode-2 (NCCL): PASS bit_exact, 1535.1 us/step   (1 GPU: 1425.6 us/step, decode-1 in z4)
- decode-2-oneshot: PASS bit_exact, 1558.3 us/step (ran tp_all_reduce v2: installed during decode-2, see NOTE)
- collective-2: PASS, f32 all_reduce 6.8 us (rank 0)
- vx_peer_buffer_check (Vx PR #3619 c6b9b6b3a, LD_LIBRARY_PATH fixed): 2 ranks "rank 0/1: ok", 8 ranks all "ok"
  (posted: https://github.com/vx-lang/Vx/pull/3619#issuecomment-6974587933)

## z4c (tp_all_reduce v2: each rank's flag waited by its own warp; all 8 loads before the adds)
- allreduce-8 v2: PASS (bits eager + graph replays), 5.2 us/call (v1: 16.8; 2 ranks v1: 4.8)
- decode-8-oneshot v2: PASS bit_exact all 8 ranks, 840.1 us/step (v1 1520.5; NCCL 968.1)
- collective-8 (fixed test): PASS
- nsys, ctx 4, 30 timed steps, --skip-reference (wall under nsys: 1 GPU 1570, 8 NCCL 1038, 8 one-shot 1193 us):
  GPU 0 per step: 1 GPU span 1133 us (kernels 1128, lm_head bf16_matvec4 433);
  8 NCCL span 637 (kernels 580; nccl AllGather 191; lm_head 71);
  8 one-shot span 547 (kernels 511; tp_all_reduce 125 of which ~100 the first call waiting for the slowest rank; 5-6 us
  each otherwise). Host work per step (~300 us: numpy embedding, input upload, eager lm_head + gather + download)
  outside the graph.

## Not reached
- GLM-5.3 download (zai-org/GLM-5.3, 755.7 GB, not gated): 196 GB / 27 of 141 shards when the node went away
  (xet transfer stalled after 23 files; plain HTTP, 16 workers: ~350 MB/s).
- SGLang 0.5.22 installed (torch 2.14.1+cu130) in ~/glm-h200/sglang-venv.
- z5.sh (real 4-layer bits, smoke, full model, SGLang greedy + bench_one_batch) copied, not started.
