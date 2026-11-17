# H200 session z (2026-11-17, aditya@204.12.188.116, Vx 6a1d8f47)

`z.sh` on repo pod-z (topk-cluster bc867d2 + moe-group). Log: `z.log`.

- Build and test-spawn-hardware: 244 passed, 1 xfailed. test-resident-graph: 3 failed, all on one bug.
  ResidentGLM's multi-block top-k buffers passed `np.uint32` where `CUDAContext.tensor` takes a dtype
  name. The same bug crashed every `topk_hist_ab` run and the 131072-token step profiles. Fixed in
  2f0a0dd; rerun as session z2.
- `moe_group_ab` (#3575 offsets): `moe-group-3575.md`.
- Step A/B at 3001 and 32768 tokens, position against ilp_hist (`ab-*.txt`): 205.0 -> 205.8 us at
  3001, 243.4 -> 228.2 at 32768.
- fold_shared, a dense layer then a MoE one with 16 experts (`ab-fold.txt`): 472.5 -> 463.2 us. The
  shared expert's gate_up, its down and the add (3 launches) are gone; the routed gate_up and down
  take one more expert.
- fuse_router, first version (bf16_matvec4's rows, then k rounds of argmax in the last block;
  `ab-fuse.txt`): router_bf16 took 18.4 us, against bf16_matvec4's 9.1 plus the router's 4.6. Slower;
  redesigned in 2f0a0dd (session z2).
