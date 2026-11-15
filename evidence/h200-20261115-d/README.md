# H200 (Nebius), 13c57ff: the #38 cvt kernels on hardware, before their register budgets

* test-spawn-hardware: 153 passed, 1 xfailed; with `test_spawn_cvt_sm90_gives_the_ref_bits`, each sm_90
  cvt kernel gives its `_ref` build's bits.
* layer-profile-fp8fma (context 64): 321 us a dense layer, 335 before the fusions of 8ca5d8a.
* layer-profile-fp8real (cvt, context 1000, index_topk 2048, BF16 head, norm block 1024): 595 us a dense
  layer, half of it mla_partial (295 us); topk_sort 27 us.
* moe-profile, k = 8: fp8cvt gate_up 99.7 / down 45.9 us, fp8fma 93.2 / 65.1, fp8x4 158.8 / 90.6.
* matvec-profile: its `matvec_cvt_tiles` column is wrong (about 140 us at every shape): the profile
  loaded the module again at each timed launch. Fixed in the next commit; h200-20261115-f has it right.
