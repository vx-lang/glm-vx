# H200 (Nebius): fp8_moe_cvt_down with four rows a subgroup and all k experts (#38)

Each subgroup now takes four output rows and every routed expert in ascending id order, instead of
one expert a subgroup and two rows a block. Each row's sums are in the same order, so the outputs are
the old kernel's bit for bit (checked on the CPU against the old `_ref` build at 5 shapes before it
was replaced).

* h200-20261115-g: test_spawn_hardware 153 passed, 1 xfailed (each sm_90 cvt kernel equals its `_ref`).
  moe-profile (ncu, k = 1 / 2 / 4 / 8): fp8cvt down 7.7 / 12.7 / 20.5 / 37.1 us, before 33.3 / 35.2 /
  40.0 / 45.9 (h200-20261115-d); gate_up 12.8 / 23.0 / 41.5 / 72.1.
* CUDA events against the old kernel, 6144 x 2048 experts: k = 8 49 -> 42 us, k = 4 41.5 -> 26,
  k = 1 32 -> 10.5. A second tile a loop pass (167 registers) gave 41.4; two rows a subgroup, 49.
* resident-fp8cvt-full4: max |logit diff| 8.8e-6 against float32, top-1 equal (layer 3 is an MoE layer).
* layer-profile-fp8real: the MoE layer's routed experts take 70.8 + 38.3 us against SGLang's 76.6 + 34.0.
  Each layer is about 550 us, of which mla_partial is 294 (#33, Vx#3469).
