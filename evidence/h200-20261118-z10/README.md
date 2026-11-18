# 8x H200: glm-vx's TP8 server, the router's projection and choice in one launch (#63), 2026-11-18

z8's best server (followers polling, `--tp-option moe_down2=true`) with `--tp-option fuse_router=true`
(gpu/spawn/router_bf16: a block a router row, the top-8 choice in the last block to finish, in place of
bf16_matvec4 and the router, 9.3 + 5.9 us a MoE layer in z5's profile). W1, the same client and flags as z6-z8.

| server | W1 TPOT median | p99 | TTFT median | vs SGLang (9.18) |
|---|---|---|---|---|
| z8 c: polling + moe_down2 | 9.84 ms | 9.88 | 9052 ms | 1.072 |
| + fuse_router | **9.53 ms** | 9.53 | 8681 ms | **1.038** |

fuse_router sums the router's logits in the hand bf16_matvec's order (not bf16_matvec4's) and takes expf from
libdevice: not z7's bits, so z11 measures this server's GSM8K, ShareGPT and W3 again.
