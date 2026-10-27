# Fused MoE expert kernels: hidden 6144, expert width 2048, 256 experts

CUDA events (no ncu counters): median of 5, L2 flushed before each; GB/s is expected MB over GPU time. Peak DRAM 3352 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f32 | moe_gate_up | 1 | 100.7 | - | 92.9 | 30.0 | 1084 | 32 | - |  x  |  |
| f32 | moe_down | 1 | 50.3 | - | 43.2 | 15.0 | 1164 | 35 | - |  x  |  |
| f32 | moe_gate_up | 2 | 201.3 | - | 104.1 | 60.1 | 1933 | 58 | - |  x  |  |
| f32 | moe_down | 2 | 100.7 | - | 71.2 | 30.0 | 1414 | 42 | - |  x  |  |
| f32 | moe_gate_up | 4 | 402.7 | - | 191.8 | 120.1 | 2100 | 63 | - |  x  |  |
| f32 | moe_down | 4 | 201.3 | - | 124.2 | 60.1 | 1622 | 48 | - |  x  |  |
| f32 | moe_gate_up | 8 | 805.3 | - | 368.2 | 240.2 | 2187 | 65 | - |  x  |  |
| f32 | moe_down | 8 | 402.7 | - | 231.7 | 120.1 | 1738 | 52 | - |  x  |  |
| bf16 | moe_gate_up | 1 | 50.3 | - | 265.0 | 15.0 | 190 | 6 | - |  x  |  |
| bf16 | moe_down | 1 | 25.2 | - | 64.2 | 7.5 | 392 | 12 | - |  x  |  |
| bf16 | moe_gate_up | 2 | 100.7 | - | 277.2 | 30.0 | 363 | 11 | - |  x  |  |
| bf16 | moe_down | 2 | 50.3 | - | 118.2 | 15.0 | 426 | 13 | - |  x  |  |
| bf16 | moe_gate_up | 4 | 201.3 | - | 532.3 | 60.1 | 378 | 11 | - |  x  |  |
| bf16 | moe_down | 4 | 100.7 | - | 233.1 | 30.0 | 432 | 13 | - |  x  |  |
| bf16 | moe_gate_up | 8 | 402.7 | - | 1041.5 | 120.1 | 387 | 12 | - |  x  |  |
| bf16 | moe_down | 8 | 201.3 | - | 468.5 | 60.1 | 430 | 13 | - |  x  |  |
| fp8 | moe_gate_up | 1 | 25.2 | - | 152.7 | 7.5 | 165 | 5 | - |  x  |  |
| fp8 | moe_down | 1 | 12.6 | - | 59.4 | 3.8 | 212 | 6 | - |  x  |  |
| fp8 | moe_gate_up | 2 | 50.3 | - | 157.1 | 15.0 | 320 | 10 | - |  x  |  |
| fp8 | moe_down | 2 | 25.2 | - | 105.1 | 7.5 | 239 | 7 | - |  x  |  |
| fp8 | moe_gate_up | 4 | 100.7 | - | 284.9 | 30.0 | 353 | 11 | - |  x  |  |
| fp8 | moe_down | 4 | 50.3 | - | 210.4 | 15.0 | 239 | 7 | - |  x  |  |
| fp8 | moe_gate_up | 8 | 201.4 | - | 554.0 | 60.1 | 363 | 11 | - |  x  |  |
| fp8 | moe_down | 8 | 100.7 | - | 405.8 | 30.0 | 248 | 7 | - |  x  |  |
| int4 | moe_gate_up | 1 | 13.0 | - | 96.6 | 3.9 | 134 | 4 | - |  x  |  |
| int4 | moe_down | 1 | 6.5 | - | 39.9 | 1.9 | 163 | 5 | - |  x  |  |
| int4 | moe_gate_up | 2 | 26.0 | - | 105.6 | 7.7 | 246 | 7 | - |  x  |  |
| int4 | moe_down | 2 | 13.0 | - | 71.6 | 3.9 | 181 | 5 | - |  x  |  |
| int4 | moe_gate_up | 4 | 51.9 | - | 187.8 | 15.5 | 276 | 8 | - |  x  |  |
| int4 | moe_down | 4 | 26.0 | - | 132.3 | 7.7 | 196 | 6 | - |  x  |  |
| int4 | moe_gate_up | 8 | 103.8 | - | 350.5 | 31.0 | 296 | 9 | - |  x  |  |
| int4 | moe_down | 8 | 51.9 | - | 256.9 | 15.5 | 202 | 6 | - |  x  |  |
