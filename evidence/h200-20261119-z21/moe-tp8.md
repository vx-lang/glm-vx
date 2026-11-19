# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt2 | moe_gate_up | 8 | 25.2 | 25.3 | 12.5 | 5.2 | 2023 | 42 | 30 | 512 x 128 | 91 |
| fp8cvt2 | moe_down | 8 | 12.6 | 12.6 | 16.2 | 2.6 | 779 | 16 | 43 | 768 x 128 | 80 |
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.7 | 13.5 | 5.9 | 2119 | 44 | 32 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 19.0 | 2.9 | 747 | 16 | 44 | 768 x 128 | 80 |
| fp8mma | moe_gate_up | 8 | 25.2 | 25.3 | 12.5 | 5.2 | 2024 | 42 | 31 | 512 x 128 | 91 |
| fp8mma | moe_down | 8 | 12.6 | 12.8 | 9.8 | 2.6 | 1308 | 27 | 22 | 384 x 288 | 72 |
| fp8mma | moe_gate_up | 9 | 28.3 | 28.4 | 13.1 | 5.9 | 2167 | 45 | 33 | 576 x 128 | 91 |
| fp8mma | moe_down | 9 | 14.2 | 14.5 | 10.4 | 2.9 | 1399 | 29 | 22 | 384 x 288 | 72 |
| fp8mma16 | moe_gate_up | 8 | 25.2 | 25.5 | 12.9 | 5.2 | 1981 | 41 | 30 | 512 x 128 | 91 |
| fp8mma16 | moe_down | 8 | 12.6 | 12.8 | 9.8 | 2.6 | 1308 | 27 | 19 | 384 x 288 | 72 |
| fp8mma16 | moe_gate_up | 9 | 28.3 | 28.6 | 13.4 | 5.9 | 2131 | 44 | 32 | 576 x 128 | 91 |
| fp8mma16 | moe_down | 9 | 14.2 | 14.6 | 10.3 | 2.9 | 1415 | 29 | 20 | 384 x 288 | 72 |
