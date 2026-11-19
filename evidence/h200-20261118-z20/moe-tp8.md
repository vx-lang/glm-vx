# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.8 | 13.5 | 5.9 | 2135 | 44 | 32 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 18.5 | 2.9 | 766 | 16 | 45 | 768 x 128 | 80 |
| fp8mma | moe_gate_up | 9 | 28.3 | 28.8 | 13.4 | 5.9 | 2142 | 44 | 32 | 576 x 128 | 91 |
| fp8mma | moe_down | 9 | 14.2 | 14.7 | 11.2 | 2.9 | 1312 | 27 | 25 | 384 x 288 | 72 |
| fp8mma16 | moe_gate_up | 9 | 28.3 | 28.7 | 13.4 | 5.9 | 2146 | 45 | 33 | 576 x 128 | 91 |
| fp8mma16 | moe_down | 9 | 14.2 | 14.9 | 10.5 | 2.9 | 1413 | 29 | 24 | 384 x 288 | 72 |
