# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt2 | moe_gate_up | 7 | 22.0 | 22.1 | 12.5 | 4.6 | 1759 | 37 | 27 | 448 x 128 | 91 |
| fp8cvt2 | moe_down | 7 | 11.0 | 11.0 | 16.1 | 2.3 | 687 | 14 | 44 | 768 x 128 | 80 |
| fp8cvt2 | moe_gate_up | 8 | 25.2 | 25.3 | 12.5 | 5.2 | 2024 | 42 | 30 | 512 x 128 | 91 |
| fp8cvt2 | moe_down | 8 | 12.6 | 12.6 | 16.1 | 2.6 | 782 | 16 | 43 | 768 x 128 | 80 |
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.6 | 13.3 | 5.9 | 2152 | 45 | 33 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 18.5 | 2.9 | 766 | 16 | 45 | 768 x 128 | 80 |
| fp8mma16pgu | moe_gate_up | 7 | 22.0 | 22.1 | 10.5 | 4.6 | 2097 | 44 | 10 | 112 x 256 | 128 |
| fp8mma16pgu | moe_down | 7 | 11.0 | 11.2 | 9.2 | 2.3 | 1219 | 25 | 18 | 384 x 288 | 72 |
| fp8mma16pgu | moe_gate_up | 8 | 25.2 | 25.4 | 12.6 | 5.2 | 2012 | 42 | 9 | 128 x 256 | 128 |
| fp8mma16pgu | moe_down | 8 | 12.6 | 12.8 | 9.8 | 2.6 | 1310 | 27 | 19 | 384 x 288 | 72 |
| fp8mma16pgu | moe_gate_up | 9 | 28.3 | 29.0 | 14.4 | 5.9 | 2015 | 42 | 9 | 144 x 256 | 128 |
| fp8mma16pgu | moe_down | 9 | 14.2 | 14.7 | 10.4 | 2.9 | 1411 | 29 | 20 | 384 x 288 | 72 |
