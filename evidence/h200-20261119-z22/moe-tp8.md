# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.9 | 13.4 | 5.9 | 2154 | 45 | 32 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 18.5 | 2.9 | 766 | 16 | 45 | 768 x 128 | 80 |
| fp8mma16gu | moe_gate_up | 9 | 28.3 | 29.4 | 15.2 | 5.9 | 1927 | 40 | 13 | 144 x 256 | 128 |
| fp8mma16gu | moe_down | 9 | 14.2 | 14.6 | 10.2 | 2.9 | 1436 | 30 | 21 | 384 x 288 | 72 |
