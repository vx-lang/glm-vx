# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.8 | 13.3 | 5.9 | 2164 | 45 | 33 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 18.6 | 2.9 | 763 | 16 | 45 | 768 x 128 | 80 |
| fp8mma16gu | moe_gate_up | 9 | 28.3 | 29.4 | 15.3 | 5.9 | 1922 | 40 | 13 | 144 x 256 | 128 |
| fp8mma16gu | moe_down | 9 | 14.2 | 14.6 | 10.3 | 2.9 | 1412 | 29 | 20 | 384 x 288 | 72 |
| fp8mma16pgu | moe_gate_up | 9 | 28.3 | 28.9 | 14.2 | 5.9 | 2029 | 42 | 9 | 144 x 256 | 128 |
| fp8mma16pgu | moe_down | 9 | 14.2 | 15.0 | 10.5 | 2.9 | 1429 | 30 | 20 | 384 x 288 | 72 |
