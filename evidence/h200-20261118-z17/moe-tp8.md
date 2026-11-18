# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt | moe_gate_up | 8 | 25.2 | 25.3 | 12.5 | 5.2 | 2027 | 42 | 30 | 512 x 128 | 91 |
| fp8cvt | moe_down | 8 | 12.6 | 12.6 | 19.4 | 2.6 | 652 | 14 | 26 | 384 x 128 | 123 |
| fp8cvt | moe_gate_up | 9 | 28.3 | 28.6 | 13.2 | 5.9 | 2156 | 45 | 33 | 576 x 128 | 91 |
| fp8cvt | moe_down | 9 | 14.2 | 14.2 | 21.7 | 2.9 | 655 | 14 | 27 | 384 x 128 | 123 |
| fp8cvt2 | moe_gate_up | 8 | 25.2 | 25.3 | 12.7 | 5.2 | 1988 | 41 | 30 | 512 x 128 | 91 |
| fp8cvt2 | moe_down | 8 | 12.6 | 12.6 | 16.3 | 2.6 | 773 | 16 | 42 | 768 x 128 | 80 |
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.5 | 13.3 | 5.9 | 2138 | 44 | 33 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 18.1 | 2.9 | 782 | 16 | 46 | 768 x 128 | 80 |
