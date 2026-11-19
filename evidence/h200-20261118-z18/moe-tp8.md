# Fused MoE expert kernels: hidden 6144, expert width 256, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt2 | moe_gate_up | 9 | 28.3 | 28.9 | 13.9 | 5.9 | 2082 | 43 | 31 | 576 x 128 | 91 |
| fp8cvt2 | moe_down | 9 | 14.2 | 14.2 | 18.3 | 2.9 | 774 | 16 | 46 | 768 x 128 | 80 |
| fp8mma | moe_gate_up | 9 | 28.3 | 28.7 | 13.5 | 5.9 | 2129 | 44 | 32 | 576 x 128 | 91 |
| fp8mma | moe_down | 9 | 14.2 | 14.2 | 13.4 | 2.9 | 1056 | 22 | 19 | 384 x 288 | 127 |
