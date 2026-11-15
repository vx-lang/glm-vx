# Fused MoE expert kernels: hidden 6144, expert width 2048, 256 experts

ncu, each kernel alone with cold caches. Peak DRAM 4814 GB/s.

| Format | Kernel | k | Expected MB | DRAM MB | GPU us | us at peak BW | GB/s | % peak BW | SM % | grid x block | regs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8cvt | moe_gate_up | 1 | 25.2 | 25.2 | 12.8 | 5.2 | 1971 | 41 | 30 | 512 x 128 | 96 |
| fp8cvt | moe_down | 1 | 12.6 | 12.6 | 7.7 | 2.6 | 1630 | 34 | 25 | 384 x 128 | 127 |
| fp8cvt | moe_gate_up | 2 | 50.3 | 52.7 | 23.0 | 10.5 | 2295 | 48 | 33 | 1024 x 128 | 96 |
| fp8cvt | moe_down | 2 | 25.2 | 25.2 | 12.7 | 5.2 | 1986 | 41 | 28 | 384 x 128 | 127 |
| fp8cvt | moe_gate_up | 4 | 100.7 | 103.4 | 41.5 | 20.9 | 2491 | 52 | 38 | 2048 x 128 | 96 |
| fp8cvt | moe_down | 4 | 50.3 | 53.0 | 20.5 | 10.5 | 2578 | 54 | 34 | 384 x 128 | 127 |
| fp8cvt | moe_gate_up | 8 | 201.4 | 204.2 | 72.1 | 41.8 | 2833 | 59 | 44 | 4096 x 128 | 96 |
| fp8cvt | moe_down | 8 | 100.7 | 103.2 | 37.1 | 20.9 | 2781 | 58 | 37 | 384 x 128 | 127 |
| fp8fma | moe_gate_up | 1 | 25.2 | 25.2 | 14.4 | 5.2 | 1757 | 36 | 45 | 512 x 128 | 122 |
| fp8fma | moe_down | 1 | 12.6 | 12.6 | 46.2 | 2.6 | 273 | 6 | 19 | 3072 x 256 | 95 |
| fp8fma | moe_gate_up | 2 | 50.3 | 52.9 | 26.6 | 10.5 | 1990 | 41 | 49 | 1024 x 128 | 122 |
| fp8fma | moe_down | 2 | 25.2 | 25.2 | 51.4 | 5.2 | 491 | 10 | 24 | 3072 x 256 | 95 |
| fp8fma | moe_gate_up | 4 | 100.7 | 103.5 | 48.4 | 20.9 | 2137 | 44 | 54 | 2048 x 128 | 122 |
| fp8fma | moe_down | 4 | 50.3 | 52.7 | 57.1 | 10.5 | 923 | 19 | 34 | 3072 x 256 | 95 |
| fp8fma | moe_gate_up | 8 | 201.4 | 203.9 | 92.7 | 41.8 | 2200 | 46 | 57 | 4096 x 128 | 122 |
| fp8fma | moe_down | 8 | 100.7 | 103.5 | 64.9 | 20.9 | 1595 | 33 | 50 | 3072 x 256 | 95 |
