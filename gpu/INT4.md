# INT4 and batched/grouped weight kernels (M5)

Sources: `int4.vx`, `fp8_batched.vx`; host wrappers `quant_ops.py` (`QuantKernels`, kept out of
`runtime.py`); tests `test_quant_simt.py` (CPU SIMT) and `test_quant_hardware.py` (A100).
Oracles are `glm_vx/quant.py`.

## INT4 layout

Codes in [-8, 7], two per byte (element 2i in the low nibble, two's complement), odd trailing
element padded with a zero nibble, packed `uint8 [N, ceil(K/2)]`; one scale per (row, 128-column
group), `[N, ceil(K/128)]`, passed as raw 16-bit words: binary16 (`scale_kind` 0) or bfloat16
(`scale_kind` 1). The kernels decode both exactly (subnormals, inf, NaN included) and multiply in
float32. `glm_vx/quant.py`'s `int4_quantize` produces this layout with fp16 scales.

### Real checkpoints: compressed-tensors pack-quantized (W4A16)

Published GLM INT4 checkpoints, including `lowbitcoffee/GLM-5.2-W4A16` for this architecture
(`glm_moe_dsa`), are llm-compressor/compressed-tensors `pack-quantized`: symmetric INT4, group 128,
no zero point, `weight_packed int32 [N, ceil(K/8)]`, `weight_scale bf16 [N, K/128]`,
`weight_shape int64 [2]` per Linear (experts stored one Linear each). Codes are stored as
`code + 8`, element k at bits `4*(k % 8)` of word `k // 8`. Read as little-endian bytes that is
this layout's nibble order exactly; only the encoding differs (offset-8 = two's complement with the
top bit flipped), so the repack is `XOR 0x88` on the bytes (`compressed_int4_to_packed`).
`SafetensorsCheckpoint.compressed_int4(prefix)` loads one Linear, validating shapes and dtypes and
rejecting a zero point, an activation-order index (`weight_g_idx`) or non-finite scales;
`compressed_int4_scheme(config["quantization_config"])` rejects every scheme the kernels do not
implement (asymmetric, other group sizes or bit widths, activation or KV quantization).

Checked against the library (`validation/compressed_int4_oracle.py`, needs torch +
compressed-tensors) on three real Linears fetched by range request (`validation/fetch_tensors.py`):
codes equal, and our weights rounded to bf16 are bitwise equal to its decompression. Note that the
library (and so the Transformers reference) rounds `code * scale` to bf16, up to ~0.3% of the
largest weight; the kernels keep the exact float32 product, so model-level parity against that
reference carries that difference. AutoAWQ (`quant_method: awq`, zero points, interleaved packing),
GPTQ and GGUF Q4_K are not supported.

## Entries (block = multiple of 32, 1-D grid, one warp per output element)

| entry | computes | grid.x |
|---|---|---|
| `glm_vx_gpu_int4_matvec(out, packed, scales, x, N, K, scale_kind)` | `out[N]` | ceil(N / warps) |
| `glm_vx_gpu_int4_matmul(out, packed, scales, x, expert_ids, M, N, K, E, scale_kind)` | `out[M,N]` | ceil(N*M / warps) |
| `glm_vx_gpu_fp8_matmul(out, weight, scales, x, expert_ids, M, N, K, E)` | `out[M,N]` | ceil(N*M / warps) |

`E == 0`: one dense matrix shared by all M rows (`expert_ids` is not read; pass a null pointer).
`E > 0` (grouped): weights/scales are stacked `[E, ...]` experts of identical shape and
`expert_ids[M]` (int32) selects each row's expert, so all experts of a layer run in **one launch**.
For top-k routing the caller expands tokens to T*k rows with one id each. An id outside `[0, E)`
writes zeros to that row (unrouted slot). Tasks are ordered n-major so the M rows sharing a weight
row are adjacent warps. FP8 semantics (E4M3FN, float32 multiplier scales per 128x128 block) match
`fp8.vx`; accumulation is per 128-column tile, then scaled, as in `fp8_matvec_blocked`.

Limits: M, N, K, E are int32; offsets are int64; `N*M` warps must fit the grid. Not fused: bias,
activation, output quantization. Experts of different shapes need separate launches.

## Not measured

No performance claim. These kernels re-read each weight row once per activation row (L2 may
absorb it); no tensor cores, no weight reuse in registers. Tune only after a measurement.
