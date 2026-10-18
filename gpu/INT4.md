# INT4 and batched/grouped weight kernels (M5)

Sources: `int4.vx`, `fp8_batched.vx`; host wrappers `quant_ops.py` (`QuantKernels`, kept out of
`runtime.py`); tests `test_quant_simt.py` (CPU SIMT) and `test_quant_hardware.py` (A100).
Oracles are `glm_vx/quant.py`.

## INT4 layout (open decision: defaulted, not final)

This is `quant.py`'s symmetric group-128 layout, **not** GPTQ/AWQ/GGUF Q4: codes in [-8, 7], two per
byte (element 2i in the low nibble, two's complement), odd trailing element padded with a zero
nibble, packed `uint8 [N, ceil(K/2)]`, scales `float16 [N, ceil(K/128)]` (passed as raw bits; the
kernel decodes binary16 itself, including subnormals/inf/NaN). A different on-disk layout would
need only a repack on load, or a change to the two nibble/scale reads in `int4.vx`.

## Entries (block = multiple of 32, 1-D grid, one warp per output element)

| entry | computes | grid.x |
|---|---|---|
| `glm_vx_gpu_int4_matvec(out, packed, scales, x, N, K)` | `out[N]` | ceil(N / warps) |
| `glm_vx_gpu_int4_matmul(out, packed, scales, x, expert_ids, M, N, K, E)` | `out[M,N]` | ceil(N*M / warps) |
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
