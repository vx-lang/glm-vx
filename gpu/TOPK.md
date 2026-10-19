# Device stable top-k (DSA indexer)

`topk.vx`, host wrapper `gpu/topk.py` (`TopK.topk_stable`, `scratch_ints`), tests
`test_topk_simt.py` (CPU SIMT) and `test_topk_hardware.py` (A100; skipped without a GPU).

## Contract

```text
void glm_vx_gpu_topk_stable(int32_t* out_ids, const int32_t* scores /* float32 bits */,
                            int32_t* scratch, int32_t tokens, int32_t k)
```

`out_ids[0 : min(k, tokens)]` equals `np.argsort(-scores, kind='stable')[:min(k, tokens)]`:
descending score, ties broken by lower position. Nothing past `min(k, tokens)` is written;
`tokens <= 0`, `k <= 0` or any block size other than 256 write nothing. Launch with
`grid.x = 1, block.x = 256`. `scores` is the float32 buffer passed as `int32` (the PTX path has
no f32 bit cast; the kernel orders by an exact monotone key from the bits). `scratch` is int32
device memory of at least `66824 + 2*min(k, tokens)` elements (`gpu.topk.scratch_ints`), contents
irrelevant on entry; it must not alias `scores` or `out_ids`.

- **-0.0 and +0.0 tie** (same key), as numpy does.
- **NaN** (any sign or payload) sorts after every number including -inf, and NaNs tie with each
  other by position; this is what `argsort(-scores)` does. Tested, though the model should not
  produce NaN.
- +/-inf are ordinary extreme values.

## Algorithm

One block, no atomics (results are deterministic). Four 8-bit radix passes find the k-th largest
key: every thread histograms a strided slice into its own 256 bins, bin owners sum across threads,
thread 0 picks the digit. The selected set is every key above the threshold plus the first R
positions equal to it (R from the last pass), compacted in position order via per-thread chunk
counts, then ranked by (key descending, index ascending) to write the sorted ids.

## Limits and cost (not measured)

Select cost is about 4 passes over `tokens`, one block only, so it does not use the rest of the
GPU; the final ranking is O(min(k,tokens)^2 / 256) per thread, which is fine for k of a few
thousand and wrong for huge k with huge tokens (use a sort there). The 64K-int scratch for
per-thread histograms is global memory traffic. No performance claim: measure on hardware before
tuning (a multi-block select would be the next step for 128K+ tokens). Validated only under the
CPU SIMT emulator so far.
