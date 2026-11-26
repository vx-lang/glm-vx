# glm-vx main's kernels under Vx main 5d313543e (one H200)

This ran on 2026-11-26 on a single Nebius H200 (gpu-h200-sxm, 1gpu-16vcpu-200gb; driver 580.196.07, CUDA 13.0's
ptxas). Vx main 5d313543e is 125 commits after 10c6f478f, the Vx that glm-vx was tested with. Its vxc was built on
the box with `cargo build --release`. The script is `z75.sh`.

1. **The build stops at E8016.** Vx #3803 (a488cc3d9, 2026-11-23) wants an `mlir!` block that has `ensures` clauses
   inside `unsafe { }`, since the compiler trusts those clauses. The fix:
   - `wrap_mlir_unsafe.py` wraps 162 helper bodies of the form `return mlir!(...) { ... };` under gpu/spawn/. That
     includes the templates in gen_index_mma.py, gen_index_chunk.py, gen_moe_mma.py and the fp8_gemm `.vx.in` files.
   - `wrap_mlir_unsafe2.py` wraps the other two: moe_mma's `let sm: i64 = mlir!(...)` and fp8_gemm's `even_pitch!`.
   - Every generator's `--check` passes after it.
2. **After the fix, every kernel builds.** 150 PTX files in all: hand kernels, spawn, index_mma, moe_mma, fp8_gemm.
3. **The PTX matches production.** Against the same kernels built by Vx 10c6f478f without the fix (`ptx-diff.txt`):
   - 131 are identical.
   - 18 differ only in the text of assert messages.
   - One has a code change: `moe_mma/fp8_moe_vxmm16_down_at`, the `moe_mma='vx16'` option, which is off by default.
     It has the same counts of mma.sync, ld.global, ldmatrix and cvt, the same 52 registers and 42176 B of shared
     memory, and 128 fewer lines (fewer register moves).
4. **The hardware tests pass:** 502 passed, 7 skipped, 1 xfailed (`glm-tests.log`). That is the same count as the
   branches built by 10c6f478f.

With Vx 10c6f478f, the fixed sources still compile, but `fp8_gemm_m64`'s PTX has no `.extern .shared`. Its build
then stops: "the manifest says 98368 dynamic shared bytes, but the PTX has no .extern .shared memory". So this change
needs Vx main from 2026-11-23 on.
