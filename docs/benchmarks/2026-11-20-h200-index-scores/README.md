# The indexer's scores and top-k kernels on one H200: MLIR / NVVM ops against inline PTX (2026-11-20)

glm-vx's DSA indexer kernels (`gpu/spawn/index_mma/`) had their warp work in inline PTX. Branch
`nvvm-ops-not-ptx` rewrote it as MLIR / NVVM ops. This folder times each kernel both ways, so that a kernel goes
to ops only where it is not slower ([#65](https://github.com/vx-lang/glm-vx/issues/65)).

## Claim

Kernel times, one launch after an L2 flush, the median of 20 launches; then the median of 3 rounds. "ops" is
the MLIR / NVVM ops build, "ptx" the inline PTX build; GLM-5.3's indexer shapes (32 heads of 128, k = 2048), one
query, on one H200.

| kernel | ops / ptx, geometric mean over 6 token counts | smallest, largest | ops faster at |
|---|---|---|---|
| `index_scores_mma_at` (f32 keys; the served `index_kernel='mma'`) | **1.046** | 1.026, 1.079 | 0 of 6 |
| `index_scores_mma_hist_at` | 0.993 | 0.985, 1.000 | 5 of 6 |
| `index_scores_mma16_at` (bf16 keys, `index_kernel='mma16'`) | 0.989 | 0.979, 1.000 | 4 of 6 |
| `index_scores_mma16_hist_at` | 0.996 | 0.986, 1.009 | 3 of 6 |
| `topk_select_ilp_at` (with `--kernel mma` / `mma16`) | 0.986 / 0.985 | 0.972-1.000 / 0.974-1.000 | 5 / 5 of 6 |
| `topk_select_ilp_hist_at` | 0.977 / 0.975 | 0.959-0.994 / 0.958-0.993 | 6 / 6 of 6 |
| `topk_select_at` | 0.999 / 0.999 | 0.996-1.001 / 0.994-1.001 | 4 / 4 of 6 |
| `topk_mb_list_at` + `topk_mb_write_at` | 0.994 / 0.995 | 0.987-1.000 / 0.991-1.000 | 5 / 5 of 6 |
| `topk_select_bitmap_at` | 0.992 / 0.992 | 0.983-1.000 / 0.983-1.000 | 2 / 3 of 4 |

So `index_scores_mma_at` is slower as ops at every context length, 2.6% to 7.9%: glm-vx main keeps it (and its
`_hist` twin, which shares the warp body) as inline PTX. Every other converted kernel is as fast as the inline
PTX or faster, and stays ops.

## Setup

What both sides share:

| | |
|---|---|
| GPU | one NVIDIA H200 (141 GB HBM3e), driver 580.173.02, CUDA 13.0 |
| machine | Nebius gpu-h200-sxm, 1gpu-16vcpu-200gb, Ubuntu 24.04.5, kernel 6.11.0-1016-nvidia |
| compiler | vxc of Vx main 10c6f478f; PTX checked by CUDA 13.1 ptxas (sm_90), compiled by the driver at load |
| script | `validation/topk_hist_ab.py` (the same file on both builds), `--kernel mma` and `--kernel mma16` |
| inputs | `numpy.random.default_rng(1)`: random keys, queries and head weights; 32 heads of 128; k = 2048; 3001, 8192, 32768, 65536, 131072 and 200000 tokens of context |
| timing | each launch alone: a 512 MB copy flushes L2, then CUDA events; the median of 20 launches a run |
| rounds | 3 a kernel option, the builds' order alternating (ops-ptx, ptx-ops, ops-ptx), the GPU idle before each run |

What differs:

| | ptx | ops |
|---|---|---|
| glm-vx commit | origin/main 7970a7b | nvvm-ops-not-ptx 8eae0c0 (now on main as fa51723, except as noted) |
| the kernels' warp work | inline PTX in `mlir!` blocks | MLIR / NVVM ops in `mlir!` blocks |
| registers (ptxas, sm_90) | as ops, for every kernel timed here | as ptx |

## Every input

`scores` is `index_scores_mma_at` with `--kernel mma` and `index_scores_mma16_at` with `--kernel mma16`;
`scores_hist` is its `_hist` twin. Each cell is ops / ptx: the median over the 3 rounds in microseconds, then
[the fastest, the slowest] round, then the ratio of the medians.

| tokens | `--kernel mma`: scores | scores_hist | `--kernel mma16`: scores | scores_hist |
|---|---|---|---|---|
| 3001 | 7.3 [7.2-7.4] / 6.9 [6.9-7.1] = 1.058 | 6.4 [6.3-6.5] / 6.5 [6.4-6.5] = 0.985 | 6.9 [6.8-6.9] / 7.0 [7.0-7.0] = 0.986 | 6.3 [6.2-6.4] / 6.3 [6.1-6.4] = 1.000 |
| 8192 | 8.2 [8.1-8.2] / 7.6 [7.6-7.8] = 1.079 | 7.3 [7.3-7.3] / 7.3 [7.3-7.6] = 1.000 | 7.2 [7.1-7.3] / 7.2 [7.2-7.2] = 1.000 | 6.8 [6.7-6.9] / 6.9 [6.8-7.2] = 0.986 |
| 32768 | 12.0 [11.9-12.1] / 11.7 [11.5-11.8] = 1.026 | 12.4 [12.4-12.5] / 12.5 [12.5-12.8] = 0.992 | 9.4 [9.4-9.7] / 9.6 [9.6-9.7] = 0.979 | 11.6 [11.4-11.6] / 11.5 [11.5-11.6] = 1.009 |
| 65536 | 19.0 [18.8-19.0] / 18.4 [18.3-18.4] = 1.033 | 20.7 [20.5-20.8] / 20.8 [20.8-20.9] = 0.995 | 13.0 [13.0-13.1] / 13.2 [13.1-13.2] = 0.985 | 18.3 [18.3-18.6] / 18.5 [18.4-18.7] = 0.989 |
| 131072 | 28.4 [28.4-28.6] / 27.3 [27.2-28.0] = 1.040 | 33.1 [32.8-33.4] / 33.5 [33.4-33.6] = 0.988 | 21.3 [21.2-21.5] / 21.6 [21.5-21.7] = 0.986 | 28.9 [28.7-28.9] / 28.9 [28.7-28.9] = 1.000 |
| 200000 | 37.0 [36.7-37.2] / 35.6 [35.6-35.9] = 1.039 | 45.3 [45.1-45.6] / 45.5 [45.2-45.5] = 0.996 | 27.9 [27.9-27.9] / 27.9 [27.9-28.3] = 1.000 | 38.5 [38.4-38.8] / 38.7 [38.6-38.8] = 0.995 |

Every other column of every run, the top-k kernels included, is in `runs.csv` (744 rows: kernel option, round,
build, tokens, column, microseconds) and in `logs/<kernel>-r<round>-<build>.md`.

## The same work

`topk_hist_ab` checks every output of every run: the `_hist` twins' scores equal the plain kernels' bit for bit,
the bins equal NumPy's, and each top-k set equals `np.sort(np.argsort(-scores, kind='stable')[:2048])`. All 12
runs passed. Both builds run the same launches on the same inputs; only the PTX inside the kernels differs.

Why `index_scores_mma_at` is slower as ops: both builds give 472 SASS instructions and 132 registers (ptxas 13.1,
sm_90), the same mix but for one branch, and about 70% of them in the same order. As ops the 16-byte global loads
come later relative to the `HMMA`s; the inline PTX kept them earlier. Loading the A rows in pairs or dropping
`.rn` from the ops did not change the SASS ([#65](https://github.com/vx-lang/glm-vx/issues/65)).

## Limits

- One GPU, one kernel at a time, warm launches after an L2 flush: not a decode step or a served token. These
  kernels are a small part of a GLM-5.3 decode step.
- The served configuration (`index_kernel='mma'`) uses `index_scores_mma_at`, which stays inline PTX; the bf16-key
  kernels (`mma16`) are not on the served path.
- 3 rounds a kernel option, on one machine on one day; earlier 2-round data from another session (z59) agrees.
- The driver compiles the PTX at load (CUDA 13.0); another driver may schedule the SASS differently.

## Commands

`run.sh` is the measurement. `env.txt` records the machine, the compiler and the commits. To rebuild a side:

```sh
git checkout 7970a7b    # or 8eae0c0
VXC=/path/to/vxc PTXAS=/path/to/ptxas PTXAS_ARCH=sm_90 gpu/build.sh && gpu/spawn/build.sh && gpu/spawn/fp8_gemm/build.sh
python -m validation.topk_hist_ab --kernel mma --output mma.md
```
