# Journal: MLA decode, from 120 us to 26 us (2026-11-15 to 2026-11-16)

This journal records the work on GLM-5.3's MLA decode attention for #33 (part of #29, beating SGLang at
batch 1). It covers what was tried, what each step measured, and what was kept or dropped, in order.
The kernels and options as they stand are described in [../MLA.md](../MLA.md).
Evidence folders:
- `evidence/h200-20261115-i`: flash
- `evidence/h200-20261115-j`: tensor cores
- `evidence/h200-20261116`: merge rewrite

All timings are from an NVIDIA H200 (Nebius VM; its disk `/mnt/disk-a4` survived restarts, and the
address changed twice). GLM-5.3's MLA shape is 64 heads, latent rank 512, rope 64, with up to 2048
selected tokens. A random permutation of a 4096-token cache stands in for the selection.

## Where we started

At the start of the day `ResidentGLM`'s best MLA kernel was `mla_heads_at` (four heads a block). It
took 45-120 us a step at 256-2048 selected tokens, against 13.9 us for SGLang's FlashMLA kernel.

ncu showed `mla_heads_at` issue-bound: about 600 warp instructions per head and token, against 34
for the arithmetic. 56% of its stalls were at barriers, and it reread each head's query from shared
memory for every token.

The natural fix, flash decoding with each lane's query and sums kept in registers for the whole
token loop, could not be written in Vx. Lane-private state did not survive between the lane loops
that a warp-wide sum separates. We filed this as Vx #3469 and the user had it fixed.

## Part 1: flash decoding in registers (`mla_kernel='flash'`)

### Vx #3475: lane-owned tiles stay in registers

vx-94's PR #3475 (merged as 243ea31c) keeps a shared-memory tile in registers when every access is
the lane's own element, `t[w][..][lane]`, with constant loops unrolled. Our first `mla_flash_at`
used it:
- four heads a subgroup, with each lane's queries, sums, max and sum in such tiles;
- `lanes::sum` for each score.

With the PR's vxc it had no local memory and no shared-memory traffic in the token loop. Four heads
a subgroup took 199 registers; two heads took 112.

The first sweep (`evidence/h200-20261115-i/flash_sweep.txt`, flash kernel plus merge, CUDA events with
L2 flushed) gave, at the best split count for each, 32.2 / 56.1 / 73.8 us at 256 / 1000 / 2048 tokens. That was 1.6x faster than
`mla_heads_at` (52 / 82 / 120), with output within 1.8e-7 of `mla_chunks_at`. We posted the numbers
on Vx #3469.

vx-7e and vx-94 then asked whether the pod's old vxc (89832571) had built that PTX. It had not: the
PTX was built on the dev box with the PR's vxc, and its sha256 matched a local 243ea31c build. We
corrected the issue comment to say so.

### The merge was the bottleneck

Timing the flash kernel and the merge apart showed the merge taking 53-59 us at 64 splits, more
than the flash kernel itself (41 us). Each thread walked the splits in a `while` loop with a run-time
bound, and there were only 64 blocks.

| attempt | result |
|---|---|
| A branch-free `for s in 0..64` with a clamped index and a select | ptxas unrolled it, but the time barely moved (53 us) |
| ncu | latency, not bandwidth: 148 GB/s at 12% occupancy; 32 registers could not keep 64 loads in flight |
| Four subgroups of 16 splits, four columns a lane, then a block-level merge of the four | 53 -> 20 us |

### The flash kernel itself

ncu put the flash kernel at 25% occupancy, waiting on two dependent global loads (the token id, then
its row) for every token.

| variant | change | 2048 tokens | kept? |
|---|---|---|---|
| flash3 | load the next token's row, and the id two tokens on, while scoring the current one | 41 -> 29 us under `.maxnreg 128` (139 registers unbudgeted, one block an SM) | yes |
| flash4 | fused dot products (`fma32!` instead of mul + add; scores no longer bit-equal to `mla_chunks_at`) | 1-4% | no |
| flash5 | two tokens a pass with swapped row buffers, to drop 18 MOVs a token | none (it spilled) | no |

After prefetching, ncu called the kernel issue-bound: 16.2 M warp instructions, about 120 per head
and token.

Final flash plus merge, as events: 22 / 29 / 40 us at 256 / 1000 / 2048. That was 3x
`mla_heads_at`'s speed, with errors of at most 1.8e-7 against `mla_chunks_at` and at most 4.6e-7
against float64.

### Into ResidentGLM

`mla_kernel='flash'` uses up to 64 splits of about 8 tokens (16 for the other kernels), with the
numerator and stats viewed per group of 16 heads.

The CPU test with a 16-head, rank-512 model ran for over 30 minutes in the emulator, which runs an
OS thread per lane, so the logits test moved to the GPU file. On the H200, 600 replayed steps through
1-64 splits gave logits within 1e-4 of `'chunks'`; the measured gap was 4.7e-7.

`validation/mla_profile.py` gained flash. At ResidentGLM's split policies it measured 21.3 / 28.7 /
39.9 us against `mla_heads_at`'s 149 / 162 / 172.

Commits: 5a97e8b, 16f10a4, 28703a5.

## Part 2: bf16 cache and tensor cores (`mla_kernel='tc'`)

### The decision

The flash kernel issued about 120 instructions per head and token, and trimming them gained little.
Closing the gap meant doing the two products, scores and P @ rows, as matrix multiplies on the
tensor cores, as FlashMLA does.

Vx's subgroup tile matmuls (`acc + (a as f32) @ (b as f32)`, lowered to ldmatrix + mma.sync) need
bf16 or f16 inputs, and our cache was f32. The options were:
1. a bf16 cache, as SGLang uses;
2. TF32 mma, which needs new Vx support;
3. staying on f32.

The user chose bf16: "we can't beat SGLang if we are stuck in a higher precision datatype".

A bf16 cache alone would not halve the time: the f32 kernel was instruction-bound, with DRAM at 3%
and L2 at 10%. The gain had to come from the tensor cores.

### The first kernel and the race proof

The design: block (16 heads, one 32-token chunk). It gathers the chunk's rows into shared tiles, then
computes S = rows @ q_T with mma, a softmax, and O = P @ rows with mma. GLM-5.3 selects at most 2048
tokens, so 64 splits of 32 always suffice, a block never sees more than one chunk, and no online
rescaling is needed.

Vx's rules shaped the code:
- a matmul operand must be a whole indexed tile (no row-range views), so tile halves became part of
  the tile shapes (`[4, 2, 16, 128]`);
- the race proof needs write indices linear in the loop variables (no `t / 16`, `w % 4`), and splits
  the work by tile shape instead;
- a lane loop inside a subgroup loop must have exactly 32 passes;
- `-1` as an `if` value needs a typed `let`.

It compiled to 38 mma.sync and 76 ldmatrix, used 96 registers, and needed 93 KB of dynamic shared
memory (over 48 KB; our launcher already handled that). It was correct on the CPU (bf16 test against
float64) and on the H200 (5e-4 to 1e-3), but slow: about 33 us for every count, against flash's 22 us
at 256 tokens.

### Finding the fixed cost

| step | change | kernel time |
|---|---|---|
| ncu | 74% of the stall samples before the first mma; the top stall a shared store waiting on a global load; the 32-row gather loop not unrolled (id, then row, then store, one row at a time) | |
| ids first | ids loaded first, gather unrolled | 33 -> 30 us |
| queries | query loads unrolled | 30 -> 27 us (still software bf16 rounding: `setp.nan` / `selp 0x7FC0`) |
| prep kernel | `mla_q_bf16_at` converts the queries once, in the tile layout, and blocks copy them with `lanes::copy_async`; blocks past the step's splits return at once | about 26 us |

Two side findings:
- **Event timing floor.** A near-empty kernel measures 3.2 us. Sequences of three
  launches "took" 160-300 us because Python's launch time leaked into the events. From then on every sequence was
  timed as a captured CUDA graph.
- **The prep kernel** took 11 us at first, with 16 serial loads per thread on 4 blocks; with 64
  blocks it took 5 us.

ncu then showed the gather writing each bf16 element with its own 16-bit shared store, plus PRMT/SHF
and address arithmetic: about 1,600 SASS instructions a warp before the first mma, at 2 warps a
scheduler. Eight subgroups helped at 256 tokens (23 us), but the unrolled gather needed 226
registers. That allowed one block an SM and two waves at 2048 (42 us).

### Asking for compiler help

The user said to ask the Vx sessions for compiler work. Three asks went out:
1. **A row-gather copy into a shared tile** (cp.async, rows chosen by an id list, zero rows for bad
   ids). vx-7e wrote `lanes::copy_rows_async` (#3487, PR #3490). We agreed the signature, with ids in
   a shared i32 tile and `[first, col]`.
2. **Wide shared stores of neighbouring bf16** (#3488). vx-94 found that the ldmatrix swizzle
   `at ^ (line << 3)` hid the adjacency from LLVM (PR #3493).
3. **`as bf16` as one `cvt.rn.bf16.f32`** (#3489). The arith-expand pass meant for x86 also ran on
   the NVIDIA kernels. vx-7e took it (PR #3492).

We sent repro kernels for 2 and 3, which showed the bad PTX. vx-94 also wrote `lanes::scale_rows`
(#3485, PR #3491) for online softmax across chunks.

On the H200 we ran each PR's GPU test for its author:
- #3491: `in registers: 32 272 64 544`, `in shared memory: 32 32 64 544`;
- #3493: four and eight bf16 per lane, on both the H100 and A100 machine files;
- the first GPU run of #3490.

All passed with both of Vx's code generators.

### `lanes::copy_rows_async` and after

| change | 256 | 2048 | notes |
|---|---|---|---|
| copy_rows_async (`cp.async.cg ... 16`) | 23 -> 12.4 us | 42 -> 21.4 us | 140 registers |
| `.maxnreg 128` | | 16.9 us | 127 registers, two blocks an SM |
| all eight subgroups in every phase | | no change | 64-column latent tiles; the score halves added in the softmax; O copied from the mma registers with `lanes::copy` (needs a flat block index b, so that the output row 16·b fixes the block loop); 115 registers, 68 KB |

As whole CUDA graphs (prep + kernel + merge), tc beat flash at 2048 tokens: 30.4 against 39.3 us.

### Tried and dropped

| idea | result |
|---|---|
| All 64 heads in a block (FlashMLA's shape), 16 subgroups | Each chunk's rows are gathered once instead of four times. Needed `machine-sm90.vx` (228 KiB, through a `spawn-variant`), because tiles kept in mma registers still count towards the shared capacity, and a merge for its own layout. A shadowed index (`i`) in that merge made every head but one wrong until it was renamed. A block took about 22 us, and the step 32-37 us. |
| Two chunks a block | One max over 64 tokens, so no rescaling, and half the splits. 7 us slower everywhere: a block's serial path doubled. Four chunks did not fit in shared memory. |
| O staged through shared memory for coalesced global stores | No better. |

### Where the time went (`phase_time.py`)

We timed the kernel cut after each phase, as graphs:

| phase | 1 split | 64 splits |
|---|---|---|
| ids + gather | 3.2 us | 3.9 us |
| scores | 1.0 | 1.4 |
| softmax | 0.6 | 0.8 |
| P @ rows + store | 2.9 | 7.4 |

The merge then read back the 8 MB f32 numerator in about 9 us.

### Into ResidentGLM

`mla_kernel='tc'` keeps the latent and rope caches as uint16 holding bf16 bits, and
`store_row_bf16_at` writes the new rows. It runs `mla_q_bf16_at`, `mla_tc_at` and
`mla_flash_merge_at` with 64 splits of 32 (index_topk must be at most 2048).

On the H200, 600 replayed steps agree with `'flash'` to 6.7e-3 relative in the logits, with the same
argmax at every step. The spawn builder and the three launchers learned the `bf16` element (as uint16
bits; loads of bf16 may be 32 bits wide).

Commits: f9935e2, e5ac9a2 (on a branch until #3490 merged), 0844719.

## Part 3: the merge, again

The tc numerator round trip looked like the next lever. Spreading the merge over 16 column blocks a
head helped at 2048 tokens but hurt at 256.

ncu with a warm cache (`--cache-control none --graph-profiling node`) showed what was really wrong.
The merge was issue- and barrier-bound: 4.6 M warp instructions, about 560 a warp, because every lane
recomputed the max and the factors for all 64 split slots branch-free.

The rewrite works the factors out once a block (one lane a split, `lanes::max` and `lanes::sum`),
keeps them in shared memory, and streams factor · numerator over the step's splits only. As graphs:

| | 256 | 1000 | 2048 |
|---|---|---|---|
| flash, before | 21.0 us | 27.7 us | 39.3 us |
| flash, after | 15.6 | 24.2 | 35.9 |
| tc, before | 22.4 | 27.6 | 30.0 |
| tc, after | 19.0 | 22.2 | 26.2 |

Commit 9b96af2.

## The comparison with SGLang, corrected

The 13.9 us we had been chasing is SGLang's flash-decode kernel alone. Its MLA block at context 1000
(evidence/h200-20261115-sglang) is:

| kernel | absorb | rope | kv store | ? | flash decode | combine | absorb |
|---|---|---|---|---|---|---|---|
| us | 5.9 | 1.6 | 1.2 | 3.3 | 13.9 | 5.8 | 6.5 |

The 3.3 us kernel turned out to be `flash::prepare_varlen_num_blocks` (split scheduling), so
SGLang's attention is three kernels.

The like-for-like profile (#40) used nsys kernel durations inside a real decode step, warm, at
GLM-5.3 width: our `mla_step_profile.py` against SGLang's `pod_sglang.sh profile` at input 1000 and
2048.

| context | ours `'tc'` | ours `'flash'` | SGLang |
|---|---|---|---|
| 1000 | 1.9 + 12.2 + 4.4 = 18.5 us | 20.8 us | 3.3 + 14.4 + 5.8 = 23.5 us |
| 2048 | 1.8 + 13.7 + 6.7 = 22.2 us | 30.8 us | 3.2 + 16.1 + 5.9 = 25.2 us |

So `'tc'` beats SGLang's MLA attention at both contexts. The same traces show the indexer's top-k
(25.6 us at 1000, 50.7 + 5.7 at 2048, against SGLang's 1.7 and 6.0) and scores (about 10.5 against
3.6) as the larger gaps now.

The larger remaining gap in the MLA block is now elsewhere: our absorb matvecs take 12.4 + 12.3 us
against SGLang's 5.9 + 6.5 (#42).

## Toolchain pins

| pin | why |
|---|---|
| 89832571 | where the day started |
| 243ea31c | #3475: lane-owned tiles in registers, needed by `mla_flash_at` |
| d75d4c60 | #3490: `lanes::copy_rows_async`, needed by `mla_tc_at`; also has #3486 |
| 56f57c3c | #3491: `lanes::scale_rows`; #3492: `as bf16` in one cvt |

Pods need bundles built on the pod (dev box glibc 2.43, pods 2.39). We built them with
`scripts/pod_build_bundle.sh` (`APT=0`, from `~/pod-toolchain/vx-src-<commit>.tar.gz`, reusing an
older bundle's llvm-22). On the pod they sit next to each other as `~/glm-h200/vxtc-<commit>`, with
`env-<commit>.sh`.

## Operational lessons

- **`/tmp` is a 31 GB RAM disk shared by every session on the dev box.** Our cargo targets (17 GB)
  hit its quota and broke tool writes for all sessions. We moved the caches to `/mnt/scratch`. Use
  `cargo sweep --time 2` to clear old ones.
- **On the pod,** `python script.py` imports the older `~/glm-h200/repo` through the venv's editable
  install. Run from `repo-flash` with `PYTHONPATH=.`.
- **Use our own watch patterns.** A watch on "rror" matched the crate name `thiserror`.
- **Share the GPU through messages.** We announced each slot to vx-7e and vx-94, and gave the GPU
  back between our timings.
- **Vx details we hit:**
  - `lanes::` reductions only as the whole value of a `let` in subgroup code;
  - a lane loop in a subgroup has 32 passes;
  - tiles that stay in mma registers still count towards the SMEM capacity;
  - `lanes::copy` straight from mma registers keeps them in registers;
  - `copy_async` tokens on computed indices of one tensor are refused by the borrow checker (give
    each its own tile index);
  - `build.py` applies a `spawn-variant` only with `--variant`.

## What remains

- #40: profile against SGLang like-for-like (done: see above).
- #32: the indexer's top-k, now the largest gap in the attention block.
- #41: shorten `mla_tc_at`'s per-block critical path (chunks with `scale_rows` and double-buffered
  row copies; fewer barriers).
- #42: the absorb matvecs, and the bf16 queries written by the absorb kernel.
- #43: decide whether to skip top-k when the context fits in index_topk.
