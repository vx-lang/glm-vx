# Faster than SGLang's MLA attention, in a language with no CUDA C++: GLM-5.3 decode on an H200 in Vx

*2026-11-16*

GLM-5.3's MLA decode attention now takes **18.5 µs at a 1000-token context and 22.2 µs at 2048** on one
NVIDIA H200 in glm-vx. SGLang 0.5.22 takes 23.5 and 25.2 µs for the same step. Two days earlier our best
kernel took about 120 µs.

Every kernel involved is written in [Vx](https://github.com/vx-lang/Vx), with no CUDA C++, no inline PTX
and no attention library. This post is about how that went: what Vx made possible, where the compiler had
to grow, how we checked every step, and where the limits still are.

## The problem

GLM-5.3 uses multi-head latent attention (MLA) with DeepSeek-style sparse selection (DSA). For each
decode step and each layer:

- 64 heads attend over up to 2048 tokens that an indexer selects from the whole context;
- every cached token is a 512-wide latent row plus a 64-wide rope row;
- each head's query is absorbed into that latent space, so attention is a dot product against the rows,
  a softmax, and a weighted sum of the same rows.

At batch 1 this is a latency problem. There is little arithmetic per launch, the selected tokens sit
scattered across the cache, and every microsecond of each layer's 78 counts.

## Where we started: 120 µs, and a language limit

Our kernel at the start, `mla_heads_at`, took 45-120 µs depending on the number of tokens. ncu showed why:
about 600 warp instructions per head and token, against 34 for the actual arithmetic, with more than half
of the stalls at barriers. Every token reread each head's query from shared memory.

The textbook fix is flash decoding: each lane keeps its slice of the query and of the running sums in
registers for the whole token loop. Vx could not express that. A warp-wide sum (`lanes::sum`) has to sit
between lane loops, and per-lane state did not survive from one lane loop to the next. We wrote up the
gap as Vx #3469.

## Step 1: flash decoding in registers (120 → 40 µs)

The Vx fix (#3475) is a compiler rule, not a new construct. A shared-memory tile whose every access is
the lane's own element, `t[w][..][lane]`, is kept in that lane's registers. The kernel reads as if the
state lived in shared memory. The compiled code has no local memory and no shared-memory traffic in the
token loop:

```
for q in 0..n {
  for lane in 0..32 across lanes {
    for h in 0..2 {
      let mut p: f32 = 0.0;
      for e in 0..16 {
        p = p + qr[w][h][e][lane] * row[w][e][lane];   // the query tile stays in registers
      }
      ...
      part[w][h][lane] = p;
    }
    // Start the next token's row while this one is scored.
    for e in 0..16 { nx[w][e][lane] = cache_latent[k1 * 512 + e * 32 + lane]; }
  }
  let x0: f32 = lanes::sum(&part[w][0], 32);             // the score, summed across the warp
  ...
}
```

Three more steps took it from there:
- **Prefetching.** We load the next token's row while the current one is scored, and the token id two
  ahead, so the loop stops waiting on two dependent loads per token. That took 41 → 29 µs.
- **A register budget.** With `.maxnreg 128`, two blocks fit on an SM.
- **A rewritten merge.** It had quietly been taking more time than the attention itself.

The result was 3× faster than where we started, with output within 1.8e-7 of the previous kernel.

## Step 2: bf16 and the tensor cores (40 → 26 µs)

At that point the f32 kernel was bound by instruction issue: about 120 instructions per head and token,
with DRAM at 3% of peak. Going faster meant doing the two products, scores and P × rows, as matrix
multiplies on the tensor cores, as FlashMLA does. That in turn meant a bf16 KV cache, as SGLang uses.

In Vx, a tensor-core matmul is an expression in a subgroup's code:

```
// Scores: 16 tokens x 8 heads per warp, over 256 latent columns plus the rope.
s[w] = (sl[th][c0] as f32) @ (qt[c0][hh] as f32);
s[w] = s[w] + (sl[th][c0 + 1] as f32) @ (qt[c0 + 1][hh] as f32);
...
// O = P @ rows, accumulated in the mma registers and copied straight to memory.
lanes::fill(&mut o[w], 0.0);
o[w] = o[w] + (p[0] as f32) @ (sl[0][w] as f32);
o[w] = o[w] + (p[1] as f32) @ (sl[1][w] as f32);
lanes::copy(numerator, &o[w], [b * 16, 64 * w]);
```

vxc lowers those lines to `ldmatrix` and `mma.sync.m16n8k16` with the tiles swizzled to avoid
shared-memory bank conflicts. It keeps `o[w]` in registers because only a fill, matmuls and a copy touch
it. We never wrote a line of PTX.

The first version was correct and slow: 42 µs, almost all of it gathering the selected rows into shared
memory through registers, one 16-bit store per element. The Vx compiler sessions added
`lanes::copy_rows_async`. It is cp.async straight from the cache into a shared tile, rows picked by an
id list, with zero rows for invalid ids:

```
let g0 = lanes::copy_rows_async(&mut sl[0][w], cache_latent, &ids[w], [0, 64 * w]);
let g1 = lanes::copy_rows_async(&mut sl[1][w], cache_latent, &ids[w], [16, 64 * w]);
...
g0.wait();
g1.wait();
```

That one routine halved the kernel (42 → 21 µs) and freed 86 registers. Its first ever GPU run was this
kernel.

## The compiler loop

The pattern that made this fast was not any single feature. It was the loop between the kernel and the
compiler. When a kernel hit something Vx could not express, or lowered badly, two Vx compiler sessions
wrote the change, usually the same day. We built each on the H200, ran its GPU tests, and reported
registers and PTX back. Seven changes, from the afternoon of 15 November to the next morning (UTC):

| Vx change | Why we needed it | Effect |
| --- | --- | --- |
| #3475: lane-owned tiles stay in registers | per-lane state across warp-wide sums | flash decoding expressible; 1.6× at once |
| #3490: `lanes::copy_rows_async` | the row gather was 74% of the stalls | kernel 42 → 21 µs |
| #3491: `lanes::scale_rows` | rescaling an mma accumulator by rows | the accumulator can stay in registers across chunks (our next step) |
| #3492: `as bf16` as one `cvt.rn.bf16.f32` | software rounding per element | GPU rounding identical to the CPU's |
| #3493: neighbouring bf16 stores joined | one 16-bit store per element | one wide store |
| #3503: the same without the register cost | #3493 cost 6 registers here | 115 → 110 registers |
| #3486: block `lanes::sort` fix | a bug only a GPU run showed | correct on the H200 |

Two other Vx properties shaped the work every day:

- **The race proof.** vxc refuses any parallel write that two passes of an `across` loop could make to
  the same element. It forced some layouts (tile halves became tile dimensions, since `w / 4` is not an
  index it can reason about). It also meant a whole class of races never reached a GPU.
- **One source, two targets.** The same kernel source compiles to PTX and to a CPU library. Every kernel
  is tested against float64 references on the CPU before a GPU session, and again on the H200.

## Measuring honestly

For most of this work we compared against "SGLang's 13.9 µs". That turned out to be only SGLang's
flash-decode kernel. Its attention is three kernels: a split-scheduling kernel (3.3 µs),
`FlashAttnFwdSm90` (14.4 µs) and a combine (5.8 µs). Our own numbers had been taken in a harsher way, as
single kernels after an L2 flush. They also counted a 3 µs graph-launch floor that a real decode step
doesn't pay.

So we measured both sides the same way: nsys kernel durations inside a real batch-1 decode step, with a
warm cache.

| Context | glm-vx (bf16, tensor cores) | glm-vx (f32 flash) | SGLang 0.5.22 |
| --- | --- | --- | --- |
| 1000 tokens | **18.5 µs** | 20.8 µs | 23.5 µs |
| 2048 tokens | **22.2 µs** | 30.8 µs | 25.2 µs |

The same traces showed where the attention block is still behind. The indexer's top-k takes 25.6 µs
against 1.7, and the query/value absorb matvecs take 23 µs against 12.

The top-k gap had a simple cause. SGLang's top-k checks `seq_len <= topk` first and just writes
`0, 1, 2, …`. Ours sorted all the scores anyway. We now skip both the scores and the sort when every
cached token fits, on the GPU and in our CPU reference model. That fix is in review.

## How reliable is testing on a CPU?

Every kernel runs twice before we trust it. The first run is on the CPU: our own emulator runs the
compiled Vx kernels with one OS thread per GPU lane, and Vx runs the `spawn` kernels on the host. Then
it runs again on the H200.

The CPU runs catch indexing, layout and logic errors against the reference bit for bit. The shadowed
loop variable in one merge kernel was caught there. They cannot see:
- compiler bugs on the GPU path (#3482 was correct on the CPU, wrong on the H200);
- real concurrency;
- hardware arithmetic (`ex2.approx`, the tensor cores' summation order);
- registers, shared-memory limits or speed.

So a CPU pass is necessary, never sufficient, and nothing counts as done until the GPU tests and a
profile agree.

## What Vx made easy, and what is still hard

Easy, or at least possible without leaving the language:
- register-resident flash decoding;
- tensor-core matmuls as one line each;
- asynchronous gathers;
- warp reductions;
- one source that runs on a CPU for testing and on an H100 or H200 for real.

Still hard:
- **Index arithmetic for the race proof.** Write indices must be linear in the loop variables, so some
  natural layouts have to be restated as tile shapes.
- **Matmul operands are whole tiles.** You cannot take a row range of a tile.
- **Shared-memory accounting.** Tiles that live only in mma registers still count against the declared
  capacity, which ruled out one 4-chunk design.
- **Slow CPU tests.** The thread-per-lane emulator is thorough but takes over an hour for the full
  decode suites.
- **Some ideas lost.** All 64 heads per block (FlashMLA's shape) and multi-chunk blocks were both slower
  here. Being able to try each in an afternoon is the point.

## What's next

- Shorten the tensor-core kernel's critical path: several chunks per block, with `lanes::scale_rows` and
  double-buffered row gathers.
- Bring the absorb matvecs and the indexer's scores to SGLang's speed.
- Make the top-k fast at contexts past 2048 (50.7 µs against SGLang's 6.0).

The kernels, a detailed journal of every step, and the evidence behind each number are in the glm-vx
repository: `docs/MLA.md`, `docs/journal/2026-11-15-mla-flash-and-tensor-cores.md`, and
`evidence/h200-20261115-*` and `evidence/h200-20261116`.
