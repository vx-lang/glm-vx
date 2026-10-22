"""Block-FP8 GEMM on H100 (glm-vx #10): the host side of gpu/spawn/fp8_gemm/.

    C[M, N] = sum_kb a_scale[m, kb] * b_scale[n // 128, kb] * sum_{k in kb} A[m, k] * B[n, k]

A [M, K] and B [N, K] are e4m3 codes (uint8), row-major; a_scale is f32 [M, ceil(K / 128)] (per-token
groups of 128 along K, as DeepGEMM and SGLang's W8A8 block FP8 make them) and b_scale f32
[ceil(N / 128), ceil(K / 128)], the checkpoint's `weight_scale_inv` (dequant = code * scale;
glm_vx/checkpoint.py dequantize_block_fp8). C is f32 or bfloat16 (uint16 bits).

- `plan` picks the kernel variant (m64: one consumer warpgroup, M <= 64; m128: two) and the split-K.
- `FP8GEMM` encodes the two TMA tensor maps (gpu/runtime.py encode_tensor_map_2d), uploads them,
  and launches the kernel, then the split-K reduction when splits > 1.
- `reference` and `error_bound` are the numerical contract: the f64 value and how far the
  hardware may be from it (module constants below say why).
- `simulate` replays the kernel's index math in NumPy (TMA box and swizzle, wgmma descriptors and
  register fragments, scale promotion, split-K, epilogue masks) so a layout mistake fails without
  a GPU (gpu/test_fp8_gemm.py).

gpu/FP8-GEMM.md has the design and the numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUILD = Path(__file__).parent / 'build' / 'spawn' / 'fp8_gemm'
BLOCK_N = 128
BLOCK_K = 128            # k per stage and per scale block; one 128-byte swizzle row of e4m3
SM90_SMS = 132           # H100 SXM


@dataclass(frozen=True)
class Variant:
    name: str
    consumers: int       # warpgroups of 64 rows
    stages: int

    @property
    def block_m(self):
        return 64 * self.consumers

    @property
    def threads(self):
        return 128 * self.consumers + 32

    @property
    def a_stage(self):
        return self.block_m * BLOCK_K

    @property
    def b_stage(self):
        return BLOCK_N * BLOCK_K

    @property
    def ctas_per_sm(self):
        """Resident CTAs per SM by shared memory (228 KiB an SM, 1 KiB reserved a CTA) and
        registers (65536 an SM, at most ~160 a thread here: gpu/FP8-GEMM.md)."""
        shared = self.stages * (self.a_stage + self.b_stage) + 1024 + 16 * self.stages
        return max(1, min((228 * 1024) // (shared + 1024), 65536 // (160 * self.threads)))


# Must match gpu/spawn/fp8_gemm/gen_fp8_gemm.py VARIANTS (gpu/test_fp8_gemm.py checks).
VARIANTS = {'m64': Variant('m64', 1, 4), 'm128': Variant('m128', 2, 4)}


@dataclass(frozen=True)
class Plan:
    variant: Variant
    m: int
    n: int
    k: int
    splits: int
    split_blocks: int    # 128-k blocks per split; the last split may have fewer

    @property
    def blocks_k(self):
        return -(-self.k // BLOCK_K)

    @property
    def tiles_m(self):
        return -(-self.m // self.variant.block_m)

    @property
    def tiles_n(self):
        return -(-self.n // BLOCK_N)

    @property
    def grid(self):
        return self.tiles_m * self.tiles_n * self.splits

    @property
    def block(self):
        return self.variant.threads


def plan(m, n, k, *, sms=SM90_SMS, variant=None, splits=None):
    """Variant and split-K for C[m, n] = A[m, k] B[n, k]^T.

    m64 for m <= 64, else m128 (B is read once for both 64-row halves). Splits: decode tiles are
    few (m = 1, n = 2048 is 16 CTAs), so K is split until the CTAs fill the GPU's CTA slots
    (sms * ctas_per_sm), with at least 2 k blocks a split, choosing the split with the fewest
    waves' idle slots and then the fewest splits. Tuning (stream-K, a persistent grid) is later.
    """
    for value, name in ((m, 'm'), (n, 'n'), (k, 'k')):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer, got {value!r}')
    if k % 16:
        raise ValueError(f'k must be a multiple of 16 (the TMA row stride of A and B is k bytes), got {k}')
    v = VARIANTS[variant] if variant else VARIANTS['m64' if m <= 64 else 'm128']
    blocks_k = -(-k // BLOCK_K)
    tiles = -(-m // v.block_m) * -(-n // BLOCK_N)
    if splits is None:
        slots = sms * v.ctas_per_sm
        splits = 1
        if tiles < slots:
            # Split counts that blocks_k divides into (s splits of ceil(blocks_k / s) blocks), up to
            # 2 blocks a split and 2 waves; the best wave efficiency, then the fewest splits.
            candidates = {-(-blocks_k // -(-blocks_k // s)) for s in range(1, max(1, blocks_k // 2) + 1)}
            best = max(((tiles * s) / (-(-tiles * s // slots) * slots), -s)
                       for s in candidates if s == 1 or tiles * s <= 2 * slots)
            splits = -best[1]
    if isinstance(splits, bool) or not isinstance(splits, int) or not 1 <= splits <= blocks_k:
        raise ValueError(f'splits must be in 1..{blocks_k}, got {splits!r}')
    split_blocks = -(-blocks_k // splits)
    return Plan(v, m, n, k, -(-blocks_k // split_blocks), split_blocks)


def check_arguments(a, a_scale, b, b_scale, out, m, n, k):
    """Shapes and types of the device tensors (gpu.runtime.DeviceTensor or alike: .dtype, .size,
    .pointer). Returns True for a bfloat16 (uint16) output, False for f32."""
    blocks_k = -(-k // BLOCK_K)
    want = {'a': (a, 'uint8', m * k), 'b': (b, 'uint8', n * k), 'a_scale': (a_scale, 'float32', m * blocks_k),
            'b_scale': (b_scale, 'float32', -(-n // BLOCK_N) * blocks_k)}
    for name, (tensor, dtype, size) in want.items():
        if str(tensor.dtype) != dtype:
            raise TypeError(f'{name} must be {dtype}, got {tensor.dtype}')
        if tensor.size != size:
            raise ValueError(f'{name} must have {size} elements for m={m}, n={n}, k={k}; got {tensor.size}')
    for name, tensor in (('a', a), ('b', b)):
        if tensor.pointer % 16:
            raise ValueError(f'{name} must be 16-byte aligned for TMA')
    if str(out.dtype) not in ('float32', 'uint16'):
        raise TypeError(f'out must be float32 or uint16 (bfloat16 bits), got {out.dtype}')
    if out.size != m * n:
        raise ValueError(f'out must have {m * n} elements, got {out.size}')
    return str(out.dtype) == 'uint16'


class FP8GEMM:
    """The kernels of gpu/build/spawn/fp8_gemm (gpu/spawn/fp8_gemm/build.sh) on one context.

    `gemm(a, a_scale, b, b_scale, out, m=, n=, k=)` launches on `stream` and returns the Plan.
    Tensor maps embed A's and B's addresses; they are encoded and uploaded once per (A, B, shape,
    variant) and cached, since the upload is blocking: it synchronizes the context, and it must
    happen outside a CUDA graph capture (capture after one warm-up call with the same buffers).
    A map is rewritten only between launches, never while a kernel that reads it may run; each
    CTA's producer thread fences it (fence.proxy.tensormap::generic.acquire.sys) before its TMA.
    """
    def __init__(self, context, build=BUILD, *, sms=None):
        from gpu.spawn.launch import SpawnKernel
        self.context = context
        self.kernels = {name: SpawnKernel(context, Path(build) / f'fp8_gemm_{name}.ptx') for name in VARIANTS}
        self.reduce = SpawnKernel(context, Path(build) / 'fp8_gemm_reduce.ptx')
        if sms is None:
            try:
                sms = context._attribute(16)       # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
            except Exception:
                sms = SM90_SMS
        self.sms = sms
        self._maps, self._workspace, self._dummy = {}, None, {}

    def tensor_maps(self, a, b, p):
        """One 256-byte device tensor: A's map at byte 0, B's at byte 128 (each 64-byte aligned)."""
        from gpu.runtime import encode_tensor_map_2d
        key = (a.pointer, b.pointer, p.m, p.n, p.k, p.variant.name)
        maps = self._maps.get(key)
        if maps is None:
            driver = self.context.driver
            raw = (encode_tensor_map_2d(driver, a.pointer, p.m, p.k, p.k, p.variant.block_m, BLOCK_K)
                   + encode_tensor_map_2d(driver, b.pointer, p.n, p.k, p.k, BLOCK_N, BLOCK_K))
            maps = self.context.tensor((256,), 'uint8', raw)
            if len(self._maps) >= 64:
                self._maps.pop(next(iter(self._maps))).close()
            self._maps[key] = maps
        return maps

    def _scratch(self, dtype):
        if dtype not in self._dummy:
            self._dummy[dtype] = self.context.tensor((1,), dtype, bytes(4 if dtype == 'float32' else 2))
        return self._dummy[dtype]

    def workspace(self, elements):
        if self._workspace is None or self._workspace.size < elements:
            if self._workspace is not None:
                self._workspace.close()
            self._workspace = self.context.tensor((elements,), 'float32')
        return self._workspace.view((elements,))

    def gemm(self, a, a_scale, b, b_scale, out, *, m, n, k, splits=None, variant=None, stream=None):
        p = plan(m, n, k, sms=self.sms, variant=variant, splits=splits)
        bf16 = check_arguments(a, a_scale, b, b_scale, out, m, n, k)
        maps = self.tensor_maps(a, b, p)
        if p.splits == 1:
            f32_out = self._scratch('float32') if bf16 else out
            bf16_out = out if bf16 else self._scratch('uint16')
        else:
            f32_out, bf16_out = self.workspace(p.splits * m * n), self._scratch('uint16')
        self.kernels[p.variant.name].launch(
            (p.grid,), p.block, stream=stream, maps=maps, a_scale=a_scale, b_scale=b_scale, out=f32_out,
            out_bf16=bf16_out, rows_m=m, rows_n=n, cols=k, splits=p.splits, split_blocks=p.split_blocks,
            bf16_out=int(bf16))
        if p.splits > 1:
            self.reduce.launch((-(-m * n // 256),), 256, stream=stream, partial=f32_out, out=self._scratch('float32') if bf16 else out,
                               out_bf16=out if bf16 else self._scratch('uint16'), count=m * n, splits=p.splits,
                               bf16_out=int(bf16))
        return p

    __call__ = gemm


# --- numerical contract ------------------------------------------------------------------------------

# Hopper's FP8 wgmma does not accumulate in full f32: DeepSeek-V3 (arXiv 2412.19437, section 3.3.2)
# measured that it keeps "around 14 bits", which is why the scales are applied, and the sum
# promoted to f32 on the CUDA cores, every 128 k (DeepGEMM does the same; so does this kernel).
# Model: a 128-k block is 4 wgmma k32 steps; each step sums 32 products and adds them to the
# running block sum, so at most 8 roundings, each at most 2^-13 of a partial sum whose magnitude
# is at most S_kb = sum_k |a_k b_k|. Hence |part - P_kb| <= 8 * 2^-13 * S_kb = 2^-10 * S_kb.
EPS_MMA = 2.0 ** -10
U32 = 2.0 ** -24         # f32 unit roundoff: s = a_s * b_s, part * s, acc + ..., split-K sums
U_BF16 = 2.0 ** -8       # bfloat16 unit roundoff (8 significant bits), round to nearest even


def e4m3_values(codes):
    from glm_vx.checkpoint import FP8_E4M3FN_TABLE
    return FP8_E4M3FN_TABLE[np.asarray(codes, np.uint8)].astype(np.float64)


def reference(a_codes, a_scale, b_codes, b_scale, *, splits=1, bf16=False, terms=False):
    """(C, bound): C in f64 and the largest |C_kernel - C| the kernel may have, elementwise.
    With terms=True, also sum_kb |s_kb| S_kb, the scale of the tensor core's rounding: a measured
    max |C_kernel - C| / that term is the effective EPS_MMA of the hardware.

    Each block's sum of e4m3 products is exact in f64 (a product has at most 8 significant bits
    between 2^-18 and 2^17.6; 128 of them need fewer than 53 bits), so C is the true value up to
    f64 rounding of the scale products. The bound is
        EPS_MMA * sum_kb |s_kb| S_kb  +  (KB + splits + 2) * U32 * sum_kb |s_kb| (|P_kb| + EPS_MMA S_kb)
    with s_kb = a_scale * b_scale, plus U_BF16 * (|C| + bound) for a bfloat16 output.
    """
    a = e4m3_values(a_codes)
    b = e4m3_values(b_codes)
    m, k = a.shape
    n = b.shape[0]
    blocks = -(-k // BLOCK_K)
    a_scale = np.asarray(a_scale, np.float64).reshape(m, blocks)
    b_scale = np.asarray(b_scale, np.float64).reshape(-(-n // BLOCK_N), blocks)
    c = np.zeros((m, n))
    mma = np.zeros((m, n))
    promo = np.zeros((m, n))
    for kb in range(blocks):
        ks = slice(kb * BLOCK_K, (kb + 1) * BLOCK_K)
        part = a[:, ks] @ b[:, ks].T
        mags = np.abs(a[:, ks]) @ np.abs(b[:, ks]).T
        s = np.abs(a_scale[:, kb:kb + 1] * np.repeat(b_scale[:, kb], BLOCK_N)[None, :n])
        sign = np.sign(a_scale[:, kb:kb + 1] * np.repeat(b_scale[:, kb], BLOCK_N)[None, :n])
        c += sign * s * part
        mma += s * mags
        promo += s * (np.abs(part) + EPS_MMA * mags)
    bound = EPS_MMA * mma + (blocks + splits + 2) * U32 * promo
    if bf16:
        bound = bound + U_BF16 * (np.abs(c) + bound)
    return (c, bound, mma) if terms else (c, bound)


def bf16_round(x):
    """f32 to bfloat16 bits, round to nearest even (as cvt.rn.bf16.f32 for finite values)."""
    bits = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_value(bits):
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


# --- NumPy replay of the kernel's index math ------------------------------------------------------------

def sw128_desc(addr):
    """sm90.vx sw128_desc: the wgmma descriptor of a K-major tile with 128-byte swizzle."""
    return ((addr & 0x3FFFF) >> 4) | (1 << 16) | (64 << 32) | (1 << 62)


def swizzle128(offset):
    """CU_TENSOR_MAP_SWIZZLE_128B on a byte offset from a 1024-aligned base: the 16-byte chunk
    (bits 4-6) XOR the 128-byte row within the 1024-byte atom (bits 7-9)."""
    offset = np.asarray(offset)
    return offset ^ (((offset >> 7) & 7) << 4)


def tma_load(smem, dst, matrix, c0, c1, box_rows, box_cols=BLOCK_K):
    """cp.async.bulk.tensor.2d with a 128B-swizzle map: box element (r, c) of matrix[c1 + r, c0 + c]
    (zero outside the matrix) goes to byte dst + swizzle128(128 r + c)."""
    rows, cols = matrix.shape
    box = np.zeros((box_rows, box_cols), np.uint8)
    r1, k1 = min(rows, c1 + box_rows), min(cols, c0 + box_cols)
    if r1 > c1 and k1 > c0:
        box[:r1 - c1, :k1 - c0] = matrix[c1:r1, c0:k1]
    r, c = np.meshgrid(np.arange(box_rows), np.arange(box_cols), indexing='ij')
    assert dst % 1024 == 0
    smem[dst + swizzle128(box_cols * r + c)] = box


def wgmma_operand(smem, desc, rows, kbytes=32):
    """The [rows, kbytes] e4m3 operand a wgmma reads through `desc` (PTX ISA 9.7.15.5.1): K-major,
    8-row groups SBO apart. Swizzle mode 1 (128B): row r of a group at 128 r, the address
    swizzled; mode 0 (none): 8 x 16-byte core matrices, 16-byte k chunks LBO apart."""
    start = (desc & 0x3FFF) << 4
    lbo, sbo = ((desc >> 16) & 0x3FFF) << 4, ((desc >> 32) & 0x3FFF) << 4
    mode = (desc >> 62) & 3
    r, c = np.meshgrid(np.arange(rows), np.arange(kbytes), indexing='ij')
    if mode == 1:
        addr = start + (r // 8) * sbo + (r % 8) * 128 + c
        base = start & ~1023
        addr = base + swizzle128(addr - base)
    elif mode == 0:
        addr = start + (r // 8) * sbo + (r % 8) * 16 + (c // 16) * lbo + c % 16
    else:
        raise ValueError(f'swizzle mode {mode} is not modelled')
    return smem[addr]


def fragment_index():
    """(row, col) in the 64 x 128 wgmma D tile of register i of thread tid (PTX ISA 9.7.15.5.2)."""
    tid, i = np.meshgrid(np.arange(128), np.arange(BLOCK_N // 2), indexing='ij')
    warp, lane = tid // 32, tid % 32
    row = 16 * warp + lane // 4 + 8 * ((i >> 1) & 1)
    col = 8 * (i >> 2) + 2 * (lane % 4) + (i & 1)
    return row, col


def simulate(a_codes, a_scale, b_codes, b_scale, p, *, bf16=False, desc=sw128_desc, k_step=2):
    """Replay fp8_gemm.vx.in for plan p, CTA by CTA, as the kernel indexes: returns C (f32, or
    bfloat16 bits). The wgmma block sums are exact (f64, then f32), so this checks the index math,
    not the tensor core's precision. `desc` and `k_step` exist to mutate the model in tests."""
    a_codes, b_codes = np.asarray(a_codes, np.uint8), np.asarray(b_codes, np.uint8)
    m, k = a_codes.shape
    n = b_codes.shape[0]
    v = p.variant
    blocks_k = p.blocks_k
    a_scale = np.asarray(a_scale, np.float32).reshape(-1)
    b_scale = np.asarray(b_scale, np.float32).reshape(-1)
    a_base = 0  # Vx's layout: the A ring, then the B ring (1024-aligned), then the barriers
    b_base = a_base + v.stages * v.a_stage
    smem = np.zeros(b_base + v.stages * v.b_stage, np.uint8)
    rows_f, cols_f = fragment_index()
    partial = np.zeros((p.splits, m, n), np.float32)
    out_bf16 = np.zeros((m, n), np.uint16)
    for bid in range(p.grid):
        m_tile, rest = bid % p.tiles_m, bid // p.tiles_m
        split, n_tile = rest % p.splits, rest // p.splits
        m0, n0 = m_tile * v.block_m, n_tile * BLOCK_N
        kb0 = split * p.split_blocks
        nkb = min(blocks_k - kb0, p.split_blocks)
        acc = np.zeros((v.consumers, 128, BLOCK_N // 2), np.float32)
        for i in range(nkb):
            s = i % v.stages
            kc = (kb0 + i) * BLOCK_K
            tma_load(smem, a_base + v.a_stage * s, a_codes, kc, m0, v.block_m)
            tma_load(smem, b_base + v.b_stage * s, b_codes, kc, n0, BLOCK_N)
            kb = kb0 + i
            sb = b_scale[(n0 // 128) * blocks_k + kb]
            for g in range(v.consumers):
                a_desc = desc(a_base + 8192 * g) + (v.a_stage // 16) * s
                b_desc = desc(b_base) + (v.b_stage // 16) * s
                part = np.zeros((64, BLOCK_N))
                for step in range(BLOCK_K // 32):
                    a_op = e4m3_values(wgmma_operand(smem, a_desc + k_step * step, 64))
                    b_op = e4m3_values(wgmma_operand(smem, b_desc + k_step * step, BLOCK_N))
                    part += a_op @ b_op.T
                reg = part[rows_f, cols_f].astype(np.float32)          # [tid, i]
                tid = np.arange(128)
                row_lo = m0 + 64 * g + 16 * (tid // 32) + (tid % 32) // 4
                srow_lo, srow_hi = np.minimum(row_lo, m - 1), np.minimum(row_lo + 8, m - 1)
                s_lo = a_scale[srow_lo * blocks_k + kb] * sb               # f32 products
                s_hi = a_scale[srow_hi * blocks_k + kb] * sb
                hi = ((np.arange(BLOCK_N // 2) >> 1) & 1).astype(bool)
                scale = np.where(hi[None, :], s_hi[:, None], s_lo[:, None]).astype(np.float32)
                acc[g] = acc[g] + reg * scale
        for g in range(v.consumers):
            rows = m0 + 64 * g + rows_f
            cols = n0 + cols_f
            ok = (rows < m) & (cols < n)
            if bf16 and p.splits == 1:
                out_bf16[rows[ok], cols[ok]] = bf16_round(acc[g][ok])
            else:
                partial[split, rows[ok], cols[ok]] = acc[g][ok]
    if p.splits == 1:
        return out_bf16 if bf16 else partial[0]
    total = np.zeros((m, n), np.float32)
    for s in range(p.splits):
        total = total + partial[s]
    return bf16_round(total) if bf16 else total


# --- test data -----------------------------------------------------------------------------------------

def random_case(m, n, k, seed=0):
    """Random e4m3 codes (no NaN codes) and positive scales in the checkpoint's layouts."""
    rng = np.random.default_rng(seed)
    blocks = -(-k // BLOCK_K)

    def codes(shape):
        c = rng.integers(0, 256, size=shape, dtype=np.uint8)
        return np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8)
    a, b = codes((m, k)), codes((n, k))
    a_scale = (2.0 ** rng.uniform(-10, -4, size=(m, blocks))).astype(np.float32)
    b_scale = (2.0 ** rng.uniform(-10, -4, size=(-(-n // BLOCK_N), blocks))).astype(np.float32)
    return a, a_scale, b, b_scale


def bytes_moved(m, n, k, *, out_bytes=4):
    """Bytes a GEMM must move at least: A, B, both scales, C."""
    blocks = -(-k // BLOCK_K)
    return m * k + n * k + 4 * (m * blocks + -(-n // BLOCK_N) * blocks) + out_bytes * m * n
