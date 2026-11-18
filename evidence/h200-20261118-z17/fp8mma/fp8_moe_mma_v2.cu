// Speed-of-light probe for docs/TP-SERVING-PLAN.md item 4 (not glm-vx code): one MoE layer of GLM-5.3 at batch 1,
// one rank of 8 (each expert's intermediate slice 2048/8 = 256), 8 routed experts + the shared one, FP8 e4m3
// weights with 128x128 float scales read straight into mma.sync.m16n8k32.e4m3 (sm_89+), activations quantized to
// e4m3 in groups of 128 (amax / 448, as SGLang's per-token-group quant before its FP8 GEMMs), f32 accumulation.
//   gate_up: h[e][i] = silu(g) * u, g = Wg[e][i] . x, u = Wu[e][i] . x    (512 rows x 6144 a expert)
//   down:    out[j]  = sum_e w[e] * Wd[e][j] . h[e]                         (6144 rows x 256 a expert)
// Checked against a double-precision host model of the same quantized inputs, then timed in a CUDA graph of 75
// layers (different random experts each, out of 256 a layer's worth of weights, so few reads hit L2).
//   nvcc -O3 -arch=sm_90a -o fp8_moe_mma fp8_moe_mma.cu && ./fp8_moe_mma
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int H = 6144, I = 256, EXPERTS = 256, ACTIVE = 9, KB = H / 128;  // KB: scale blocks along the hidden size

__device__ __forceinline__ void mma_e4m3(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_e4m3(float a, float b, float c, float d) {
  // byte i of the word = the i-th value
  __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// A lane (g = lane/4, t = lane%4) loads 16 contiguous bytes of rows g and g+8 at k = base + t*16; the two mmas take
// bytes 0-7 and 8-15 of them. The mma reads a0/a2 (row g) and b0/b1 as k = t*4..+3 and 16+t*4..+3; here those are
// k = base+t*16+0..3 and +4..7 (first mma), +8..11 and +12..15 (second): the same permutation of k for A and B,
// so each dot product is unchanged. B's 8 columns all hold x: every column of C is the product, c0 = row g, c2 = g+8.
__device__ __forceinline__ void mma16(float c[4], uint4 r0, uint4 r8, uint32_t b[4]) {
  mma_e4m3(c, r0.x, r8.x, r0.y, r8.y, b[0], b[1]);
  mma_e4m3(c, r0.z, r8.z, r0.w, r8.w, b[2], b[3]);
}

// gate_up: a block of 8 warps for one expert's 16 rows of gate and the same 16 of up; warp w takes the scale blocks
// kb = w*6 .. w*6+5 of the hidden size. Each warp quantizes its own x (the 4 lanes t of a 128-group hold it all).
#ifndef GUW
#define GUW 8
#endif
constexpr int GU_WARPS = GUW, GU_KB = KB / GU_WARPS;  // -DGUW=N: N warps a block, KB/N scale blocks each
__global__ void __launch_bounds__(32 * GUW) gate_up(const uint8_t* __restrict__ w, const float* __restrict__ ws,  // [E][512][H], [E][4][KB]
                                               const float* __restrict__ x, const int* __restrict__ ids, float* __restrict__ h) {
  const int e = ids[blockIdx.y], tile = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const uint8_t* wg = w + (size_t)e * 512 * H + (size_t)(tile * 16 + g) * H;
  const uint8_t* wu = wg + (size_t)256 * H;
  const float* sg = ws + (size_t)e * 4 * KB + (tile * 16 / 128) * KB;
  const float* su = sg + 2 * KB;
  float accg[2] = {0, 0}, accu[2] = {0, 0};
  uint4 nxt[8];
  auto load = [&](int kb, uint4* d) {
#pragma unroll
    for (int c = 0; c < 2; c++) {
      const int k = kb * 128 + c * 64 + t * 16;
      d[c * 4 + 0] = ldg_stream(wg + k); d[c * 4 + 1] = ldg_stream(wg + 8 * H + k);
      d[c * 4 + 2] = ldg_stream(wu + k); d[c * 4 + 3] = ldg_stream(wu + 8 * H + k);
    }
  };
  load(warp * GU_KB, nxt);
#pragma unroll
  for (int j = 0; j < GU_KB; j++) {
    const int kb = warp * GU_KB + j;
    uint4 cur[8];
#pragma unroll
    for (int q = 0; q < 8; q++) cur[q] = nxt[q];
    if (j + 1 < GU_KB) load(kb + 1, nxt);
    float v[32], amax = 0.f;
#pragma unroll
    for (int c = 0; c < 2; c++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        float4 f = *reinterpret_cast<const float4*>(x + kb * 128 + c * 64 + t * 16 + q * 4);
        v[c * 16 + q * 4 + 0] = f.x; v[c * 16 + q * 4 + 1] = f.y; v[c * 16 + q * 4 + 2] = f.z; v[c * 16 + q * 4 + 3] = f.w;
      }
#pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float sx = fmaxf(amax, 1e-10f) / 448.f, inv = 1.f / sx;
    float cg[4] = {0, 0, 0, 0}, cu[4] = {0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < 2; c++) {
      uint32_t b[4];
#pragma unroll
      for (int q = 0; q < 4; q++) b[q] = pack_e4m3(v[c * 16 + q * 4] * inv, v[c * 16 + q * 4 + 1] * inv, v[c * 16 + q * 4 + 2] * inv, v[c * 16 + q * 4 + 3] * inv);
      mma16(cg, cur[c * 4 + 0], cur[c * 4 + 1], b);
      mma16(cu, cur[c * 4 + 2], cur[c * 4 + 3], b);
    }
    const float fg = sg[kb] * sx, fu = su[kb] * sx;
    accg[0] += fg * cg[0]; accg[1] += fg * cg[2];
    accu[0] += fu * cu[0]; accu[1] += fu * cu[2];
  }
  __shared__ float part[GU_WARPS][2][16];
  if (t == 0) { part[warp][0][g] = accg[0]; part[warp][0][g + 8] = accg[1]; part[warp][1][g] = accu[0]; part[warp][1][g + 8] = accu[1]; }
  __syncthreads();
  if (threadIdx.x < 16) {
    float gs = 0, us = 0;
    for (int q = 0; q < GU_WARPS; q++) { gs += part[q][0][threadIdx.x]; us += part[q][1][threadIdx.x]; }
    h[blockIdx.y * I + tile * 16 + threadIdx.x] = gs / (1.f + expf(-gs)) * us;
  }
}

// down: a block of 9 warps (one an active expert) for 16 output rows; each warp quantizes its expert's h (two
// 128-groups), 8 mmas, then the warps' sums are added in expert order with the routing weights.
__global__ void __launch_bounds__(32 * ACTIVE) down(const uint8_t* __restrict__ w, const float* __restrict__ ws,  // [E][H][I], [E][KB][2]
                                                    const float* __restrict__ h, const int* __restrict__ ids,
                                                    const float* __restrict__ route, float* __restrict__ out) {
  const int a = threadIdx.x / 32, e = ids[a], tile = blockIdx.x, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const uint8_t* wr = w + (size_t)e * H * I + (size_t)(tile * 16 + g) * I;
  uint4 r0[4], r8[4];
#pragma unroll
  for (int c = 0; c < 4; c++) { r0[c] = ldg_stream(wr + c * 64 + t * 16); r8[c] = ldg_stream(wr + 8 * I + c * 64 + t * 16); }
  const float* he = h + a * I;
  float acc[2] = {0, 0};
#pragma unroll
  for (int kb = 0; kb < 2; kb++) {
    float v[32], amax = 0.f;
#pragma unroll
    for (int c = 0; c < 2; c++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        float4 f = *reinterpret_cast<const float4*>(he + kb * 128 + c * 64 + t * 16 + q * 4);
        v[c * 16 + q * 4 + 0] = f.x; v[c * 16 + q * 4 + 1] = f.y; v[c * 16 + q * 4 + 2] = f.z; v[c * 16 + q * 4 + 3] = f.w;
      }
#pragma unroll
    for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float sx = fmaxf(amax, 1e-10f) / 448.f, inv = 1.f / sx;
    float cc[4] = {0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < 2; c++) {
      uint32_t b[4];
#pragma unroll
      for (int q = 0; q < 4; q++) b[q] = pack_e4m3(v[c * 16 + q * 4] * inv, v[c * 16 + q * 4 + 1] * inv, v[c * 16 + q * 4 + 2] * inv, v[c * 16 + q * 4 + 3] * inv);
      mma16(cc, r0[kb * 2 + c], r8[kb * 2 + c], b);
    }
    const float f = ws[(size_t)e * KB * 2 + (tile * 16 / 128) * 2 + kb] * sx;
    acc[0] += f * cc[0]; acc[1] += f * cc[2];
  }
  __shared__ float part[ACTIVE][16];
  if (t == 0) { part[a][g] = acc[0] * route[a]; part[a][g + 8] = acc[1] * route[a]; }
  __syncthreads();
  if (threadIdx.x < 16) {
    float s = 0;
    for (int q = 0; q < ACTIVE; q++) s += part[q][threadIdx.x];
    out[tile * 16 + threadIdx.x] = s;
  }
}

// ---- host model: the same quantization, double sums
static float e4m3_to_float(uint8_t b) { __nv_fp8_e4m3 v; v.__x = b; return float(v); }
static uint8_t float_to_e4m3(float f) { __nv_fp8_e4m3 v(f); return v.__x; }
static void quant_group(const float* v, int n, std::vector<double>& q, double& s) {
  float amax = 0; for (int i = 0; i < n; i++) amax = std::max(amax, std::fabs(v[i]));
  float sx = std::max(amax, 1e-10f) / 448.f, inv = 1.f / sx; s = sx;
  q.resize(n); for (int i = 0; i < n; i++) q[i] = e4m3_to_float(float_to_e4m3(v[i] * inv));
}

int main(int argc, char** argv) {
  const int layers = 75;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  // weights: e4m3 codes of N(0,1) values, scales ~1e-3 (the checkpoint's order); 256 experts
  size_t gu_bytes = (size_t)EXPERTS * 512 * H, dn_bytes = (size_t)EXPERTS * H * I;
  std::vector<uint8_t> wgu(gu_bytes), wdn(dn_bytes);
  for (auto& b : wgu) b = float_to_e4m3(nd(rng));
  for (auto& b : wdn) b = float_to_e4m3(nd(rng));
  std::vector<float> sgu((size_t)EXPERTS * 4 * KB), sdn((size_t)EXPERTS * KB * 2);
  std::uniform_real_distribution<float> ud(0.5e-3f, 2e-3f);
  for (auto& s : sgu) s = ud(rng);
  for (auto& s : sdn) s = ud(rng);
  std::vector<float> x(H); for (auto& v : x) v = nd(rng);
  std::vector<int> ids_all(layers * ACTIVE);
  for (int l = 0; l < layers; l++) {
    std::vector<int> p(EXPERTS); for (int i = 0; i < EXPERTS; i++) p[i] = i;
    std::shuffle(p.begin(), p.end(), rng);
    for (int a = 0; a < ACTIVE; a++) ids_all[l * ACTIVE + a] = p[a];
  }
  std::vector<float> route(ACTIVE); for (int a = 0; a < ACTIVE; a++) route[a] = a == ACTIVE - 1 ? 1.f : 0.05f + 0.02f * a;

  uint8_t *d_wgu, *d_wdn; float *d_sgu, *d_sdn, *d_x, *d_h, *d_out, *d_route; int* d_ids;
  CK(cudaMalloc(&d_wgu, gu_bytes)); CK(cudaMalloc(&d_wdn, dn_bytes));
  CK(cudaMalloc(&d_sgu, sgu.size() * 4)); CK(cudaMalloc(&d_sdn, sdn.size() * 4));
  CK(cudaMalloc(&d_x, H * 4)); CK(cudaMalloc(&d_h, layers * ACTIVE * I * 4)); CK(cudaMalloc(&d_out, layers * H * 4));
  CK(cudaMalloc(&d_route, ACTIVE * 4)); CK(cudaMalloc(&d_ids, ids_all.size() * 4));
  CK(cudaMemcpy(d_wgu, wgu.data(), gu_bytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_wdn, wdn.data(), dn_bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_sgu, sgu.data(), sgu.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_sdn, sdn.data(), sdn.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_x, x.data(), H * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_route, route.data(), ACTIVE * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_ids, ids_all.data(), ids_all.size() * 4, cudaMemcpyHostToDevice));

  auto launch = [&](int l, cudaStream_t s) {
    gate_up<<<dim3(I / 16, ACTIVE), 32 * GU_WARPS, 0, s>>>(d_wgu, d_sgu, d_x, d_ids + l * ACTIVE, d_h + l * ACTIVE * I);
    down<<<H / 16, 32 * ACTIVE, 0, s>>>(d_wdn, d_sdn, d_h + l * ACTIVE * I, d_ids + l * ACTIVE, d_route, d_out + l * H);
  };
  cudaStream_t s; CK(cudaStreamCreate(&s));
  launch(0, s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());

  // check layer 0 against the host model (the kernel's own h for down's input, so each stage is checked alone)
  std::vector<float> h(ACTIVE * I), out(H);
  CK(cudaMemcpy(h.data(), d_h, h.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(out.data(), d_out, H * 4, cudaMemcpyDeviceToHost));
  double worst_h = 0, worst_o = 0, scale_h = 0, scale_o = 0, worst_h_f32 = 0;
  std::vector<std::vector<double>> qx(KB); std::vector<double> sx(KB);
  for (int kb = 0; kb < KB; kb++) quant_group(&x[kb * 128], 128, qx[kb], sx[kb]);
  for (int a = 0; a < ACTIVE; a++) {
    int e = ids_all[a];
    for (int i = 0; i < I; i++) {
      double gq = 0, uq = 0, gf = 0, uf = 0;
      for (int kb = 0; kb < KB; kb++) {
        double sg = sgu[(size_t)e * 4 * KB + (i / 128) * KB + kb], su = sgu[(size_t)e * 4 * KB + (2 + i / 128) * KB + kb];
        double dg = 0, du = 0, fg = 0, fu = 0;
        for (int k = 0; k < 128; k++) {
          double wgv = e4m3_to_float(wgu[(size_t)e * 512 * H + (size_t)i * H + kb * 128 + k]);
          double wuv = e4m3_to_float(wgu[(size_t)e * 512 * H + (size_t)(256 + i) * H + kb * 128 + k]);
          dg += wgv * qx[kb][k]; du += wuv * qx[kb][k];
          fg += wgv * x[kb * 128 + k]; fu += wuv * x[kb * 128 + k];
        }
        gq += sg * sx[kb] * dg; uq += su * sx[kb] * du; gf += sg * fg; uf += su * fu;
      }
      double ref = gq / (1 + std::exp(-gq)) * uq, reff = gf / (1 + std::exp(-gf)) * uf;
      worst_h = std::max(worst_h, std::fabs(h[a * I + i] - ref)); worst_h_f32 = std::max(worst_h_f32, std::fabs(h[a * I + i] - reff));
      scale_h = std::max(scale_h, std::fabs(ref));
    }
  }
  for (int j = 0; j < H; j++) {
    double o = 0;
    for (int a = 0; a < ACTIVE; a++) {
      int e = ids_all[a];
      for (int kb = 0; kb < 2; kb++) {
        std::vector<double> q; double shx; quant_group(&h[a * I + kb * 128], 128, q, shx);
        double d = 0;
        for (int k = 0; k < 128; k++) d += e4m3_to_float(wdn[(size_t)e * H * I + (size_t)j * I + kb * 128 + k]) * q[k];
        o += route[a] * sdn[(size_t)e * KB * 2 + (j / 128) * 2 + kb] * shx * d;
      }
    }
    worst_o = std::max(worst_o, std::fabs(out[j] - o)); scale_o = std::max(scale_o, std::fabs(o));
  }
  printf("{\"check\": {\"gate_up_max_abs_err\": %.3g, \"gate_up_max_abs\": %.3g, \"gate_up_vs_unquantized_x_max_abs_err\": %.3g, "
         "\"down_max_abs_err\": %.3g, \"down_max_abs\": %.3g}", worst_h, scale_h, worst_h_f32, worst_o, scale_o);

  // time: a graph of `layers` layers, replayed
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int l = 0; l < layers; l++) launch(l, s);
  CK(cudaStreamEndCapture(s, &graph)); CK(cudaGraphInstantiate(&exec, graph, 0));
  // a gate_up-only and a down-only graph, to split the time
  cudaGraph_t g1, g2; cudaGraphExec_t e1, e2;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int l = 0; l < layers; l++) gate_up<<<dim3(I / 16, ACTIVE), 32 * GU_WARPS, 0, s>>>(d_wgu, d_sgu, d_x, d_ids + l * ACTIVE, d_h + l * ACTIVE * I);
  CK(cudaStreamEndCapture(s, &g1)); CK(cudaGraphInstantiate(&e1, g1, 0));
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int l = 0; l < layers; l++) down<<<H / 16, 32 * ACTIVE, 0, s>>>(d_wdn, d_sdn, d_h + l * ACTIVE * I, d_ids + l * ACTIVE, d_route, d_out + l * H);
  CK(cudaStreamEndCapture(s, &g2)); CK(cudaGraphInstantiate(&e2, g2, 0));
  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  auto time = [&](cudaGraphExec_t ex) {
    for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ex, s));
    std::vector<float> ms;
    for (int i = 0; i < 20; i++) { CK(cudaEventRecord(t0, s)); CK(cudaGraphLaunch(ex, s)); CK(cudaEventRecord(t1, s)); CK(cudaEventSynchronize(t1)); float m; CK(cudaEventElapsedTime(&m, t0, t1)); ms.push_back(m); }
    std::sort(ms.begin(), ms.end()); return 1e3 * ms[ms.size() / 2] / layers;
  };
  double both = time(exec), gu = time(e1), dn = time(e2);
  double gu_mb = ACTIVE * 512.0 * H / 1e6, dn_mb = ACTIVE * (double)H * I / 1e6;
  printf(", \"us_per_layer\": {\"gate_up_and_down\": %.2f, \"gate_up\": %.2f, \"down\": %.2f}, \"weight_MB_per_layer\": {\"gate_up\": %.1f, \"down\": %.1f}, "
         "\"TB_per_s\": {\"gate_up\": %.2f, \"down\": %.2f}}\n", both, gu, dn, gu_mb, dn_mb, gu_mb / gu, dn_mb / dn);
  return 0;
}
