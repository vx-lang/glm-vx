// The floor for one MoE layer's weight reads at TP8 batch 1: a kernel that only streams MB megabytes (16-byte
// loads, UNROLL in flight a lane, one store a block), timed as the probe times its kernels: a CUDA graph of 75
// launches, each over its own part of a 4 GB buffer (so few reads hit L2), median of 20 replays.
//   nvcc -O3 -arch=sm_90a -o stream_floor stream_floor.cu && ./stream_floor
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

template <int UNROLL>
__global__ void __launch_bounds__(256) stream(const uint4* __restrict__ p, size_t n16, unsigned* out) {
  unsigned acc = 0;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride * UNROLL) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; u++) v[u] = i + u * stride < n16 ? __ldg(p + i + u * stride) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int u = 0; u < UNROLL; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  acc = __reduce_xor_sync(0xffffffff, acc);
  if (threadIdx.x == 0) atomicXor(out, acc);
}

int main() {
  const int layers = 75;
  const size_t total = (size_t)4 << 30;  // 75 layers of up to 42.5 MB, each its own part
  uint8_t* buf; unsigned* out;
  CK(cudaMalloc(&buf, total)); CK(cudaMemset(buf, 1, total)); CK(cudaMalloc(&out, 4));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  printf("{\"sms\": %d, \"rows\": [", sms);
  bool first = true;
  for (double mb : {14.2, 28.3, 42.5}) {
    const size_t bytes = (size_t)(mb * 1e6) / 16 * 16, n16 = bytes / 16, step = total / layers / 256 * 256;
    for (int blocks_per_sm : {2, 4, 8}) {
      for (int unroll : {2, 4}) {
        const int grid = sms * blocks_per_sm;
        cudaGraph_t g; cudaGraphExec_t ex;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int l = 0; l < layers; l++) {
          const uint4* p = reinterpret_cast<const uint4*>(buf + l * step);
          if (unroll == 2) stream<2><<<grid, 256, 0, s>>>(p, n16, out); else stream<4><<<grid, 256, 0, s>>>(p, n16, out);
        }
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ex, g, 0));
        for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ex, s));
        std::vector<float> ms;
        for (int i = 0; i < 20; i++) {
          CK(cudaEventRecord(t0, s)); CK(cudaGraphLaunch(ex, s)); CK(cudaEventRecord(t1, s)); CK(cudaEventSynchronize(t1));
          float m; CK(cudaEventElapsedTime(&m, t0, t1)); ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        const double us = 1e3 * ms[ms.size() / 2] / layers;
        printf("%s{\"MB\": %.1f, \"blocks_per_sm\": %d, \"unroll\": %d, \"us\": %.2f, \"TB_per_s\": %.2f}", first ? "" : ", ", mb, blocks_per_sm, unroll, us, bytes / us / 1e6);
        first = false;
        CK(cudaGraphExecDestroy(ex)); CK(cudaGraphDestroy(g));
      }
    }
  }
  printf("]}\n");
  return 0;
}
