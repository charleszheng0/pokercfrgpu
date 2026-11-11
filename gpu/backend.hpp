// Two back ends for the level-by-level solver: CUDA (nvcc) and CPU threads (any compiler).
// The solver is written once with for_each(n, LAMBDA(int64_t i) {...}) and Buf<T>.
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#define LAMBDA [=] __host__ __device__

namespace poker {

inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

template <class F>
__global__ void k_for_each(int64_t n, F f) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) f(i);
}

template <class F>
void for_each(int64_t n, F f) {
  if (n <= 0) return;
  const int t = 256;
  const int64_t b = (n + t - 1) / t;
  k_for_each<<<static_cast<unsigned>(b), t>>>(n, f);
  cuda_check(cudaGetLastError(), "launch");
}

inline void device_sync() { cuda_check(cudaDeviceSynchronize(), "sync"); }
inline const char* backend_name() { return "cuda"; }

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  ~Buf() {
    if (p) cudaFree(p);
  }
  void alloc(size_t k) {
    if (p) cudaFree(p), p = nullptr;
    n = k;
    if (k) cuda_check(cudaMalloc(&p, k * sizeof(T)), "malloc");
  }
  void zero() {
    if (n) cuda_check(cudaMemset(p, 0, n * sizeof(T)), "memset");
  }
  void upload(const std::vector<T>& v) {
    alloc(v.size());
    if (n) cuda_check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  void upload_at(size_t off, const T* src, size_t k) {
    cuda_check(cudaMemcpy(p + off, src, k * sizeof(T), cudaMemcpyHostToDevice), "upload_at");
  }
  void download_at(size_t off, T* dst, size_t k) const {
    cuda_check(cudaMemcpy(dst, p + off, k * sizeof(T), cudaMemcpyDeviceToHost), "download_at");
  }
  size_t bytes() const { return n * sizeof(T); }
};

}  // namespace poker

#else  // ---------------------------------------------------------------- CPU threads

#include <algorithm>
#include <thread>

#include "thread_pool.hpp"
#define HD
#define LAMBDA [=]

namespace poker {

inline ThreadPool& cpu_pool() {
  static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
  return pool;
}

template <class F>
void for_each(int64_t n, F f) {
  if (n <= 0) return;
  if (n < 4096) {
    for (int64_t i = 0; i < n; ++i) f(i);
    return;
  }
  const int chunks = 64;
  cpu_pool().parallel_for(chunks, [&](int c) {
    const int64_t lo = n * c / chunks, hi = n * (c + 1) / chunks;
    for (int64_t i = lo; i < hi; ++i) f(i);
  });
}

inline void device_sync() {}
inline const char* backend_name() { return "cpu"; }

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  std::vector<T> v;
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void alloc(size_t k) {
    v.assign(k, T{});
    n = k;
    p = v.data();
  }
  void zero() { std::fill(v.begin(), v.end(), T{}); }
  void upload(const std::vector<T>& src) {
    v = src;
    n = v.size();
    p = v.data();
  }
  void upload_at(size_t off, const T* src, size_t k) { std::memcpy(p + off, src, k * sizeof(T)); }
  void download_at(size_t off, T* dst, size_t k) const { std::memcpy(dst, p + off, k * sizeof(T)); }
  size_t bytes() const { return n * sizeof(T); }
};

}  // namespace poker
#endif
