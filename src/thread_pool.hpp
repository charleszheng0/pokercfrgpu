// A small fixed thread pool with one operation: parallel_for.
// Nested calls from inside a running job run on the calling thread (no nesting).
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace poker {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void parallel_for(int n, const std::function<void(int)>& fn);
  int size() const { return static_cast<int>(workers_.size()) + 1; }

 private:
  void run_items();
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable start_cv_, done_cv_;
  const std::function<void(int)>* job_ = nullptr;
  int n_ = 0;
  std::atomic<int> next_{0};
  int active_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace poker
