#include "thread_pool.hpp"

namespace poker {

namespace {
thread_local bool tl_in_job = false;
}

ThreadPool::ThreadPool(int num_threads) {
  for (int i = 0; i + 1 < num_threads; ++i) {
    workers_.emplace_back([this] {
      tl_in_job = true;
      uint64_t seen = 0;
      for (;;) {
        std::unique_lock<std::mutex> lk(mu_);
        start_cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
        ++active_;
        lk.unlock();
        run_items();
        lk.lock();
        if (--active_ == 0) done_cv_.notify_all();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  start_cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::run_items() {
  for (;;) {
    int i = next_.fetch_add(1);
    if (i >= n_) return;
    (*job_)(i);
  }
}

void ThreadPool::parallel_for(int n, const std::function<void(int)>& fn) {
  if (workers_.empty() || n <= 1 || tl_in_job) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  std::unique_lock<std::mutex> lk(mu_);
  done_cv_.wait(lk, [&] { return active_ == 0; });  // previous job fully drained
  job_ = &fn;
  n_ = n;
  next_ = 0;
  ++generation_;
  lk.unlock();
  start_cv_.notify_all();
  tl_in_job = true;
  run_items();
  tl_in_job = false;
  lk.lock();
  done_cv_.wait(lk, [&] { return active_ == 0; });
}

}  // namespace poker
