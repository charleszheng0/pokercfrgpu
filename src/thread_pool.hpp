// A small fixed thread pool with one operation: parallel_for.
//
// Every parallel_for call makes a Group on the caller's stack, and the caller runs its own
// items. A top-level call publishes its group at once, so the workers help. A nested call
// (from inside an item) runs on its own thread, but publishes its group when it sees an idle
// thread; idle threads then take its remaining items. A nested owner never runs other
// groups' items while it waits, so the stack does not grow. Only a top-level owner helps.
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
  struct Group {
    const std::function<void(int)>* fn = nullptr;
    int n = 0;
    std::atomic<int> next{0};  // next item index to claim
    std::atomic<int> done{0};  // items finished, by any thread
    bool published = false;    // read and written only by the owner
  };

  void worker_loop();
  void publish(Group* g);
  void unpublish(Group* g);
  bool try_claim(Group*& g, int& i);
  bool any_claimable_locked() const;
  void run_claimed(Group* g, int i);
  void wait_done(Group* g, bool help);

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;      // group published, group finished, or stop
  std::vector<Group*> groups_;      // published groups (guarded by mu_)
  std::atomic<int> num_groups_{0};  // groups_.size(), for a check without the lock
  std::atomic<int> idle_{0};        // threads that look for work or sleep
  bool stop_ = false;
};

}  // namespace poker
