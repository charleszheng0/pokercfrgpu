#include "thread_pool.hpp"

#include <algorithm>
#include <chrono>

namespace poker {

namespace {
// Depth of parallel_for items on this thread. 0 = not inside an item (top level).
thread_local int tl_depth = 0;

// Time an idle thread spins (with yield) before it blocks on the condition variable.
// Default 0 (block at once): on a shared, oversubscribed machine a 50 us spin made the
// turn benchmark slower (spinning threads take time slices from working threads).
#ifndef POOL_SPIN_US
#define POOL_SPIN_US 0
#endif
constexpr auto kSpin = std::chrono::microseconds(POOL_SPIN_US);
}  // namespace

ThreadPool::ThreadPool(int num_threads) {
  const int nw = std::max(0, num_threads - 1);
  idle_.store(nw);  // workers start idle
  for (int i = 0; i < nw; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

bool ThreadPool::any_claimable_locked() const {
  for (const Group* g : groups_)
    if (g->next.load(std::memory_order_relaxed) < g->n) return true;
  return false;
}

// Claims one item of a published group (oldest group first: its items are usually larger).
// The claim is made under mu_, and an owner removes its group under mu_ before it returns,
// so the group is alive here. After a claim with i < n the owner waits for our done
// increment, so the group stays alive until then.
bool ThreadPool::try_claim(Group*& g, int& i) {
  if (num_groups_.load(std::memory_order_relaxed) == 0) return false;
  std::lock_guard<std::mutex> lk(mu_);
  for (Group* c : groups_) {
    if (c->next.load(std::memory_order_relaxed) >= c->n) continue;
    const int k = c->next.fetch_add(1, std::memory_order_relaxed);
    if (k < c->n) {
      g = c;
      i = k;
      return true;
    }
  }
  return false;
}

// Runs item i of another thread's group, then more items of the same group while there are
// any. While we hold an unreported item, done < n, so the owner waits and g is alive: we
// claim the next item without the lock, before we report the current one. After the last
// done.fetch_add we do not touch g: the owner can return and destroy it.
void ThreadPool::run_claimed(Group* g, int i) {
  const int n = g->n;
  const std::function<void(int)>& fn = *g->fn;
  ++tl_depth;
  for (;;) {
    fn(i);
    const int j = g->next.fetch_add(1, std::memory_order_relaxed);
    if (j >= n) break;
    g->done.fetch_add(1, std::memory_order_release);  // done < n: we still hold item j
    i = j;
  }
  --tl_depth;
  // release: the owner's acquire load of done == n sees our writes from all our fn calls.
  if (g->done.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
    // The owner can be blocked on cv_ with predicate done == n. Take the lock so the
    // notification cannot fall between its check and its wait.
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();
  }
}

void ThreadPool::publish(Group* g) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    groups_.push_back(g);
    num_groups_.store(static_cast<int>(groups_.size()), std::memory_order_relaxed);
  }
  g->published = true;
  cv_.notify_all();
}

void ThreadPool::unpublish(Group* g) {
  std::lock_guard<std::mutex> lk(mu_);
  groups_.erase(std::find(groups_.begin(), groups_.end(), g));
  num_groups_.store(static_cast<int>(groups_.size()), std::memory_order_relaxed);
  g->published = false;
}

void ThreadPool::worker_loop() {
  bool searching = true;  // counted in idle_ (the constructor counted us)
  bool spinning = false;
  std::chrono::steady_clock::time_point spin_end;
  for (;;) {
    Group* g;
    int i;
    if (try_claim(g, i)) {
      if (searching) {
        idle_.fetch_sub(1, std::memory_order_relaxed);
        searching = false;
      }
      spinning = false;
      run_claimed(g, i);
      continue;
    }
    if (!searching) {
      idle_.fetch_add(1, std::memory_order_relaxed);
      searching = true;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!spinning) {
      spinning = true;
      spin_end = now + kSpin;
    }
    if (now < spin_end) {
      std::this_thread::yield();
      continue;
    }
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || any_claimable_locked(); });
    if (stop_) return;
    spinning = false;
  }
}

// Waits until all items of our own group are done. Our group is no longer published.
// A top-level owner (help = true) runs items of other groups meanwhile. A nested owner
// does not, so its stack does not grow.
void ThreadPool::wait_done(Group* g, bool help) {
  auto finished = [&] { return g->done.load(std::memory_order_acquire) == g->n; };
  bool searching = false;
  bool spinning = false;
  std::chrono::steady_clock::time_point spin_end;
  while (!finished()) {
    Group* h;
    int i;
    if (help && try_claim(h, i)) {
      if (searching) {
        idle_.fetch_sub(1, std::memory_order_relaxed);
        searching = false;
      }
      spinning = false;
      run_claimed(h, i);
      continue;
    }
    if (help && !searching) {
      idle_.fetch_add(1, std::memory_order_relaxed);
      searching = true;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!spinning) {
      spinning = true;
      spin_end = now + kSpin;
    }
    if (now < spin_end) {
      std::this_thread::yield();
      continue;
    }
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return finished() || (help && any_claimable_locked()); });
    spinning = false;
  }
  if (searching) idle_.fetch_sub(1, std::memory_order_relaxed);
}

void ThreadPool::parallel_for(int n, const std::function<void(int)>& fn) {
  if (workers_.empty() || n <= 1) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  const bool top = tl_depth == 0;
  Group g;
  g.fn = &fn;
  g.n = n;
  if (top) publish(&g);

  // Run our own items. A nested group is published only when a thread is idle, so in a
  // busy pool a nested loop costs one relaxed load per item.
  ++tl_depth;
  for (;;) {
    const int i = g.next.fetch_add(1, std::memory_order_relaxed);
    if (i >= n) break;
    if (!g.published && i + 1 < n && idle_.load(std::memory_order_relaxed) > 0) publish(&g);
    fn(i);
    g.done.fetch_add(1, std::memory_order_relaxed);  // only this thread needs to see it
  }
  --tl_depth;

  if (g.published) unpublish(&g);
  wait_done(&g, top);
}

}  // namespace poker
