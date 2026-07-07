#ifndef SECP256K1_DETAIL_BATCH_POOL_HPP
#define SECP256K1_DETAIL_BATCH_POOL_HPP

// Persistent worker pool for CPU batch verification.
//
// Rationale: the batch-verify _mt paths used to spawn a fresh std::thread set on EVERY
// call and join them at the end. For libbitcoin IBD (one bridge call per block, many
// blocks) that meant a thread create+join storm AND cold thread_local state (e.g. the
// pubkey decompress/verify caches) on every block. A single persistent pool, created
// once and reused, removes the per-call spawn cost and keeps worker thread_locals warm
// across calls. This matches what libbitcoin's libsecp path gets from std::for_each(par)
// (a persistent thread pool under the hood).
//
// Model: one job at a time (callers serialize). Work [0,n) is handed out in steal-sized
// chunks via an atomic cursor, so threads keep taking chunks "until exhausted" rather
// than getting a fixed slice. The CALLING thread also participates as a worker, so a
// pool of N background threads + the caller gives up to N+1-way parallelism but the
// pool is sized to hardware_concurrency (caller + N-1 background = N).

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace secp256k1 {
namespace detail {

class BatchWorkerPool {
public:
    BatchWorkerPool() {
        unsigned hw = std::thread::hardware_concurrency();
        n_workers_ = hw ? hw : 1u;
        // n_workers_-1 background threads; the caller thread is the remaining worker.
        for (unsigned i = 1; i < n_workers_; ++i)
            workers_.emplace_back([this, i] { worker_loop(i); });
    }
    ~BatchWorkerPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++gen_;
        }
        cv_start_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }
    BatchWorkerPool(const BatchWorkerPool&) = delete;
    BatchWorkerPool& operator=(const BatchWorkerPool&) = delete;

    unsigned size() const noexcept { return n_workers_; }

    // Run fn over [0,n) in steal-sized chunks across up to `want` workers (incl. the
    // caller). Blocks until complete. Early-out: if any fn(start,end) returns false the
    // run stops as soon as workers observe it and run() returns false.
    bool run(std::size_t n, std::size_t steal, unsigned want,
             const std::function<bool(std::size_t, std::size_t)>& fn) {
        if (steal == 0) steal = n ? n : 1;
        // Serial fast path: tiny work, single worker, or a degenerate pool.
        if (want <= 1 || n_workers_ <= 1 || n <= steal) {
            for (std::size_t s = 0; s < n; s += steal)
                if (!fn(s, std::min(s + steal, n))) return false;
            return true;
        }
        std::unique_lock<std::mutex> job(job_m_);  // one job at a time
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            n_ = n;
            steal_ = steal;
            next_.store(0, std::memory_order_relaxed);
            ok_.store(true, std::memory_order_relaxed);
            want_ = std::min(want, n_workers_);
            pending_.store(want_ - 1, std::memory_order_relaxed);  // background participants
            ++gen_;
        }
        cv_start_.notify_all();
        drain();  // caller participates
        if (want_ > 1) {
            std::unique_lock<std::mutex> lk(done_m_);
            cv_done_.wait(lk, [this] { return pending_.load(std::memory_order_acquire) == 0; });
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = nullptr;
        }
        return ok_.load(std::memory_order_acquire);
    }

private:
    void drain() {
        const std::function<bool(std::size_t, std::size_t)>& fn = *fn_;
        for (;;) {
            if (!ok_.load(std::memory_order_relaxed)) return;
            const std::size_t s = next_.fetch_add(1, std::memory_order_relaxed) * steal_;
            if (s >= n_) return;
            const std::size_t e = std::min(s + steal_, n_);
            if (!fn(s, e)) {
                ok_.store(false, std::memory_order_relaxed);
                return;
            }
        }
    }

    void worker_loop(unsigned id) {
        std::uint64_t seen = 0;
        for (;;) {
            bool participate;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_start_.wait(lk, [this, &seen] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                participate = (id < want_);
            }
            if (participate) {
                drain();
                if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    std::lock_guard<std::mutex> dlk(done_m_);
                    cv_done_.notify_one();
                }
            }
        }
    }

    unsigned                 n_workers_ = 1;
    std::vector<std::thread>  workers_;
    std::mutex                m_, job_m_, done_m_;
    std::condition_variable   cv_start_, cv_done_;
    const std::function<bool(std::size_t, std::size_t)>* fn_ = nullptr;
    std::size_t               n_ = 0, steal_ = 0;
    std::atomic<std::size_t>  next_{0};
    std::atomic<bool>         ok_{true};
    std::atomic<unsigned>     pending_{0};
    unsigned                  want_ = 0;
    std::uint64_t             gen_ = 0;
    bool                      stop_ = false;
};

// Single shared instance for the whole program (defined in batch_verify.cpp).
BatchWorkerPool& batch_worker_pool();

}  // namespace detail
}  // namespace secp256k1

#endif  // SECP256K1_DETAIL_BATCH_POOL_HPP
