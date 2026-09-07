#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "modn_add_control.hpp"

// F1 research contract: return only (x0 + sum(rhs[0..count))) mod n.
// The canonical final value, not intermediate recurrence states, is observable.
// All operands must already be canonical pa_modn values; this is unchecked.
// Every arithmetic operation below uses the unchanged Original modular adder.
// Associativity permits summaries of contiguous sections of ONE input stream;
// these are not unrelated independent recurrences. No constant-time claim.
//
// Nonempty rhs must designate count readable Limbs. tree's scratch must have
// scratch_count writable Limbs, disjoint from rhs[0..count) AND x0. Pointer
// extents, lifetimes and disjointness are preconditions, not validated here.
// Only the final returned value is an output; no input element is modified.
//
// Checked errors: excessive count -> length_error; positive-count null rhs,
// insufficient/null tree scratch, empty/duplicate/out-of-range multicore CPU
// IDs -> invalid_argument. count==0 returns x0 without inspecting any pointer,
// scratch argument or CPU list. Allocation/thread failures propagate; started
// threads are joined even if a later launch fails. Affinity failures become
// system_error after all launched threads have joined.
namespace pa_f1 {

using Word = pa_modn::Word;
using Limbs = pa_modn::Limbs;

inline constexpr std::size_t max_count =
    std::numeric_limits<std::size_t>::max() / sizeof(Limbs);

namespace detail {

inline void validate_input(const Limbs* rhs, std::size_t count) {
    // Check before any dereference or count-derived indexing/allocation.
    if (count > max_count) {
        throw std::length_error("F1 count exceeds addressable Limbs capacity");
    }
    if (count != 0 && rhs == nullptr) {
        throw std::invalid_argument("F1 nonempty input requires rhs");
    }
}

inline Limbs plus(const Limbs& a, const Limbs& b) noexcept {
    return pa_modn::add<pa_modn::Method::Original>(a, b);
}

// Requires count>0. Compact adjacent pairs in place, preserving left-to-right
// order and carrying an odd tail unchanged. Exactly count-1 modular adds;
// no allocation. Writes stay inside values[0..count).
inline Limbs balanced_merge(Limbs* values, std::size_t count) noexcept {
    while (count > 1) {
        const std::size_t pairs = count / 2;
        for (std::size_t i = 0; i < pairs; ++i) {
            values[i] = plus(values[2 * i], values[2 * i + 1]);
        }
        if ((count & 1U) != 0) {
            values[pairs] = values[count - 1];
        }
        count = pairs + (count & 1U);
    }
    return values[0];
}

struct Chunk {
    std::size_t begin;
    std::size_t size;
};

// Requires 0<=index<chunks<=max_count. The first count%chunks sections get
// one extra element. Products/additions are bounded by count (no rounding-up
// addition to count); empty sections are valid when chunks>count.
inline Chunk chunk(std::size_t count, std::size_t chunks,
                   std::size_t index) noexcept {
    const std::size_t base = count / chunks;
    const std::size_t extra = count % chunks;
    return {index * base + std::min(index, extra),
            base + std::size_t(index < extra)};
}

// Constructed before the first thread starts. This owner can only run on the
// launching thread; every stored thread is distinct, joinable at most once,
// and never the owner itself. It also covers partial thread-creation failure.
struct JoinAll {
    std::vector<std::thread>& threads;
    ~JoinAll() {
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
};

} // namespace detail

// Baseline: count modular additions, no allocation, one dependency chain.
// Sequential execution is intentional: it is the recurrence reference for
// the representations below, not a production parallelism policy choice.
inline Limbs serial(const Limbs& x0, const Limbs* rhs, std::size_t count) {
    detail::validate_input(rhs, count);
    Limbs value = x0;
    for (std::size_t i = 0; i < count; ++i) {
        value = detail::plus(value, rhs[i]);
    }
    return value;
}

// Four is the representation's interleaving width, NOT a hardcoded worker
// count. Runs on one caller thread. Four balanced contiguous chunks are
// reduced from zero in interleaved order, merged as ordered pairs, and x0 is
// incorporated exactly once. count>0 costs count+4 modular additions, even
// when some chunks are empty; count==0 costs zero. No dynamic allocation.
inline Limbs chunk4_interleaved(const Limbs& x0, const Limbs* rhs,
                               std::size_t count) {
    detail::validate_input(rhs, count);
    if (count == 0) {
        return x0;
    }
    std::array<Limbs, 4> partials{};
    const auto c0 = detail::chunk(count, 4, 0);
    const auto c1 = detail::chunk(count, 4, 1);
    const auto c2 = detail::chunk(count, 4, 2);
    const auto c3 = detail::chunk(count, 4, 3);
    const std::size_t common = count / 4;
    for (std::size_t i = 0; i < common; ++i) {
        partials[0] = detail::plus(partials[0], rhs[c0.begin + i]);
        partials[1] = detail::plus(partials[1], rhs[c1.begin + i]);
        partials[2] = detail::plus(partials[2], rhs[c2.begin + i]);
        partials[3] = detail::plus(partials[3], rhs[c3.begin + i]);
    }
    if (c0.size > common) {
        partials[0] = detail::plus(partials[0], rhs[c0.begin + common]);
    }
    if (c1.size > common) {
        partials[1] = detail::plus(partials[1], rhs[c1.begin + common]);
    }
    if (c2.size > common) {
        partials[2] = detail::plus(partials[2], rhs[c2.begin + common]);
    }
    // The final balanced chunk always has size floor(count/4).
    return detail::plus(x0, detail::balanced_merge(partials.data(), 4));
}

// Caller owns the scratch allocation. Copying all count operands and all
// bottom-up work happen in this call; a full-job benchmark must also time its
// allocation/initialization/free if scratch is not a reusable public resource.
// count>0 costs count-1 tree adds plus one x0 add, i.e. count in total.
inline Limbs tree(const Limbs& x0, const Limbs* rhs, std::size_t count,
                  Limbs* scratch, std::size_t scratch_count) {
    detail::validate_input(rhs, count);
    if (count == 0) {
        return x0;
    }
    if (scratch == nullptr || scratch_count < count) {
        throw std::invalid_argument("F1 tree needs at least count scratch Limbs");
    }
    for (std::size_t i = 0; i < count; ++i) {
        scratch[i] = rhs[i];
    }
    return detail::plus(x0, detail::balanced_merge(scratch, count));
}

// CPU count and headroom policy belong to the DRIVER, based on the observed
// allowed CPU set. For count>0 this launches min(count,cpu_ids.size()) threads
// on the first CPU IDs, each owning a balanced contiguous input chunk. The
// entire list must be unique and in the Linux fixed CPU-set range; the driver
// must additionally validate current availability before timing. Actual
// affinity-setting errors are checked in each worker, never silently ignored.
//
// count>0 costs count+workers modular additions: zero-origin chunks cost count,
// their ordered balanced merge costs workers-1, and including x0 costs one.
// Three vector storage requests (partials, exception slots, reserved threads),
// platform thread-start storage, creation, affinity, joining, and destruction
// all occur INSIDE this call. The partials are zero-initialized; workers use
// private zero-origin state and publish once. No persistent pool/preprocessing.
inline Limbs multicore_cold(const Limbs& x0, const Limbs* rhs,
                           std::size_t count, const std::vector<int>& cpu_ids) {
    detail::validate_input(rhs, count);
    if (count == 0) {
        return x0;
    }
#if defined(__linux__)
    if (cpu_ids.empty()) {
        throw std::invalid_argument("F1 multicore needs at least one CPU ID");
    }
    cpu_set_t seen;
    CPU_ZERO(&seen);
    for (const int cpu : cpu_ids) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            throw std::invalid_argument("F1 CPU ID is outside CPU_SETSIZE");
        }
        if (CPU_ISSET(cpu, &seen)) {
            throw std::invalid_argument("F1 CPU IDs must be unique");
        }
        CPU_SET(cpu, &seen);
    }
    const std::size_t workers = std::min(count, cpu_ids.size());
    std::vector<Limbs> partials(workers);
    std::vector<std::exception_ptr> failures(workers);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    {
        detail::JoinAll join_all{threads};
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&, worker]() {
                try {
                    cpu_set_t mask;
                    CPU_ZERO(&mask);
                    CPU_SET(cpu_ids[worker], &mask);
                    const int status = pthread_setaffinity_np(
                        pthread_self(), sizeof(mask), &mask);
                    if (status != 0) {
                        throw std::system_error(status, std::generic_category(),
                                                "F1 worker affinity");
                    }
                    const auto section = detail::chunk(count, workers, worker);
                    Limbs value{};
                    for (std::size_t i = 0; i < section.size; ++i) {
                        value = detail::plus(value, rhs[section.begin + i]);
                    }
                    partials[worker] = value;
                } catch (...) {
                    failures[worker] = std::current_exception();
                }
            });
        }
    } // Join every thread before reading either results or exception slots.
    for (const auto& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
    return detail::plus(x0, detail::balanced_merge(partials.data(), workers));
#else
    (void)cpu_ids;
    throw std::runtime_error("F1 multicore affinity requires Linux");
#endif
}

} // namespace pa_f1
