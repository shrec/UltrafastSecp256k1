// ============================================================================
// Pippenger Bucket Method -- Multi-Scalar Multiplication
// ============================================================================
// Reference: Bernstein et al. "Faster batch forgery identification" (2012)
//
// GLV note: GLV-decomposition was evaluated but found counterproductive
// for Pippenger: doubling point count (2N) increases scatter/aggregate
// cost more than the saved window-doublings (ceil(128/c) vs ceil(256/c)).
// Individual scalar_mul already uses GLV internally.
//
// Bucket method for computing sum(s_i * P_i):
//   For each window of c bits, scatter points into 2^c buckets by digit,
//   aggregate buckets bottom-up (running sum trick), then combine windows.

#include "secp256k1/pippenger.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/config.hpp"
#include <algorithm>
#include <cstring>
#include <memory>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// -- Optimal Window Width -----------------------------------------------------
// Empirical CPU heuristic after affine fast-path + touched-bucket optimizations.
// Measured crossover bands on current x86-64 path:
//   n=48..72   -> c=5
//   n=80..384  -> c=6
//   n=512      -> c=7
//   n=1024     -> c=8
// Larger bands stay conservative and can be retuned again with hardware data.

unsigned pippenger_optimal_window(std::size_t n) {
    if (n <= 1)    return 1;
    if (n <= 4)    return 2;
    if (n <= 8)    return 3;
    if (n <= 16)   return 4;
    if (n <= 72)   return 5;
    if (n <= 384)  return 6;
    if (n <= 768)  return 7;
    if (n <= 2048) return 8;
    if (n <= 8192) return 9;
    if (n <= 32768) return 10;
    if (n <= 131072) return 12;
    return 14;
}

// -- Extract c-bit digit at position `bit_offset` from scalar -----------------
// Extracts bits [bit_offset, bit_offset+width) from the scalar.
// Returns unsigned digit in [0, 2^width).
// Word-level extraction: 1-2 limb reads instead of `width` calls to s.bit().
static inline uint32_t extract_digit(const Scalar& s, unsigned bit_offset, unsigned width) {
    auto const& limbs = s.limbs();
    unsigned const limb_idx = bit_offset >> 6;   // / 64
    unsigned const bit_idx  = bit_offset & 63;   // % 64

    // Primary word: shift down to align desired bits
    std::uint64_t word = limbs[limb_idx] >> bit_idx;

    // If window crosses a limb boundary, OR in bits from next limb
    if (bit_idx + width > 64 && limb_idx < 3) {
        word |= limbs[limb_idx + 1] << (64 - bit_idx);
    }

    return static_cast<uint32_t>(word) & ((1U << width) - 1);
}

// -- Pippenger Core -----------------------------------------------------------
Point pippenger_msm(const Scalar* scalars,
                    const Point* points,
                    std::size_t n) {
    // Trivial cases
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);

    // For small n, fall back to Strauss (lower constant factor).
    // Empirical crossover on the current CPU path is around n ~= 48.
    if (n < 48) {
        return multi_scalar_mul(scalars, points, n);
    }

    unsigned const c = pippenger_optimal_window(n);
    std::size_t const num_buckets = static_cast<std::size_t>(1) << c; // 2^c
    unsigned const num_windows = (256 + c - 1) / c;                   // ceil(256/c)

    // Pre-allocate bucket array (reused per window).
    // Keep small window sizes (c<=6, 64 buckets) on stack; larger windows go to
    // heap to cap stack usage at ~7 KB for the bucket array (M-05: avoid large
    // stack frames -- 256-bucket stack path was ~26-35 KB depending on Point size).
    constexpr std::size_t STACK_BUCKETS = 64;
    Point stack_buckets[STACK_BUCKETS];
    std::unique_ptr<Point[]> heap_buckets;
    Point* buckets = stack_buckets;
    if (num_buckets > STACK_BUCKETS) {
        heap_buckets = std::make_unique<Point[]>(num_buckets);
        buckets = heap_buckets.get();
    }
    // touched[] and used[] track which buckets are non-empty this window,
    // so we can reset only those (avoids O(2^c) clear per window).
    std::size_t touched_stack[STACK_BUCKETS];
    std::unique_ptr<std::size_t[]> touched_heap;
    std::size_t* touched = touched_stack;
    if (num_buckets > STACK_BUCKETS) {
        touched_heap = std::make_unique<std::size_t[]>(num_buckets);
        touched = touched_heap.get();
    }
    std::uint8_t used_stack[STACK_BUCKETS];
    std::unique_ptr<std::uint8_t[]> used_heap;
    std::uint8_t* used = used_stack;
    if (num_buckets > STACK_BUCKETS) {
        used_heap = std::make_unique<std::uint8_t[]>(num_buckets);
        used = used_heap.get();
    }
    std::memset(used, 0, num_buckets * sizeof(std::uint8_t));

    // Pre-extract all digits to avoid per-window scalar bit extraction.
    const auto digits =
        std::make_unique<std::uint16_t[]>(n * static_cast<std::size_t>(num_windows));
    for (std::size_t i = 0; i < n; ++i) {
        for (unsigned w = 0; w < num_windows; ++w) {
            digits[i * static_cast<std::size_t>(num_windows) + w] =
                static_cast<std::uint16_t>(extract_digit(scalars[i], w * c, c));
        }
    }
    bool all_affine = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (!points[i].is_infinity() && !points[i].is_normalized()) {
            all_affine = false;
            break;
        }
    }

    // Result accumulator
    Point result = Point::infinity();

    // Process windows from MSB to LSB
    for (int w = static_cast<int>(num_windows) - 1; w >= 0; --w) {
        // If not the first window, shift result left by c bits
        if (w < static_cast<int>(num_windows) - 1) {
            for (unsigned shift = 0; shift < c; ++shift) {
                result.dbl_inplace();
            }
        }

        std::size_t touched_count = 0;
        std::size_t max_touched_digit = 0;

        // -- Scatter: distribute points into buckets --
        // Prefetch distance: 8 points ahead keeps L2/L3 latency hidden for
        // the ~40-byte affine FE52 point load.
        constexpr std::size_t PREFETCH_DIST = 8;
        if (all_affine) {
            for (std::size_t i = 0; i < n; ++i) {
                if (SECP256K1_LIKELY(i + PREFETCH_DIST < n)) {
#ifdef __GNUC__
                    __builtin_prefetch(&points[i + PREFETCH_DIST], 0, 1);
#endif
                }
                std::uint32_t const digit = digits[i * static_cast<std::size_t>(num_windows) +
                                                          static_cast<std::size_t>(w)];
                if (SECP256K1_UNLIKELY(digit == 0) || SECP256K1_UNLIKELY(points[i].is_infinity())) continue;
                if (!used[digit]) {
                    used[digit] = 1;
                    touched[touched_count++] = static_cast<std::size_t>(digit);
                    max_touched_digit = std::max(max_touched_digit, static_cast<std::size_t>(digit));
#if defined(SECP256K1_FAST_52BIT)
                    buckets[digit] = Point::from_affine52(points[i].X52(), points[i].Y52());
#else
                    buckets[digit] = Point::from_affine(points[i].X(), points[i].Y());
#endif
                    continue;
                }
#if defined(SECP256K1_FAST_52BIT)
                buckets[digit].add_mixed52_inplace(points[i].X52(), points[i].Y52());
#else
                buckets[digit].add_mixed_inplace(points[i].X(), points[i].Y());
#endif
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if (SECP256K1_LIKELY(i + PREFETCH_DIST < n)) {
#ifdef __GNUC__
                    __builtin_prefetch(&points[i + PREFETCH_DIST], 0, 1);
#endif
                }
                std::uint32_t const digit = digits[i * static_cast<std::size_t>(num_windows) +
                                                          static_cast<std::size_t>(w)];
                if (SECP256K1_UNLIKELY(digit == 0)) continue;  // bucket[0] is unused (identity)
                if (!used[digit]) {
                    used[digit] = 1;
                    touched[touched_count++] = static_cast<std::size_t>(digit);
                    max_touched_digit = std::max(max_touched_digit, static_cast<std::size_t>(digit));
                    buckets[digit] = points[i];
                    continue;
                }
                buckets[digit].add_inplace(points[i]);
            }
        }

        // -- Aggregate buckets (running-sum trick) --
        // Computes sum_{b=1}^{2^c-1} b * bucket[b] efficiently:
        //   running_sum starts at bucket[2^c-1]
        //   partial_sum accumulates running_sum at each step
        //   This gives: partial_sum = 1*bucket[1] + 2*bucket[2] + ... = Sum b*bucket[b]
        Point running_sum = Point::infinity();
        Point partial_sum = Point::infinity();

        for (std::size_t b = max_touched_digit; b >= 1; --b) {
            // Only read from buckets that were explicitly written this window.
            // Untouched slots remain uninitialized on the stack; adding the
            // identity element would be a no-op, so skipping them is correct
            // and avoids MSan uninitialized-read false positives.
            if (SECP256K1_LIKELY(used[b] != 0)) {
                running_sum.add_inplace(buckets[b]);
            }
            partial_sum.add_inplace(running_sum);
        }

        // Combine this window's contribution
        result.add_inplace(partial_sum);

        // Reset only touched buckets (O(touched) instead of O(2^c))
        for (std::size_t i = 0; i < touched_count; ++i) {
            buckets[touched[i]] = Point::infinity();
            used[touched[i]] = 0;
        }
    }

    return result;
}

// -- Signed-digit Pippenger (halved bucket count) -----------------------------
// Uses signed digits [-2^(c-1), ..., -1, 0, 1, ..., 2^(c-1)]
// This halves the number of buckets (2^(c-1) instead of 2^c) at the cost
// of a carry propagation pass. Very effective for large n.
//
// Not yet enabled by default -- the unsigned version above is simpler and
// already very fast. This is provided for future optimization.

// -- Vector convenience -------------------------------------------------------
Point pippenger_msm(const std::vector<Scalar>& scalars,
                    const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return pippenger_msm(scalars.data(), points.data(), n);
}

// -- Unified MSM (auto-select) ------------------------------------------------
// Strauss for very small MSMs, Pippenger from n >= 48.
// Current crossover on the optimized CPU path is ~48 points.
// N=64 Schnorr batch -> 128 points in MSM -> Pippenger path.
Point msm(const Scalar* scalars,
          const Point* points,
          std::size_t n) {
    if (n < 48) {
        return multi_scalar_mul(scalars, points, n);
    }
    return pippenger_msm(scalars, points, n);
}

Point msm(const std::vector<Scalar>& scalars,
          const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return msm(scalars.data(), points.data(), n);
}

} // namespace secp256k1
