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
    if (n <= 65536) return 12;
    if (n <= 262144) return 13;
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
    std::size_t const num_buckets_unsigned = static_cast<std::size_t>(1) << c; // 2^c
    unsigned const num_windows = (256 + c - 1) / c;                   // ceil(256/c)
    // eff_buckets: 2^c unsigned, or 2^(c-1) signed — set after signed-digit init
    std::size_t num_buckets = num_buckets_unsigned;

    // Pre-allocate bucket / scratch arrays.
    // Stack for small windows (c<=6, 64 entries); thread_local pool for larger
    // windows — avoids malloc/free on every Pippenger call (V-PERF-02/P1-3).
    constexpr std::size_t STACK_BUCKETS = 64;
    Point stack_buckets[STACK_BUCKETS];
    static thread_local std::vector<Point>          tl_buckets;
    static thread_local std::vector<std::size_t>    tl_touched;
    static thread_local std::vector<std::uint8_t>   tl_used;
    Point*        buckets = stack_buckets;
    std::size_t   touched_stack[STACK_BUCKETS];
    std::size_t*  touched = touched_stack;
    std::uint8_t  used_stack[STACK_BUCKETS];
    std::uint8_t* used = used_stack;
    if (num_buckets_unsigned > STACK_BUCKETS) {
        if (tl_buckets.size() < num_buckets_unsigned) tl_buckets.resize(num_buckets_unsigned);
        if (tl_touched.size() < num_buckets_unsigned) tl_touched.resize(num_buckets_unsigned);
        if (tl_used.size()    < num_buckets_unsigned) tl_used.resize(num_buckets_unsigned);
        buckets = tl_buckets.data();
        touched = tl_touched.data();
        used    = tl_used.data();
    }
    std::memset(used, 0, num_buckets_unsigned * sizeof(std::uint8_t));

    // Pre-extract all scalar digits — thread_local pool avoids 208KB+ malloc
    // per call (n=4096, c=10, num_windows=26 → 212992 bytes).
    // Layout: window-major digits[w * n + i] so the scatter inner loop reads sequentially.
    // Extraction order: scalar-major (outer=scalar, inner=window) so each scalar's 32 bytes
    // stay hot in L1 cache across all num_windows extractions (B-3: eliminates n-1 reloads/window).
    static thread_local std::vector<std::uint16_t> tl_digits;
    std::size_t const digits_count = n * static_cast<std::size_t>(num_windows);
    if (tl_digits.size() < digits_count) tl_digits.resize(digits_count);
    std::uint16_t* digits = tl_digits.data();
    for (std::size_t i = 0; i < n; ++i) {
        for (unsigned w = 0; w < num_windows; ++w) {
            digits[static_cast<std::size_t>(w) * n + i] =
                static_cast<std::uint16_t>(extract_digit(scalars[i], w * c, c));
        }
    }
    // Signed-digit conversion for c >= 7: halves bucket count from 2^c to 2^(c-1).
    // Carry propagation: for each digit d > 2^(c-1), d -= 2^c and carry +1 to next window.
    // Scatter: positive digits add point, negative digits add negated point.
    // Savings: ~50% fewer buckets → ~50% less aggregate work per window.
    // Overhead: O(n × num_windows) carry compare-and-branch (~2ns each, negligible).
    bool const use_signed = (c >= 7);
    static thread_local std::vector<std::int16_t> tl_sdigits;
    std::int16_t* sdigits = nullptr;
    if (use_signed) {
        if (tl_sdigits.size() < digits_count) tl_sdigits.resize(digits_count);
        sdigits = tl_sdigits.data();
        // Copy unsigned digits to signed buffer
        for (std::size_t k = 0; k < digits_count; ++k) {
            sdigits[k] = static_cast<std::int16_t>(digits[k]);
        }
        // Carry propagation (window-major, LSB→MSB)
        int16_t const half  = static_cast<int16_t>(1 << (c - 1));
        int16_t const base  = static_cast<int16_t>(1 << c);
        for (unsigned w = 0; w < num_windows; ++w) {
            std::int16_t* row      = sdigits + w * n;
            std::int16_t* next_row = (w + 1 < num_windows) ? sdigits + (w + 1) * n : nullptr;
            for (std::size_t i = 0; i < n; ++i) {
                if (row[i] > half) {
                    row[i] = static_cast<std::int16_t>(row[i] - base);
                    if (next_row) next_row[i]++;
                }
            }
        }
        // Halve bucket count: only need [1, 2^(c-1)]. TLS pools already sized above.
        num_buckets >>= 1;
        std::memset(used, 0, num_buckets * sizeof(std::uint8_t));
    }

    // Scan ALL points to determine if all non-infinity points are affine.
    // The first-point heuristic (B-11) was incorrect: mixed affine/Jacobian
    // input caused wrong results when the first point was affine but later ones
    // were Jacobian (add_mixed52_inplace gives wrong result on Jacobian input).
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
        constexpr std::size_t PREFETCH_DIST = 8;
        const std::uint16_t* const wrow  = digits  + static_cast<std::size_t>(w) * n;
        const std::int16_t*  const swrow = sdigits ? sdigits + static_cast<std::size_t>(w) * n : nullptr;
        if (use_signed) {
            // Signed scatter: bucket[|d|] += (d>0 ? P : -P)
            for (std::size_t i = 0; i < n; ++i) {
                if (SECP256K1_LIKELY(i + PREFETCH_DIST < n)) {
#ifdef __GNUC__
                    __builtin_prefetch(&points[i + PREFETCH_DIST], 0, 1);
#endif
                }
                std::int16_t const sd = swrow[i];
                if (SECP256K1_UNLIKELY(sd == 0) || SECP256K1_UNLIKELY(points[i].is_infinity())) continue;
                bool const is_neg = sd < 0;
                std::size_t const abs_d = is_neg ? static_cast<std::size_t>(-sd)
                                                  : static_cast<std::size_t>(sd);
                if (!used[abs_d]) {
                    used[abs_d] = 1;
                    touched[touched_count++] = abs_d;
                    max_touched_digit = std::max(max_touched_digit, abs_d);
#if defined(SECP256K1_FAST_52BIT)
                    buckets[abs_d] = Point::from_affine52(points[i].X52(), points[i].Y52());
#else
                    buckets[abs_d] = Point::from_affine(points[i].X(), points[i].Y());
#endif
                    if (is_neg) buckets[abs_d].negate_inplace();
                    continue;
                }
#if defined(SECP256K1_FAST_52BIT)
                if (is_neg) {
                    buckets[abs_d].add_mixed52_neg_inplace(points[i].X52(), points[i].Y52());
                } else {
                    buckets[abs_d].add_mixed52_inplace(points[i].X52(), points[i].Y52());
                }
#else
                if (is_neg) {
                    Point neg = points[i]; neg.negate_inplace();
                    buckets[abs_d].add_inplace(neg);
                } else {
                    buckets[abs_d].add_inplace(points[i]);
                }
#endif
                used[abs_d] = 2;
            }
        } else if (all_affine) {
            for (std::size_t i = 0; i < n; ++i) {
                if (SECP256K1_LIKELY(i + PREFETCH_DIST < n)) {
#ifdef __GNUC__
                    __builtin_prefetch(&points[i + PREFETCH_DIST], 0, 1);
#endif
                }
                std::uint32_t const digit = wrow[i];
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
                used[digit] = 2;  // bucket is now Jacobian
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if (SECP256K1_LIKELY(i + PREFETCH_DIST < n)) {
#ifdef __GNUC__
                    __builtin_prefetch(&points[i + PREFETCH_DIST], 0, 1);
#endif
                }
                std::uint32_t const digit = wrow[i];
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
        // bool flags replace is_infinity() calls: avoids per-bucket function call overhead
        // (for c=8: 256 buckets × 43 windows = ~11K calls eliminated per MSM).
        Point running_sum = Point::infinity();
        Point partial_sum = Point::infinity();
        bool running_sum_nonempty = false;
        bool partial_sum_nonempty = false;

        for (std::size_t b = max_touched_digit; b >= 1; --b) {
            // Only read from buckets that were explicitly written this window.
            // Untouched slots remain uninitialized on the stack; adding the
            // identity element would be a no-op, so skipping them is correct
            // and avoids MSan uninitialized-read false positives.
            if (SECP256K1_LIKELY(used[b] != 0)) {
                if (running_sum_nonempty) {
#if defined(SECP256K1_FAST_52BIT)
                    // used[b]==1: bucket set exactly once (from_affine52, z=1) — cheaper mixed-add
                    if (all_affine && used[b] == 1) {
                        running_sum.add_mixed52_inplace(buckets[b].X52(), buckets[b].Y52());
                    } else {
                        running_sum.add_inplace(buckets[b]);
                    }
#else
                    running_sum.add_inplace(buckets[b]);
#endif
                } else {
                    running_sum = buckets[b];
                    running_sum_nonempty = true;
                }
            }
            if (running_sum_nonempty) {
                if (partial_sum_nonempty) {
                    partial_sum.add_inplace(running_sum);
                } else {
                    partial_sum = running_sum;
                    partial_sum_nonempty = true;
                }
            }
        }

        // Combine this window's contribution (skip if window had no non-zero buckets)
        if (partial_sum_nonempty) {
            result.add_inplace(partial_sum);
        }

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
