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
#include "secp256k1/glv.hpp"
#include "secp256k1/config.hpp"
#include <algorithm>
#include <cstring>
#include <memory>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// -- Optimal Window Width -----------------------------------------------------
// The bands below were re-measured on x86-64 (i5-14400F, GCC 14.2, performance
// governor, turbo off, pinned to cpu0, nice -20) by linking the SAME
// pippenger.cpp with the window forced to a fixed c and timing the identical
// input set, so the only difference between the numbers is c. Every timing was
// gated on pippenger_msm agreeing with multi_scalar_mul -- an independent
// implementation -- so a window cannot be reported fast because it is wrong.
//
// Microseconds per MSM, forced window (lower is better):
//
//     n     c=5      c=6      c=7      c=8      c=9
//    128  2327.5   2588.3   2344.8   2804.3   3836.6
//    200  3169.8   3381.6   3107.1   3508.9   4475.4
//    384  5287.0   5251.2   4810.4   5162.2   6139.4
//    768  9659.1   9038.5   8307.4   8294.6   9293.0
//   1024 12814.6  11667.7  10648.0  10354.3  11377.1
//
// The result that changed the table: **c = 6 is not optimal at any size.** It
// was previously selected for the whole 80..384 band, which is exactly the band
// schnorr_batch_verify lands in (N signatures build an MSM over n = 2N points,
// so a 100-signature batch is n = 200). c = 6 loses to c = 7 by 8% there.
//
// The reason is that use_signed turns on at c >= 7. Signed digits halve the
// bucket count, so c = 7 signed has 2^6 = 64 effective buckets -- the SAME
// count as c = 6 unsigned -- but needs only 256/7 + 1 = 37 windows instead of
// ceil(256/6) = 43. Same work per window, 14% fewer windows. The old bands
// predate the signed-digit path and never charged c = 7 for the halving it
// gets for free.
//
// c = 5 stays for the small band, where the 2^c aggregation term still
// dominates; it is only reachable through a direct pippenger_msm call, since
// msm() routes everything below kStraussCrossover to Strauss.
// Bands above 2048 are unmeasured and stay as they were.

unsigned pippenger_optimal_window(std::size_t n) {
    if (n <= 1)    return 1;
    if (n <= 4)    return 2;
    if (n <= 8)    return 3;
    if (n <= 16)   return 4;
    if (n <= 104)  return 5;
    if (n <= 896)  return 7;
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
namespace {

// -- Bucket engine ------------------------------------------------------------
// The window/bucket machinery, with the two things the callers differ in lifted
// out: the window width `c`, and `scalar_bits` -- an upper bound on the bit
// length of every scalar in `scalars`.
//
// `scalar_bits` exists because the GLV entry point below feeds this the same
// machinery over HALF-length scalars. Everything here is generic in the bit
// length; only the window count and the digit-extraction bound ever looked at
// 256, and both now read the parameter.
//
// Window count. For the signed path the count is floor(scalar_bits / c) + 1,
// and the +1 is not slack -- it is what makes the last window unable to carry.
// The top window starts at bit (W-1)*c = floor(scalar_bits/c)*c, so the widest
// value it can hold is 2^(scalar_bits mod c) - 1, and scalar_bits mod c <= c-1
// means that value is always below the 2^(c-1) half point. No carry can leave
// it, for ANY scalar_bits. When c divides scalar_bits exactly the +1 window is
// the one that catches the carry out of the last real window (BUG-01); when it
// does not, floor+1 == ceil and the +1 costs nothing.
Point pippenger_core(const Scalar* scalars,
                     const Point* points,
                     std::size_t n,
                     unsigned c,
                     unsigned scalar_bits) {
    std::size_t const num_buckets_unsigned = static_cast<std::size_t>(1) << c; // 2^c
    // BUG-01 fix (signed-digit carry overflow — last-window carry lost):
    // When c exactly divides 256 (e.g. c=8: 256/8=32 windows, top byte [0,255]),
    // carry propagation from window 31 has nowhere to go. For a digit d > half,
    // subtracting 2^c makes it negative but the carry of +1 to window 32 is silently
    // dropped, corrupting the MSM result by sum_{i: carry lost} * 2^256 * P_i.
    // Fix: add one extra window (256/c + 1 total) so the carry lands in window 32
    // (which starts at 0 and ends at 0 or 1, always < half → no further carry).
    // For c that don't divide 256 (c=7,9,...), 256/c + 1 == ceil(256/c) already
    // (floor(256/c) + 1 = ceil(256/c) when 256%c != 0), so no extra work is done.
    // The `>= 7` here is a PERFORMANCE bound. It was previously documented as a
    // correctness bound, on the evidence that lowering it to 6 made pippenger_msm
    // disagree with a naive sum of scalar_mul at n = 100, 256 and 384 while
    // c = 7 and c = 8 stayed exact. That evidence was real; the conclusion was
    // not. There is no c = 6 defect.
    //
    // The actual defect was in the signed SCATTER, which read X and Y and handed
    // them to from_affine52 / add_mixed52_inplace with no all_affine guard --
    // so it was wrong for every non-affine input set at every c that reached it.
    // The sweep that "found a c = 6 defect" used Jacobian points, and c = 6 was
    // simply the first window anyone tried to route through the signed path.
    // Fixed 2026-09-03; see the all_affine split in the scatter below.
    //
    // The bound stays at 7 because c = 6 is slower, not because it is wrong:
    // c = 6 is the widest window that still runs the UNSIGNED bucket path, so it
    // pays ceil(256/6) = 43 windows for the same 64 effective buckets that
    // c = 7 signed gets in 256/7 + 1 = 37. It lost at every size measured.
    bool const use_signed = (c >= 7);
    unsigned const num_windows = use_signed
        ? (scalar_bits / c) + 1        // +1 absorbs last-window carry; see note above
        : (scalar_bits + c - 1) / c;   // ceil(scalar_bits/c) for unsigned path
    // eff_buckets: 2^c unsigned, or 2^(c-1) signed — set after signed-digit init
    std::size_t num_buckets = num_buckets_unsigned;
    std::size_t const tls_alloc_size = num_buckets_unsigned + (use_signed ? std::size_t{1} : std::size_t{0});

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
        if (tl_buckets.size() < tls_alloc_size) tl_buckets.resize(tls_alloc_size);
        if (tl_touched.size() < tls_alloc_size) tl_touched.resize(tls_alloc_size);
        if (tl_used.size()    < tls_alloc_size) tl_used.resize(tls_alloc_size);
        buckets = tl_buckets.data();
        touched = tl_touched.data();
        used    = tl_used.data();
    }
    // NOTE: used[] is re-zeroed at the top of every window iteration below.
    // The single pre-loop memset is removed; per-window zeroing is the fix for
    // the stale-bucket bug where window W's dirty bits polluted window W+1.

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
            unsigned const bit_off = w * c;
            // Extra carry-overflow window (bit_off >= scalar_bits): every scalar bit
            // at or above scalar_bits is 0 — carry propagation will set these to 0
            // or 1, never above half.
            digits[static_cast<std::size_t>(w) * n + i] =
                (bit_off < scalar_bits)
                    ? static_cast<std::uint16_t>(extract_digit(scalars[i], bit_off, c))
                    : 0;
        }
    }
    // Signed-digit conversion for c >= 7: halves bucket count from 2^c to 2^(c-1).
    // Carry propagation: for each digit d > 2^(c-1), d -= 2^c and carry +1 to next window.
    // Scatter: positive digits add point, negative digits add negated point.
    // Savings: ~50% fewer buckets → ~50% less aggregate work per window.
    // Overhead: O(n × num_windows) carry compare-and-branch (~2ns each, negligible).
    // use_signed already declared above (moved with num_windows).
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
        // Halve bucket count for aggregate loop bounds. TLS pools sized to
        // num_buckets_unsigned+1 above to accommodate the carry-overflow slot.
        num_buckets >>= 1;
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

        // Zero used[] at the start of every window so stale bits from window W
        // do not cause window W+1's first-touch path to skip bucket initialization.
        // For the signed-digit path, zero tls_alloc_size bytes (= num_buckets_unsigned+1)
        // to also cover the overflow slot at index num_buckets_unsigned, which can be
        // reached when the last window receives a carry and abs_d == num_buckets_unsigned.
        std::size_t const memset_size = (num_buckets_unsigned > STACK_BUCKETS)
            ? tls_alloc_size : num_buckets;
        std::memset(used, 0, memset_size * sizeof(std::uint8_t));
        std::size_t touched_count = 0;
        std::size_t max_touched_digit = 0;

        // -- Scatter: distribute points into buckets --
        constexpr std::size_t PREFETCH_DIST = 8;
        const std::uint16_t* const wrow  = digits  + static_cast<std::size_t>(w) * n;
        const std::int16_t*  const swrow = sdigits ? sdigits + static_cast<std::size_t>(w) * n : nullptr;
        if (use_signed && all_affine) {
            // Signed scatter, affine inputs: bucket[|d|] += (d>0 ? P : -P)
            //
            // The all_affine guard is NOT optional and was missing here until
            // 2026-09-03. This loop reads X() and Y() and hands them to
            // from_affine52 / add_mixed52_inplace, both of which assume z = 1.
            // For a Jacobian input the true affine x is X/Z^2, so dropping Z
            // does not fail -- it silently substitutes a different point, and
            // the MSM returns a well-formed wrong answer.
            //
            // The unsigned scatter below has always had this split. The signed
            // one did not, so pippenger_msm was wrong for every non-affine
            // input set at c >= 7, i.e. from n = 512 up under the old window
            // table. It was never caught because both MSM tests in the suite
            // ran at n = 64 and n = 256, which the old table put on c = 5 and
            // c = 6 -- the unsigned path.
            //
            // It also explains the note that said "the signed path has a defect
            // that only shows at c = 6". It shows at every c that reaches this
            // branch; c = 6 was simply the first window someone tried to enable
            // it for, using a Jacobian input set.
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
        } else if (use_signed) {
            // Signed scatter, general inputs: same digits, full Jacobian adds.
            // Slower per point, but it is the only correct thing to do when the
            // caller's points carry a Z. Note used[] is still set to 1 on first
            // touch; the aggregation below only reads that as "affine" under an
            // all_affine guard, which is false here.
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
                    buckets[abs_d] = points[i];
                    if (is_neg) buckets[abs_d].negate_inplace();
                    continue;
                }
                if (is_neg) {
                    Point neg = points[i];
                    neg.negate_inplace();
                    buckets[abs_d].add_inplace(neg);
                } else {
                    buckets[abs_d].add_inplace(points[i]);
                }
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

} // namespace

// -- Public entry point -------------------------------------------------------
Point pippenger_msm(const Scalar* scalars,
                    const Point* points,
                    std::size_t n) {
    // Trivial cases
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);

    // For small n, fall back to Strauss (lower constant factor). This threshold
    // is deliberately NOT msm()'s kStraussCrossover: the audit suite calls
    // pippenger_msm directly at n = 64, 100 and 128 to exercise the bucket path,
    // and raising it here would route those tests to Strauss and stop testing
    // Pippenger at all. See the note on kStraussCrossover.
    if (n < 48) {
        return multi_scalar_mul(scalars, points, n);
    }

    return pippenger_core(scalars, points, n, pippenger_optimal_window(n), 256);
}

// -- GLV Pippenger ------------------------------------------------------------
// pippenger_core is generic in the scalar length, and GLV is the cheapest way
// to halve it: k*P = k1*Q1 + k2*Q2 with |k1|, |k2| < 2^128, where Q1 = ±P and
// Q2 = ±phi(P). The MSM doubles in width (2n points) but the scalars halve, so
// the FILL term is unchanged -- 2n points over half the windows is the same
// number of bucket writes -- while the AGGREGATION term, which is
// windows * 2^c and does not depend on n at all, is cut in half with it.
//
// That is the whole gain, and it is why GLV pays off most where aggregation is
// the largest share of the work, i.e. at moderate n. It is not free: n scalar
// decompositions, n endomorphisms and 2n point copies are paid up front.
namespace {

// phi(x, y, z) = (beta*x, y, z). z is untouched, so an affine input stays
// affine and keeps pippenger_core's all_affine mixed-add fast path -- which
// matters, because every caller that reaches here feeds affine points.
//
// fast::apply_endomorphism does the same thing but goes through x_raw(), which
// on the 5x52 build converts out of and back into the internal representation
// for a single multiply. This stays in FE52 throughout.
Point endomorphism_inplace_repr(const Point& P) {
#if defined(SECP256K1_FAST_52BIT)
    if (P.is_infinity()) return P;
    static const fast::FieldElement52 beta52 =
        fast::FieldElement52::from_fe(fast::FieldElement::from_bytes(fast::glv_constants::BETA));
    fast::FieldElement52 bx = P.X52();
    bx.mul_assign(beta52);
    return P.is_normalized() ? Point::from_affine52(bx, P.Y52())
                             : Point::from_jacobian52(bx, P.Y52(), P.Z52(), false);
#else
    return fast::apply_endomorphism(P);
#endif
}

// Window width for the GLV shape: m = 2n points, 129-bit scalars. Separate
// from pippenger_optimal_window because both inputs to the cost model changed
// -- twice the points, half the windows -- so the old bands do not transfer.
// Re-measured the same way as those: same file, window forced, affine inputs.
// Microseconds per MSM, forced window, affine inputs, same conditions as the
// non-GLV table above (lower is better):
//
//     m     c=5      c=6      c=7      c=8      c=9
//    128  1093.8   1212.0   1062.4   1292.9   1764.4
//    512  3245.3   3099.8   2816.0   2915.0   3345.9
//   1024  6197.2   5531.6   5071.4   4833.8   5281.8
//   2048 11902.5  10519.4   9549.9   8929.4   8925.8
//   4096 23341.8  20267.6  18743.4  16614.0  16195.2
//
// The bands sit higher than the non-GLV ones at equal m because the scalars are
// half as long: the aggregation term is windows * 2^c and windows have halved,
// so a wider window costs half what it used to and pays for itself sooner.
// c = 6 is dominated here too, for the same reason as above -- it is the widest
// window that still runs the unsigned bucket path.
// m > 4096 is unmeasured; c = 9 is the largest window with data behind it.
unsigned pippenger_glv_window(std::size_t m) {
    if (m <= 16)    return 4;
    if (m <= 96)    return 5;
    if (m <= 640)   return 7;
    if (m <= 2048)  return 8;
    return 9;
}

} // namespace

Point pippenger_msm_glv(const Scalar* scalars,
                        const Point* points,
                        std::size_t n) {
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);

    std::size_t const m = 2 * n;

    // thread_local, like every other scratch buffer in this file: an MSM in a
    // verification loop must not allocate.
    static thread_local std::vector<Scalar> tl_half;
    static thread_local std::vector<Point>  tl_pts;
    if (tl_half.size() < m) tl_half.resize(m);
    if (tl_pts.size()  < m) tl_pts.resize(m);
    Scalar* const half = tl_half.data();
    Point*  const pts  = tl_pts.data();

    // k1 stream at [0, n), k2 stream at [n, 2n) -- same split multi_scalar_mul
    // uses, so the two are directly comparable when one is checked against the
    // other.
    for (std::size_t i = 0; i < n; ++i) {
        fast::GLVDecomposition const d = fast::glv_decompose(scalars[i]);
        half[i]     = d.k1;
        half[n + i] = d.k2;
        pts[i]      = d.k1_neg ? points[i].negate() : points[i];
        Point const e = endomorphism_inplace_repr(points[i]);
        pts[n + i]  = d.k2_neg ? e.negate() : e;
    }

    // 129, not 128: the decomposition bound is |k1|, |k2| < 2^128 and one bit
    // of margin costs nothing here. floor(129/c) + 1 == floor(128/c) + 1 for
    // every c in [4, 9], so the window count is identical either way -- the
    // margin only widens the digit-extraction bound.
    return pippenger_core(half, pts, m, pippenger_glv_window(m), 129);
}

// -- Signed-digit Pippenger (halved bucket count) -----------------------------
// Uses signed digits [-2^(c-1), ..., -1, 0, 1, ..., 2^(c-1)]
// This halves the number of buckets (2^(c-1) instead of 2^c) at the cost
// of a carry propagation pass. Very effective for large n.
//
// Enabled for c >= 7 because c = 6 is the widest window that still runs the
// unsigned bucket path, and it loses: same effective bucket count as c = 7
// signed, 14% more windows. The earlier reading of the same evidence -- that
// c = 6 was a correctness bound -- was wrong; see the note at the use_signed
// definition, and the all_affine split in the signed scatter that it points to.

// -- Vector convenience -------------------------------------------------------
Point pippenger_msm(const std::vector<Scalar>& scalars,
                    const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return pippenger_msm(scalars.data(), points.data(), n);
}

// -- Unified MSM (auto-select) ------------------------------------------------
// Strauss below the crossover, GLV Pippenger at or above it.
//
// Three implementations, same inputs, same process; affine input points, as
// every caller feeds. Microseconds per MSM:
//
//      n   Strauss   Pippenger   GLV Pippenger   best previous -> GLV
//     48     814.9      1192.3           899.2        +10.3%  (Strauss wins)
//     56     950.7      1306.3           974.4         +2.5%  (Strauss wins)
//     64    1084.1      1406.2          1051.7         -3.0%
//     96    1651.2      1787.4          1344.2        -18.6%
//    128    2209.2      2081.7          1626.2        -21.9%
//    200    3476.3      2750.1          2261.0        -17.8%
//    512    9031.0      5383.7          4811.7        -10.6%
//   1024   18989.2      9511.7          8756.1         -7.9%
//   2048   40659.4     17233.5         16166.7         -6.2%
//
// The crossover sits at n ~= 60. Below it the GLV setup -- n decompositions,
// n endomorphisms, 2n point copies -- is not yet amortised over enough bucket
// work; above it the halved aggregation term wins and keeps winning.
//
// This is the routing schnorr_batch_verify uses (N signatures -> n = 2N
// points), so the whole batch-verification range from ~30 signatures upward
// runs through the last column.
//
// pippenger_msm keeps its OWN fallback at n < 48 on purpose: the audit suite
// calls it directly to exercise the bucket path at n = 64, 100 and 128, and
// raising its internal threshold would silently route those tests to Strauss
// and stop testing Pippenger at all.
//
// Embedded targets keep the old routing. GLV Pippenger holds 2n points and
// 2n half-scalars in thread_local scratch -- more than plain Pippenger, not
// less -- and Strauss holds n GLV wNAF streams plus n tables. Neither budget
// has been measured on the device, so the ESP32 path stays where it was
// instead of inheriting a desktop measurement.
#if defined(SECP256K1_ESP32)
constexpr std::size_t kStraussCrossover = 48;
#else
constexpr std::size_t kStraussCrossover = 60;
#endif

Point msm(const Scalar* scalars,
          const Point* points,
          std::size_t n) {
    if (n < kStraussCrossover) {
        return multi_scalar_mul(scalars, points, n);
    }
#if defined(SECP256K1_ESP32)
    return pippenger_msm(scalars, points, n);
#else
    return pippenger_msm_glv(scalars, points, n);
#endif
}

Point msm(const std::vector<Scalar>& scalars,
          const std::vector<Point>& points) {
    std::size_t const n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return msm(scalars.data(), points.data(), n);
}

} // namespace secp256k1
