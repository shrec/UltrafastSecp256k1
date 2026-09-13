// ============================================================================
// field_52_magnitude.hpp -- the 5x52 magnitude model, written down
// ============================================================================
// GitHub issue #396: FieldElement52 carries no magnitude information, so a
// magnitude precondition violation produces a silently wrong field element --
// no assertion, no crash, nothing for a sanitizer to report. The bounds that
// hold the formulas together live only in comments and in the integer literals
// at the negate() call sites in point.cpp, and nothing checks that the two
// agree. #397 was an instance of them disagreeing, by 4 magnitudes, for as long
// as it took someone to work the arithmetic out by hand.
//
// This header is the first half of the answer: the model as code rather than as
// prose, so it can be asserted. It does NOT track magnitudes on the values
// themselves -- that needs shadow fields on FieldElement52, which changes its
// size and is a separate change (see the issue). What it gives is a way to ASK
// a value what its magnitude actually is, which is enough to pin the live
// formulas against their declared bounds in CI. That is what
// audit/test_regression_fe52_magnitude_model.cpp does.
//
// ---------------------------------------------------------------------------
// The definition, and how it differs from libsecp256k1
// ---------------------------------------------------------------------------
// A value has magnitude m when every limb fits m times its own mask:
//
//     n[0..3] <= m * M52          n[4] <= m * M48
//
// libsecp256k1's negate computes 2*(m+1)*p - a; ours computes (m+1)*p - a
// (field_52_impl.hpp, FieldElement52::negate). So our slack is exactly half of
// theirs and their published magnitude ceilings do NOT transfer -- do not
// "correct" the bounds here against upstream's numbers.
//
// ---------------------------------------------------------------------------
// Measured on this tree, 2026-09-06, g++-14 -O2 x86-64
// ---------------------------------------------------------------------------
// Kernel postconditions, over random inputs at magnitude 1 (20,000 trials):
//
//     fe52_mul_inner   out:  n[0..3] <= 1*M52,  n[4] <= 1*M48
//     fe52_sqr_inner   out:  n[0..3] <= 1*M52,  n[4] <= 2*M48     <-- note the 2
//     normalize_weak   out:  n[0..3] <= 1*M52,  n[4] <= 1*M48
//
// The squaring kernel's top limb reaching 2*M48 is why a magnitude model
// derived from the low limbs alone is wrong for this tree; it is asserted in
// the audit module rather than assumed here.
//
// Where the declared bounds actually start to bite (3,000 trials per magnitude,
// comparing negate(declared) * w against a normalize-then-negate(1) reference):
//
//     declared negate(8):  correct through actual magnitude 9,
//                          1199/3000 products WRONG at actual magnitude 10
//
// So GEJ_X_MAGNITUDE_MAX = 8 is honest with one magnitude of margin, and the
// EFD alternatives named in #396 -- dbl-2009-l, dbl-2007-bl, mdbl-2007-bl, all
// mathematically correct formulas with steady-state X 22 and Y 10 -- are past
// that threshold. Dropping one of them into jac52_double_coords without also
// widening the negate() arguments corrupts silently. That is now a measured
// fact rather than a static model.
// ============================================================================
#ifndef SECP256K1_FIELD_52_MAGNITUDE_HPP
#define SECP256K1_FIELD_52_MAGNITUDE_HPP

#include <cstdint>

namespace secp256k1::fast {

namespace fe52_magnitude {

constexpr std::uint64_t kM52 = 0xFFFFFFFFFFFFFULL;  // (1 << 52) - 1
constexpr std::uint64_t kM48 = 0xFFFFFFFFFFFFULL;   // (1 << 48) - 1

// Smallest m for which `limbs` satisfies the magnitude-m bound. A zero value
// has magnitude 0; every other value has magnitude >= 1. Saturates rather than
// wrapping, so a limb near 2^64 reports a large magnitude instead of a small
// one -- a wrapped limb is exactly the state this model exists to catch.
[[nodiscard]] constexpr unsigned magnitude_of(const std::uint64_t limbs[5]) noexcept {
    // ceil(limb / mask) written as div + remainder, NOT as (limb + mask - 1) /
    // mask: the value being classified may be a wrapped limb near 2^64 -- which
    // is exactly the state this model exists to catch -- and the rounding form
    // would overflow and report a small magnitude for it.
    auto const ceil_div = [](std::uint64_t limb, std::uint64_t mask) -> std::uint64_t {
        return limb / mask + (limb % mask != 0 ? 1ULL : 0ULL);
    };
    std::uint64_t m = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const mi = ceil_div(limbs[i], kM52);
        if (mi > m) m = mi;
    }
    std::uint64_t const m4 = ceil_div(limbs[4], kM48);
    if (m4 > m) m = m4;
    return m > 0xFFFFFFFFULL ? 0xFFFFFFFFu : static_cast<unsigned>(m);
}

// Does `limbs` satisfy the magnitude-m bound? m == 0 admits only zero.
[[nodiscard]] constexpr bool magnitude_ok(const std::uint64_t limbs[5], unsigned m) noexcept {
    return magnitude_of(limbs) <= m;
}

}  // namespace fe52_magnitude

}  // namespace secp256k1::fast

#endif  // SECP256K1_FIELD_52_MAGNITUDE_HPP
