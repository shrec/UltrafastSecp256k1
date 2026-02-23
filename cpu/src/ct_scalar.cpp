// ============================================================================
// Constant-Time Scalar Arithmetic -- Implementation
// ============================================================================
// All operations have data-independent execution traces.
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// ============================================================================

#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/ops.hpp"
#include <cstring>

namespace secp256k1::ct {

// secp256k1 curve order n in 4 x 64-bit limbs (little-endian)
static constexpr std::uint64_t N[4] = {
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

// --- Internal helpers --------------------------------------------------------

static inline std::uint64_t add256_scalar(std::uint64_t r[4],
                                           const std::uint64_t a[4],
                                           const std::uint64_t b[4]) noexcept {
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t sum_lo = a[i] + b[i];
        std::uint64_t c1 = static_cast<std::uint64_t>(sum_lo < a[i]);
        std::uint64_t sum = sum_lo + carry;
        std::uint64_t c2 = static_cast<std::uint64_t>(sum < sum_lo);
        r[i] = sum;
        carry = c1 + c2;
    }
    return carry;
}

static inline std::uint64_t sub256_scalar(std::uint64_t r[4],
                                           const std::uint64_t a[4],
                                           const std::uint64_t b[4]) noexcept {
    std::uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = a[i] - b[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(a[i] < b[i]);
        std::uint64_t result = diff - borrow;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < borrow);
        r[i] = result;
        borrow = b1 + b2;
    }
    return borrow;
}

[[maybe_unused]] static inline void ct_reduce_once_n(std::uint64_t r[4]) noexcept {
    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256_scalar(tmp, r, N);
    std::uint64_t mask = is_zero_mask(borrow);  // no borrow = r >= n
    cmov256(r, tmp, mask);
}

// --- Public API --------------------------------------------------------------

Scalar scalar_add(const Scalar& a, const Scalar& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t carry = add256_scalar(r, a.limbs().data(), b.limbs().data());

    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256_scalar(tmp, r, N);

    std::uint64_t no_borrow = is_zero_mask(borrow);
    std::uint64_t has_carry = is_nonzero_mask(carry);
    std::uint64_t mask = no_borrow | has_carry;
    cmov256(r, tmp, mask);

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

Scalar scalar_sub(const Scalar& a, const Scalar& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t borrow = sub256_scalar(r, a.limbs().data(), b.limbs().data());

    std::uint64_t tmp[4];
    add256_scalar(tmp, r, N);

    std::uint64_t mask = is_nonzero_mask(borrow);
    cmov256(r, tmp, mask);

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

Scalar scalar_neg(const Scalar& a) noexcept {
    // -a mod n = n - a (if a != 0), 0 (if a == 0)
    std::uint64_t r[4];
    sub256_scalar(r, N, a.limbs().data());

    std::uint64_t zero_mask = scalar_is_zero(a);
    std::uint64_t z[4] = {0, 0, 0, 0};
    cmov256(r, z, zero_mask);

    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

Scalar scalar_half(const Scalar& a) noexcept {
    // Compute a/2 mod n.
    // If a is odd: r = (a + n) >> 1  (n is odd, so a+n is even)
    // If a is even: r = a >> 1
    const auto& al = a.limbs();
    std::uint64_t odd = al[0] & 1;  // 1 if odd, 0 if even
    // Branchless: conditionally add n
    std::uint64_t t[4];
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t n_masked = N[i] & static_cast<std::uint64_t>(-odd);
        std::uint64_t sum_lo = al[i] + n_masked;
        std::uint64_t c1 = static_cast<std::uint64_t>(sum_lo < al[i]);
        std::uint64_t sum = sum_lo + carry;
        std::uint64_t c2 = static_cast<std::uint64_t>(sum < sum_lo);
        t[i] = sum;
        carry = c1 + c2;
    }
    // Right shift by 1 (carry bit goes into top)
    std::uint64_t r[4];
    r[0] = (t[0] >> 1) | (t[1] << 63);
    r[1] = (t[1] >> 1) | (t[2] << 63);
    r[2] = (t[2] >> 1) | (t[3] << 63);
    r[3] = (t[3] >> 1) | (carry << 63);
    return Scalar::from_limbs({r[0], r[1], r[2], r[3]});
}

// --- Conditional Operations --------------------------------------------------

void scalar_cmov(Scalar* r, const Scalar& a, std::uint64_t mask) noexcept {
    *r = scalar_select(a, *r, mask);
}

void scalar_cswap(Scalar* a, Scalar* b, std::uint64_t mask) noexcept {
    Scalar old_a = *a;
    Scalar old_b = *b;
    *a = scalar_select(old_b, old_a, mask);
    *b = scalar_select(old_a, old_b, mask);
}

Scalar scalar_select(const Scalar& a, const Scalar& b,
                     std::uint64_t mask) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    return Scalar::from_limbs({
        ct_select(al[0], bl[0], mask),
        ct_select(al[1], bl[1], mask),
        ct_select(al[2], bl[2], mask),
        ct_select(al[3], bl[3], mask)
    });
}

Scalar scalar_cneg(const Scalar& a, std::uint64_t mask) noexcept {
    Scalar neg = scalar_neg(a);
    return scalar_select(neg, a, mask);
}

// --- Comparison --------------------------------------------------------------

std::uint64_t scalar_is_zero(const Scalar& a) noexcept {
    const auto& l = a.limbs();
    std::uint64_t z = l[0] | l[1] | l[2] | l[3];
    return is_zero_mask(z);
}

std::uint64_t scalar_eq(const Scalar& a, const Scalar& b) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    std::uint64_t diff = (al[0] ^ bl[0]) | (al[1] ^ bl[1]) |
                         (al[2] ^ bl[2]) | (al[3] ^ bl[3]);
    return is_zero_mask(diff);
}

// --- Bit Access --------------------------------------------------------------

std::uint64_t scalar_bit(const Scalar& a, std::size_t index) noexcept {
    // CT w.r.t. scalar value (the secret). Position is public (loop counter).
    // index / 64 = limb, index % 64 = bit within limb
    std::size_t limb_idx = index >> 6;
    std::size_t bit_idx  = index & 63;

    // CT: always read all 4 limbs, select the right one via branchless mask
    std::uint64_t val = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t mask = eq_mask(static_cast<std::uint64_t>(i),
                                     static_cast<std::uint64_t>(limb_idx));
        val |= a.limbs()[i] & mask;
    }
    return (val >> bit_idx) & 1;
}

std::uint64_t scalar_window(const Scalar& a, std::size_t pos,
                            unsigned width) noexcept {
    // Direct limb access: pos and width are public (derived from loop counter),
    // so variable-time access to the limb index is safe. Only the scalar VALUE
    // is secret, and shift+mask doesn't leak it.
    const auto& limbs = a.limbs();
    std::size_t limb_idx = pos >> 6;
    std::size_t bit_idx  = pos & 63;
    std::uint64_t mask   = (1ULL << width) - 1;

    // Fast path: window doesn't span limb boundary
    if (bit_idx + width <= 64) {
        return (limbs[limb_idx] >> bit_idx) & mask;
    }
    // Slow path: window crosses limb boundary (only when bit_idx + width > 64)
    std::uint64_t lo = limbs[limb_idx] >> bit_idx;
    std::uint64_t hi = (limb_idx + 1 < 4) ? limbs[limb_idx + 1] : 0;
    return (lo | (hi << (64 - bit_idx))) & mask;
}

} // namespace secp256k1::ct
