// ============================================================================
// lanes/lanes_kernels.hpp -- array-wise test / benchmark kernels over a lane type
// ============================================================================
//
// make_lane_kernels<F>() builds a LaneFieldKernels table (secp256k1/lanes.hpp)
// for one lane field type F. Include it after the F header, only from that
// ISA's translation unit. Everything has internal linkage.
//
// The kernels take arrays of 4x64 elements of any length n: full blocks of
// F::kLanes go straight through fe_load / fe_store, a short tail is padded
// with zero elements.
//
// LIMB-BOUND ASSERT: every kernel runs fe_bound_ok() on each mul/sqr input and
// on each op result, and counts failures in a TU-local counter exposed as
// LaneFieldKernels::bound_violations. The test requires it to stay 0. It is a
// counter rather than a trap so the differential test can report which
// backend broke the invariant instead of dying.
// ============================================================================

#ifndef SECP256K1_LANES_KERNELS_HPP
#define SECP256K1_LANES_KERNELS_HPP

#include <cstddef>
#include <cstdint>

#include "secp256k1/lanes.hpp"
#include "field_common.hpp"

namespace secp256k1::lanes::field {

static std::uint64_t g_lane_bound_violations = 0;

static std::uint64_t lk_bound_violations() noexcept { return g_lane_bound_violations; }

template <class F>
static inline void lk_check(const F& x) noexcept {
    if (!fe_bound_ok(x)) ++g_lane_bound_violations;
}

// Call body(offset, count) for each block of up to F::kLanes elements.
template <class F, class Body>
static inline void lk_blocks(std::size_t n, Body&& body) {
    for (std::size_t off = 0; off < n; off += F::kLanes) {
        const std::size_t m = (n - off < F::kLanes) ? (n - off) : std::size_t{F::kLanes};
        body(off, m);
    }
}

template <class F>
static inline void lk_load(F& x, const std::uint64_t* src, std::size_t m) noexcept {
    if (m == F::kLanes) {
        fe_load(x, src);
        return;
    }
    std::uint64_t tmp[4 * F::kLanes] = {};
    for (std::size_t i = 0; i < 4 * m; ++i) tmp[i] = src[i];
    fe_load(x, tmp);
}

template <class F>
static inline void lk_load_limbs(F& x, const std::uint64_t* src, std::size_t m) noexcept {
    if (m == F::kLanes) {
        fe_load_limbs(x, src);
        return;
    }
    std::uint64_t tmp[F::kLimbCount * F::kLanes] = {};
    for (std::size_t i = 0; i < F::kLimbCount * m; ++i) tmp[i] = src[i];
    fe_load_limbs(x, tmp);
}

template <class F>
static inline void lk_store_with(std::uint64_t* dst, const F& x, std::size_t m, bool raw) noexcept {
    std::uint64_t tmp[4 * F::kLanes];
    std::uint64_t* out = (m == F::kLanes) ? dst : tmp;
    if (raw) fe_store_raw(out, x);
    else fe_store(out, x);
    if (out == tmp)
        for (std::size_t i = 0; i < 4 * m; ++i) dst[i] = tmp[i];
}

template <class F>
static inline void lk_store(std::uint64_t* dst, const F& x, std::size_t m) noexcept {
    lk_store_with(dst, x, m, false);
}

template <class F>
static inline F lk_mul(const F& x, const F& y) noexcept {
    lk_check(x);
    lk_check(y);
    const F r = fe_mul(x, y);
    lk_check(r);
    return r;
}

template <class F>
static inline F lk_sqr(const F& x) noexcept {
    lk_check(x);
    const F r = fe_sqr(x);
    lk_check(r);
    return r;
}

// ---- Kernels -----------------------------------------------------------------

template <class F>
static void lk_load_store(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load(x, a + 4 * off, m);
        lk_check(x);
        lk_store(r + 4 * off, x, m);
    });
}

// fe_sub(a, 0) computes a + 4p: a weak-normal but non-canonical form of a,
// which fe_normalize must bring back to a mod p. The
// result is written WITHOUT another normalization, so a non-canonical
// fe_normalize output shows up as a mismatch.
template <class F>
static void lk_normalize(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, zero;
        lk_load(x, a + 4 * off, m);
        const std::uint64_t zeros[4 * F::kLanes] = {};
        fe_load(zero, zeros);
        const F y = fe_sub(x, zero);
        lk_check(y);
        const F z = fe_normalize(y);
        lk_check(z);
        lk_store_with(r + 4 * off, z, m, true);
    });
}

template <class F>
static void lk_add(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        const F z = fe_add(x, y);
        lk_check(z);
        lk_store(r + 4 * off, z, m);
    });
}

template <class F>
static void lk_sub(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        const F z = fe_sub(x, y);
        lk_check(z);
        lk_store(r + 4 * off, z, m);
    });
}

template <class F>
static void lk_mul_kernel(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b,
                          std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        lk_store(r + 4 * off, lk_mul(x, y), m);
    });
}

template <class F>
static void lk_sqr_kernel(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load(x, a + 4 * off, m);
        lk_store(r + 4 * off, lk_sqr(x), m);
    });
}

template <class F>
static void lk_negate(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load(x, a + 4 * off, m);
        const F z = fe_negate(x);
        lk_check(z);
        lk_store(r + 4 * off, z, m);
    });
}

template <class F>
static void lk_mul_small(std::uint64_t* r, const std::uint64_t* a, std::uint32_t k, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load(x, a + 4 * off, m);
        const F z = fe_mul_small(x, k);
        lk_check(z);
        lk_store(r + 4 * off, z, m);
    });
}

template <class F>
static void lk_inverse(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load(x, a + 4 * off, m);
        lk_check(x);
        const F z = fe_inverse(x);
        lk_check(z);
        lk_store(r + 4 * off, z, m);
    });
}

template <class F>
static void lk_is_zero(std::uint8_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load(x, a + 4 * off, m);
        const unsigned bits = F::mask_bits(fe_is_zero(x));
        for (std::size_t l = 0; l < m; ++l) r[off + l] = static_cast<std::uint8_t>((bits >> l) & 1u);
    });
}

template <class F>
static void lk_equal(std::uint8_t* r, const std::uint64_t* a, const std::uint64_t* b, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        const unsigned bits = F::mask_bits(fe_equal(x, y));
        for (std::size_t l = 0; l < m; ++l) r[off + l] = static_cast<std::uint8_t>((bits >> l) & 1u);
    });
}

template <class F>
static void lk_select(std::uint64_t* r, const std::uint8_t* mask, const std::uint64_t* a,
                      const std::uint64_t* b, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        unsigned bits = 0;
        for (std::size_t l = 0; l < m; ++l)
            if (mask[off + l] != 0) bits |= 1u << l;
        const F z = fe_select(F::mask_from_bits(bits), x, y);
        lk_check(z);
        lk_store(r + 4 * off, z, m);
    });
}

template <class F>
static void lk_chain(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b,
                     const std::uint8_t* ops, std::size_t steps, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        for (std::size_t s = 0; s < steps; ++s) {
            switch (ops[s] & 7u) {
            case 0: x = fe_add(x, y); break;
            case 1: x = fe_sub(x, y); break;
            case 2: x = lk_mul(x, y); break;
            case 3: x = lk_sqr(x); break;
            case 4: x = fe_negate(x); break;
            case 5: x = fe_mul_small(x, 3u); break;
            case 6: y = fe_add(y, x); break;
            default: y = lk_mul(x, y); break;
            }
            lk_check(x);
            lk_check(y);
        }
        lk_store(r + 4 * off, x, m);
    });
}

template <class F>
static void lk_mul_limbs(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b,
                         std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load_limbs(x, a + F::kLimbCount * off, m);
        lk_load_limbs(y, b + F::kLimbCount * off, m);
        lk_store(r + 4 * off, lk_mul(x, y), m);
    });
}

template <class F>
static void lk_sqr_limbs(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load_limbs(x, a + F::kLimbCount * off, m);
        lk_store(r + 4 * off, lk_sqr(x), m);
    });
}

template <class F>
static void lk_normalize_limbs(std::uint64_t* r, const std::uint64_t* a, std::size_t n) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x;
        lk_load_limbs(x, a + F::kLimbCount * off, m);
        lk_check(x);
        const F z = fe_normalize(x);
        lk_check(z);
        lk_store_with(r + 4 * off, z, m, true);
    });
}

// No bound checks here: this is the timed loop.
template <class F>
static void lk_bench(int op, std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b,
                     std::size_t n, std::size_t iters) {
    lk_blocks<F>(n, [&](std::size_t off, std::size_t m) {
        F x, y;
        lk_load(x, a + 4 * off, m);
        lk_load(y, b + 4 * off, m);
        if (op == 0) {
            for (std::size_t i = 0; i < iters; ++i) x = fe_mul(x, y);
        } else if (op == 1) {
            for (std::size_t i = 0; i < iters; ++i) x = fe_sqr(x);
        } else {
            for (std::size_t i = 0; i < iters; ++i) x = fe_add(x, y);
        }
        lk_store(r + 4 * off, x, m);
    });
}

template <class F>
static constexpr LaneFieldKernels make_lane_kernels(LaneBackend backend) noexcept {
    return LaneFieldKernels{
        backend,
        F::kLanes,
        F::kLimbBits,
        F::kLimbCount,
        F::kTopLimbMax,
        &lk_load_store<F>,
        &lk_normalize<F>,
        &lk_add<F>,
        &lk_sub<F>,
        &lk_mul_kernel<F>,
        &lk_sqr_kernel<F>,
        &lk_negate<F>,
        &lk_mul_small<F>,
        &lk_inverse<F>,
        &lk_is_zero<F>,
        &lk_equal<F>,
        &lk_select<F>,
        &lk_chain<F>,
        &lk_mul_limbs<F>,
        &lk_sqr_limbs<F>,
        &lk_normalize_limbs<F>,
        &lk_bench<F>,
        &lk_bound_violations,
    };
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_KERNELS_HPP
