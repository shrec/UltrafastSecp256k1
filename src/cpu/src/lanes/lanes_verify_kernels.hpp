// ============================================================================
// lanes/lanes_verify_kernels.hpp -- per-ISA ECDSA / BIP-340 / BCH 2019 Schnorr
//                                   lane verify kernels
// ============================================================================
//
// make_lane_verify_kernels<F>() builds the LaneVerifyKernels table
// (lanes_verify_abi.hpp) for one lane field type F. Include it after the F
// header, only from that ISA's translation unit. Everything has internal
// linkage. The scalar side (parse, policy, s^-1, lift_x / key decompression,
// challenge, fallback) is in lanes_verify.cpp.
//
// PUBLIC DATA ONLY: variable time (CT-VERIFY).
// ============================================================================

#ifndef SECP256K1_LANES_VERIFY_KERNELS_HPP
#define SECP256K1_LANES_VERIFY_KERNELS_HPP

#include <cstdint>

#include "group_lanes.hpp"
#include "lanes_verify_abi.hpp"

namespace secp256k1::lanes::field {

// bit l: X == r*Z^2 (mod p), or rn_ok_l and X == rn*Z^2 (mod p). Both compares
// run in every lane; rn_ok selects per lane.
template <class F>
static inline unsigned lv_xcheck_bits(const F& X, const F& Z, const std::uint64_t* r,
                                      const std::uint64_t* rn, unsigned rn_ok) noexcept {
    F rv, rnv;
    fe_load(rv, r);
    fe_load(rnv, rn);
    const F z2 = fe_sqr(Z);
    const unsigned e1 = F::mask_bits(fe_equal(X, fe_mul(rv, z2)));
    const unsigned e2 = F::mask_bits(fe_equal(X, fe_mul(rnv, z2)));
    return (e1 | (e2 & rn_ok)) & lg_all_lanes<F>();
}

template <class F>
static unsigned lv_ecdsa_xcheck(const std::uint64_t* X, const std::uint64_t* Z, const std::uint64_t* r,
                                const std::uint64_t* rn, unsigned rn_ok) {
    F x, z;
    fe_load(x, X);
    fe_load(z, Z);
    return lv_xcheck_bits(x, z, r, rn, rn_ok);
}

// R = u1*G + u2*Q from the group's GLV halves (see LaneVerifyGroup).
template <class F>
static inline unsigned lv_dual_mul(LaneJac<F>& R, const detail::LaneVerifyGroup& in) noexcept {
    const std::uint64_t* const mag[4] = {in.k[0], in.k[1], in.k[2], in.k[3]};
    return lg_dual_mul<F>(R, mag, in.k_neg, in.qx, in.qy);
}

template <class F>
static void lv_ecdsa_group(const detail::LaneVerifyGroup& in, detail::LaneVerifyResult& out) {
    LaneJac<F> R;
    const unsigned flagged = lv_dual_mul<F>(R, in);
    const unsigned ok = lv_xcheck_bits(R.x, R.z, in.r, in.rn, in.rn_ok);
    out.flagged = flagged;
    out.ok = ok & ~flagged & lg_all_lanes<F>();
}

template <class F>
static void lv_schnorr_group(const detail::LaneVerifyGroup& in, detail::LaneVerifyResult& out) {
    constexpr unsigned W = F::kLanes;
    LaneJac<F> R;
    const unsigned flagged = lv_dual_mul<F>(R, in);
    // Affine R: one lane-parallel inversion (0 -> 0 on flagged lanes; unused).
    const F zi = fe_inverse(R.z);
    LaneAff<F> a;
    lg_to_affine(a, R, zi);
    alignas(64) std::uint64_t xs[4 * W];
    alignas(64) std::uint64_t ys[4 * W];
    fe_store(xs, a.x);   // canonical
    fe_store(ys, a.y);
    unsigned ok = 0;
    for (unsigned l = 0; l < W; ++l) {
        const std::uint64_t* x = xs + 4u * l;
        const std::uint64_t* rr = in.r + 4u * l;
        const bool x_match = x[0] == rr[0] && x[1] == rr[1] && x[2] == rr[2] && x[3] == rr[3];
        const bool y_even = (ys[4u * l] & 1u) == 0;
        ok |= static_cast<unsigned>(x_match && y_even) << l;
    }
    out.flagged = flagged;
    out.ok = ok & ~flagged & lg_all_lanes<F>();
}

// a^((p-1)/2), Euler's criterion: 1 when a is a nonzero quadratic residue,
// p - 1 when it is not, 0 for a == 0. In binary, (p-1)/2 = 2^255 - 2^31 - 489
// is 223 ones, 0, 22 ones, 0000, 1, 0, 111, so the chain reuses fe_inverse's
// blocks x_k = a^(2^k - 1) (field_common.hpp):
// ((x223 * 2^23 + x22) * 2^5 + 1) * 2^4 + x3 -- 254 squarings and 14
// multiplications, the same for every lane.
template <class F>
static inline F lv_fe_euler(const F& a) noexcept {
    const F x2 = fe_mul(fe_sqr(a), a);
    const F x3 = fe_mul(fe_sqr(x2), a);
    const F x6 = fe_mul(fe_sqr_n(x3, 3), x3);
    const F x9 = fe_mul(fe_sqr_n(x6, 3), x3);
    const F x11 = fe_mul(fe_sqr_n(x9, 2), x2);
    const F x22 = fe_mul(fe_sqr_n(x11, 11), x11);
    const F x44 = fe_mul(fe_sqr_n(x22, 22), x22);
    const F x88 = fe_mul(fe_sqr_n(x44, 44), x44);
    const F x176 = fe_mul(fe_sqr_n(x88, 88), x88);
    const F x220 = fe_mul(fe_sqr_n(x176, 44), x44);
    const F x223 = fe_mul(fe_sqr_n(x220, 3), x3);
    F t = fe_mul(fe_sqr_n(x223, 23), x22);
    t = fe_mul(fe_sqr_n(t, 5), a);
    return fe_mul(fe_sqr_n(t, 4), x3);
}

// BCH 2019 Schnorr: R = s*G - e*P; ok iff R != O, y(R) is a quadratic residue
// and x(R) == r, decided on the Jacobian R = (X, Y, Z) with no inversion and
// no affine conversion:
//   y = Y/Z^3 = (Y*Z) * (Z^-2)^2, so y is a residue iff Y*Z is (Z != 0);
//   x == r iff X == r*Z^2 (lv_xcheck_bits with the r + n branch off).
// A finite R has Y != 0 (no point of order 2), so Y*Z != 0 and the criterion
// gives exactly 1 or p - 1. The x test runs first: a group in which no lane
// passes it skips the residue chain (variable time over public data).
template <class F>
static void lv_bch_group(const detail::LaneVerifyGroup& in, detail::LaneVerifyResult& out) {
    LaneJac<F> R;
    const unsigned flagged = lv_dual_mul<F>(R, in);
    const unsigned finite = ~lg_zero_bits(R.z) & lg_all_lanes<F>();
    unsigned ok = lv_xcheck_bits(R.x, R.z, in.r, in.rn, 0u) & finite & ~flagged;
    if (ok != 0u) {
        const F chi = lv_fe_euler(fe_mul(R.y, R.z));
        ok &= F::mask_bits(fe_equal(chi, lg_one<F>()));
    }
    out.flagged = flagged;
    out.ok = ok & lg_all_lanes<F>();
}

template <class F>
static constexpr detail::LaneVerifyKernels make_lane_verify_kernels(LaneBackend backend) noexcept {
    static_assert(F::kLanes <= detail::kLaneVerifyMaxLanes, "lane count exceeds the verify ABI");
    return detail::LaneVerifyKernels{
        backend,
        F::kLanes,
        &lv_ecdsa_group<F>,
        &lv_schnorr_group<F>,
        &lv_bch_group<F>,
        &lv_ecdsa_xcheck<F>,
    };
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_VERIFY_KERNELS_HPP
