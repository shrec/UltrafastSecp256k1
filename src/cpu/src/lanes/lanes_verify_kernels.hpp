// ============================================================================
// lanes/lanes_verify_kernels.hpp -- per-ISA ECDSA / BIP-340 lane verify kernels
// ============================================================================
//
// make_lane_verify_kernels<F>() builds the LaneVerifyKernels table
// (lanes_verify_abi.hpp) for one lane field type F. Include it after the F
// header, only from that ISA's translation unit. Everything has internal
// linkage. The scalar side (parse, policy, s^-1, lift_x, challenge, fallback)
// is in lanes_verify.cpp.
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

template <class F>
static void lv_ecdsa_group(const detail::LaneVerifyGroup& in, detail::LaneVerifyResult& out) {
    LaneJac<F> R;
    const unsigned flagged = lg_dual_mul<F>(R, in.u1, in.u2, in.qx, in.qy);
    const unsigned ok = lv_xcheck_bits(R.x, R.z, in.r, in.rn, in.rn_ok);
    out.flagged = flagged;
    out.ok = ok & ~flagged & lg_all_lanes<F>();
}

template <class F>
static void lv_schnorr_group(const detail::LaneVerifyGroup& in, detail::LaneVerifyResult& out) {
    constexpr unsigned W = F::kLanes;
    LaneJac<F> R;
    const unsigned flagged = lg_dual_mul<F>(R, in.u1, in.u2, in.qx, in.qy);
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

template <class F>
static constexpr detail::LaneVerifyKernels make_lane_verify_kernels(LaneBackend backend) noexcept {
    static_assert(F::kLanes <= detail::kLaneVerifyMaxLanes, "lane count exceeds the verify ABI");
    return detail::LaneVerifyKernels{
        backend,
        F::kLanes,
        &lv_ecdsa_group<F>,
        &lv_schnorr_group<F>,
        &lv_ecdsa_xcheck<F>,
    };
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_VERIFY_KERNELS_HPP
