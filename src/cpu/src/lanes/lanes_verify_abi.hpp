// ============================================================================
// lanes/lanes_verify_abi.hpp -- plain-data interface of the lane verify kernels
// ============================================================================
//
// The scalar side of lane verification (parsing, policy, s^-1, lift_x, the
// BIP-340 challenge, scalar fallback) lives in the generic translation unit
// lanes_verify.cpp, compiled with the library's normal flags. The lane side
// (group_lanes.hpp, lanes_verify_kernels.hpp) lives in the per-ISA TUs. They
// exchange only the plain arrays below, so no library header with inline
// functions is ever compiled with ISA flags (see field_common.hpp for why that
// matters).
//
// PUBLIC DATA ONLY: verification inputs.
// ============================================================================

#ifndef SECP256K1_LANES_VERIFY_ABI_HPP
#define SECP256K1_LANES_VERIFY_ABI_HPP

#include <cstdint>

#include "secp256k1/lanes.hpp"

namespace secp256k1::lanes::detail {

constexpr unsigned kLaneVerifyMaxLanes = 8;

// One group of up to `lanes` signatures. Lane l's values are at [4*l, 4*l+4),
// 4x64 little endian. Every lane the kernel computes must hold valid inputs;
// the caller fills unused lanes with a copy of a used one.
struct LaneVerifyGroup {
    alignas(64) std::uint64_t u1[4 * kLaneVerifyMaxLanes];  // G coefficient, [1, n-1]
    alignas(64) std::uint64_t u2[4 * kLaneVerifyMaxLanes];  // Q coefficient, [1, n-1]
    alignas(64) std::uint64_t qx[4 * kLaneVerifyMaxLanes];  // affine Q, on the curve
    alignas(64) std::uint64_t qy[4 * kLaneVerifyMaxLanes];
    // ECDSA: r (< n) and, where rn_ok bit l is set, r + n (< p).
    // Schnorr: r (< p) in `r`; `rn` and rn_ok unused.
    alignas(64) std::uint64_t r[4 * kLaneVerifyMaxLanes];
    alignas(64) std::uint64_t rn[4 * kLaneVerifyMaxLanes];
    unsigned rn_ok;
};

struct LaneVerifyResult {
    unsigned ok;       // bit l: lane l verified (never set for a flagged lane)
    unsigned flagged;  // bit l: exceptional, recompute with the scalar verifier
};

struct LaneVerifyKernels {
    LaneBackend backend;
    unsigned lanes;
    // R = u1*G + u2*Q; ok iff R != O and x(R) == r (mod n), compared
    // projectively: X == r*Z^2, or X == (r+n)*Z^2 where rn_ok.
    void (*ecdsa_group)(const LaneVerifyGroup& in, LaneVerifyResult& out);
    // R = u1*G + u2*Q (u1 = s, u2 = -e); ok iff R != O, y(R) even, x(R) == r.
    void (*schnorr_group)(const LaneVerifyGroup& in, LaneVerifyResult& out);
    // The ECDSA projective x-compare alone, on arbitrary X, Z (any 256-bit
    // values), for the test: bit l = X == r*Z^2 || (rn_ok_l && X == rn*Z^2).
    unsigned (*ecdsa_xcheck)(const std::uint64_t* X, const std::uint64_t* Z, const std::uint64_t* r,
                             const std::uint64_t* rn, unsigned rn_ok);
};

} // namespace secp256k1::lanes::detail

#endif // SECP256K1_LANES_VERIFY_ABI_HPP
