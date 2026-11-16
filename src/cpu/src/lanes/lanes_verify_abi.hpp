// ============================================================================
// lanes/lanes_verify_abi.hpp -- plain-data interface of the lane verify kernels
// ============================================================================
//
// The scalar side of lane verification (parsing, policy, s^-1, lift_x and the
// BCH key decompression, the BIP-340 and BCH challenges, scalar fallback)
// lives in the generic translation unit lanes_verify.cpp, compiled with the
// library's normal flags. The lane side
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

// One group of up to `lanes` signatures. Lane l's 256-bit values are at
// [4*l, 4*l+4), 4x64 little endian; its GLV halves at [2*l, 2*l+2), 2x64.
// Every lane the kernel computes must hold valid inputs; the caller fills
// unused lanes with a copy of a used one.
//
// The coefficients arrive GLV-split (glv_decompose, scalar side):
//   u1 = k0 + k1*lambda  (G coefficient),  u2 = k2 + k3*lambda  (Q coefficient),
// each k_h a magnitude below 2^128 in k[h], negative in lane l where bit l of
// k_neg[h] is set. Any half may be zero (u1 == 0 included).
struct LaneVerifyGroup {
    alignas(64) std::uint64_t k[4][2 * kLaneVerifyMaxLanes];
    unsigned k_neg[4];
    alignas(64) std::uint64_t qx[4 * kLaneVerifyMaxLanes];  // affine Q, on the curve
    alignas(64) std::uint64_t qy[4 * kLaneVerifyMaxLanes];
    // ECDSA: r (< n) and, where rn_ok bit l is set, r + n (< p).
    // Schnorr (BIP-340 and BCH 2019): r (< p) in `r`; `rn` (a copy of r) and
    // rn_ok (0) unused.
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
    // BCH 2019 Schnorr: R = u1*G + u2*Q (u1 = s, u2 = -e); ok iff R != O,
    // y(R) is a quadratic residue and x(R) == r. Both tests run on the
    // Jacobian R without an inversion: the residue test is Euler's criterion
    // on Y*Z, the x test X == r*Z^2.
    void (*bch_group)(const LaneVerifyGroup& in, LaneVerifyResult& out);
    // The ECDSA projective x-compare alone, on arbitrary X, Z (any 256-bit
    // values), for the test: bit l = X == r*Z^2 || (rn_ok_l && X == rn*Z^2).
    unsigned (*ecdsa_xcheck)(const std::uint64_t* X, const std::uint64_t* Z, const std::uint64_t* r,
                             const std::uint64_t* rn, unsigned rn_ok);
};

} // namespace secp256k1::lanes::detail

#endif // SECP256K1_LANES_VERIFY_ABI_HPP
