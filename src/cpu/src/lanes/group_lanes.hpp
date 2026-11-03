// ============================================================================
// lanes/group_lanes.hpp -- lane-parallel secp256k1 group ops and dual scalar mul
// ============================================================================
//
// One independent secp256k1 point per SIMD lane, on top of any lane field type
// F (Avx2x4, AvxIfma256x4, Ifma256x4, Ifma512x8 -- secp256k1/lanes.hpp). Include
// it after the F header, and only from that ISA's translation unit
// (lanes_<isa>.cpp). Everything has internal linkage, for the reason given in
// field_common.hpp: no ISA-specific copy may leak into a generic TU.
//
// PUBLIC DATA ONLY: variable time by design. These routines exist for
// signature VERIFICATION (CT-VERIFY: every input is public). Never feed a
// private key, nonce, signing share or ECDH scalar through them.
//
// ---- Group operations ---------------------------------------------------------
//
// Jacobian coordinates (X, Y, Z) represent the affine point (X/Z^2, Y/Z^3); the
// point at infinity is any Z == 0 (mod p). Every input and output is a weak
// normal field element (field_r52.hpp / field_avx2x4.hpp), so any op output is a
// valid input of any other op.
//
//   lg_double     dbl-2009-l (a = 0), 2M + 5S. No exceptional case: secp256k1
//                 has prime order, so no point of order 2 (Y != 0 for every
//                 finite point), and Z == 0 in gives Z3 = 2*Y*Z == 0 out, i.e.
//                 2*O = O is computed correctly.
//   lg_add_mixed  Jacobian + affine, 8M + 3S.
//   lg_add        Jacobian + Jacobian, 12M + 4S.
//
// The two additions are incomplete: they are wrong when the operands are equal
// (needs a doubling), opposite (result O) or when the Jacobian operand is O.
// Each of those cases makes the output Z3 = Z1 * H (resp. Z1 * Z2 * H) zero
// mod p, where H = U2 - U1 is zero exactly when the x coordinates agree. So
// both additions return the bit mask of lanes whose Z3 is zero: such a lane is
// EXCEPTIONAL -- its value must not be used, and the verifier recomputes that
// signature with the scalar code. A non-flagged lane is a correct finite sum.
//
// ---- Dual scalar multiplication R = u1*G + u2*Q (lg_dual_mul) -----------------
//
// GLV: secp256k1 has the endomorphism phi(x, y) = (beta*x, y) = lambda*(x, y)
// (beta^3 = 1 mod p, lambda^3 = 1 mod n; the constants of secp256k1/glv.hpp).
// The scalar side (lanes_verify.cpp) splits each coefficient with the library's
// glv_decompose:
//
//     u1 = a1 + b1*lambda,  u2 = a2 + b2*lambda  (mod n),  |a|, |b| < 2^128,
//
// and hands the kernel four magnitudes (2x64) and four sign masks per lane, so
//
//     R = a1*G + b1*phi(G) + a2*Q + b2*phi(Q)
//
// is four ~128-bit multiplications sharing ONE doubling chain of about 128
// doublings instead of 256 -- the same saving dual_scalar_mul_gen_point gets
// in the scalar verifier.
//
// Control flow is identical in every lane and depends only on the window
// widths: per-lane data changes table indices, negation masks and select
// masks, never branches.
//
//   * Recoding: fixed-length signed fixed-window recoding of each magnitude
//     (lg_recode): digits d in [-(2^(w-1) - 1), 2^(w-1)], ZERO ALLOWED, so an
//     even or zero half needs no correction. Every lane uses the same number
//     of digits, ceil(129 / w), for every half (a magnitude below 2^128 always
//     fits: the top digit is at most 2^(128 - w*(M-1)) <= 2^(w-1)), so the
//     doubling schedule is uniform.
//   * Signs: a lookup is negated (y by mask, fe_select) where the digit sign
//     XOR the half's sign is negative.
//   * G: one shared affine table of the multiples G, 2G, ..., 2^(wG-1) G and,
//     sharing its y column, the x column of phi of each entry (beta*x), built
//     once per ISA translation unit on first use (lg_g_table), looked up per
//     lane with a gather.
//   * Q: a per-lane table of the multiples Q, 2Q, ..., 2^(wQ-1) Q, built in
//     Jacobian form (one doubling, then mixed additions of the affine Q), made
//     affine with ONE lane-parallel field inversion (Montgomery batch
//     inversion over the table, per lane), plus the phi(Q) x column at one
//     field multiplication per entry (phi(Q) has the same y).
//   * Strauss interleaving: one accumulator, one doubling per bit, and at
//     each digit position a mixed add of the a1 (G), b1 (phi(G)), a2 (Q) and
//     b2 (phi(Q)) digit. A zero digit computes the addition anyway and keeps
//     the accumulator by select. A lane whose accumulator has not started
//     (only zero digits so far) takes the looked-up point itself by select;
//     the addition flags count only lanes that really added. A lane that
//     never starts (every half zero, R = O) is flagged.
//
// ---- Window widths (LgWindows) ----------------------------------------------
//
// Field-operation counts (multiplications + squarings), from the formulas
// above; they justify the choice, they are not a measurement.
//
// Per lane, one Q table entry costs a mixed addition (11) + 3 for the batch
// inversion + 4 for the affine conversion + 1 for beta*x: about 19. The main
// loop adds 2 halves * ceil(129/wQ) digits at 11 each:
//
//     wQ = 4:   8 entries * 19 + 2 * 33 * 11 =  878
//     wQ = 5:  16 entries * 19 + 2 * 26 * 11 =  876
//     wQ = 6:  32 entries * 19 + 2 * 22 * 11 = 1092
//
// so 4 and 5 are equal on operation count and 6 is worse; the footprint
// decides. The per-lane tables hold 2^(wQ-1) entries * 3 columns (x, y,
// beta*x) * kLimbCount * kLanes * 8 bytes. wQ = 5 for the 4-lane radix-2^52
// types (20 limb-lanes: 7.5 KiB) and wQ = 4 for Ifma512x8 (5 limbs x 8 lanes)
// and Avx2x4 (10 limbs x 4 lanes) (40 limb-lanes: 7.5 KiB at wQ = 4, twice
// that at wQ = 5), so the Q tables stay at 7.5 KiB next to the G table.
//
// The shared G table is built once, so only its footprint matters: 2^(wG-1)
// entries * 3 columns * kLimbCount * 8 bytes. wG = 8 for radix 2^52 (128
// entries, 15 KiB, 2 * 17 additions) and wG = 7 for radix 2^26 (64 entries,
// 15 KiB, 2 * 19 additions). Together with the Q tables that is 22.5 KiB, which
// stays inside a 32 KiB L1D.
// ============================================================================

#ifndef SECP256K1_LANES_GROUP_LANES_HPP
#define SECP256K1_LANES_GROUP_LANES_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "field_common.hpp"

namespace secp256k1::lanes::field {

template <class F>
struct LaneJac {
    F x, y, z;
};

template <class F>
struct LaneAff {
    F x, y;
};

// F is exactly kLimbCount vectors of kLanes uint64 and nothing else, so its
// object representation is limb[i][lane] at uint64 index i * kLanes + lane.
// The gather/scatter helpers below rely on that.
template <class F>
struct LgLayout {
    static constexpr unsigned K = F::kLimbCount;
    static constexpr unsigned W = F::kLanes;
    static_assert(sizeof(F) == sizeof(std::uint64_t) * K * W, "lane field type must be K*W uint64");
    static_assert(std::is_trivially_copyable_v<F>, "lane field type must be trivially copyable");
};

template <class F>
static inline unsigned lg_all_lanes() noexcept {
    return (1u << F::kLanes) - 1u;
}

template <class F>
static inline unsigned lg_zero_bits(const F& a) noexcept {
    return F::mask_bits(fe_is_zero(a)) & lg_all_lanes<F>();
}

// The same 4x64 value in every lane.
template <class F>
static inline F lg_broadcast(const std::uint64_t* c4) noexcept {
    alignas(64) std::uint64_t t[4 * F::kLanes];
    for (unsigned l = 0; l < F::kLanes; ++l)
        for (unsigned i = 0; i < 4; ++i) t[4 * l + i] = c4[i];
    F r;
    fe_load(r, t);
    return r;
}

template <class F>
static inline F lg_one() noexcept {
    static constexpr std::uint64_t kOne[4] = {1, 0, 0, 0};
    return lg_broadcast<F>(kOne);
}

// Internal limbs of one lane (K values).
template <class F>
static inline void lg_lane_limbs(const F& a, unsigned lane, std::uint64_t* out) noexcept {
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    alignas(64) std::uint64_t t[K * W];
    std::memcpy(t, &a, sizeof(F));
    for (unsigned i = 0; i < K; ++i) out[i] = t[i * W + lane];
}

// Lane l <- shared entry idx[l]; a shared entry is K internal limbs of one
// weak-normal element at tbl + idx * K.
template <class F>
static inline F lg_gather_shared(const std::uint64_t* tbl, const unsigned* idx) noexcept {
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    alignas(64) std::uint64_t t[K * W];
    for (unsigned l = 0; l < W; ++l) {
        const std::uint64_t* src = tbl + static_cast<std::size_t>(idx[l]) * K;
        for (unsigned i = 0; i < K; ++i) t[i * W + l] = src[i];
    }
    F r;
    std::memcpy(&r, t, sizeof(F));
    return r;
}

// Lane l <- lane l of per-lane entry idx[l]; a per-lane entry is the object
// representation of one F (K * W uint64) at tbl + idx * K * W.
template <class F>
static inline F lg_gather_lanes(const std::uint64_t* tbl, const unsigned* idx) noexcept {
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    alignas(64) std::uint64_t t[K * W];
    for (unsigned l = 0; l < W; ++l) {
        const std::uint64_t* src = tbl + static_cast<std::size_t>(idx[l]) * K * W;
        for (unsigned i = 0; i < K; ++i) t[i * W + l] = src[i * W + l];
    }
    F r;
    std::memcpy(&r, t, sizeof(F));
    return r;
}

// ---- Group operations -----------------------------------------------------------

// r = 2p (r may alias p). See the header comment: never exceptional.
template <class F>
static inline void lg_double(LaneJac<F>& r, const LaneJac<F>& p) noexcept {
    const F a = fe_sqr(p.x);                                   // X^2
    const F b = fe_sqr(p.y);                                   // Y^2
    const F c = fe_sqr(b);                                     // Y^4
    F d = fe_sub(fe_sub(fe_sqr(fe_add(p.x, b)), a), c);        // 2*X*Y^2
    d = fe_add(d, d);                                          // 4*X*Y^2
    const F e = fe_mul_small(a, 3u);                           // 3*X^2
    const F f = fe_sqr(e);
    const F x3 = fe_sub(f, fe_add(d, d));
    const F y3 = fe_sub(fe_mul(e, fe_sub(d, x3)), fe_mul_small(c, 8u));
    const F yz = fe_mul(p.y, p.z);
    r.x = x3;
    r.y = y3;
    r.z = fe_add(yz, yz);
}

// r = p + q, q affine (r may alias p). Returns the exceptional lanes: Z3 == 0,
// i.e. p == q, p == -q or p == O. Their r is not the sum.
template <class F>
static inline unsigned lg_add_mixed(LaneJac<F>& r, const LaneJac<F>& p, const LaneAff<F>& q) noexcept {
    const F z1z1 = fe_sqr(p.z);
    const F u2 = fe_mul(q.x, z1z1);
    const F s2 = fe_mul(q.y, fe_mul(p.z, z1z1));
    const F h = fe_sub(u2, p.x);
    const F rr = fe_sub(s2, p.y);
    const F hh = fe_sqr(h);
    const F hhh = fe_mul(h, hh);
    const F v = fe_mul(p.x, hh);
    const F x3 = fe_sub(fe_sub(fe_sqr(rr), hhh), fe_add(v, v));
    const F y3 = fe_sub(fe_mul(rr, fe_sub(v, x3)), fe_mul(p.y, hhh));
    const F z3 = fe_mul(p.z, h);
    r.x = x3;
    r.y = y3;
    r.z = z3;
    return lg_zero_bits(z3);
}

// r = p + q, both Jacobian (r may alias either). Returns the exceptional lanes:
// Z3 == 0, i.e. p == q, p == -q, p == O or q == O. Their r is not the sum.
template <class F>
static inline unsigned lg_add(LaneJac<F>& r, const LaneJac<F>& p, const LaneJac<F>& q) noexcept {
    const F z1z1 = fe_sqr(p.z);
    const F z2z2 = fe_sqr(q.z);
    const F u1 = fe_mul(p.x, z2z2);
    const F u2 = fe_mul(q.x, z1z1);
    const F s1 = fe_mul(p.y, fe_mul(q.z, z2z2));
    const F s2 = fe_mul(q.y, fe_mul(p.z, z1z1));
    const F h = fe_sub(u2, u1);
    const F rr = fe_sub(s2, s1);
    const F hh = fe_sqr(h);
    const F hhh = fe_mul(h, hh);
    const F v = fe_mul(u1, hh);
    const F x3 = fe_sub(fe_sub(fe_sqr(rr), hhh), fe_add(v, v));
    const F y3 = fe_sub(fe_mul(rr, fe_sub(v, x3)), fe_mul(s1, hhh));
    const F z3 = fe_mul(fe_mul(p.z, q.z), h);
    r.x = x3;
    r.y = y3;
    r.z = z3;
    return lg_zero_bits(z3);
}

// ---- Constants ----------------------------------------------------------------------

namespace lg {
// Generator, affine, 4x64 little endian.
constexpr std::uint64_t kGx[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL,
};
constexpr std::uint64_t kGy[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL,
};
// beta, 4x64 little endian: phi(x, y) = (beta*x, y) = lambda*(x, y) for the
// lambda that glv_decompose splits by (glv_constants::BETA / ::LAMBDA in
// secp256k1/glv.hpp; that header is not included here, see field_common.hpp).
constexpr std::uint64_t kBeta[4] = {
    0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL,
};
} // namespace lg

// ---- Scalar recoding --------------------------------------------------------------

// Number of digits of a window-Wd recoding that holds any magnitude below
// 2^128 (see lg_recode): Wd * M >= 129.
constexpr unsigned lg_digits_128(unsigned wd) noexcept {
    return (129u + wd - 1u) / wd;
}

// Signed fixed-window recoding of a magnitude m < 2^128 (2x64 little endian)
// with window Wd and M digits:
//
//   for j < M-1: v = m mod 2^Wd
//                d_j = v            if v <= 2^(Wd-1)
//                d_j = v - 2^Wd     otherwise           (|d_j| <= 2^(Wd-1))
//                m   = (m - d_j) / 2^Wd                  (exact)
//   d_{M-1} = m
//
// so m = sum d_j * 2^(Wd*j). Digits may be zero or even: the tables hold every
// multiple 1..2^(Wd-1), not only the odd ones. Each step leaves
// m_j <= floor(m / 2^(Wd*j)) + 1 (the digits below j sum to less than half a
// step in magnitude), so the top digit is at most 2^(128 - Wd*(M-1)) <= 2^(Wd-1)
// when Wd * M >= 129. Returns false if the top digit is still out of range
// (cannot happen for m < 2^128); the caller then treats the lane as exceptional.
template <unsigned Wd, unsigned M>
static inline bool lg_recode(const std::uint64_t* m2, std::int32_t* d) noexcept {
    static_assert(Wd >= 2 && Wd <= 16 && Wd * M >= 129, "window/length mismatch");
    std::uint64_t k0 = m2[0], k1 = m2[1], k2 = 0;
    constexpr std::uint64_t kMask = (std::uint64_t{1} << Wd) - 1u;
    constexpr std::uint64_t kHalf = std::uint64_t{1} << (Wd - 1u);
    for (unsigned j = 0; j + 1 < M; ++j) {
        const std::uint64_t v = k0 & kMask;
        const bool carry = v > kHalf;
        d[j] = carry ? static_cast<std::int32_t>(v) - static_cast<std::int32_t>(kMask + 1u)
                     : static_cast<std::int32_t>(v);
        // m = (m - d) / 2^Wd = floor(m / 2^Wd) + carry.
        k0 = (k0 >> Wd) | (k1 << (64u - Wd));
        k1 = (k1 >> Wd) | (k2 << (64u - Wd));
        k2 = k2 >> Wd;
        if (carry && ++k0 == 0 && ++k1 == 0) ++k2;
    }
    if ((k1 | k2) != 0 || k0 > kHalf) return false;
    d[M - 1] = static_cast<std::int32_t>(k0);
    return true;
}

// ---- Windows and tables ------------------------------------------------------------

// See "Window widths" in the header comment.
template <class F>
struct LgWindows {
    static constexpr unsigned kWQ = (F::kLimbCount * F::kLanes <= 20u) ? 5u : 4u;
    static constexpr unsigned kWG = (F::kLimbCount <= 5u) ? 8u : 7u;
    static constexpr unsigned kEQ = 1u << (kWQ - 1u);   // per-lane Q table entries: Q .. kEQ*Q
    static constexpr unsigned kEG = 1u << (kWG - 1u);   // shared G table entries: G .. kEG*G
    static constexpr unsigned kMQ = lg_digits_128(kWQ);
    static constexpr unsigned kMG = lg_digits_128(kWG);
    static constexpr int kTopQ = static_cast<int>(kWQ * (kMQ - 1u));
    static constexpr int kTopG = static_cast<int>(kWG * (kMG - 1u));
    static constexpr int kStart = (kTopQ > kTopG) ? kTopQ : kTopG;
};

// Affine (x, y) = (X / Z^2, Y / Z^3) given zi = 1 / Z.
template <class F>
static inline void lg_to_affine(LaneAff<F>& out, const LaneJac<F>& p, const F& zi) noexcept {
    const F zi2 = fe_sqr(zi);
    out.x = fe_mul(p.x, zi2);
    out.y = fe_mul(p.y, fe_mul(zi2, zi));
}

// Per-lane table of the multiples (k+1)*p, k < E, affine, and the x of
// phi((k+1)*p) = (beta*x, y). Entry k is the object representation of one F at
// out_x + k*K*W (and out_y, out_xb). Returns the exceptional lanes (an addition
// hit Z3 == 0, or the product of all Z is 0 -- impossible for a finite on-curve
// p: entry k >= 2 is k*p + p, and k*p != +-p for 2 <= k < E << n).
template <class F, unsigned E>
static inline unsigned lg_multiples(const LaneAff<F>& p, std::uint64_t* out_x, std::uint64_t* out_y,
                                    std::uint64_t* out_xb) noexcept {
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    static_assert(E >= 2, "table needs at least p and 2p");
    LaneJac<F> t[E];
    F prefix[E];
    unsigned bad = 0;
    t[0].x = p.x;
    t[0].y = p.y;
    t[0].z = lg_one<F>();
    lg_double(t[1], t[0]);
    for (unsigned k = 2; k < E; ++k) bad |= lg_add_mixed(t[k], t[k - 1], p);

    // Montgomery batch inversion of t[0..E-1].z, per lane: one field inversion.
    prefix[0] = t[0].z;
    for (unsigned k = 1; k < E; ++k) prefix[k] = fe_mul(prefix[k - 1], t[k].z);
    bad |= lg_zero_bits(prefix[E - 1]);
    const F beta = lg_broadcast<F>(lg::kBeta);
    F inv = fe_inverse(prefix[E - 1]);
    for (unsigned k = E; k-- > 0;) {
        LaneAff<F> a;
        if (k > 0) {
            const F zi = fe_mul(inv, prefix[k - 1]);
            inv = fe_mul(inv, t[k].z);
            lg_to_affine(a, t[k], zi);
        } else {
            lg_to_affine(a, t[0], inv);
        }
        const F xb = fe_mul(a.x, beta);
        const std::size_t off = static_cast<std::size_t>(k) * K * W;
        std::memcpy(out_x + off, &a.x, sizeof(F));
        std::memcpy(out_y + off, &a.y, sizeof(F));
        std::memcpy(out_xb + off, &xb, sizeof(F));
    }
    return bad;
}

// The shared table of the multiples of G: entry k = (k+1)G, plus the x of
// phi((k+1)G) (same y); K internal limbs per coordinate (weak normal).
template <class F>
struct LgGTable {
    static constexpr unsigned kEntries = LgWindows<F>::kEG;
    alignas(64) std::uint64_t x[kEntries * F::kLimbCount];
    alignas(64) std::uint64_t y[kEntries * F::kLimbCount];
    alignas(64) std::uint64_t xb[kEntries * F::kLimbCount];
};

// Built once. Every lane computes the same chain (G broadcast), so the Jacobian
// multiples are read from lane 0; the affine conversion then runs lane
// parallel, entry s*W + l in lane l, with one field inversion for the table.
// Exceptional flags are not needed: entry k >= 2 is kG + G, and kG != +-G for
// 2 <= k < 128 (G has order n).
template <class F>
static LgGTable<F> lg_build_g_table() noexcept {
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    constexpr unsigned E = LgGTable<F>::kEntries;
    static_assert(E % W == 0, "G table entries must be a multiple of the lane count");
    constexpr unsigned S = E / W;

    LgGTable<F> out{};
    LaneAff<F> g;
    g.x = lg_broadcast<F>(lg::kGx);
    g.y = lg_broadcast<F>(lg::kGy);
    const F beta = lg_broadcast<F>(lg::kBeta);

    alignas(64) std::uint64_t jx[E * K];
    alignas(64) std::uint64_t jy[E * K];
    alignas(64) std::uint64_t jz[E * K];
    LaneJac<F> cur;
    cur.x = g.x;
    cur.y = g.y;
    cur.z = lg_one<F>();
    for (unsigned k = 0; k < E; ++k) {
        if (k == 1) lg_double(cur, cur);
        else if (k > 1) (void)lg_add_mixed(cur, cur, g);
        lg_lane_limbs(cur.x, 0, jx + k * K);
        lg_lane_limbs(cur.y, 0, jy + k * K);
        lg_lane_limbs(cur.z, 0, jz + k * K);
    }

    F zs[S];
    F prefix[S];
    unsigned idx[W];
    for (unsigned s = 0; s < S; ++s) {
        for (unsigned l = 0; l < W; ++l) idx[l] = s * W + l;
        zs[s] = lg_gather_shared<F>(jz, idx);
        prefix[s] = (s == 0) ? zs[0] : fe_mul(prefix[s - 1], zs[s]);
    }
    F inv = fe_inverse(prefix[S - 1]);
    for (unsigned s = S; s-- > 0;) {
        const F zi = (s > 0) ? fe_mul(inv, prefix[s - 1]) : inv;
        if (s > 0) inv = fe_mul(inv, zs[s]);
        for (unsigned l = 0; l < W; ++l) idx[l] = s * W + l;
        LaneJac<F> p;
        p.x = lg_gather_shared<F>(jx, idx);
        p.y = lg_gather_shared<F>(jy, idx);
        p.z = zs[s];
        LaneAff<F> a;
        lg_to_affine(a, p, zi);
        const F xb = fe_normalize(fe_mul(a.x, beta));
        a.x = fe_normalize(a.x);
        a.y = fe_normalize(a.y);
        for (unsigned l = 0; l < W; ++l) {
            lg_lane_limbs(a.x, l, out.x + (s * W + l) * K);
            lg_lane_limbs(a.y, l, out.y + (s * W + l) * K);
            lg_lane_limbs(xb, l, out.xb + (s * W + l) * K);
        }
    }
    return out;
}

// First call builds the table (thread-safe static initialisation); it runs only
// once a dispatcher has handed out this ISA's kernels, never at load time.
template <class F>
static const LgGTable<F>& lg_g_table() noexcept {
    static const LgGTable<F> table = lg_build_g_table<F>();
    return table;
}

// ---- Dual scalar multiplication ------------------------------------------------------

// Digit column j of one half: lane l uses table entry |d| - 1 (entry 0 for a
// zero digit, whose addition is discarded). Returns the lanes with a non-zero
// digit; `neg` gets the lanes whose lookup is negated (digit sign XOR the
// half's sign).
template <class F, unsigned M>
static inline unsigned lg_digit_column(const std::int32_t (*digits)[M], unsigned j, unsigned half_neg,
                                       unsigned* idx, unsigned& neg) noexcept {
    unsigned nz = 0;
    neg = 0;
    for (unsigned l = 0; l < F::kLanes; ++l) {
        const std::int32_t dv = digits[l][j];
        const std::uint32_t mag = static_cast<std::uint32_t>(dv < 0 ? -dv : dv);
        idx[l] = (mag == 0u) ? 0u : mag - 1u;
        nz |= static_cast<unsigned>(mag != 0u) << l;
        neg |= (static_cast<unsigned>(dv < 0) ^ ((half_neg >> l) & 1u)) << l;
    }
    return nz;
}

// Accumulate the affine point t in the lanes of nz. A started lane adds t
// (mixed addition), a lane that has not started takes t itself (Z = 1), every
// other lane keeps acc -- by select, so control flow does not depend on lane
// data. Returns the exceptional lanes: started lanes of nz whose addition hit
// Z3 == 0 (a discarded addition never flags).
template <class F>
static inline unsigned lg_accumulate(LaneJac<F>& acc, unsigned& started, const LaneAff<F>& t, unsigned nz,
                                     const F& one) noexcept {
    const unsigned add = started & nz;
    const unsigned init = ~started & nz & lg_all_lanes<F>();
    unsigned exc = 0;
    if (started != 0) {   // a group-wide value: identical in every lane
        LaneJac<F> sum;
        exc = lg_add_mixed(sum, acc, t) & add;
        const auto ma = F::mask_from_bits(add);
        acc.x = fe_select(ma, sum.x, acc.x);
        acc.y = fe_select(ma, sum.y, acc.y);
        acc.z = fe_select(ma, sum.z, acc.z);
    }
    const auto mi = F::mask_from_bits(init);
    acc.x = fe_select(mi, t.x, acc.x);
    acc.y = fe_select(mi, t.y, acc.y);
    acc.z = fe_select(mi, one, acc.z);
    started |= nz;
    return exc;
}

// R_l = u1_l * G + u2_l * Q_l for every lane l, through the GLV halves
//   u1 = s0*m0 + s1*m1*lambda,  u2 = s2*m2 + s3*m3*lambda  (mod n):
//   mag[h]: magnitude m_h of lane l at mag[h] + 2*l, 2x64 little endian, < 2^128;
//   neg[h]: bit l set where s_h = -1 in lane l;
//   qx, qy: 4x64 affine Q per lane, on the curve (any value < 2^256 is a valid
//           field input; the caller checked the curve equation).
// Any half may be zero. Returns the exceptional lanes: a magnitude the
// recoding cannot hold (never for m < 2^128), any addition that hit Z3 == 0
// (equal / opposite operands, or an intermediate O), R == O. R is a correct
// finite point in every other lane.
template <class F>
static inline unsigned lg_dual_mul(LaneJac<F>& R, const std::uint64_t* const* mag, const unsigned* neg,
                                   const std::uint64_t* qx, const std::uint64_t* qy) noexcept {
    using Win = LgWindows<F>;
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    const LgGTable<F>& gtab = lg_g_table<F>();

    // Digits: halves 0, 1 (on G, phi(G)) with wG; halves 2, 3 (on Q, phi(Q)) with wQ.
    unsigned bad = 0;
    std::int32_t dg[2][W][Win::kMG];
    std::int32_t dq[2][W][Win::kMQ];
    for (unsigned l = 0; l < W; ++l) {
        for (unsigned h = 0; h < 2; ++h) {
            if (!lg_recode<Win::kWG, Win::kMG>(mag[h] + 2u * l, dg[h][l])) {
                bad |= 1u << l;
                for (unsigned j = 0; j < Win::kMG; ++j) dg[h][l][j] = 0;
            }
            if (!lg_recode<Win::kWQ, Win::kMQ>(mag[2 + h] + 2u * l, dq[h][l])) {
                bad |= 1u << l;
                for (unsigned j = 0; j < Win::kMQ; ++j) dq[h][l][j] = 0;
            }
        }
    }

    LaneAff<F> q;
    fe_load(q.x, qx);
    fe_load(q.y, qy);
    alignas(64) std::uint64_t qtx[Win::kEQ * K * W];
    alignas(64) std::uint64_t qty[Win::kEQ * K * W];
    alignas(64) std::uint64_t qtxb[Win::kEQ * K * W];
    bad |= lg_multiples<F, Win::kEQ>(q, qtx, qty, qtxb);

    const F one = lg_one<F>();
    LaneJac<F> acc;   // any valid field values until a lane starts
    acc.x = one;
    acc.y = one;
    acc.z = one;
    unsigned started = 0;
    unsigned idx[W];
    for (int b = Win::kStart; b >= 0; --b) {
        if (b != Win::kStart) lg_double(acc, acc);   // never exceptional; unstarted lanes are overwritten
        if (b % static_cast<int>(Win::kWG) == 0) {
            const unsigned j = static_cast<unsigned>(b) / Win::kWG;
            if (j < Win::kMG) {
                for (unsigned h = 0; h < 2; ++h) {
                    unsigned ng = 0;
                    const unsigned nz = lg_digit_column<F, Win::kMG>(dg[h], j, neg[h], idx, ng);
                    LaneAff<F> t;
                    t.x = lg_gather_shared<F>(h == 0 ? gtab.x : gtab.xb, idx);
                    t.y = lg_gather_shared<F>(gtab.y, idx);
                    t.y = fe_select(F::mask_from_bits(ng), fe_negate(t.y), t.y);
                    bad |= lg_accumulate(acc, started, t, nz, one);
                }
            }
        }
        if (b % static_cast<int>(Win::kWQ) == 0) {
            const unsigned j = static_cast<unsigned>(b) / Win::kWQ;
            if (j < Win::kMQ) {
                for (unsigned h = 0; h < 2; ++h) {
                    unsigned ng = 0;
                    const unsigned nz = lg_digit_column<F, Win::kMQ>(dq[h], j, neg[2 + h], idx, ng);
                    LaneAff<F> t;
                    t.x = lg_gather_lanes<F>(h == 0 ? qtx : qtxb, idx);
                    t.y = lg_gather_lanes<F>(qty, idx);
                    t.y = fe_select(F::mask_from_bits(ng), fe_negate(t.y), t.y);
                    bad |= lg_accumulate(acc, started, t, nz, one);
                }
            }
        }
    }
    // A lane that never started has every half zero: R = O, nothing to compare.
    bad |= ~started & lg_all_lanes<F>();
    // Belt and braces: a zero Z can only come from a flagged addition.
    bad |= lg_zero_bits(acc.z);
    R = acc;
    return bad & lg_all_lanes<F>();
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_GROUP_LANES_HPP
