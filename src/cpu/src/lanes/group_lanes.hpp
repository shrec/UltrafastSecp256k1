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
// Control flow is identical in every lane and depends only on the window
// widths: per-lane data changes table indices and negation masks, never
// branches.
//
//   * Recoding: fixed-length regular signed-window recoding (all digits odd,
//     |d| <= 2^w - 1, every digit non-zero, see lg_recode). It needs an odd
//     scalar; an even u is replaced by n - u (odd, n is odd) and the base point
//     negated per lane: u*P = (n - u)*(-P). Every lane uses the same number of
//     digits, ceil(257 / w), so the doubling schedule is uniform.
//   * G: one shared affine table of the odd multiples G, 3G, ..., (2^wG - 1)G,
//     built once per ISA translation unit on first use (lg_g_table), looked up
//     per lane with a gather.
//   * Q: a per-lane table of the odd multiples Q, 3Q, ..., (2^wQ - 1)Q, built in
//     Jacobian form and made affine with ONE lane-parallel field inversion
//     (Montgomery batch inversion over the table, per lane).
//   * A lookup of a negative digit negates y by mask (fe_select).
//   * Strauss interleaving: one accumulator, one doubling per bit, a mixed add
//     at every G digit position and at every Q digit position. The first
//     (top) digit initialises the accumulator instead of adding to O.
//   * GLV is not used: it would halve the doublings but needs per-lane
//     decomposition and two more tables per lane; it is left for a later card.
//
// ---- Window widths (LgWindows) ----------------------------------------------
//
// Per lane, the Q table costs 2^(wQ-1) table additions + 3 multiplications
// each for the batch inversion + 3 for the affine conversion (about 19 field
// mul/sqr per entry), and saves additions in the main loop (11 mul/sqr each,
// ceil(257/wQ) of them):
//
//     wQ = 4:   8 entries * 19 + 65 * 11 =  867
//     wQ = 5:  16 entries * 19 + 52 * 11 =  876
//     wQ = 6:  32 entries * 19 + 43 * 11 = 1081
//
// so 4 and 5 are equal on operation count and 6 is worse. The lane width
// decides between them: the per-lane table holds 2^(wQ-1) * 2 coordinates *
// kLimbCount * kLanes * 8 bytes, which at wQ = 5 is 10 KiB for Ifma512x8 (5
// limbs x 8 lanes) and Avx2x4 (10 limbs x 4 lanes) and 5 KiB for the 4-lane
// radix-2^52 types -- small enough next to the G table to stay in a 32 KiB L1D,
// and the shorter digit string (52 vs 65 additions) wins. A wider type (more
// than 40 limb-lanes) would take wQ = 4 to keep that footprint.
//
// The shared G table is built once, so only its footprint matters: 2^(wG-1) *
// 2 * kLimbCount * 8 bytes. wG = 8 for radix 2^52 (128 entries, 10 KiB, 33
// additions) and wG = 7 for radix 2^26 (64 entries, 10 KiB, 37 additions).
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

// ---- Scalar recoding --------------------------------------------------------------

namespace lg {
// Group order n, 4x64 little endian.
constexpr std::uint64_t kOrderN[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL,
};
// Generator, affine, 4x64 little endian.
constexpr std::uint64_t kGx[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL,
};
constexpr std::uint64_t kGy[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL,
};
} // namespace lg

static inline bool lg_lt256(const std::uint64_t* a, const std::uint64_t* b) noexcept {
    for (int i = 3; i >= 0; --i)
        if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

// Regular signed odd-digit recoding (Joye-Tunstall) of a scalar u in [1, n-1],
// 4x64 little endian, with window Wd and M digits (Wd * M >= 257):
//
//   k = u if u is odd, else n - u (and negate = true: u*P = k*(-P));
//   for j < M-1: d_j = (k mod 2^(Wd+1)) - 2^Wd   (odd, |d_j| <= 2^Wd - 1)
//                k   = (k - d_j) / 2^Wd          (exact, and odd again)
//   d_{M-1} = k                                   (odd, 1 <= k <= 2^Wd - 1)
//
// so k = sum d_j * 2^(Wd*j). k stays >= 1 and shrinks by 2^Wd per step up to
// +1, so after M-1 steps it is below 2^(256 - Wd*(M-1)) + 2 <= 2^Wd.
// Returns false if u is outside [1, n-1] or the final digit is not in range
// (cannot happen for such u); the caller then treats the lane as exceptional.
template <unsigned Wd, unsigned M>
static inline bool lg_recode(const std::uint64_t* u, std::int32_t* d, bool& negate) noexcept {
    static_assert(Wd >= 2 && Wd <= 16 && Wd * M >= 257, "window/length mismatch");
    std::uint64_t k[4] = {u[0], u[1], u[2], u[3]};
    if ((k[0] | k[1] | k[2] | k[3]) == 0 || !lg_lt256(k, lg::kOrderN)) return false;
    negate = (k[0] & 1u) == 0;
    if (negate) {
        std::uint64_t borrow = 0;
        for (int i = 0; i < 4; ++i) {
            const std::uint64_t t1 = lg::kOrderN[i] - k[i];
            const std::uint64_t b1 = (lg::kOrderN[i] < k[i]) ? 1u : 0u;
            const std::uint64_t t2 = t1 - borrow;
            const std::uint64_t b2 = (t1 < borrow) ? 1u : 0u;
            k[i] = t2;
            borrow = b1 | b2;
        }
    }
    constexpr std::uint64_t kMask = (std::uint64_t{1} << (Wd + 1)) - 1u;
    constexpr std::int64_t kHalf = std::int64_t{1} << Wd;
    for (unsigned j = 0; j + 1 < M; ++j) {
        const std::int64_t dj = static_cast<std::int64_t>(k[0] & kMask) - kHalf;
        d[j] = static_cast<std::int32_t>(dj);
        if (dj < 0) {
            // k += -dj; k < n <= 2^256 - 2^128, so this cannot overflow 256 bits.
            std::uint64_t add = static_cast<std::uint64_t>(-dj);
            for (int i = 0; i < 4 && add != 0; ++i) {
                const std::uint64_t t = k[i] + add;
                add = (t < k[i]) ? 1u : 0u;
                k[i] = t;
            }
        } else {
            // dj <= k mod 2^(Wd+1) <= k[0]: no borrow.
            k[0] -= static_cast<std::uint64_t>(dj);
        }
        k[0] = (k[0] >> Wd) | (k[1] << (64u - Wd));
        k[1] = (k[1] >> Wd) | (k[2] << (64u - Wd));
        k[2] = (k[2] >> Wd) | (k[3] << (64u - Wd));
        k[3] = k[3] >> Wd;
    }
    if ((k[1] | k[2] | k[3]) != 0) return false;
    if ((k[0] & 1u) == 0 || k[0] >= (std::uint64_t{1} << Wd)) return false;
    d[M - 1] = static_cast<std::int32_t>(k[0]);
    return true;
}

// ---- Windows and tables ------------------------------------------------------------

// See "Window widths" in the header comment.
template <class F>
struct LgWindows {
    static constexpr unsigned kWQ = (F::kLimbCount * F::kLanes <= 40u) ? 5u : 4u;
    static constexpr unsigned kWG = (F::kLimbCount <= 5u) ? 8u : 7u;
    static constexpr unsigned kEQ = 1u << (kWQ - 1u);   // per-lane Q table entries
    static constexpr unsigned kEG = 1u << (kWG - 1u);   // shared G table entries
    static constexpr unsigned kMQ = (257u + kWQ - 1u) / kWQ;
    static constexpr unsigned kMG = (257u + kWG - 1u) / kWG;
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

// Per-lane table of the odd multiples (2k+1)*p, k < E, affine. Entry k is the
// object representation of one F at out_x + k*K*W (and out_y). Returns the
// exceptional lanes (an addition hit Z3 == 0, or the product of all Z is 0 --
// impossible for a finite on-curve p, since (2k+1)p != +-2p for k < E << n).
template <class F, unsigned E>
static inline unsigned lg_odd_multiples(const LaneAff<F>& p, std::uint64_t* out_x,
                                        std::uint64_t* out_y) noexcept {
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    static_assert(E >= 2, "table needs at least p and 3p");
    LaneJac<F> t[E];
    F prefix[E];
    unsigned bad = 0;
    t[0].x = p.x;
    t[0].y = p.y;
    t[0].z = lg_one<F>();
    LaneJac<F> d2;
    lg_double(d2, t[0]);
    bad |= lg_add_mixed(t[1], d2, p);
    for (unsigned k = 2; k < E; ++k) bad |= lg_add(t[k], t[k - 1], d2);

    // Montgomery batch inversion of t[0..E-1].z, per lane: one field inversion.
    prefix[0] = t[0].z;
    for (unsigned k = 1; k < E; ++k) prefix[k] = fe_mul(prefix[k - 1], t[k].z);
    bad |= lg_zero_bits(prefix[E - 1]);
    F inv = fe_inverse(prefix[E - 1]);
    for (unsigned k = E - 1; k > 0; --k) {
        const F zi = fe_mul(inv, prefix[k - 1]);
        inv = fe_mul(inv, t[k].z);
        LaneAff<F> a;
        lg_to_affine(a, t[k], zi);
        std::memcpy(out_x + static_cast<std::size_t>(k) * K * W, &a.x, sizeof(F));
        std::memcpy(out_y + static_cast<std::size_t>(k) * K * W, &a.y, sizeof(F));
    }
    LaneAff<F> a0;
    lg_to_affine(a0, t[0], inv);
    std::memcpy(out_x, &a0.x, sizeof(F));
    std::memcpy(out_y, &a0.y, sizeof(F));
    return bad;
}

// The shared table of odd multiples of G: entry k = (2k+1)G, K internal limbs
// per coordinate (weak normal).
template <class F>
struct LgGTable {
    static constexpr unsigned kEntries = LgWindows<F>::kEG;
    alignas(64) std::uint64_t x[kEntries * F::kLimbCount];
    alignas(64) std::uint64_t y[kEntries * F::kLimbCount];
};

// Built once. Every lane computes the same chain (G broadcast), so the Jacobian
// multiples are read from lane 0; the affine conversion then runs lane
// parallel, entry s*W + l in lane l, with one field inversion for the table.
// Exceptional flags are not needed: the odd multiples (2k+1)G, k < 128, never
// equal +-2G or each other (G has order n).
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

    alignas(64) std::uint64_t jx[E * K];
    alignas(64) std::uint64_t jy[E * K];
    alignas(64) std::uint64_t jz[E * K];
    LaneJac<F> cur;
    cur.x = g.x;
    cur.y = g.y;
    cur.z = lg_one<F>();
    LaneJac<F> d2;
    lg_double(d2, cur);
    for (unsigned k = 0; k < E; ++k) {
        if (k == 1) (void)lg_add_mixed(cur, d2, g);
        else if (k > 1) (void)lg_add(cur, cur, d2);
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
        a.x = fe_normalize(a.x);
        a.y = fe_normalize(a.y);
        for (unsigned l = 0; l < W; ++l) {
            lg_lane_limbs(a.x, l, out.x + (s * W + l) * K);
            lg_lane_limbs(a.y, l, out.y + (s * W + l) * K);
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

// Table index and negation mask of digit column j: lane l uses entry
// (|d| - 1) / 2, negated when the digit sign and the lane's base negation differ.
template <class F, unsigned M>
static inline unsigned lg_digit_column(const std::int32_t (*digits)[M], unsigned j, unsigned base_neg,
                                       unsigned* idx) noexcept {
    unsigned neg = 0;
    for (unsigned l = 0; l < F::kLanes; ++l) {
        const std::int32_t dv = digits[l][j];
        const std::uint32_t mag = static_cast<std::uint32_t>(dv < 0 ? -dv : dv);
        idx[l] = (mag - 1u) >> 1;
        neg |= (static_cast<unsigned>(dv < 0) ^ ((base_neg >> l) & 1u)) << l;
    }
    return neg;
}

// Accumulate an affine point: the first one initialises A (Z = 1).
template <class F>
static inline unsigned lg_accumulate(LaneJac<F>& acc, bool& started, const LaneAff<F>& t,
                                     const F& one) noexcept {
    if (!started) {
        acc.x = t.x;
        acc.y = t.y;
        acc.z = one;
        started = true;
        return 0;
    }
    return lg_add_mixed(acc, acc, t);
}

// R_l = u1_l * G + u2_l * Q_l for every lane l.
//   u1, u2: 4x64 little-endian scalars per lane (u1 + 4*l), in [1, n-1];
//   qx, qy: 4x64 affine Q per lane, on the curve (any value < 2^256 is a valid
//           field input; the caller checked the curve equation).
// Returns the exceptional lanes: a scalar outside [1, n-1], any addition that
// hit Z3 == 0 (equal / opposite operands, or R == O). R is a correct finite
// point in every other lane.
template <class F>
static inline unsigned lg_dual_mul(LaneJac<F>& R, const std::uint64_t* u1, const std::uint64_t* u2,
                                   const std::uint64_t* qx, const std::uint64_t* qy) noexcept {
    using Win = LgWindows<F>;
    constexpr unsigned K = LgLayout<F>::K, W = LgLayout<F>::W;
    const LgGTable<F>& gtab = lg_g_table<F>();

    unsigned bad = 0, neg_g = 0, neg_q = 0;
    std::int32_t dg[W][Win::kMG];
    std::int32_t dq[W][Win::kMQ];
    for (unsigned l = 0; l < W; ++l) {
        bool ng = false, nq = false;
        if (!lg_recode<Win::kWG, Win::kMG>(u1 + 4u * l, dg[l], ng)) {
            bad |= 1u << l;
            ng = false;
            for (unsigned j = 0; j < Win::kMG; ++j) dg[l][j] = 1;
        }
        if (!lg_recode<Win::kWQ, Win::kMQ>(u2 + 4u * l, dq[l], nq)) {
            bad |= 1u << l;
            nq = false;
            for (unsigned j = 0; j < Win::kMQ; ++j) dq[l][j] = 1;
        }
        neg_g |= static_cast<unsigned>(ng) << l;
        neg_q |= static_cast<unsigned>(nq) << l;
    }

    LaneAff<F> q;
    fe_load(q.x, qx);
    fe_load(q.y, qy);
    alignas(64) std::uint64_t qtx[Win::kEQ * K * W];
    alignas(64) std::uint64_t qty[Win::kEQ * K * W];
    bad |= lg_odd_multiples<F, Win::kEQ>(q, qtx, qty);

    const F one = lg_one<F>();
    LaneJac<F> acc{};
    bool started = false;
    unsigned idx[W];
    for (int b = Win::kStart; b >= 0; --b) {
        if (started) lg_double(acc, acc);
        if (b % static_cast<int>(Win::kWG) == 0) {
            const unsigned j = static_cast<unsigned>(b) / Win::kWG;
            if (j < Win::kMG) {
                const unsigned neg = lg_digit_column<F, Win::kMG>(dg, j, neg_g, idx);
                LaneAff<F> t;
                t.x = lg_gather_shared<F>(gtab.x, idx);
                t.y = lg_gather_shared<F>(gtab.y, idx);
                t.y = fe_select(F::mask_from_bits(neg), fe_negate(t.y), t.y);
                bad |= lg_accumulate(acc, started, t, one);
            }
        }
        if (b % static_cast<int>(Win::kWQ) == 0) {
            const unsigned j = static_cast<unsigned>(b) / Win::kWQ;
            if (j < Win::kMQ) {
                const unsigned neg = lg_digit_column<F, Win::kMQ>(dq, j, neg_q, idx);
                LaneAff<F> t;
                t.x = lg_gather_lanes<F>(qtx, idx);
                t.y = lg_gather_lanes<F>(qty, idx);
                t.y = fe_select(F::mask_from_bits(neg), fe_negate(t.y), t.y);
                bad |= lg_accumulate(acc, started, t, one);
            }
        }
    }
    // Belt and braces: a zero Z can only come from a flagged addition.
    bad |= lg_zero_bits(acc.z);
    R = acc;
    return bad & lg_all_lanes<F>();
}

} // namespace secp256k1::lanes::field

#endif // SECP256K1_LANES_GROUP_LANES_HPP
