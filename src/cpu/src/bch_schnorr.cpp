// ============================================================================
// Bitcoin Cash 2019 Schnorr: single and batch verification
// ============================================================================
// Spec: https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/ (NOT
// BIP-340; see secp256k1/bch_schnorr.hpp for the verification rule).
//
// Public-data path only. Signature, message and public key are all public, so
// everything below is variable-time on purpose (CLAUDE.md CT-VERIFY). Do not
// introduce ct:: primitives here.
//
// The BIP-340 batch verifier (batch_verify.cpp) is untouched. This file reuses
// its pieces: the 128-bit odd weight shape with a_0 = 1, the seed binding +
// CSPRNG XOR, and pippenger_msm_glv_short for the 3n-point MSM at large n.
// Small and mid n use this file's own interleaved Strauss (strauss_sum)
// instead; see batch_route().
// ============================================================================

#include "secp256k1/bch_schnorr.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/detail/csprng.hpp"
#include "secp256k1/detail/secure_erase.hpp"

// The Pippenger MSM lives in pippenger.cpp, compiled only under
// SECP256K1_BUILD_PIPPENGER. src/cpu/CMakeLists.txt defines
// SECP256K1_BCH_SCHNORR_HAS_MSM for this file in exactly that case; without it
// every batch is verified entry by entry, which has the same verdict. The
// Strauss route shares the one guard, so a build has either both MSM routes
// or neither. It calls glv_decompose and (without FE52) multi_scalar_mul, which
// pippenger.cpp calls too, and compute_wnaf_into, which point.cpp calls.
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
#include "secp256k1/glv.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/precompute.hpp"
#if defined(SECP256K1_FAST_52BIT)
#include "secp256k1/field_52.hpp"
#endif
#endif

// FE52 Jacobi symbol (SafeGCD posdivstep) needs __int128. Same guard as
// batch_verify.cpp; elsewhere the residue test falls back to sqrt-and-square.
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_PLATFORM_ESP32) && \
    !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
#define SECP256K1_BCH_SCHNORR_FE52_JACOBI 1
#include "secp256k1/field_52.hpp"
#endif

#include <cstring>
#include <limits>
#include <vector>

namespace secp256k1 {

using fast::FieldElement;
using fast::Point;
using fast::Scalar;

namespace {

// Routing of one chunk of n entries. The thresholds are
// detail::kBchBatchIndividualCutoff and detail::kBchBatchStraussMax in
// bch_schnorr.hpp: placeholder until measured on the host; the manager sets
// the final values from bench data.
//
// Throughput: not yet measured -- benchmark required. The verdict does not
// depend on the thresholds: every route returns the AND of the single
// verdicts (see batch_msm_impl).
//
// This is the only place that decides a route; batch_chunk_impl, batch_msm_impl
// and the audit view detail::bch_schnorr_batch_route_for_test all go through it.
detail::BchBatchRoute batch_route(std::size_t n) noexcept {
    if (n <= 1) return detail::BchBatchRoute::kSingle;
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
    if (n < detail::kBchBatchIndividualCutoff) return detail::BchBatchRoute::kIndividual;
    if (n <= detail::kBchBatchStraussMax) return detail::BchBatchRoute::kStrauss;
    return detail::BchBatchRoute::kPippenger;
#else
    return detail::BchBatchRoute::kIndividual;
#endif
}

// Weight indices are 32-bit; a batch longer than this is verified as several
// batches, each with its own fresh seed, so no two entries of one batch ever
// share a weight index.
constexpr std::size_t kBchBatchMaxChunk =
    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

#if !defined(SECP256K1_FAST_52BIT)
// Quadratic-residue test on a public field element. Zero is not a residue
// (Jacobi(0) == 0, never 1). Only the non-52-bit verifier tail calls it.
bool fe_is_qr_var(const FieldElement& v) noexcept {
#if defined(SECP256K1_BCH_SCHNORR_FE52_JACOBI)
    return fast::FieldElement52::from_fe(v).jacobi_var() == 1;
#else
    if (v == FieldElement::zero()) return false;
    FieldElement const t = v.sqrt();
    return t.square() == v;
#endif
}
#endif

// e = SHA256(r[32] || compressed33(P) || msg32) mod n.
Scalar bch_challenge(const std::uint8_t* r32, const std::uint8_t* pub33,
                     const std::uint8_t* msg32) noexcept {
    SHA256 h;
    h.update(r32, 32);
    h.update(pub33, 33);
    h.update(msg32, 32);
    auto const digest = h.finalize();
    return Scalar::from_bytes(digest);
}

// SEC1 compressed parse: prefix 0x02/0x03, x < p, x^3 + 7 a square.
bool parse_pubkey33(const std::uint8_t* pub33, Point& out) noexcept {
    if (pub33[0] != 0x02 && pub33[0] != 0x03) return false;
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(pub33 + 1, x)) return false;
    FieldElement const y2 = x.square() * x + FieldElement::from_uint64(7);
    FieldElement y = y2.sqrt();
    if (!(y.square() == y2)) return false;
    if (((y.limbs()[0] & 1u) != 0u) != (pub33[0] == 0x03)) {
        y = FieldElement::zero() - y;
    }
    out = Point::from_affine(x, y);
    return !out.is_infinity();
}

// The lift of x whose y is a quadratic residue. p == 3 (mod 4), so sqrt() is
// y2^((p+1)/4); (p+1)/4 is even (p + 1 = 2^256 - 2^32 - 976 and 16 | 976), so
// the root is (y2^((p+1)/8))^2 -- itself a square, i.e. the residue root. It
// is a root at all only if it squares back to x^3 + 7; otherwise x is not the
// x-coordinate of any curve point and no valid signature can carry it.
bool lift_x_residue(const FieldElement& x, Point& out) noexcept {
    FieldElement const y2 = x.square() * x + FieldElement::from_uint64(7);
    FieldElement const y = y2.sqrt();
    if (!(y.square() == y2)) return false;
    out = Point::from_affine(x, y);
    return true;
}

// The single verifier. One dual multiplication, no inversion, no sqrt.
bool bch_verify_impl(const std::uint8_t* sig64, const std::uint8_t* msg32,
                     const Point& P, const std::uint8_t* pub33) noexcept {
    FieldElement r_fe;
    if (!FieldElement::parse_bytes_strict(sig64, r_fe)) return false;   // r >= p
    Scalar s;
    // s is public: VT is_zero() is correct. s == 0 is rejected on top of the
    // spec's s >= n; see SHIM_KNOWN_DIVERGENCES.md (unobservable).
    if (!Scalar::parse_bytes_strict(sig64 + 32, s) || s.is_zero()) return false;
    if (P.is_infinity()) return false;

    Scalar const e = bch_challenge(sig64, pub33, msg32);
    Point const R = Point::dual_scalar_mul_gen_point(s, e.negate_var(), P);
    if (R.is_infinity()) return false;

    // R is Jacobian (X, Y, Z): x = X/Z^2, y = Y/Z^3.
    //   x == r       <=>  r*Z^2 == X
    //   y is a QR    <=>  Y*Z is a QR, because Y/Z^3 = (Y*Z) * (Z^-2)^2
    // The x test is first because it is two multiplications; the Jacobi
    // symbol costs more. The verdict is the AND either way.
#if defined(SECP256K1_FAST_52BIT)
    using FE52 = fast::FieldElement52;
    FE52 const& Z = R.Z52();
    FE52 diff = FE52::from_fe(r_fe) * Z.square();   // magnitude 1
    diff.negate_assign(1);                           // magnitude 2
    diff.add_assign(R.X52());                        // X - r*Z^2, magnitude <= 2 + GEJ_X_MAGNITUDE_MAX
    if (!diff.normalizes_to_zero_var()) return false;
    FE52 yz = R.Y52() * Z;
    yz.normalize();
    return yz.jacobi_var() == 1;
#else
    FieldElement const& Z = R.z_raw();
    if (!(r_fe * Z.square() == R.x_raw())) return false;
    return fe_is_qr_var(R.y_raw() * Z);
#endif
}

// -- Entry accessors ----------------------------------------------------------
struct WireAccess {
    static const std::uint8_t* sig(const BchSchnorrBatchEntry& e) noexcept { return e.signature.data(); }
    static const std::uint8_t* msg(const BchSchnorrBatchEntry& e) noexcept { return e.message.data(); }
    static const std::uint8_t* pub33(const BchSchnorrBatchEntry& e) noexcept { return e.pubkey.data(); }
    static bool resolve(const BchSchnorrBatchEntry& e, Point& out) noexcept {
        return parse_pubkey33(e.pubkey.data(), out);
    }
};

struct PointAccess {
    static const std::uint8_t* sig(const BchSchnorrBatchPointEntry& e) noexcept { return e.signature.data(); }
    static const std::uint8_t* msg(const BchSchnorrBatchPointEntry& e) noexcept { return e.message.data(); }
    static const std::uint8_t* pub33(const BchSchnorrBatchPointEntry& e) noexcept { return e.pubkey33.data(); }
    static bool resolve(const BchSchnorrBatchPointEntry& e, Point& out) noexcept {
        out = e.point;
        return !out.is_infinity();
    }
};

template <typename Access, typename Entry>
bool entry_valid_impl(const Entry& e) noexcept {
    Point P;
    if (!Access::resolve(e, P)) return false;
    return bch_verify_impl(Access::sig(e), Access::msg(e), P, Access::pub33(e));
}

// -- Weights ------------------------------------------------------------------
// Binding digest: SHA256 over a domain tag, the caller seed (flagged, so "no
// seed" and "a seed" never hash alike), then r || s || pubkey33 || msg of
// every entry in order. Every input that determines a verdict is bound.
template <typename Access, typename Entry>
std::array<std::uint8_t, 32> batch_digest(const Entry* entries, std::size_t n,
                                          const std::uint8_t* seed32) {
    static constexpr char kTag[] = "BCH-Schnorr-2019/batch-weights";
    SHA256 h;
    h.update(kTag, sizeof(kTag) - 1);
    std::uint8_t const has_seed = (seed32 != nullptr) ? 1u : 0u;
    h.update(&has_seed, 1);
    if (seed32 != nullptr) h.update(seed32, 32);
    for (std::size_t i = 0; i < n; ++i) {
        h.update(Access::sig(entries[i]), 64);
        h.update(Access::pub33(entries[i]), 33);
        h.update(Access::msg(entries[i]), 32);
    }
    return h.finalize();
}

// Per-call seed = digest XOR 32 CSPRNG bytes (P2-SEC-002 in batch_verify.cpp).
// The digest binds the weights to the batch; the CSPRNG bytes make them
// unpredictable to whoever chose the batch. NEVER drop either.
template <typename Access, typename Entry>
std::array<std::uint8_t, 32> batch_seed(const Entry* entries, std::size_t n,
                                        const std::uint8_t* seed32) {
    std::array<std::uint8_t, 32> seed = batch_digest<Access>(entries, n, seed32);
    std::uint8_t rnd[32];
    detail::csprng_fill(rnd, sizeof(rnd));
    for (std::size_t j = 0; j < 32; ++j) seed[j] ^= rnd[j];
    detail::secure_erase(rnd, sizeof(rnd));
    return seed;
}

// a_0 = 1; a_i = low 128 bits of SHA256(seed || i_le32), bit 0 forced to 1.
// 36 bytes hash in one SHA-256 compression. (Same shape as batch_weight() in
// batch_verify.cpp; do not capture a SHA256::Midstate after the 32-byte seed,
// it would silently drop the seed -- see the note there.)
Scalar batch_weight(std::uint8_t (&seed_and_index)[36], std::uint32_t index) {
    if (index == 0) return Scalar::one();
    seed_and_index[32] = static_cast<std::uint8_t>(index & 0xFF);
    seed_and_index[33] = static_cast<std::uint8_t>((index >> 8) & 0xFF);
    seed_and_index[34] = static_cast<std::uint8_t>((index >> 16) & 0xFF);
    seed_and_index[35] = static_cast<std::uint8_t>((index >> 24) & 0xFF);
    auto const h = SHA256::hash(seed_and_index, sizeof(seed_and_index));
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    for (unsigned b = 0; b < 8; ++b) {
        lo |= static_cast<std::uint64_t>(h[b]) << (8 * b);
        hi |= static_cast<std::uint64_t>(h[8 + b]) << (8 * b);
    }
    return Scalar::from_limbs({lo | 1u, hi, 0, 0});
}

// -- The MSM sum ----------------------------------------------------------------
// Every engine below computes, exactly,
//     g_coeff*G + sum_i full_k[i]*full_P[i] + sum_i short_k[i]*short_P[i]
// where the "short" scalars are expected below 2^128 (the batch weights a_i).
// A zero scalar or a point at infinity contributes O.
//
// naive_sum: separate multiplications. The Strauss engine's fallback for a
// degenerate table (unreachable for finite input, see build_odd_multiples) and
// the test views' answer in a build without the MSM.
// [[maybe_unused]]: an MSM build without FE52 has no production caller.
[[maybe_unused]] Point naive_sum(const Scalar& g_coeff,
                const Scalar* full_k, const Point* full_P, std::size_t n_full,
                const Scalar* short_k, const Point* short_P, std::size_t n_short) {
    Point acc = g_coeff.is_zero() ? Point::infinity() : Point::generator().scalar_mul(g_coeff);
    for (std::size_t i = 0; i < n_full; ++i) {
        if (full_k[i].is_zero() || full_P[i].is_infinity()) continue;
        acc = acc.add(full_P[i].scalar_mul(full_k[i]));
    }
    for (std::size_t i = 0; i < n_short; ++i) {
        if (short_k[i].is_zero() || short_P[i].is_infinity()) continue;
        acc = acc.add(short_P[i].scalar_mul(short_k[i]));
    }
    return acc;
}

#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
// -- Pippenger-short engine -----------------------------------------------------
// (full_k, full_P) GLV-split, (short_k, short_P) unsplit, one bucket MSM; the G
// term on the fixed-base generator path.
Point pippenger_sum(const Scalar& g_coeff,
                    const Scalar* full_k, const Point* full_P, std::size_t n_full,
                    const Scalar* short_k, const Point* short_P, std::size_t n_short) {
    Point const rest = pippenger_msm_glv_short(full_k, full_P, n_full, short_k, short_P, n_short);
    if (g_coeff.is_zero()) return rest;
    // All public data; VT generator multiplication is correct here.
    return Point::generator().scalar_mul(g_coeff).add(rest);
}

// -- Strauss engine -------------------------------------------------------------
// One interleaved wNAF scan with ONE shared doubling chain of about 129
// positions over every variable point:
//   - a scalar of 2^128 or more (a_i*e_i against -P_i) is GLV-split,
//     k = +-k1 + lambda*(+-k2) with k1, k2 below 2^128: two streams on one table
//     of Q, the second reading phi(Q) = (beta*x, y) from it via one beta
//     multiplication per entry;
//   - a scalar below 2^128 (a_i against -R_i) is ONE unsplit stream on its
//     table, no phi. A "short" scalar that is not below 2^128 is therefore split
//     like a full one -- the same precondition check pippenger_msm_glv_short
//     makes, so a wrong caller costs speed, never correctness;
//   - the G term stays on the fixed-base generator path, added once at the end.
//     Interleaving it would need the w=15 G/H tables of dual_scalar_mul_gen_point,
//     which are file-static in point.cpp (get_dual_mul_gen_tables); this file
//     does not build a second copy.
// Tables: the odd multiples {1, 3, ..., 2^(w-1) - 1}*Q of every point are built
// for all points at once, each step a mixed addition on the point's isomorphic
// curve (build_odd_multiples), then converted to affine with ONE batch inversion
// over every entry (effective-affine, as multiscalar.cpp does). The scan uses
// mixed additions only. No per-call heap allocation once the thread_local
// scratch has grown to the batch size.
//
// The window constants below and the whole FE52 engine are one block: a build
// without FE52 takes the multi_scalar_mul arm at the end.
#if defined(SECP256K1_FAST_52BIT)
// wNAF window of the per-point streams; tables hold 2^(w-2) odd multiples.
// Chosen by measurement, the same host run as the thresholds in
// bch_schnorr.hpp. w = 4 was at or below w = 5 at every N from 2 to 2048, and
// w = 6 was slower than both at every N.
constexpr unsigned kBchStraussWindow = 4;
constexpr std::size_t kBchStraussTable = std::size_t{1} << (kBchStraussWindow - 2);
// Digits per stream. A scalar below 2^128 has at most 130 wNAF digits; 135 is
// the bound of the int8 compute_wnaf_into (the capacity multiscalar.cpp uses).
constexpr std::size_t kBchStraussWnafCap = 135;
static_assert(kBchStraussWindow >= 2 && kBchStraussWindow <= 8,
              "wNAF digits |d| <= 2^(w-1) - 1 must fit in int8_t");

using FE52 = fast::FieldElement52;

const FE52 kBchBeta52 = FE52::from_fe(FieldElement::from_bytes(fast::glv_constants::BETA));

struct StraussStream {
    std::size_t len;   // wNAF digit count; digits at [len, cap) are not written
    std::size_t tbl;   // the point's first table entry in tx / ty / phx
    bool phi;          // the lambda half: x from phx
    bool neg;          // the half-scalar is negative: flip y on top of the digit sign
};

struct StraussScratch {
    std::vector<Point> jac;            // odd multiples, Jacobian on the iso curve
    std::vector<FE52> ez;              // their Z on secp256k1 (iso Z * C)
    std::vector<FE52> prefix;          // batch-inversion prefix products
    std::vector<FE52> tx, ty, phx;     // affine tables; phx = beta * tx
    std::vector<std::uint8_t> split;   // per point: 1 when it has a phi stream
    std::vector<std::int8_t> wnaf;     // stream s at [s * cap, s * cap + len)
    std::vector<StraussStream> streams;
};

StraussScratch& strauss_scratch(std::size_t points) {
    static thread_local StraussScratch s;
    std::size_t const entries = points * kBchStraussTable;
    std::size_t const streams = 2 * points;
    if (s.jac.size() < entries) {
        s.jac.resize(entries);
        s.ez.resize(entries);
        s.prefix.resize(entries);
        s.tx.resize(entries);
        s.ty.resize(entries);
        s.phx.resize(entries);
    }
    if (s.split.size() < points) s.split.resize(points);
    if (s.streams.size() < streams) {
        s.streams.resize(streams);
        s.wnaf.resize(streams * kBchStraussWnafCap);
    }
    return s;
}

// (2j+1)*Q for j < kBchStraussTable into jac / ez at index t. Q is finite and
// the group has prime order, so 2Q != O and no step (2j+1)Q + 2Q is O or a
// doubling while 2j+3 < n; a zero Z (never reached) is caught by strauss_sum.
//
// Iso-curve trick (dual_mul_build_table in point.cpp): with D = 2Q = (X, Y, C)
// Jacobian, (x, y, z) -> (x*C^2, y*C^3, z) maps secp256k1 onto
// y^2 = x^3 + 7*C^6 and D onto the affine (X, Y). The a = 0 addition and
// doubling formulas never read the curve constant, so every step is one mixed
// addition, and the iso-curve point (x, y, z) is the secp256k1 point (x, y, z*C).
void build_odd_multiples(const Point& Q, StraussScratch& sc, std::size_t t) {
    Point const D = Q.dbl();
    FE52 dx = D.X52();
    FE52 dy = D.Y52();
    dx.normalize_weak();
    dy.normalize_weak();
    FE52 const C = D.Z52();
    FE52 const C2 = C.square();
    Point acc = Point::from_jacobian52(Q.X52() * C2, Q.Y52() * (C2 * C), Q.Z52(), false);
    for (std::size_t j = 0; j < kBchStraussTable; ++j) {
        if (j > 0) acc.add_mixed52_inplace(dx, dy);
        sc.jac[t + j] = acc;
        sc.ez[t + j] = acc.Z52() * C;
    }
}

Point strauss_sum(const Scalar& g_coeff,
                  const Scalar* full_k, const Point* full_P, std::size_t n_full,
                  const Scalar* short_k, const Point* short_P, std::size_t n_short) {
    StraussScratch& sc = strauss_scratch(n_full + n_short);
    std::size_t npts = 0;
    std::size_t nstreams = 0;
    std::size_t max_len = 0;

    auto add_stream = [&](const Scalar& k, std::size_t tbl, bool phi, bool neg) {
        StraussStream& st = sc.streams[nstreams];
        compute_wnaf_into(k, kBchStraussWindow,
                          sc.wnaf.data() + nstreams * kBchStraussWnafCap,
                          kBchStraussWnafCap, st.len);
        if (st.len == 0) return;                 // zero half: no digit, no stream
        st.tbl = tbl;
        st.phi = phi;
        st.neg = neg;
        if (st.len > max_len) max_len = st.len;
        ++nstreams;
    };
    auto add_term = [&](const Scalar& k, const Point& Q) {
        if (k.is_zero() || Q.is_infinity()) return;   // contributes O
        std::size_t const t = npts * kBchStraussTable;
        build_odd_multiples(Q, sc, t);
        auto const& l = k.limbs();
        bool const split = (l[2] | l[3]) != 0;          // k >= 2^128
        sc.split[npts] = split ? 1u : 0u;
        if (split) {
            fast::GLVDecomposition const d = fast::glv_decompose(k);
            add_stream(d.k1, t, false, d.k1_neg);
            add_stream(d.k2, t, true, d.k2_neg);
        } else {
            add_stream(k, t, false, false);
        }
        ++npts;
    };
    for (std::size_t i = 0; i < n_full; ++i) add_term(full_k[i], full_P[i]);
    for (std::size_t i = 0; i < n_short; ++i) add_term(short_k[i], short_P[i]);

    Point acc = Point::infinity();
    std::size_t const total = npts * kBchStraussTable;
    if (total != 0) {
        // One inversion for every table entry (Montgomery's trick).
        sc.prefix[0] = sc.ez[0];
        for (std::size_t k = 1; k < total; ++k) sc.prefix[k] = sc.prefix[k - 1] * sc.ez[k];
        if (sc.prefix[total - 1].normalizes_to_zero_var()) {
            return naive_sum(g_coeff, full_k, full_P, n_full, short_k, short_P, n_short);
        }
        FE52 inv = sc.prefix[total - 1].inverse_safegcd();   // VT: public data
        for (std::size_t k = total; k-- > 0;) {
            FE52 const zi = (k > 0) ? sc.prefix[k - 1] * inv : inv;
            if (k > 0) inv = inv * sc.ez[k];
            FE52 const zi2 = zi.square();
            sc.tx[k] = sc.jac[k].X52() * zi2;
            sc.ty[k] = sc.jac[k].Y52() * (zi2 * zi);
        }
        // phi(Q) table x = beta*x; its y is Q's, with the stream's own sign.
        for (std::size_t p = 0; p < npts; ++p) {
            if (sc.split[p] == 0) continue;
            std::size_t const t = p * kBchStraussTable;
            for (std::size_t j = 0; j < kBchStraussTable; ++j) {
                sc.phx[t + j] = sc.tx[t + j] * kBchBeta52;
            }
        }

        const std::int8_t* const wnaf = sc.wnaf.data();
        for (std::size_t bit = max_len; bit-- > 0;) {
            acc.dbl_inplace();
            for (std::size_t s = 0; s < nstreams; ++s) {
                StraussStream const& st = sc.streams[s];
                if (bit >= st.len) continue;
                int const d = wnaf[s * kBchStraussWnafCap + bit];
                if (d == 0) continue;
                bool const dneg = d < 0;
                std::size_t const idx =
                    st.tbl + static_cast<std::size_t>(((dneg ? -d : d) - 1) >> 1);
                FE52 const& x = st.phi ? sc.phx[idx] : sc.tx[idx];
                if (dneg != st.neg) {
                    acc.add_mixed52_neg_inplace(x, sc.ty[idx]);
                } else {
                    acc.add_mixed52_inplace(x, sc.ty[idx]);
                }
            }
        }
    }
    if (g_coeff.is_zero()) return acc;
    // All public data; VT generator multiplication is correct here.
    return Point::generator().scalar_mul(g_coeff).add(acc);
}
#else
// No FE52 point arithmetic in this build: the two halves through the
// multiscalar.cpp Strauss. The same sum; only the speed differs.
Point strauss_sum(const Scalar& g_coeff,
                  const Scalar* full_k, const Point* full_P, std::size_t n_full,
                  const Scalar* short_k, const Point* short_P, std::size_t n_short) {
    Point const acc = multi_scalar_mul(full_k, full_P, n_full)
                          .add(multi_scalar_mul(short_k, short_P, n_short));
    if (g_coeff.is_zero()) return acc;
    return Point::generator().scalar_mul(g_coeff).add(acc);
}
#endif // SECP256K1_FAST_52BIT

struct BchBatchScratch {
    std::vector<Scalar> scalars;
    std::vector<Point> points;
};

BchBatchScratch& bch_batch_scratch(std::size_t n) {
    static thread_local BchBatchScratch scratch;
    if (scratch.scalars.size() < n) scratch.scalars.resize(n);
    if (scratch.points.size() < n) scratch.points.resize(n);
    return scratch;
}

// The randomized check
//     (sum a_i*s_i)*G + sum (a_i*e_i)*(-P_i) + sum a_i*(-R_i) == O
// i.e. sum a_i*D_i == O with D_i = s_i*G - e_i*P_i - R_i and R_i the residue
// lift of r_i.
//
// Equality with the single verifier. Entry i verifies iff s_i*G - e_i*P_i is
// the point with x == r_i and a residue y, i.e. iff D_i == O (r_i has exactly
// one residue lift, and R' = O is excluded because then D_i = -R_i != O).
// Every structural rejection of the single verifier -- r >= p, s >= n, s == 0,
// bad or infinite key -- is applied here before any arithmetic, and r_i with
// no lift cannot verify singly either (x(R') == r_i would put r_i on the curve).
//
// Soundness (as in batch_verify.cpp). The group has prime order n, cofactor 1;
// 0 < a_i < 2^128 < n. One invalid entry j: a_j*D_j = O needs a_j = 0 mod n,
// impossible, so it is rejected with certainty at every position, j = 0 with
// a_0 = 1 included. Two or more: fix every weight but one invalid j >= 1;
// a_j*D_j = -T has at most one solution and a_j ranges over 2^127 values, so
// the batch accepts with probability <= 2^-127 -- PROVIDED a_j is
// unpredictable when the entries are chosen, which the CSPRNG XOR in
// batch_seed() guarantees.
//
// MSM layout: [0, n) a_i*e_i against -P_i, full width; [n, 2n) a_i against
// -R_i, below 2^128. The G term stays on the fixed-base generator path on
// every route; the 2n variable terms go to the engine batch_route() picked:
//
//   kStrauss    strauss_sum: one interleaved scan, (a_i*e_i, -P_i) GLV-split
//               into two streams, (a_i, -R_i) one unsplit stream, 3n streams.
//   kPippenger  pippenger_sum: pippenger_msm_glv_short, the same split, 3n
//               bucket points.
//
// Both engines compute the same sum exactly, so the equation, the weights and
// the soundness argument above -- which depend on nothing else -- are the same
// on both routes, and so is the verdict.
template <typename Access, typename Entry>
bool batch_msm_impl(const Entry* entries, std::size_t n, const std::uint8_t* seed32,
                    detail::BchBatchRoute route) {
    std::array<std::uint8_t, 32> const seed = batch_seed<Access>(entries, n, seed32);
    std::uint8_t bw_buf[36];
    std::memcpy(bw_buf, seed.data(), seed.size());

    auto& scratch = bch_batch_scratch(2 * n);
    Scalar* const scalars = scratch.scalars.data();
    Point* const points = scratch.points.data();
    Scalar g_coeff = Scalar::zero();

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* const sig = Access::sig(entries[i]);

        FieldElement r_fe;
        if (!FieldElement::parse_bytes_strict(sig, r_fe)) return false;
        Scalar s;
        if (!Scalar::parse_bytes_strict(sig + 32, s) || s.is_zero()) return false;

        Point P;
        if (!Access::resolve(entries[i], P) || P.is_infinity()) return false;

        Point R;
        if (!lift_x_residue(r_fe, R)) return false;

        Scalar const e = bch_challenge(sig, Access::pub33(entries[i]),
                                       Access::msg(entries[i]));
        Scalar const a = batch_weight(bw_buf, static_cast<std::uint32_t>(i));

        g_coeff += a * s;
        scalars[i] = a * e;
        points[i] = P.negate();
        scalars[n + i] = a;
        points[n + i] = R.negate();
    }

    Point const sum = (route == detail::BchBatchRoute::kStrauss)
        ? strauss_sum(g_coeff, scalars, points, n, scalars + n, points + n, n)
        : pippenger_sum(g_coeff, scalars, points, n, scalars + n, points + n, n);
    return sum.is_infinity();
}
#endif // SECP256K1_BCH_SCHNORR_HAS_MSM

template <typename Access, typename Entry>
bool batch_chunk_impl(const Entry* entries, std::size_t n, const std::uint8_t* seed32) {
    detail::BchBatchRoute const route = batch_route(n);
    if (route == detail::BchBatchRoute::kSingle) return entry_valid_impl<Access>(entries[0]);
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
    if (route != detail::BchBatchRoute::kIndividual) {
        return batch_msm_impl<Access>(entries, n, seed32, route);
    }
#else
    (void)seed32;
#endif
    for (std::size_t i = 0; i < n; ++i) {
        if (!entry_valid_impl<Access>(entries[i])) return false;
    }
    return true;
}

template <typename Access, typename Entry>
bool batch_impl(const Entry* entries, std::size_t n, const std::uint8_t* seed32) {
    if (n == 0) return true;            // empty batch: vacuously valid
    if (entries == nullptr) return false;
    for (std::size_t off = 0; off < n; off += kBchBatchMaxChunk) {
        std::size_t const m = (n - off < kBchBatchMaxChunk) ? (n - off) : kBchBatchMaxChunk;
        if (!batch_chunk_impl<Access>(entries + off, m, seed32)) return false;
    }
    return true;
}

// Batch first; per-entry only when it rejects. out_results is written in full
// on every path that returns after the argument checks.
template <typename Access, typename Entry>
bool identify_impl(const Entry* entries, std::size_t n, std::uint8_t* out_results,
                   const std::uint8_t* seed32) {
    if (n == 0) return true;
    if (entries == nullptr || out_results == nullptr) return false;
    if (batch_impl<Access>(entries, n, seed32)) {
        std::memset(out_results, 1, n);
        return true;
    }
    bool all_valid = true;
    for (std::size_t i = 0; i < n; ++i) {
        bool const ok = entry_valid_impl<Access>(entries[i]);
        out_results[i] = ok ? 1u : 0u;
        all_valid = all_valid && ok;
    }
    return all_valid;
}

} // anonymous namespace

// -- Public API ---------------------------------------------------------------

bool bch_schnorr_verify(const std::uint8_t sig64[64],
                        const std::uint8_t msg32[32],
                        const std::uint8_t pubkey33[33]) noexcept {
    if (sig64 == nullptr || msg32 == nullptr || pubkey33 == nullptr) return false;
    Point P;
    if (!parse_pubkey33(pubkey33, P)) return false;
    return bch_verify_impl(sig64, msg32, P, pubkey33);
}

bool bch_schnorr_verify(const std::uint8_t sig64[64],
                        const std::uint8_t msg32[32],
                        const fast::Point& P,
                        const std::uint8_t pubkey33[33]) noexcept {
    if (sig64 == nullptr || msg32 == nullptr || pubkey33 == nullptr) return false;
    return bch_verify_impl(sig64, msg32, P, pubkey33);
}

bool bch_schnorr_batch_verify(const BchSchnorrBatchEntry* entries, std::size_t n,
                              const std::uint8_t* seed32) {
    return batch_impl<WireAccess>(entries, n, seed32);
}

bool bch_schnorr_batch_verify(const std::vector<BchSchnorrBatchEntry>& entries,
                              const std::uint8_t* seed32) {
    return batch_impl<WireAccess>(entries.data(), entries.size(), seed32);
}

bool bch_schnorr_batch_verify(const BchSchnorrBatchPointEntry* entries, std::size_t n,
                              const std::uint8_t* seed32) {
    return batch_impl<PointAccess>(entries, n, seed32);
}

bool bch_schnorr_batch_verify_identify(const BchSchnorrBatchEntry* entries,
                                       std::size_t n,
                                       std::uint8_t* out_results,
                                       const std::uint8_t* seed32) {
    return identify_impl<WireAccess>(entries, n, out_results, seed32);
}

bool bch_schnorr_batch_verify_identify(const BchSchnorrBatchPointEntry* entries,
                                       std::size_t n,
                                       std::uint8_t* out_results,
                                       const std::uint8_t* seed32) {
    return identify_impl<PointAccess>(entries, n, out_results, seed32);
}

bool bch_schnorr_pubkey_parse(const std::uint8_t pubkey33[33],
                              fast::Point& out) noexcept {
    if (pubkey33 == nullptr) return false;
    Point P;
    if (!parse_pubkey33(pubkey33, P)) return false;
    out = P;
    return true;
}

// -- Test views ---------------------------------------------------------------
namespace detail {

std::array<std::uint8_t, 32> bch_schnorr_batch_digest_for_test(
    const BchSchnorrBatchEntry* entries, std::size_t n, const std::uint8_t* seed32) {
    return batch_digest<WireAccess>(entries, n, seed32);
}

std::array<std::uint8_t, 32> bch_schnorr_batch_seed_for_test(
    const BchSchnorrBatchEntry* entries, std::size_t n, const std::uint8_t* seed32) {
    return batch_seed<WireAccess>(entries, n, seed32);
}

fast::Scalar bch_schnorr_batch_weight_for_test(
    const std::array<std::uint8_t, 32>& seed, std::uint32_t index) {
    std::uint8_t buf[36];
    std::memcpy(buf, seed.data(), seed.size());
    return batch_weight(buf, index);
}

BchBatchRoute bch_schnorr_batch_route_for_test(std::size_t n) noexcept {
    return batch_route(n);
}

fast::Point bch_schnorr_strauss_sum_for_test(
    const fast::Scalar& g_coeff,
    const fast::Scalar* full_scalars, const fast::Point* full_points, std::size_t n_full,
    const fast::Scalar* short_scalars, const fast::Point* short_points, std::size_t n_short) {
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
    return strauss_sum(g_coeff, full_scalars, full_points, n_full,
                       short_scalars, short_points, n_short);
#else
    return naive_sum(g_coeff, full_scalars, full_points, n_full,
                     short_scalars, short_points, n_short);
#endif
}

fast::Point bch_schnorr_pippenger_sum_for_test(
    const fast::Scalar& g_coeff,
    const fast::Scalar* full_scalars, const fast::Point* full_points, std::size_t n_full,
    const fast::Scalar* short_scalars, const fast::Point* short_points, std::size_t n_short) {
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
    return pippenger_sum(g_coeff, full_scalars, full_points, n_full,
                         short_scalars, short_points, n_short);
#else
    return naive_sum(g_coeff, full_scalars, full_points, n_full,
                     short_scalars, short_points, n_short);
#endif
}

} // namespace detail

} // namespace secp256k1
