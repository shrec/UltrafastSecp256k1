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
// CSPRNG XOR, and pippenger_msm_glv_short for the 3n-point MSM.
// ============================================================================

#include "secp256k1/bch_schnorr.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/detail/csprng.hpp"
#include "secp256k1/detail/secure_erase.hpp"

// The MSM lives in pippenger.cpp, compiled only under SECP256K1_BUILD_PIPPENGER.
// src/cpu/CMakeLists.txt defines SECP256K1_BCH_SCHNORR_HAS_MSM for this file in
// exactly that case; without it every batch is verified entry by entry, which
// has the same verdict.
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
#include "secp256k1/pippenger.hpp"
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

// Below this many signatures the batch verifies one at a time.
//
// Throughput: not yet measured -- benchmark required. The value is the BIP-340
// cutoff (kSchnorrBatchIndividualCutoff in batch_verify.cpp, measured for the
// same 3N-point pippenger_msm_glv_short shape) and is a placeholder until a
// bench_unified A/B of this file with the cutoff forced to 1 is run. The
// verdict does not depend on it: both paths return the AND of the single
// verdicts.
constexpr std::size_t kBchBatchIndividualCutoff = 27;

// Weight indices are 32-bit; a batch longer than this is verified as several
// batches, each with its own fresh seed, so no two entries of one batch ever
// share a weight index.
constexpr std::size_t kBchBatchMaxChunk =
    static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

// Quadratic-residue test on a public field element. Zero is not a residue
// (Jacobi(0) == 0, never 1).
bool fe_is_qr_var(const FieldElement& v) noexcept {
#if defined(SECP256K1_BCH_SCHNORR_FE52_JACOBI)
    return fast::FieldElement52::from_fe(v).jacobi_var() == 1;
#else
    if (v == FieldElement::zero()) return false;
    FieldElement const t = v.sqrt();
    return t.square() == v;
#endif
}

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

#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
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
// MSM layout: [0, n) a_i*e_i against -P_i, full width, GLV-split; [n, 2n) a_i
// against -R_i, below 2^128, unsplit. pippenger_msm_glv_short runs 3n points.
template <typename Access, typename Entry>
bool batch_msm_impl(const Entry* entries, std::size_t n, const std::uint8_t* seed32) {
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

    // All public data; VT generator multiplication is correct here.
    Point const G_term = Point::generator().scalar_mul(g_coeff);
    Point const rest = pippenger_msm_glv_short(scalars, points, n,
                                               scalars + n, points + n, n);
    return G_term.add(rest).is_infinity();
}
#endif // SECP256K1_BCH_SCHNORR_HAS_MSM

template <typename Access, typename Entry>
bool batch_chunk_impl(const Entry* entries, std::size_t n, const std::uint8_t* seed32) {
    if (n == 1) return entry_valid_impl<Access>(entries[0]);
#if defined(SECP256K1_BCH_SCHNORR_HAS_MSM)
    if (n > kBchBatchIndividualCutoff) {
        return batch_msm_impl<Access>(entries, n, seed32);
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

} // namespace detail

} // namespace secp256k1
