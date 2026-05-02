#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/config.hpp"    // SECP256K1_FAST_52BIT
#include "secp256k1/field_52.hpp"
#include "secp256k1/debug_invariants.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/ct/point.hpp"  // ct::generator_mul for secret-bearing paths
#include "secp256k1/ct/scalar.hpp" // ct::scalar_cneg, ct::bool_to_mask
#include <cstring>
#include <string_view>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;
#if defined(SECP256K1_FAST_52BIT)
using FE52 = fast::FieldElement52;
#endif

// -- FE52 sqrt() and inverse() available as FieldElement52 class methods ------
// sqrt() uses FE52 ops (~4us, faster than 4x64 ~6.8us).
// inverse() uses FE52 Fermat (~4us) -- but SafeGCD (~2-3us) is faster for
// variable-time paths (point.cpp batch inverse, verify Y-parity).

// -- lift_x: shared BIP-340 x-only -> affine Point ----------------------------
// Input must be strict x in [0, p), represented as 4x64 LE limbs.
// Returns Point::infinity() if x is not on the curve.
static Point lift_x_from_limbs(const std::uint64_t* px_limb_le) {
#if defined(SECP256K1_FAST_52BIT)
#if defined(__aarch64__)
    // On current ARM64 targets, 4x64 sqrt path benchmarks faster than FE52
    // for lift_x; use it for raw Schnorr verify input decoding.
    FieldElement const px_fe = FieldElement::from_limbs_raw({
        px_limb_le[0], px_limb_le[1], px_limb_le[2], px_limb_le[3]});
    auto x3 = px_fe * px_fe * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y_fe = y2.sqrt();
    auto chk = y_fe * y_fe;
    if (!(chk == y2)) return Point::infinity();
    if (y_fe.limbs()[0] & 1) y_fe = y_fe.negate();
    return Point::from_affine(px_fe, y_fe);
#else
    FE52 const px52 = FE52::from_4x64_limbs(px_limb_le);

    // y^2 = x^3 + 7
    FE52 const x3 = px52.square() * px52;
    static const FE52 seven52 = FE52::from_fe(FieldElement::from_uint64(7));
    FE52 const y2 = x3 + seven52;

    // sqrt via FE52 addition chain: a^((p+1)/4), ~253 sqr + 13 mul
    FE52 y52 = y2.sqrt();

    // Verify sqrt without fully normalizing both operands.
    FE52 check = y52.square();
    check.negate_assign(1);
    check.add_assign(y2);
    if (!check.normalizes_to_zero_var()) return Point::infinity();

    // Ensure even Y (BIP-340 convention): check parity of normalized y
    FE52 y_norm = y52;
    y_norm.normalize();
    if (y_norm.n[0] & 1) {
        // Negate: y = p - y
        y52 = y52.negate(1);
        y52.normalize_weak();
    }

    // Zero-conversion: construct Point directly from FE52 affine coordinates
    return Point::from_affine52(px52, y52);
#endif
#else
    FieldElement const px_fe = FieldElement::from_limbs_raw({
        px_limb_le[0], px_limb_le[1], px_limb_le[2], px_limb_le[3]});
    auto x3 = px_fe * px_fe * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y_fe = y2.sqrt();
    auto chk = y_fe * y_fe;
    if (!(chk == y2)) return Point::infinity();
    // 4x64 mul_impl Barrett-reduces to [0, p), so limbs()[0] & 1 is
    // the true parity -- no serialization needed.
    if (y_fe.limbs()[0] & 1) y_fe = y_fe.negate();
    return Point::from_affine(px_fe, y_fe);
#endif
}

static inline std::uint64_t load_be64_unaligned(const uint8_t* p) {
    std::uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
#if defined(_MSC_VER)
    return _byteswap_uint64(v);
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    return __builtin_bswap64(v);
#else
    return v;
#endif
}

static inline void parse_be32_to_le64(const uint8_t* in32, std::uint64_t* out4) {
    out4[3] = load_be64_unaligned(in32 + 0);
    out4[2] = load_be64_unaligned(in32 + 8);
    out4[1] = load_be64_unaligned(in32 + 16);
    out4[0] = load_be64_unaligned(in32 + 24);
}

static inline bool limbs_lt_p(const std::uint64_t* x4) {
    constexpr std::uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    return x4[3] != 0xFFFFFFFFFFFFFFFFULL ||
           x4[2] != 0xFFFFFFFFFFFFFFFFULL ||
           x4[1] != 0xFFFFFFFFFFFFFFFFULL ||
           x4[0] < P0;
}

static inline bool parse_and_check_lt_p(const uint8_t* in32, std::uint64_t* out4) {
    parse_be32_to_le64(in32, out4);
    return limbs_lt_p(out4);
}

static inline bool parse2_and_check_lt_p(const uint8_t* a32,
                                         const uint8_t* b32,
                                         std::uint64_t* out_a4,
                                         std::uint64_t* out_b4) {
    out_a4[3] = load_be64_unaligned(a32 + 0);
    out_a4[2] = load_be64_unaligned(a32 + 8);
    out_a4[1] = load_be64_unaligned(a32 + 16);
    out_a4[0] = load_be64_unaligned(a32 + 24);

    out_b4[3] = load_be64_unaligned(b32 + 0);
    out_b4[2] = load_be64_unaligned(b32 + 8);
    out_b4[1] = load_be64_unaligned(b32 + 16);
    out_b4[0] = load_be64_unaligned(b32 + 24);

    return limbs_lt_p(out_a4) && limbs_lt_p(out_b4);
}

// Tiny thread-local cache for raw BIP-340 verification path.
// Public-key input is non-secret, so memoizing lifted x-only pubkeys is safe.
struct XOnlyLiftCacheEntry {
    std::array<uint8_t, 32> x{};
    Point p = Point::infinity();
    bool valid = false;
};

static inline Scalar schnorr_challenge_scalar(const uint8_t* r32,
                                              const uint8_t* pubkey_x32,
                                              const uint8_t* msg32) {
    alignas(16) uint8_t challenge_input[96];
    std::memcpy(challenge_input + 0, r32, 32);
    std::memcpy(challenge_input + 32, pubkey_x32, 32);
    std::memcpy(challenge_input + 64, msg32, 32);
    return Scalar::from_bytes(detail::cached_tagged_hash(detail::g_challenge_midstate,
                                                         challenge_input,
                                                         sizeof(challenge_input)));
}

#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(SECP256K1_PLATFORM_STM32)

static constexpr std::size_t kLiftXCacheSlots = 1024;

// Single file-scope thread_local cache shared by both lookup and store.
// Bug fix: previously each function had its own function-scope thread_local
// array — C++ function-scope thread_locals are independent symbols, so
// store wrote to array B while lookup read from array A (always empty).
// The cache was completely dead. Moving to file scope fixes the sharing.
static thread_local std::array<XOnlyLiftCacheEntry, kLiftXCacheSlots> g_lift_x_cache{};

static inline std::size_t lift_x_cache_index(const uint8_t* pubkey_x32) {
    std::size_t const h0 = pubkey_x32[0] ^ pubkey_x32[7] ^ pubkey_x32[15] ^ pubkey_x32[23] ^ pubkey_x32[31];
    std::size_t const h1 = pubkey_x32[1] ^ pubkey_x32[9] ^ pubkey_x32[17] ^ pubkey_x32[25];
    return (h0 | (h1 << 8)) & (kLiftXCacheSlots - 1);
}

// Returns slot index so lift_x_cached can reuse it for the store path,
// avoiding a second call to lift_x_cache_index when the lookup misses.
static inline bool lift_x_cache_lookup(const uint8_t* pubkey_x32, Point& out,
                                        std::size_t& idx_out) {
    idx_out = lift_x_cache_index(pubkey_x32);
    auto& slot = g_lift_x_cache[idx_out];
    if (!slot.valid || std::memcmp(slot.x.data(), pubkey_x32, 32) != 0) {
        return false;
    }
    out = slot.p;
    return !out.is_infinity();
}

static inline void lift_x_cache_store_at(const uint8_t* pubkey_x32,
                                           const Point& lifted,
                                           std::size_t idx) {
    auto& slot = g_lift_x_cache[idx];
    std::memcpy(slot.x.data(), pubkey_x32, 32);
    slot.p = lifted;
    slot.valid = true;
}
#endif

static inline bool lift_x_cached(const uint8_t* pubkey_x32,
                                 Point& out) {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(SECP256K1_PLATFORM_STM32)
    std::uint64_t pkL[4];
    if (!parse_and_check_lt_p(pubkey_x32, pkL)) return false;
    Point const lifted = lift_x_from_limbs(pkL);
    if (lifted.is_infinity()) return false;
    out = lifted;
    return true;
#else
    std::size_t cache_idx = 0;
    if (lift_x_cache_lookup(pubkey_x32, out, cache_idx)) {
        return true;
    }

    std::uint64_t pkL[4];
    if (!parse_and_check_lt_p(pubkey_x32, pkL)) return false;

    Point const lifted = lift_x_from_limbs(pkL);
    if (lifted.is_infinity()) {
        return false;
    }

    lift_x_cache_store_at(pubkey_x32, lifted, cache_idx);
    out = lifted;
    return true;
#endif
}

// -- Shared BIP-340 tagged-hash midstates (from tagged_hash.hpp) ---------------
using detail::g_aux_midstate;
using detail::g_nonce_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

// -- Tagged Hash (BIP-340) -- generic fallback ---------------------------------

std::array<uint8_t, 32> tagged_hash(const char* tag,
                                     const void* data, std::size_t len) {
    std::string_view const sv(tag);
    auto tag_hash = SHA256::hash(sv.data(), sv.size());
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    ctx.update(data, len);
    return ctx.finalize();
}

// -- Schnorr Signature --------------------------------------------------------

std::array<uint8_t, 64> SchnorrSignature::to_bytes() const {
    std::array<uint8_t, 64> out{};
    std::memcpy(out.data(), r.data(), 32);
    auto s_bytes = s.to_bytes();
    std::memcpy(out.data() + 32, s_bytes.data(), 32);
    return out;
}

SchnorrSignature SchnorrSignature::from_bytes(const uint8_t* data64) {
    SchnorrSignature sig{};
    std::memcpy(sig.r.data(), data64, 32);
    sig.s = Scalar::from_bytes(data64 + 32);
    return sig;
}

SchnorrSignature SchnorrSignature::from_bytes(const std::array<uint8_t, 64>& data) {
    return from_bytes(data.data());
}

// -- BIP-340 strict signature parsing (r < p, 0 < s < n) ---------------------

bool SchnorrSignature::parse_strict(const uint8_t* data64, SchnorrSignature& out) noexcept {
    // BIP-340: fail if r >= p
    FieldElement r_fe;
    if (!FieldElement::parse_bytes_strict(data64, r_fe)) return false;

    // BIP-340: fail if s >= n; also reject s == 0
    Scalar s_val;
    if (!Scalar::parse_bytes_strict_nonzero(data64 + 32, s_val)) return false;

    std::memcpy(out.r.data(), data64, 32);
    out.s = s_val;
    return true;
}

bool SchnorrSignature::parse_strict(const std::array<uint8_t, 64>& data,
                                     SchnorrSignature& out) noexcept {
    return parse_strict(data.data(), out);
}

// -- X-only pubkey ------------------------------------------------------------

std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    SECP_ASSERT_SCALAR_VALID(private_key);
    auto P = ct::generator_mul(private_key);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    (void)p_y_odd;
    return px;
}

// -- SchnorrKeypair Creation --------------------------------------------------

SchnorrKeypair schnorr_keypair_create(const Scalar& private_key) {
    SECP_ASSERT_SCALAR_VALID(private_key);
    SchnorrKeypair kp{};
    auto d_prime = private_key;
    if (d_prime.is_zero()) return kp;

    auto P = ct::generator_mul(d_prime);
    auto [px, p_y_odd] = P.x_bytes_and_parity();

    kp.d = ct::scalar_cneg(d_prime, ct::bool_to_mask(p_y_odd));
    kp.px = px;
    return kp;
}

// -- BIP-340 Sign (keypair variant, fast) -------------------------------------
// Uses pre-computed keypair: only 1 gen_mul + 1 FE52 inverse per sign.

SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    SECP_ASSERT_SCALAR_VALID(kp.d);
    if (kp.d.is_zero()) return SchnorrSignature{};

    // Step 1: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 2: k' = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, kp.px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, 96);
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) {
        // Zeroize all secret-derived data before early return (~2^-128 probability).
        detail::secure_erase(d_bytes.data(), d_bytes.size());
        detail::secure_erase(t_hash.data(), t_hash.size());
        detail::secure_erase(t, sizeof(t));
        detail::secure_erase(nonce_input, sizeof(nonce_input));
        detail::secure_erase(rand_hash.data(), rand_hash.size());
        detail::secure_erase(&k_prime, sizeof(k_prime));
        return SchnorrSignature{};
    }

    // Step 3: R = k' * G (CT Hamburg comb, blinded when context_randomize active)
    auto R = ct::generator_mul_blinded(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 4: k = k' if has_even_y(R), else n - k'  [CT: branchless negate]
    auto k = ct::scalar_cneg(k_prime, ct::bool_to_mask(r_y_odd));

    // Step 5: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg)
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, rx.data(), 32);
    std::memcpy(challenge_input + 32, kp.px.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 6: sig = (R.x, k + e * d)
    // CT: use ct::scalar_mul/add — fast::Scalar operator*/+ have a
    // secret-dependent branch in the final modular reduction (see ct_scalar.cpp).
    SchnorrSignature sig{};
    sig.r = rx;
    sig.s = ct::scalar_add(k, ct::scalar_mul(e, kp.d));

    // Erase all secret-derived stack buffers (matches ct::schnorr_sign cleanup)
    detail::secure_erase(d_bytes.data(), d_bytes.size());
    detail::secure_erase(t_hash.data(), t_hash.size());
    detail::secure_erase(t, sizeof(t));
    detail::secure_erase(nonce_input, sizeof(nonce_input));
    detail::secure_erase(rand_hash.data(), rand_hash.size());
    detail::secure_erase(challenge_input, sizeof(challenge_input));
    detail::secure_erase(&k_prime, sizeof(k_prime));
    detail::secure_erase(&k, sizeof(k));

    // Reject degenerate output: r==all-zeros is astronomically rare (~2^-128)
    // but must never be serialised as a valid signature (Rule 14).
    bool r_zero = true;
    for (auto b : sig.r) r_zero &= (b == 0);
    if (SECP256K1_UNLIKELY(r_zero || sig.s.is_zero())) return SchnorrSignature{};

    return sig;
}

// -- BIP-340 Sign + Verify (fault attack countermeasure) ----------------------

SchnorrSignature schnorr_sign_verified(const SchnorrKeypair& kp,
                                       const std::array<uint8_t, 32>& msg,
                                       const std::array<uint8_t, 32>& aux_rand) {
    const auto sig = schnorr_sign(kp, msg, aux_rand);

    if (sig.s.is_zero()) {
        return SchnorrSignature{};
    }

    if (!schnorr_verify(kp.px, msg, sig)) {
        return SchnorrSignature{};
    }

    return sig;
}

// -- BIP-340 Sign (raw key, convenience) --------------------------------------

SchnorrSignature schnorr_sign(const Scalar& private_key,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    const auto kp = schnorr_keypair_create(private_key);
    return schnorr_sign(kp, msg, aux_rand);
}

// -- BIP-340 Sign (raw key) + Verify ------------------------------------------

SchnorrSignature schnorr_sign_verified(const Scalar& private_key,
                                       const std::array<uint8_t, 32>& msg,
                                       const std::array<uint8_t, 32>& aux_rand) {
    const auto kp = schnorr_keypair_create(private_key);
    return schnorr_sign_verified(kp, msg, aux_rand);
}

// -- BIP-340 Verify -----------------------------------------------------------

bool schnorr_verify(const uint8_t* pubkey_x32,
                    const uint8_t* msg32,
                    const SchnorrSignature& sig) {
    // Step 0: BIP-340 strict range checks
    // Check s: must be in [1, n-1] -- enforced at parse time by parse_strict,
    // but also guard here for callers using from_bytes (reducing parser).
    if (sig.s.is_zero()) return false;

    // Check r < p: parse r bytes to 4x64 LE limbs + strict check, no FieldElement.
    std::uint64_t rL[4];
    if (!parse_and_check_lt_p(sig.r.data(), rL)) return false;

    // Step 2: Lift x-only pubkey to point (cached for repeated pubkeys)
    Point P;
    if (!lift_x_cached(pubkey_x32, P)) return false;

    // Step 3: e = tagged_hash("BIP0340/challenge", r || pubkey_x || msg) mod n
    const auto e = schnorr_challenge_scalar(sig.r.data(), pubkey_x32, msg32);

    // Step 4: R = s*G - e*P  (4-stream GLV Strauss: s*G + (-e)*P in one pass)
    const auto neg_e = e.negate();
    const auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, P);

    if (R.is_infinity()) return false;

    // Steps 5+6: Single affine conversion (matches libsecp256k1 approach).
    // Z^{-1} -> Z^{-2}, Z^{-3} -> x_aff, y_aff -> check both X match and Y-parity.
    // Since Y-parity requires Z^{-3} anyway, computing X from Z^{-2} is free.
#if defined(SECP256K1_FAST_52BIT)
    FE52 const z_inv = R.Z52().inverse_safegcd();
    FE52 const z_inv2 = z_inv.square();
    FE52 x_aff = R.X52() * z_inv2;       // magnitude 1
    FE52 const z_inv3 = z_inv * z_inv2;
    FE52 y_aff = R.Y52() * z_inv3;       // magnitude 1

    // X-check: r parsed directly to FE52 (no FieldElement intermediate)
    const FE52 r52 = FE52::from_4x64_limbs(rL);
    x_aff.negate_assign(1);               // magnitude 2
    x_aff.add_assign(r52);                // magnitude 3 (r52 - x_aff)
    const bool x_match = x_aff.normalizes_to_zero_var();

    // Y-parity: must fully normalize to check lowest bit reliably.
    y_aff.normalize();
    return x_match & ((y_aff.n[0] & 1) == 0);
#else
    FieldElement r_fe_check = FieldElement::from_limbs_raw({rL[0], rL[1], rL[2], rL[3]});
    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = R.x_raw() * z_inv2;
    FieldElement z_inv3 = z_inv * z_inv2;
    FieldElement y_aff = R.y_raw() * z_inv3;
    return (x_aff == r_fe_check) & ((y_aff.limbs()[0] & 1) == 0);
#endif
}

// -- Pre-cached X-only Pubkey -------------------------------------------------

bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const uint8_t* pubkey_x32) {
    Point P = Point::infinity();
    if (!lift_x_cached(pubkey_x32, P)) return false;
    out.point = P;
    std::memcpy(out.x_bytes.data(), pubkey_x32, 32);

#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    // Build GLV P/phi(P) tables once — eliminates ~1,954 ns rebuild per verify.
    out.tables_valid = Point::build_schnorr_verify_tables(
        P, out.tbl_P, out.tbl_phi_base, out.Z_shared);
#endif
    return true;
}

bool schnorr_xonly_pubkey_parse(SchnorrXonlyPubkey& out,
                                const std::array<uint8_t, 32>& pubkey_x) {
    return schnorr_xonly_pubkey_parse(out, pubkey_x.data());
}

SchnorrXonlyPubkey schnorr_xonly_from_keypair(const SchnorrKeypair& kp) {
    SchnorrXonlyPubkey pub{};
    auto P = ct::generator_mul(kp.d);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    if (p_y_odd) {
#if defined(SECP256K1_FAST_52BIT)
        FE52 neg_y = P.Y52().negate(1);
        neg_y.normalize_weak();
        P = Point::from_jacobian52(P.X52(), neg_y, P.Z52(), false);
#else
        const auto y_neg = P.y().negate();
        P = Point::from_jacobian_coords(P.x(), y_neg, P.z(), false);
#endif
    }
    P.normalize();
    pub.point = P;
    pub.x_bytes = px;
    return pub;
}

// -- BIP-340 Verify (fast, pre-cached pubkey) ---------------------------------
// Skips lift_x sqrt (~1.6us savings). Same algorithm, just uses cached Point.

bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const uint8_t* msg32,
                    const SchnorrSignature& sig) {
    // BIP-340 strict: s must be nonzero
    if (sig.s.is_zero()) return false;

    // BIP-340 strict: r < p
    // Parse r bytes directly to 4x64 LE limbs, check against prime, then
    // convert to FE52 in one shot -- no FieldElement intermediate object.
    std::uint64_t rL[4];
    if (!parse_and_check_lt_p(sig.r.data(), rL)) return false;

    const auto e = schnorr_challenge_scalar(sig.r.data(), pubkey.x_bytes.data(), msg32);

    // R = s*G - e*P  (direct Point -- no sqrt needed)
    const auto neg_e = e.negate();

#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    // Fast path: use cached GLV P/phi(P) tables (skips ~1,954 ns table rebuild).
    // flip_phi = (decomp_e.k1_neg != decomp_e.k2_neg) for neg_e scalar.
    // Computed inside dual_scalar_mul_gen_prebuilt from GLV(neg_e).
    const auto R = pubkey.tables_valid
        ? Point::dual_scalar_mul_gen_prebuilt(sig.s, neg_e,
                                               pubkey.tbl_P, pubkey.tbl_phi_base,
                                               pubkey.Z_shared)
        : Point::dual_scalar_mul_gen_point(sig.s, neg_e, pubkey.point);
#else
    const auto R = Point::dual_scalar_mul_gen_point(sig.s, neg_e, pubkey.point);
#endif

    if (R.is_infinity()) return false;

    // Single affine conversion: Z^{-1} -> (x_aff, y_aff) -> check both.
    // Since Y-parity requires Z^{-3} anyway, computing X from Z^{-2} is free.
#if defined(SECP256K1_FAST_52BIT)
    FE52 const z_inv = R.Z52().inverse_safegcd();
    FE52 const z_inv2 = z_inv.square();
    FE52 x_aff = R.X52() * z_inv2;       // magnitude 1
    FE52 const z_inv3 = z_inv * z_inv2;
    FE52 y_aff = R.Y52() * z_inv3;       // magnitude 1

    // X-check: parse r directly to FE52 (no FieldElement intermediate)
    const FE52 r52 = FE52::from_4x64_limbs(rL);
    x_aff.negate_assign(1);               // magnitude 2
    x_aff.add_assign(r52);                // magnitude 3 (r52 - x_aff)
    const bool x_match = x_aff.normalizes_to_zero_var();

    // Y-parity: must fully normalize to check lowest bit reliably.
    y_aff.normalize();
    return x_match & ((y_aff.n[0] & 1) == 0);
#else
    FieldElement r_fe_check = FieldElement::from_limbs_raw({rL[0], rL[1], rL[2], rL[3]});
    FieldElement z_inv = R.z_raw().inverse();
    FieldElement z_inv2 = z_inv;
    z_inv2.square_inplace();
    FieldElement x_aff = R.x_raw() * z_inv2;
    FieldElement z_inv3 = z_inv * z_inv2;
    FieldElement y_aff = R.y_raw() * z_inv3;
    return (x_aff == r_fe_check) & ((y_aff.limbs()[0] & 1) == 0);
#endif
}

// -- Array wrappers (delegate to raw-pointer implementations) -----------------

bool schnorr_verify(const std::array<uint8_t, 32>& pubkey_x,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    return schnorr_verify(pubkey_x.data(), msg.data(), sig);
}

bool schnorr_verify(const std::array<uint8_t, 32>& pubkey_x,
                    const uint8_t* msg32,
                    const SchnorrSignature& sig) {
    return schnorr_verify(pubkey_x.data(), msg32, sig);
}

bool schnorr_verify(const SchnorrXonlyPubkey& pubkey,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    return schnorr_verify(pubkey, msg.data(), sig);
}

} // namespace secp256k1
