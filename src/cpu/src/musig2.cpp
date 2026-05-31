// ============================================================================
// MuSig2: Two-Round Multi-Signature Scheme (BIP-327)
// ============================================================================

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/detail/csprng.hpp"
#include <cstring>
#include <algorithm>

namespace {
using secp256k1::detail::secure_erase;
} // anonymous namespace

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;
using detail::g_keyagg_list_midstate;
using detail::g_keyagg_coeff_midstate;
using detail::g_musig_nonceblinding_midstate;
using detail::g_musig_aux_midstate;
using detail::g_musig_nonce_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

// -- Helpers ------------------------------------------------------------------

namespace {

// Decompress a 33-byte compressed point
Point decompress_point(const std::array<uint8_t, 33>& compressed) {
    if (compressed[0] != 0x02 && compressed[0] != 0x03) {
        return Point::infinity();
    }

    // Strict: reject x >= p
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(compressed.data() + 1, x)) {
        return Point::infinity();
    }

    // y^2 = x^3 + 7
    auto x3 = x.square() * x;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain
    auto y = y2.sqrt();

    if (y.square() != y2) return Point::infinity();

    // Select parity (limbs()[0] LSB == big-endian byte[31] LSB for normalized FE)
    bool const y_odd = (y.limbs()[0] & 1) != 0;
    bool const want_odd = (compressed[0] == 0x03);
    if (y_odd != want_odd) {
        y = y.negate();
    }

    return Point::from_affine(x, y);
}

// Check if point has even Y
bool has_even_y(const Point& P) {
    return P.has_even_y();
}

} // anonymous namespace

// -- Key Aggregation (KeyAgg) -------------------------------------------------
// BIP-327 KeyAgg: Q = sum(a_i * P_i)
// a_i = tagged_hash("KeyAgg coefficient", L || pk_i)
// where L = tagged_hash("KeyAgg list", pk_1 || ... || pk_n)
// pk_i are 33-byte compressed pubkeys (prefix byte preserves Y parity).

MuSig2KeyAggCtx musig2_key_agg(const std::vector<std::array<uint8_t, 33>>& pubkeys) {
    MuSig2KeyAggCtx ctx{};
    std::size_t const n = pubkeys.size();
    if (n == 0) return ctx;

    // Validate ALL pubkeys upfront AND cache the decompressed points.
    // BIP-327 cpoint(P_i) decompresses the 33-byte compressed key, respecting
    // Y parity. The partial_sign step adjusts d for each P_i's Y parity
    // independently, so we must NOT force even-Y here.
    //
    // static thread_local: retains capacity across calls (hot-path scanner
    // allows this; local vector construction per call is flagged as HEAP_VEC).
    // clear() before push_back() ensures MSan shadow bits are clean on reuse —
    // stale shadow state from a previous call's Point objects is avoided because
    // each element is move-constructed fresh rather than copy-assigned in place.
    static thread_local std::vector<Point> points;
    points.clear();
    points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto pt = decompress_point(pubkeys[i]);
        if (pt.is_infinity()) return ctx;
        points.push_back(std::move(pt));
    }

    // BIP-327: L = tagged_hash("KeyAgg list", pk_1 || ... || pk_n) — 33 bytes each
    SHA256 l_ctx = g_keyagg_list_midstate;
    for (std::size_t i = 0; i < n; ++i) {
        l_ctx.update(pubkeys[i].data(), 33);
    }
    auto L = l_ctx.finalize();

    // BIP-327: pk2 = first key in list that differs from pk_1.
    // Keys equal to pk2 get coefficient 1; all others get the hash coefficient.
    ctx.key_coefficients.resize(n);
    const std::array<uint8_t, 33>* pk2 = nullptr;
    for (std::size_t i = 1; i < n; ++i) {
        if (pubkeys[i] != pubkeys[0]) {
            pk2 = &pubkeys[i];
            break;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (pk2 && pubkeys[i] == *pk2) {
            ctx.key_coefficients[i] = Scalar::one();
        } else {
            // a_i = tagged_hash("KeyAgg coefficient", L || pk_i) — pk_i is 33 bytes
            uint8_t coeff_input[65];
            std::memcpy(coeff_input, L.data(), 32);
            std::memcpy(coeff_input + 32, pubkeys[i].data(), 33);
            auto coeff_hash = cached_tagged_hash(g_keyagg_coeff_midstate, coeff_input, 65);
            ctx.key_coefficients[i] = Scalar::from_bytes(coeff_hash);
        }
    }

    // Q = sum(a_i * P_i) via MSM (Shamir trick for n=2, Pippenger for n>=48).
    // Cached points from validation loop — no double sqrt.
    Point Q = msm(ctx.key_coefficients, points);

    ctx.Q = Q;

    // Ensure even Y for BIP-340 compatibility
    ctx.Q_negated = !has_even_y(Q);
    if (ctx.Q_negated) {
        ctx.Q = Q.negate();
    }

    ctx.Q_x = ctx.Q.x().to_bytes();

    // Store individual pubkeys for signer_index validation in musig2_partial_sign (Rule 13).
    ctx.individual_pubkeys = pubkeys;
    return ctx;
}

// -- Nonce Generation ---------------------------------------------------------

std::pair<MuSig2SecNonce, MuSig2PubNonce> musig2_nonce_gen(
    const Scalar& secret_key,
    const std::array<uint8_t, 32>& pub_key,
    const std::array<uint8_t, 32>& agg_pub_key,
    const std::array<uint8_t, 32>& msg,
    const uint8_t* extra_input,
    const uint8_t* nonce_extra) {

    MuSig2SecNonce sec{};
    MuSig2PubNonce pub{};

    // t = secret_key XOR tagged_hash("MuSig/aux", extra_input or fresh_random)
    // When extra_input==NULL we fill with OS randomness so nonces are never
    // deterministic. A caller always passing NULL would otherwise produce the
    // same k1/k2 for identical (secret_key, msg) inputs — enabling nonce reuse.
    auto sk_bytes = secret_key.to_bytes();
    std::array<uint8_t, 32> aux{};
    if (extra_input) {
        std::memcpy(aux.data(), extra_input, 32);
    } else {
        secp256k1::detail::csprng_fill(aux.data(), 32);
    }
    auto aux_hash = cached_tagged_hash(g_musig_aux_midstate, aux.data(), 32);

    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = sk_bytes[i] ^ aux_hash[i];

    // k1 = tagged_hash("MuSig/nonce", t || pub_key || agg_pub_key || msg || [nonce_extra] || 0x01)
    // CT: fixed 2-iteration approach — data-dependent retry loop is non-CT because
    // the iteration count leaks whether k1_hash >= n or == 0 (both functions of
    // the private-key-derived t). Probability of retry ~2^-128. Fix: always run
    // exactly 2 iterations and use ct::scalar_select (matches rfc6979_nonce fix).
    // nonce_extra: when non-NULL, included in nonce_input before counter (BIP-327 extra_input32
    // defense-in-depth). nonce_input is 161 bytes when set, 129 when NULL. Backward-compatible.
    {
        uint8_t nonce_input[161];
        std::memcpy(nonce_input, t, 32);
        std::memcpy(nonce_input + 32, pub_key.data(), 32);
        std::memcpy(nonce_input + 64, agg_pub_key.data(), 32);
        std::memcpy(nonce_input + 96, msg.data(), 32);
        std::size_t ni_len;
        if (nonce_extra) {
            std::memcpy(nonce_input + 128, nonce_extra, 32);
            nonce_input[160] = 0x01;
            ni_len = 161;
        } else {
            nonce_input[128] = 0x01;
            ni_len = 129;
        }
        auto k1_hash = cached_tagged_hash(g_musig_nonce_midstate, nonce_input, ni_len);
        Scalar cand1{}, cand2{};
        bool const ok1 = Scalar::parse_bytes_strict_nonzero(k1_hash.data(), cand1);
        // Advance hash state unconditionally: XOR byte 31 of the candidate hash to
        // derive the second candidate. This matches the original loop's first retry step.
        auto k1_hash2 = k1_hash;
        k1_hash2[31] ^= 0x01u;
        (void)Scalar::parse_bytes_strict_nonzero(k1_hash2.data(), cand2);
        std::uint64_t const mask = static_cast<std::uint64_t>(
            -static_cast<std::int64_t>(static_cast<int>(ok1)));
        sec.k1 = ct::scalar_select(cand1, cand2, mask);
        // P2-CT-003: erase both candidate scalars — both hold nonce-derived
        // secret material and must not persist as stack residue.
        secure_erase(&cand1, sizeof(cand1));
        secure_erase(&cand2, sizeof(cand2));
        secure_erase(nonce_input, sizeof(nonce_input));
        secure_erase(k1_hash.data(), k1_hash.size());
        secure_erase(k1_hash2.data(), k1_hash2.size());
    }

    // k2 = tagged_hash("MuSig/nonce", t || pub_key || agg_pub_key || msg || [nonce_extra] || 0x02)
    // Same CT fixed 2-iteration approach as k1 above.
    {
        uint8_t nonce_input[161];
        std::memcpy(nonce_input, t, 32);
        std::memcpy(nonce_input + 32, pub_key.data(), 32);
        std::memcpy(nonce_input + 64, agg_pub_key.data(), 32);
        std::memcpy(nonce_input + 96, msg.data(), 32);
        std::size_t ni_len;
        if (nonce_extra) {
            std::memcpy(nonce_input + 128, nonce_extra, 32);
            nonce_input[160] = 0x02;
            ni_len = 161;
        } else {
            nonce_input[128] = 0x02;
            ni_len = 129;
        }
        auto k2_hash = cached_tagged_hash(g_musig_nonce_midstate, nonce_input, ni_len);
        Scalar cand1{}, cand2{};
        bool const ok1 = Scalar::parse_bytes_strict_nonzero(k2_hash.data(), cand1);
        auto k2_hash2 = k2_hash;
        k2_hash2[31] ^= 0x02u;
        (void)Scalar::parse_bytes_strict_nonzero(k2_hash2.data(), cand2);
        std::uint64_t const mask = static_cast<std::uint64_t>(
            -static_cast<std::int64_t>(static_cast<int>(ok1)));
        sec.k2 = ct::scalar_select(cand1, cand2, mask);
        // P2-CT-003: erase both candidate scalars — both hold nonce-derived
        // secret material and must not persist as stack residue.
        secure_erase(&cand1, sizeof(cand1));
        secure_erase(&cand2, sizeof(cand2));
        secure_erase(nonce_input, sizeof(nonce_input));
        secure_erase(k2_hash.data(), k2_hash.size());
        secure_erase(k2_hash2.data(), k2_hash2.size());
    }

    // Zeroize secret key material now that nonces are derived
    secure_erase(sk_bytes.data(), sk_bytes.size());
    secure_erase(aux_hash.data(), aux_hash.size());
    secure_erase(t, sizeof(t));

    // R1 = k1 * G, R2 = k2 * G (CT-004: use generator_mul_blinded for DPA defense;
    // blinding is transparent — blinded(k)*G == k*G mathematically, so nonces
    // remain deterministic while gaining protection against power/EM side channels
    // when secp256k1_context_randomize() has been called).
    auto R1 = ct::generator_mul_blinded(sec.k1);
    auto R2 = ct::generator_mul_blinded(sec.k2);
    pub.R1 = R1.to_compressed();
    pub.R2 = R2.to_compressed();

    return {sec, pub};
}

// -- Nonce Serialization ------------------------------------------------------

std::array<uint8_t, 66> MuSig2PubNonce::serialize() const {
    std::array<uint8_t, 66> out{};
    std::memcpy(out.data(), R1.data(), 33);
    std::memcpy(out.data() + 33, R2.data(), 33);
    return out;
}

MuSig2PubNonce MuSig2PubNonce::deserialize(const std::array<uint8_t, 66>& data) {
    MuSig2PubNonce nonce{};
    std::memcpy(nonce.R1.data(), data.data(), 33);
    std::memcpy(nonce.R2.data(), data.data() + 33, 33);
    return nonce;
}

// -- Nonce Aggregation --------------------------------------------------------

MuSig2AggNonce musig2_nonce_agg(const std::vector<MuSig2PubNonce>& pub_nonces) {
    // SEC-009: empty vector guard — return infinity-bearing struct so that
    // musig2_start_sign_session rejects it via the SEC-005 check below.
    if (pub_nonces.empty()) {
        MuSig2AggNonce empty{};
        empty.R1 = Point::infinity();
        empty.R2 = Point::infinity();
        return empty;
    }

    MuSig2AggNonce agg{};
    agg.R1 = Point::infinity();
    agg.R2 = Point::infinity();

    for (const auto& nonce : pub_nonces) {
        auto r1 = decompress_point(nonce.R1);
        auto r2 = decompress_point(nonce.R2);
        agg.R1 = agg.R1.add(r1);
        agg.R2 = agg.R2.add(r2);
    }

    return agg;
}

// Fast overload: skip decompress — Points are already in affine form (SHIM-007).
MuSig2AggNonce musig2_nonce_agg_points(
    const std::vector<std::pair<fast::Point, fast::Point>>& pts)
{
    MuSig2AggNonce agg{};
    agg.R1 = Point::infinity();
    agg.R2 = Point::infinity();
    for (const auto& [r1, r2] : pts) {
        agg.R1 = agg.R1.add(r1);
        agg.R2 = agg.R2.add(r2);
    }
    return agg;
}

// -- Session Start ------------------------------------------------------------

MuSig2Session musig2_start_sign_session(
    const MuSig2AggNonce& agg_nonce,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const std::array<uint8_t, 32>& msg) {

    MuSig2Session session{};

    // SEC-005: BIP-327 §GetSessionValues step 2 — abort if either aggregated nonce
    // component is infinity (indicates empty input, nonce cancellation, or invalid input).
    // A zero Scalar for session.b and session.e signals an invalid session to the caller
    // (musig2_partial_sign checks: if session.e.is_zero() the session is degenerate).
    // Returning a default-constructed session (all-zero scalars, infinity R) is the
    // agreed invalid-session convention used throughout this codebase.
    if (agg_nonce.R1.is_infinity() || agg_nonce.R2.is_infinity()) {
        return session;  // default-constructed: R=infinity, b=0, e=0, R_negated=false
    }

    // b = tagged_hash("MuSig/nonceblinding", cbytes_ext(R1)||cbytes_ext(R2)||xbytes(Q)||msg)
    // BIP-327 §GetSessionValues: tag MUST be "MuSig/nonceblinding" (not "noncecoef").
    auto R1_comp = agg_nonce.R1.to_compressed();
    auto R2_comp = agg_nonce.R2.to_compressed();

    uint8_t b_input[130]; // 33 + 33 + 32 + 32
    std::memcpy(b_input, R1_comp.data(), 33);
    std::memcpy(b_input + 33, R2_comp.data(), 33);
    std::memcpy(b_input + 66, key_agg_ctx.Q_x.data(), 32);
    std::memcpy(b_input + 98, msg.data(), 32);
    auto b_hash = cached_tagged_hash(g_musig_nonceblinding_midstate, b_input, 130);
    session.b = Scalar::from_bytes(b_hash);

    // R = R1 + b * R2
    auto bR2 = agg_nonce.R2.scalar_mul(session.b);  // nosemgrep: secret-scalar-variable-time-mul
    session.R = agg_nonce.R1.add(bR2);

    // Negate R if needed for even Y
    session.R_negated = !has_even_y(session.R);
    if (session.R_negated) {
        session.R = session.R.negate();
    }

    // e = tagged_hash("BIP0340/challenge", R.x || Q_x || msg)
    auto R_x = session.R.x().to_bytes();
    uint8_t e_input[96];
    std::memcpy(e_input, R_x.data(), 32);
    std::memcpy(e_input + 32, key_agg_ctx.Q_x.data(), 32);
    std::memcpy(e_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, e_input, 96);
    session.e = Scalar::from_bytes(e_hash);

    return session;
}

// -- Partial Signing ----------------------------------------------------------

Scalar musig2_partial_sign(
    MuSig2SecNonce& sec_nonce,
    const Scalar& secret_key,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index) {

    // Bounds check: signer_index out of range (distinct from degenerate arithmetic)
    if (signer_index >= key_agg_ctx.key_coefficients.size()) {
        return Scalar::zero();  // caller checks for zero and returns UFSECP_ERR_BAD_INPUT
    }

    // Rule 13 (MANDATORY, fail-closed): validate that secret_key actually corresponds
    // to the claimed signer_index BEFORE signing. The C++ API path (musig2_key_agg)
    // always populates individual_pubkeys; the ABI path populates it in
    // ufsecp_musig2_partial_sign_v2 from the caller's pubkeys[] (the v1 ABI, which has
    // no pubkeys parameter, is hard-failed at entry — see src/impl/ufsecp_musig2.cpp).
    //
    // P1-SEC-01 / MED-3 (closed): this check was previously SKIPPED when
    // individual_pubkeys was empty, which let a C++ caller that manually cleared the
    // field sign as any signer_index. We now fail closed: if the per-signer pubkeys
    // are absent or too short we cannot validate the signer index, so we refuse to
    // sign rather than signing blind. signer_index and the container size are public
    // (not secret-derived), so branching on them is not a timing concern.
    if (signer_index >= key_agg_ctx.individual_pubkeys.size()) {
        return Scalar::zero();  // cannot validate signer_index → refuse to sign
    }
    {
        Point const derived = ct::generator_mul_blinded(secret_key);
        if (SECP256K1_UNLIKELY(derived.is_infinity())) {
            return Scalar::zero();
        }
        // CT invariant: to_compressed() calls SafeGCD field inverse (Bernstein-Yang,
        // fixed 59 divstep iterations) on the Jacobian Z from ct::generator_mul.
        // The Z value is secret-dependent, but SafeGCD runs a fixed number of steps
        // regardless of input, so this path is constant-time. The byte comparison is
        // a branchless XOR-accumulate — no early exit to avoid leaking which bytes
        // differ; for a correct (matching) signer it always reaches diff == 0.
        auto const derived_c = derived.to_compressed();
        const auto& expected = key_agg_ctx.individual_pubkeys[signer_index];
        std::uint64_t diff = 0;
        for (std::size_t i = 0; i < 33; ++i) {
            diff |= static_cast<std::uint64_t>(derived_c[i] ^ expected[i]);
        }
        if (diff != 0) {
            return Scalar::zero();
        }
    }

    // Reject degenerate session: e==0 means the challenge hash is zero (impossible
    // in a correct FROST/MuSig2 run). If e==0, s_i = k + 0*a_i*d = k, which
    // would expose the secret nonce. Erase and fail-closed.
    if (session.e.is_zero_ct()) {
        secure_erase(&sec_nonce.k1, sizeof(sec_nonce.k1));
        secure_erase(&sec_nonce.k2, sizeof(sec_nonce.k2));
        return Scalar::zero();
    }

    // k = k1 + b * k2
    // Use ct:: primitives: fast::Scalar +/* have a secret-dependent branch in the
    // final modular reduction that can leak nonce bits via timing side-channels.
    Scalar k = ct::scalar_add(sec_nonce.k1, ct::scalar_mul(session.b, sec_nonce.k2));

    // CT conditional negate k if R was negated (R_negated is public,
    // but keep branchless for consistency and to avoid pipeline leaks).
    {
        std::uint64_t const mask = ct::bool_to_mask(session.R_negated);
        Scalar const neg_k = k.negate();
        k = ct::scalar_select(neg_k, k, mask);
    }

    // Adjust secret key -- fully constant-time path:
    // BIP-327 §signing: s_i = k_i_eff + e * a_i * g * d_i
    // g = -1 if Q has odd Y, else 1. No per-signer P_i parity adjustment.
    // (Individual P_i parity is NOT part of the BIP-327 signing formula.)
    Scalar d = secret_key;

    // CT negate d if aggregate key Q was negated for even-Y (the g factor).
    {
        std::uint64_t const mask = ct::bool_to_mask(key_agg_ctx.Q_negated);
        Scalar const neg_d = d.negate();
        d = ct::scalar_select(neg_d, d, mask);
    }

    // s_i = k + e * a_i * d  (mod n)
    // ct::scalar_mul/add: branchless modular arithmetic -- no secret-dependent
    // branches in the final reduction, unlike fast::Scalar operator*/ operator+.
    Scalar const ea = ct::scalar_mul(session.e, key_agg_ctx.key_coefficients[signer_index]);
    Scalar const ead = ct::scalar_mul(ea, d);
    Scalar const result = ct::scalar_add(k, ead);

    // Erase secret nonce and adjusted signing key from stack, then consume
    // the caller's secret nonce to enforce single-use (M-03).
    secure_erase(&k, sizeof(k));
    secure_erase(&d, sizeof(d));
    secure_erase(&sec_nonce.k1, sizeof(sec_nonce.k1));
    secure_erase(&sec_nonce.k2, sizeof(sec_nonce.k2));

    return result;
}

// -- Partial Verification -----------------------------------------------------

bool musig2_partial_verify(
    const Scalar& partial_sig,
    const MuSig2PubNonce& pub_nonce,
    const std::array<uint8_t, 32>& pubkey,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index) {

    // Bounds check: signer_index must be valid for the key_coefficients vector.
    // musig2_partial_sign already has this guard; verify must be consistent.
    if (signer_index >= key_agg_ctx.key_coefficients.size()) {
        return false;
    }

    // s_i * G should equal R_i + b * R2_i + e * a_i * P_i
    // (with appropriate negation adjustments)

    // CT-003: partial_sig is a public value — use variable-time scalar_mul, not CT.
    auto sG = Point::generator().scalar_mul(partial_sig);

    auto R1_i = decompress_point(pub_nonce.R1);
    auto R2_i = decompress_point(pub_nonce.R2);
    // BIP-327 §4 PartialSigVerify: reject invalid (infinity) nonce points.
    if (R1_i.is_infinity() || R2_i.is_infinity()) {
        return false;
    }

    // Effective nonce: R_i = R1_i + b * R2_i
    auto R_eff = R1_i.add(R2_i.scalar_mul(session.b));
    if (session.R_negated) {
        R_eff = R_eff.negate();
    }

    // Key contribution: e * a_i * P_i
    // Lift pubkey (strict: reject x >= p). Use BOTH possible Y parities
    // because the partial sign API takes a raw secret key (original parity),
    // not an x-only key. BIP-327 signing uses d_i directly (no per-signer
    // Y-parity flip), so P_i can have either Y parity.
    FieldElement px;
    if (!FieldElement::parse_bytes_strict(pubkey, px)) return false;
    auto x3 = px.square() * px;
    auto y2 = x3 + FieldElement::from_uint64(7);
    auto y = y2.sqrt();
    if (y.square() != y2) return false;  // x not on curve -- reject

    Scalar ea = session.e * key_agg_ctx.key_coefficients[signer_index];
    if (key_agg_ctx.Q_negated) ea = ea.negate();

    // Helper: check if sG == R_eff + ea * Point::from_affine(px, y_candidate)
    auto jacobian_eq = [](const Point& A, const Point& B) -> bool {
        auto const z1sq = A.z().square();
        auto const z2sq = B.z().square();
        if (A.X() * z2sq != B.X() * z1sq) return false;
        auto const z1cu = z1sq * A.z();
        auto const z2cu = z2sq * B.z();
        return A.Y() * z2cu == B.Y() * z1cu;
    };

    // Try even-Y candidate; ea * (-Pi) = -(ea * Pi) so only one scalar_mul needed.
    FieldElement y_even = (y.limbs()[0] & 1) ? y.negate() : y;
    auto Pi_even = Point::from_affine(px, y_even);
    auto eaP = Pi_even.scalar_mul(ea);
    if (jacobian_eq(sG, R_eff.add(eaP))) return true;
    return jacobian_eq(sG, R_eff.add(eaP.negate()));
}

// -- Signature Aggregation ----------------------------------------------------

std::array<uint8_t, 64> musig2_partial_sig_agg(
    const std::vector<Scalar>& partial_sigs,
    const MuSig2Session& session) {

    // s = sum(s_i) — CT: use ct::scalar_add instead of fast::operator+=
    // fast::Scalar::operator+= has a data-dependent ge(ORDER) branch in its
    // final modular reduction. Partial sigs contain signer-secret contributions;
    // use branchless ct::scalar_add throughout the accumulation (CT-002 fix).
    Scalar s = Scalar::zero();
    for (const auto& si : partial_sigs) {
        s = secp256k1::ct::scalar_add(s, si);
    }

    // Fail-closed: degenerate aggregated signature is a security failure
    if (s.is_zero_ct() || session.R.is_infinity()) {
        return {};  // all-zero array signals failure to caller
    }

    // Final signature: (R.x, s)
    auto R_x = session.R.x().to_bytes();
    auto s_bytes = s.to_bytes();
    secure_erase(&s, sizeof(s));

    std::array<uint8_t, 64> sig{};
    std::memcpy(sig.data(), R_x.data(), 32);
    std::memcpy(sig.data() + 32, s_bytes.data(), 32);
    return sig;
}

} // namespace secp256k1
