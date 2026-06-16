// ============================================================================
// shim_extrakeys.cpp -- x-only pubkeys and keypairs
// ============================================================================
#include "secp256k1_extrakeys.h"
#include "secp256k1.h"
#include "shim_internal.hpp"
#include "shim_pubkey_helpers.hpp"

#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using namespace secp256k1::fast;

// point_to_pubkey_data from shim_pubkey_helpers.hpp
using secp256k1_shim_internal::point_to_pubkey_data;
using secp256k1_shim_internal::pubkey_data_to_point;
// x-only / keypair derivation is fail-closed on an off-curve opaque pubkey.
using secp256k1_shim_internal::pubkey_data_to_point_checked;

// Helper: reconstruct a Point from an xonly_pubkey using cached X||Y -- no sqrt.
// SHIM-CURVE-CHECK-XONLY: validate y^2=x^3+7 before use. A hostile caller could
// write arbitrary bytes into secp256k1_xonly_pubkey.data[32..63], bypassing
// secp256k1_xonly_pubkey_from_pubkey / secp256k1_xonly_pubkey_parse.
// Off-curve input -> Point::infinity(); callers return 0 and clear outputs.
static Point xonly_to_point(const secp256k1_xonly_pubkey *xp)
{
    std::array<uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), xp->data,      32);
    std::memcpy(yb.data(), xp->data + 32, 32);
    FieldElement x, y;
    if (!FieldElement::parse_bytes_strict(xb, x)) return Point::infinity();
    if (!FieldElement::parse_bytes_strict(yb, y)) return Point::infinity();
    auto b7 = FieldElement::from_uint64(7);
    if (y * y != x * x * x + b7) return Point::infinity();
    return Point::from_affine(x, y);
}

extern "C" {

// -- X-only public key ----------------------------------------------------
// Opaque layout: data[0..31] = X big-endian, data[32..63] = Y big-endian (even parity)
// Caching Y avoids re-lifting (sqrt) on every Schnorr verify — saves ~1.5 us/verify.

int secp256k1_xonly_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    const unsigned char *input32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !input32) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_xonly_pubkey_parse: NULL argument");
        return 0;
    }

    // PASS4-003: use schnorr_xonly_pubkey_parse (FE52 + Jacobi QR pre-rejection +
    // lift_x/GLV cache) instead of manual FieldElement::sqrt(). Benefits:
    //   1. FE52 Jacobi (~900 ns) pre-rejects invalid x-coords before 3.8 µs sqrt.
    //   2. Populates lift_x cache → first secp256k1_schnorrsig_verify (ShimSchnorrCache
    //      miss) finds lift_x and GLV tables already built, saving ~3.5 µs per call.
    //   3. Builds GLV tables on first encounter (one-phase design).
    secp256k1::SchnorrXonlyPubkey epk{};
    if (!secp256k1::schnorr_xonly_pubkey_parse(epk, input32)) return 0;
    // epk.point: Z=1 (affine from lift_x), even Y (BIP-340 convention).
    // point_to_pubkey_data uses the fast Z=1 path: no field inversion needed.
    point_to_pubkey_data(epk.point, pubkey->data);
    return 1;
}

int secp256k1_xonly_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output32,
    const secp256k1_xonly_pubkey *pubkey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!output32 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_xonly_pubkey_serialize: NULL argument");
        return 0;
    }
    auto P = xonly_to_point(pubkey);
    if (P.is_infinity()) {
        std::memset(output32, 0, 32);
        return 0;
    }
    std::memcpy(output32, pubkey->data, 32);
    return 1;
}

int secp256k1_xonly_pubkey_cmp(
    const secp256k1_context *ctx,
    const secp256k1_xonly_pubkey *pk1,
    const secp256k1_xonly_pubkey *pk2)
{
    // SHIM-A05: NULL ctx must fire callback (libsecp aborts via ARG_CHECK).
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr,
            "secp256k1_xonly_pubkey_cmp: NULL context");
        return 0;
    }
    if (!pk1 || !pk2) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_xonly_pubkey_cmp: invalid pubkey argument");
        return 0;
    }
    return std::memcmp(pk1->data, pk2->data, 32);
}

int secp256k1_xonly_pubkey_from_pubkey(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *xonly_pubkey,
    int *pk_parity, const secp256k1_pubkey *pubkey)
{
    SHIM_REQUIRE_CTX(ctx);  // SHIM-NEW-003: NULL ctx fires illegal callback (abort)
    // COMPAT-006: NULL xonly_pubkey or pubkey must fire illegal callback (ARG_CHECK in upstream).
    if (!xonly_pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_xonly_pubkey_from_pubkey: xonly_pubkey is NULL");
        return 0;
    }
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_xonly_pubkey_from_pubkey: pubkey is NULL");
        return 0;
    }

    // SHIM-A10: validate curve membership before trusting the stored bytes.
    // pubkey_data_to_point checks y²=x³+7; returns infinity if off-curve.
    auto pt = pubkey_data_to_point_checked(pubkey->data);
    if (pt.is_infinity()) { std::memset(xonly_pubkey->data, 0, sizeof(xonly_pubkey->data)); return 0; }

    // pubkey layout: data[0..31] = X, data[32..63] = Y (both big-endian)
    int y_is_odd = (pubkey->data[63] & 1) ? 1 : 0;

    // X-only key always uses even Y: if pubkey Y is odd, negate Y
    std::memcpy(xonly_pubkey->data, pubkey->data, 32); // copy X
    if (!y_is_odd) {
        std::memcpy(xonly_pubkey->data + 32, pubkey->data + 32, 32); // even Y, use as-is
    } else {
        // Negate Y: -Y mod p
        std::array<uint8_t, 32> yb{};
        std::memcpy(yb.data(), pubkey->data + 32, 32);
        auto y = FieldElement::from_bytes(yb);
        y = y.negate();
        auto yn = y.to_bytes();
        std::memcpy(xonly_pubkey->data + 32, yn.data(), 32);
    }

    if (pk_parity) *pk_parity = y_is_odd;
    return 1;
}

// -- Keypair --------------------------------------------------------------
// Layout: data[0..31] = secret key, data[32..95] = pubkey opaque (X || Y)

int secp256k1_keypair_create(
    const secp256k1_context *ctx, secp256k1_keypair *keypair,
    const unsigned char *seckey)
{
    // SHIM-007: require sign-capable context (NULL ctx fires illegal callback/abort)
    if (!secp256k1_shim_internal::ctx_can_sign(ctx)) return 0;
    if (!keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_create: keypair is NULL");
        return 0;
    }
    if (!seckey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_create: seckey is NULL");
        return 0;
    }
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);

    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) {
        std::memset(keypair->data, 0, sizeof(keypair->data));
        secp256k1::detail::secure_erase(&k, sizeof(k));   // secret-residue sweep
        return 0;
    }

    auto P = secp256k1::ct::generator_mul(k);   // CT: Rule 12 — sk is a private key
    if (P.is_infinity()) {
        std::memset(keypair->data, 0, sizeof(keypair->data));
        secp256k1::detail::secure_erase(&k, sizeof(k));   // secret-residue sweep
        return 0;
    }

    // BIP-340 normalization: Schnorr signing requires d to produce an even-Y pubkey.
    // Negate k (CT) when P.y is odd so that k*G has even Y, matching libsecp256k1
    // keypair_create behavior. Y-parity of P is a public output — bool_to_mask is
    // safe here. ct::scalar_cneg is branchless (follows shim_schnorr.cpp NEW-006 pattern).
    bool const y_odd = !P.has_even_y();
    k = secp256k1::ct::scalar_cneg(k, secp256k1::ct::bool_to_mask(y_odd));
    // If y was odd, negate P to produce the even-Y point matching the new k.
    if (y_odd) P = P.negate();

    // Store BIP-340-normalized seckey, then erase the stack copies (both the
    // serialized bytes AND the Scalar k holding the normalized private key).
    auto k_bytes = k.to_bytes();
    std::memcpy(keypair->data, k_bytes.data(), 32);
    secp256k1::detail::secure_erase(k_bytes.data(), k_bytes.size());
    secp256k1::detail::secure_erase(&k, sizeof(k));   // secret-residue sweep

    // Store pubkey (X || Y). point_to_pubkey_data avoids the 65-byte to_uncompressed()
    // allocation (PERF pattern from keypair_xonly_tweak_add / NEW-PERF-002).
    point_to_pubkey_data(P, keypair->data + 32);
    return 1;
}

int secp256k1_keypair_sec(
    const secp256k1_context *ctx, unsigned char *seckey,
    const secp256k1_keypair *keypair)
{
    SHIM_REQUIRE_CTX(ctx);  // SHIM-NEW-003: NULL ctx fires illegal callback
    if (!seckey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_sec: seckey is NULL");
        return 0;
    }
    if (!keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_sec: keypair is NULL");
        return 0;
    }
    Scalar sk_check;
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk_check)) {
        std::memset(seckey, 0, 32);
        secp256k1::detail::secure_erase(&sk_check, sizeof(sk_check));
        return 0;
    }
    secp256k1::detail::secure_erase(&sk_check, sizeof(sk_check));
    std::memcpy(seckey, keypair->data, 32);
    return 1;
}

int secp256k1_keypair_pub(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const secp256k1_keypair *keypair)
{
    SHIM_REQUIRE_CTX(ctx);  // SHIM-NEW-003: NULL ctx fires illegal callback
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_pub: pubkey is NULL");
        return 0;
    }
    if (!keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_pub: keypair is NULL");
        return 0;
    }
    auto P = pubkey_data_to_point_checked(keypair->data + 32);
    if (P.is_infinity()) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    std::memcpy(pubkey->data, keypair->data + 32, 64);
    return 1;
}

int secp256k1_keypair_xonly_pub(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    int *pk_parity, const secp256k1_keypair *keypair)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_xonly_pub: pubkey is NULL");
        return 0;
    }
    if (!keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_keypair_xonly_pub: keypair is NULL");
        return 0;
    }
    auto P = pubkey_data_to_point_checked(keypair->data + 32);
    if (P.is_infinity()) {
        std::memset(pubkey->data, 0, sizeof(pubkey->data));
        if (pk_parity) *pk_parity = 0;
        return 0;
    }

    // keypair layout: data[0..31]=sk, data[32..63]=X, data[64..95]=Y (big-endian)
    int y_is_odd = (keypair->data[95] & 1) ? 1 : 0;

    std::memcpy(pubkey->data, keypair->data + 32, 32); // X

    // x-only canonical form: use even Y
    if (!y_is_odd) {
        std::memcpy(pubkey->data + 32, keypair->data + 64, 32);
    } else {
        std::array<uint8_t, 32> yb{};
        std::memcpy(yb.data(), keypair->data + 64, 32);
        FieldElement y;
        if (!FieldElement::parse_bytes_strict(yb, y)) {
            std::memset(pubkey->data, 0, sizeof(pubkey->data));
            if (pk_parity) *pk_parity = 0;
            return 0;
        }
        y = y.negate();
        auto yn = y.to_bytes();
        std::memcpy(pubkey->data + 32, yn.data(), 32);
    }

    if (pk_parity) *pk_parity = y_is_odd;
    return 1;
}

// -- Taproot tweak operations -----------------------------------------------

int secp256k1_xonly_pubkey_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_pubkey *output_pubkey,
    const secp256k1_xonly_pubkey *internal_pubkey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);  // SHIM-006: NULL ctx must fire illegal callback
    if (!output_pubkey || !internal_pubkey || !tweak32) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_xonly_pubkey_tweak_add: NULL argument");
        return 0;
    }

    // PASS-COMPAT-004: upstream xonly_pubkey_tweak_add memsets output_pubkey to 0
    // on every failure path. output_pubkey is a separate buffer from the inputs
    // (no in-place hazard), so zero it before each failure return to match.
    auto P = xonly_to_point(internal_pubkey);
    if (P.is_infinity()) { std::memset(output_pubkey->data, 0, 64); return 0; }

    // Reject tweak >= n (libsecp uses scalar_set_b32_limit)
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) { std::memset(output_pubkey->data, 0, 64); return 0; }

    auto tG = scalar_mul_generator(t);
    auto Q  = P.add(tG);
    if (Q.is_infinity()) { std::memset(output_pubkey->data, 0, 64); return 0; }

    // PERF-006: use point_to_pubkey_data instead of to_uncompressed() to avoid
    // a 65-byte allocation. point_to_pubkey_data uses the fast affine path when
    // Z=1, falling back to to_uncompressed() only for Jacobian results.
    point_to_pubkey_data(Q, output_pubkey->data);
    return 1;
}

int secp256k1_xonly_pubkey_tweak_add_check(
    const secp256k1_context *ctx,
    const unsigned char *tweaked_pubkey32,
    int tweaked_pk_parity,
    const secp256k1_xonly_pubkey *internal_pubkey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);  // SHIM-006: NULL ctx must fire illegal callback
    if (!tweaked_pubkey32 || !internal_pubkey || !tweak32) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_xonly_pubkey_tweak_add_check: NULL argument");
        return 0;
    }

    auto P = xonly_to_point(internal_pubkey);
    if (P.is_infinity()) return 0;

    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;

    auto tG = scalar_mul_generator(t);
    auto Q  = P.add(tG);
    if (Q.is_infinity()) return 0;

    // PERF-006: extract X and Y via point_to_pubkey_data instead of to_uncompressed().
    // point_to_pubkey_data writes X[0..31] || Y[32..63] — avoids 65-byte allocation.
    unsigned char qdata[64];
    point_to_pubkey_data(Q, qdata);
    int q_parity = (qdata[63] & 1) ? 1 : 0;

    if (q_parity != tweaked_pk_parity) return 0;
    return std::memcmp(qdata, tweaked_pubkey32, 32) == 0 ? 1 : 0;
}

int secp256k1_keypair_xonly_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_keypair *keypair,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);  // SHIM-006: NULL ctx must fire illegal callback
    if (!keypair || !tweak32) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_keypair_xonly_tweak_add: NULL argument");
        return 0;
    }

    // CT-04: tweaked private-key material (sk, new_sk, new_skb) is the BIP-341
    // key-path-spend secret. It MUST be secure_erase'd on EVERY return path —
    // mirroring the discipline already applied to keypair_create (line ~168) and
    // the shim ECDSA/ellswift sign paths (CT-01/SHIM-01/02). The CT-01 sweep
    // (commit 24b29021) covered shim_ecdsa/ellswift/recovery + bip32 but not this
    // function.
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk)) {
        std::memset(keypair->data, 0, sizeof(keypair->data));
        secp256k1::detail::secure_erase(&sk, sizeof(sk));   // CT-04
        return 0;
    }

    // If Y is odd, negate the secret key (so the x-only key has even Y).
    // Use ct::scalar_cneg to avoid branching on keypair state.
    std::uint64_t y_odd_mask = (keypair->data[95] & 1) ? ~std::uint64_t(0) : std::uint64_t(0);
    sk = secp256k1::ct::scalar_cneg(sk, y_odd_mask);

    // tweak in [0, n-1]; libsecp allows 0 (keypair unchanged)
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) {
        std::memset(keypair->data, 0, sizeof(keypair->data));
        secp256k1::detail::secure_erase(&sk, sizeof(sk));   // CT-04
        return 0;
    }

    auto new_sk = secp256k1::ct::scalar_add(sk, t);
    // CT-002: is_zero_ct() reads all limbs unconditionally before comparing.
    // is_zero() (fast::) has a data-dependent early-exit; new_sk is a secret.
    if (new_sk.is_zero_ct()) {
        std::memset(keypair->data, 0, sizeof(keypair->data));
        secp256k1::detail::secure_erase(&sk, sizeof(sk));         // CT-04
        secp256k1::detail::secure_erase(&new_sk, sizeof(new_sk));
        return 0;
    }

    auto P = secp256k1::ct::generator_mul(new_sk);   // CT: Rule 12 — new_sk is secret
    if (P.is_infinity()) {
        std::memset(keypair->data, 0, sizeof(keypair->data));
        secp256k1::detail::secure_erase(&sk, sizeof(sk));         // CT-04
        secp256k1::detail::secure_erase(&new_sk, sizeof(new_sk));
        return 0;
    }

    auto new_skb = new_sk.to_bytes();
    std::memcpy(keypair->data, new_skb.data(), 32);
    // NEW-PERF-002: use point_to_pubkey_data to avoid to_uncompressed() 65-byte
    // allocation + extra memcpy. secp256k1_xonly_pubkey_tweak_add and
    // secp256k1_xonly_pubkey_tweak_add_check already use this pattern (PERF-004 fix).
    point_to_pubkey_data(P, keypair->data + 32);
    // CT-04: erase the tweaked-private-key stack residue after its last use.
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&new_sk, sizeof(new_sk));
    secp256k1::detail::secure_erase(new_skb.data(), new_skb.size());
    return 1;
}

} // extern "C"
