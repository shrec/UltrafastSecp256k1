/* ============================================================================
 * UltrafastSecp256k1 -- ufsecp C ABI Implementation
 * ============================================================================
 * Wraps the C++ UltrafastSecp256k1 library behind the opaque ufsecp_ctx and
 * the ufsecp_* function surface.  All conversions between opaque byte arrays
 * and internal C++ types happen here -- nothing leaks.
 *
 * Build with:  -DUFSECP_BUILDING   (sets dllexport on Windows)
 * ============================================================================ */

#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif

#include "ufsecp.h"

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <string>
#include <new>

/* -- UltrafastSecp256k1 C++ headers ---------------------------------------- */
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/taproot.hpp"
#include "secp256k1/init.hpp"

using Scalar = secp256k1::fast::Scalar;
using Point  = secp256k1::fast::Point;
using FE     = secp256k1::fast::FieldElement;

/* ===========================================================================
 * Context definition (opaque to callers)
 * =========================================================================== */

struct ufsecp_ctx {
    ufsecp_error_t   last_err;
    char             last_msg[128];
    bool             selftest_ok;
};

static void ctx_clear_err(ufsecp_ctx* ctx) {
    ctx->last_err  = UFSECP_OK;
    ctx->last_msg[0] = '\0';
}

static ufsecp_error_t ctx_set_err(ufsecp_ctx* ctx, ufsecp_error_t err, const char* msg) {
    ctx->last_err = err;
    if (msg) {
        /* Portable safe copy without MSVC deprecation warning */
        size_t i = 0;
        /* cppcheck-suppress arrayIndexOutOfBoundsCond ; i bounded by sizeof(last_msg)-1 */
        for (; i < sizeof(ctx->last_msg) - 1 && msg[i]; ++i) {
            ctx->last_msg[i] = msg[i];
}
        ctx->last_msg[i] = '\0';
    } else {
        ctx->last_msg[0] = '\0';
    }
    return err;
}

/* ===========================================================================
 * Internal helpers (same pattern as existing c_api, but with error model)
 * =========================================================================== */

// All scalar parsing uses the strict variants below.
// Message hashes (32-byte) are handled as raw byte arrays (no scalar reduction).

// Strict parser for secret keys: rejects 0, values >= n. No reduction.
static inline bool scalar_parse_strict_nonzero(const uint8_t b[32], Scalar& out) {
    std::array<uint8_t, 32> arr;
    std::memcpy(arr.data(), b, 32);
    return Scalar::parse_bytes_strict_nonzero(arr, out);
}

// Strict parser for tweaks: rejects values >= n, allows 0. No reduction.
static inline bool scalar_parse_strict(const uint8_t b[32], Scalar& out) {
    std::array<uint8_t, 32> arr;
    std::memcpy(arr.data(), b, 32);
    return Scalar::parse_bytes_strict(arr, out);
}

static inline void scalar_to_bytes(const Scalar& s, uint8_t out[32]) {
    auto arr = s.to_bytes();
    std::memcpy(out, arr.data(), 32);
}

static inline Point point_from_compressed(const uint8_t pub[33]) {
    // Strict: reject x >= p (no reduction)
    FE x;
    if (!FE::parse_bytes_strict(pub + 1, x)) return Point::infinity();

    /* y^2 = x^3 + 7 */
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto y2 = x3 + FE::from_uint64(7);

    /* sqrt via addition chain for (p+1)/4 */
    auto t = y2;
    auto a = t.square() * t;
    auto b = a.square() * t;
    auto c = b.square().square().square() * b;
    auto d = c.square().square().square() * b;
    auto e = d.square().square() * a;
    auto f = e;
    for (int i = 0; i < 11; ++i) f = f.square();
    f = f * e;
    auto g = f;
    for (int i = 0; i < 22; ++i) g = g.square();
    g = g * f;
    auto h = g;
    for (int i = 0; i < 44; ++i) h = h.square();
    h = h * g;
    auto j = h;
    for (int i = 0; i < 88; ++i) j = j.square();
    j = j * h;
    auto k = j;
    for (int i = 0; i < 44; ++i) k = k.square();
    k = k * g;
    auto m = k.square().square().square() * b;
    auto y = m;
    for (int i = 0; i < 23; ++i) y = y.square();
    y = y * f;
    for (int i = 0; i < 6; ++i) y = y.square();
    y = y * a;
    y = y.square().square();

    // Verify sqrt: y^2 must equal y2 (reject if x has no valid y on curve)
    if (y * y != y2) return Point::infinity();

    auto y_bytes = y.to_bytes();
    bool const y_is_odd = (y_bytes[31] & 1) != 0;
    bool const want_odd = (pub[0] == 0x03);
    if (y_is_odd != want_odd) {
        y = FE::from_uint64(0) - y;
}

    return Point::from_affine(x, y);
}

static inline void point_to_compressed(const Point& p, uint8_t out[33]) {
    auto comp = p.to_compressed();
    std::memcpy(out, comp.data(), 33);
}

static secp256k1::Network to_network(int n) {
    return n == UFSECP_NET_TESTNET ? secp256k1::Network::Testnet
                                   : secp256k1::Network::Mainnet;
}

/* ===========================================================================
 * Version / error (stateless, no ctx needed)
 * =========================================================================== */

unsigned int ufsecp_version(void) {
    return UFSECP_VERSION_PACKED;
}

unsigned int ufsecp_abi_version(void) {
    return UFSECP_ABI_VERSION;
}

const char* ufsecp_version_string(void) {
    return UFSECP_VERSION_STRING;
}

const char* ufsecp_error_str(ufsecp_error_t err) {
    switch (err) {
    case UFSECP_OK:                return "OK";
    case UFSECP_ERR_NULL_ARG:      return "NULL argument";
    case UFSECP_ERR_BAD_KEY:       return "invalid private key";
    case UFSECP_ERR_BAD_PUBKEY:    return "invalid public key";
    case UFSECP_ERR_BAD_SIG:       return "invalid signature";
    case UFSECP_ERR_BAD_INPUT:     return "malformed input";
    case UFSECP_ERR_VERIFY_FAIL:   return "verification failed";
    case UFSECP_ERR_ARITH:         return "arithmetic error";
    case UFSECP_ERR_SELFTEST:      return "self-test failed";
    case UFSECP_ERR_INTERNAL:      return "internal error";
    case UFSECP_ERR_BUF_TOO_SMALL: return "buffer too small";
    default:                       return "unknown error";
    }
}

/* ===========================================================================
 * Context lifecycle
 * =========================================================================== */

ufsecp_error_t ufsecp_ctx_create(ufsecp_ctx** ctx_out) {
    if (!ctx_out) return UFSECP_ERR_NULL_ARG;
    *ctx_out = nullptr;

    auto* ctx = static_cast<ufsecp_ctx*>(std::calloc(1, sizeof(ufsecp_ctx)));
    if (!ctx) return UFSECP_ERR_INTERNAL;

    ctx->last_err   = UFSECP_OK;
    ctx->last_msg[0] = '\0';

    /* Run selftest once (cached globally by ensure_library_integrity) */
    ctx->selftest_ok = secp256k1::fast::ensure_library_integrity(false);
    if (!ctx->selftest_ok) {
        std::free(ctx);
        return UFSECP_ERR_SELFTEST;
    }

    *ctx_out = ctx;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ctx_clone(const ufsecp_ctx* src, ufsecp_ctx** ctx_out) {
    if (!src || !ctx_out) return UFSECP_ERR_NULL_ARG;
    *ctx_out = nullptr;

    auto* dst = static_cast<ufsecp_ctx*>(std::malloc(sizeof(ufsecp_ctx)));
    if (!dst) return UFSECP_ERR_INTERNAL;

    std::memcpy(dst, src, sizeof(ufsecp_ctx));
    ctx_clear_err(dst);

    *ctx_out = dst;
    return UFSECP_OK;
}

void ufsecp_ctx_destroy(ufsecp_ctx* ctx) {
    std::free(ctx);  // free(NULL) is a no-op per C standard
}

ufsecp_error_t ufsecp_last_error(const ufsecp_ctx* ctx) {
    return ctx ? ctx->last_err : UFSECP_ERR_NULL_ARG;
}

const char* ufsecp_last_error_msg(const ufsecp_ctx* ctx) {
    if (!ctx) return "NULL context";
    return ctx->last_msg[0] ? ctx->last_msg : ufsecp_error_str(ctx->last_err);
}

size_t ufsecp_ctx_size(void) {
    return sizeof(ufsecp_ctx);
}

/* ===========================================================================
 * Private key utilities
 * =========================================================================== */

ufsecp_error_t ufsecp_seckey_verify(const ufsecp_ctx* ctx,
                                    const uint8_t privkey[32]) {
    if (!privkey) return UFSECP_ERR_NULL_ARG;
    (void)ctx;
    // BIP-340 strict: reject if privkey == 0 or privkey >= n (no reduction)
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        return UFSECP_ERR_BAD_KEY;
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_seckey_negate(ufsecp_ctx* ctx, uint8_t privkey[32]) {
    if (!ctx || !privkey) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto neg = sk.negate();
    // negate of valid nonzero scalar is always nonzero
    scalar_to_bytes(neg, privkey);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_seckey_tweak_add(ufsecp_ctx* ctx, uint8_t privkey[32],
                                       const uint8_t tweak[32]) {
    if (!ctx || !privkey || !tweak) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    Scalar tw;
    if (!scalar_parse_strict(tweak, tw)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak >= n");
    }
    auto result = sk + tw;
    if (result.is_zero()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_add resulted in zero");
    }
    scalar_to_bytes(result, privkey);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_seckey_tweak_mul(ufsecp_ctx* ctx, uint8_t privkey[32],
                                       const uint8_t tweak[32]) {
    if (!ctx || !privkey || !tweak) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    Scalar tw;
    // tweak_mul: reject tweak==0 (result would be zero) and tweak >= n
    if (!scalar_parse_strict_nonzero(tweak, tw)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak is zero or >= n");
    }
    auto result = sk * tw;
    if (result.is_zero()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_mul resulted in zero");
    }
    scalar_to_bytes(result, privkey);
    return UFSECP_OK;
}

/* ===========================================================================
 * Public key
 * =========================================================================== */

ufsecp_error_t ufsecp_pubkey_create(ufsecp_ctx* ctx,
                                    const uint8_t privkey[32],
                                    uint8_t pubkey33_out[33]) {
    if (!ctx || !privkey || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    // CT path: private key is secret, use constant-time generator_mul
    auto pk = secp256k1::ct::generator_mul(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "pubkey at infinity");
}

    point_to_compressed(pk, pubkey33_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_create_uncompressed(ufsecp_ctx* ctx,
                                                 const uint8_t privkey[32],
                                                 uint8_t pubkey65_out[65]) {
    if (!ctx || !privkey || !pubkey65_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    // CT path: private key is secret, use constant-time generator_mul
    auto pk = secp256k1::ct::generator_mul(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "pubkey at infinity");
}

    auto uncomp = pk.to_uncompressed();
    std::memcpy(pubkey65_out, uncomp.data(), 65);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_parse(ufsecp_ctx* ctx,
                                   const uint8_t* input, size_t input_len,
                                   uint8_t pubkey33_out[33]) {
    if (!ctx || !input || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (input_len == 33 && (input[0] == 0x02 || input[0] == 0x03)) {
        auto p = point_from_compressed(input);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "decompression failed");
}
        point_to_compressed(p, pubkey33_out);
        return UFSECP_OK;
    }
    if (input_len == 65 && input[0] == 0x04) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), input + 1, 32);
        std::memcpy(y_bytes.data(), input + 33, 32);
        // Strict: reject x >= p or y >= p
        FE x, y;
        if (!FE::parse_bytes_strict(x_bytes, x) ||
            !FE::parse_bytes_strict(y_bytes, y)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "coordinate >= p");
        }
        // On-curve check: y^2 == x^3 + 7
        auto lhs = y * y;
        auto rhs = x * x * x + FE::from_uint64(7);
        if (lhs != rhs) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "point not on curve");
        }
        auto p = Point::from_affine(x, y);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "point at infinity");
}
        point_to_compressed(p, pubkey33_out);
        return UFSECP_OK;
    }
    return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "expected 33 or 65 byte pubkey");
}

ufsecp_error_t ufsecp_pubkey_xonly(ufsecp_ctx* ctx,
                                   const uint8_t privkey[32],
                                   uint8_t xonly32_out[32]) {
    if (!ctx || !privkey || !xonly32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto xonly = secp256k1::schnorr_pubkey(sk);
    std::memcpy(xonly32_out, xonly.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * ECDSA
 * =========================================================================== */

ufsecp_error_t ufsecp_ecdsa_sign(ufsecp_ctx* ctx,
                                 const uint8_t msg32[32],
                                 const uint8_t privkey[32],
                                 uint8_t sig64_out[64]) {
    if (!ctx || !msg32 || !privkey || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    // CT path returns already-normalized (low-S) signature
    auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sign_verified(ufsecp_ctx* ctx,
                                          const uint8_t msg32[32],
                                          const uint8_t privkey[32],
                                          uint8_t sig64_out[64]) {
    if (!ctx || !msg32 || !privkey || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto sig = secp256k1::ct::ecdsa_sign_verified(msg, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_verify(ufsecp_ctx* ctx,
                                   const uint8_t msg32[32],
                                   const uint8_t sig64[64],
                                   const uint8_t pubkey33[33]) {
    if (!ctx || !msg32 || !sig64 || !pubkey33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature ecdsasig;
    if (!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }
    auto pk = point_from_compressed(pubkey33);

    if (!secp256k1::ecdsa_verify(msg, pk, ecdsasig)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECDSA verify failed");
    }

    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sig_to_der(ufsecp_ctx* ctx,
                                        const uint8_t sig64[64],
                                        uint8_t* der_out, size_t* der_len) {
    if (!ctx || !sig64 || !der_out || !der_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature ecdsasig;
    if (!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }

    auto [der, actual_len] = ecdsasig.to_der();
    if (*der_len < actual_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "DER buffer too small");
}

    std::memcpy(der_out, der.data(), actual_len);
    *der_len = actual_len;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sig_from_der(ufsecp_ctx* ctx,
                                         const uint8_t* der, size_t der_len,
                                         uint8_t sig64_out[64]) {
    if (!ctx || !der || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    /* Strict DER parser for ECDSA secp256k1 signatures.
     * Format: 0x30 <total_len> 0x02 <r_len> <r_bytes...> 0x02 <s_len> <s_bytes...>
     *
     * Enforces:
     * - Single-byte length encoding only (no long form)
     * - No negative integers (high bit of first data byte must be 0)
     * - No unnecessary leading zero padding
     * - Exact total length (no trailing bytes)
     * - r, s must be in [1, n-1] (canonical, nonzero)
     * - Max total DER length: 72 bytes */

    /* Max DER ECDSA sig: 2 + 2 + 33 + 2 + 33 = 72 */
    if (der_len < 8 || der_len > 72 || der[0] != 0x30) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: missing/oversized SEQUENCE");
    }

    /* Reject long-form length encoding (bit 7 set = multi-byte length) */
    if (der[1] & 0x80) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: long-form length");
    }

    size_t const seq_len = der[1];
    if (seq_len + 2 != der_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: length mismatch");
    }

    size_t pos = 2;

    /* --- Helper lambda: parse one INTEGER component strictly --- */
    auto parse_int = [&](const char* name, const uint8_t*& out_ptr,
                         size_t& out_len) -> ufsecp_error_t {
        if (pos >= der_len || der[pos] != 0x02) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: missing INTEGER");
        }
        pos++;
        if (pos >= der_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: truncated");
        }
        /* Reject long-form length for component */
        if (der[pos] & 0x80) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: long-form int length");
        }
        size_t const int_len = der[pos++];
        if (int_len == 0 || pos + int_len > der_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: int length out of bounds");
        }
        /* Reject negative: high bit set on first data byte means negative in DER */
        if (der[pos] & 0x80) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: negative integer");
        }
        /* Reject unnecessary leading zero: 0x00 prefix only valid when next byte
         * has high bit set (positive number needs padding to stay positive).
         * If next byte has high bit clear, the 0x00 is superfluous padding.  */
        if (int_len > 1 && der[pos] == 0x00 && !(der[pos + 1] & 0x80)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: unnecessary leading zero");
        }

        out_ptr = der + pos;
        out_len = int_len;
        /* Strip valid leading zero pad (high bit of next byte is set) */
        if (out_len > 0 && out_ptr[0] == 0x00) { out_ptr++; out_len--; }
        pos += int_len;
        (void)name;
        return UFSECP_OK;
    };

    /* Read R */
    const uint8_t* r_ptr = nullptr;
    size_t r_data_len = 0;
    {
        auto rc = parse_int("R", r_ptr, r_data_len);
        if (rc != UFSECP_OK) return rc;
    }

    /* Read S */
    const uint8_t* s_ptr = nullptr;
    size_t s_data_len = 0;
    {
        auto rc = parse_int("S", s_ptr, s_data_len);
        if (rc != UFSECP_OK) return rc;
    }

    /* Reject trailing bytes after S (must consume entire SEQUENCE) */
    if (pos != der_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: trailing bytes");
    }

    if (r_data_len > 32 || s_data_len > 32) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: component > 32 bytes");
    }

    /* Build compact sig64 (big-endian, right-aligned in 32-byte slots) */
    std::memset(sig64_out, 0, 64);
    /* Explicit null checks for static analyzer (r_ptr/s_ptr guaranteed non-null
     * when *_data_len > 0 by parse_int() success, but SonarCloud can't track it) */
    if (r_data_len > 0 && r_ptr) {
        std::memcpy(sig64_out + (32 - r_data_len), r_ptr, r_data_len);
    }
    if (s_data_len > 0 && s_ptr) {
        std::memcpy(sig64_out + 32 + (32 - s_data_len), s_ptr, s_data_len);
    }

    /* Range check: r and s must be in [1, n-1] (strict nonzero, no reduce) */
    Scalar r_sc, s_sc;
    if (!Scalar::parse_bytes_strict_nonzero(sig64_out, r_sc) ||
        !Scalar::parse_bytes_strict_nonzero(sig64_out + 32, s_sc)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: r or s out of range [1,n-1]");
    }

    return UFSECP_OK;
}

/* -- ECDSA Recovery -------------------------------------------------------- */

ufsecp_error_t ufsecp_ecdsa_sign_recoverable(ufsecp_ctx* ctx,
                                             const uint8_t msg32[32],
                                             const uint8_t privkey[32],
                                             uint8_t sig64_out[64],
                                             int* recid_out) {
    if (!ctx || !msg32 || !privkey || !sig64_out || !recid_out) {
        return UFSECP_ERR_NULL_ARG;
}
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    // NOTE: No ct::ecdsa_sign_recoverable exists yet. Using fast path with
    // zeroization. Recovery signing is inherently non-CT due to recid computation.
    // TODO: Implement ct::ecdsa_sign_recoverable when CT recovery is needed.
    auto rsig = secp256k1::ecdsa_sign_recoverable(msg, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto normalized = rsig.sig.normalize();
    auto compact = normalized.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    *recid_out = rsig.recid;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_recover(ufsecp_ctx* ctx,
                                    const uint8_t msg32[32],
                                    const uint8_t sig64[64],
                                    int recid,
                                    uint8_t pubkey33_out[33]) {
    if (!ctx || !msg32 || !sig64 || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (recid < 0 || recid > 3) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "recid must be 0..3");
    }

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature ecdsasig;
    if (!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }

    auto [point, ok] = secp256k1::ecdsa_recover(msg, ecdsasig, recid);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "recovery failed");
    }

    point_to_compressed(point, pubkey33_out);
    return UFSECP_OK;
}

/* ===========================================================================
 * Schnorr (BIP-340)
 * =========================================================================== */

ufsecp_error_t ufsecp_schnorr_sign(ufsecp_ctx* ctx,
                                   const uint8_t msg32[32],
                                   const uint8_t privkey[32],
                                   const uint8_t aux_rand[32],
                                   uint8_t sig64_out[64]) {
    if (!ctx || !msg32 || !privkey || !aux_rand || !sig64_out) {
        return UFSECP_ERR_NULL_ARG;
}
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_sign_verified(ufsecp_ctx* ctx,
                                            const uint8_t msg32[32],
                                            const uint8_t privkey[32],
                                            const uint8_t aux_rand[32],
                                            uint8_t sig64_out[64]) {
    if (!ctx || !msg32 || !privkey || !aux_rand || !sig64_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    auto sig = secp256k1::ct::schnorr_sign_verified(kp, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_verify(ufsecp_ctx* ctx,
                                     const uint8_t msg32[32],
                                     const uint8_t sig64[64],
                                     const uint8_t pubkey_x[32]) {
    if (!ctx || !msg32 || !sig64 || !pubkey_x) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    // BIP-340 strict parse: reject non-canonical r >= p, s >= n, or s == 0
    secp256k1::SchnorrSignature schnorr_sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, schnorr_sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "Non-canonical Schnorr sig (r>=p or s>=n)");
    }

    // BIP-340 strict: reject pubkey x >= p
    FE pk_fe;
    if (!FE::parse_bytes_strict(pubkey_x, pk_fe)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "Non-canonical pubkey (x>=p)");
    }

    std::array<uint8_t, 32> pk_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey_x, 32);
    std::memcpy(msg_arr.data(), msg32, 32);

    if (!secp256k1::schnorr_verify(pk_arr, msg_arr, schnorr_sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "Schnorr verify failed");
}

    return UFSECP_OK;
}

/* ===========================================================================
 * ECDH
 * =========================================================================== */

ufsecp_error_t ufsecp_ecdh(ufsecp_ctx* ctx,
                           const uint8_t privkey[32],
                           const uint8_t pubkey33[33],
                           uint8_t secret32_out[32]) {
    if (!ctx || !privkey || !pubkey33 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid or infinity pubkey");
    }
    auto secret = secp256k1::ecdh_compute(sk, pk);
    std::memcpy(secret32_out, secret.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh_xonly(ufsecp_ctx* ctx,
                                 const uint8_t privkey[32],
                                 const uint8_t pubkey33[33],
                                 uint8_t secret32_out[32]) {
    if (!ctx || !privkey || !pubkey33 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid or infinity pubkey");
    }
    auto secret = secp256k1::ecdh_compute_xonly(sk, pk);
    std::memcpy(secret32_out, secret.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh_raw(ufsecp_ctx* ctx,
                               const uint8_t privkey[32],
                               const uint8_t pubkey33[33],
                               uint8_t secret32_out[32]) {
    if (!ctx || !privkey || !pubkey33 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid or infinity pubkey");
    }
    auto secret = secp256k1::ecdh_compute_raw(sk, pk);
    std::memcpy(secret32_out, secret.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * Hashing (stateless -- no ctx required, but returns error_t for consistency)
 * =========================================================================== */

ufsecp_error_t ufsecp_sha256(const uint8_t* data, size_t len,
                             uint8_t digest32_out[32]) {
    if (!data || !digest32_out) return UFSECP_ERR_NULL_ARG;
    secp256k1::SHA256 hasher;
    hasher.update(data, len);
    auto digest = hasher.finalize();
    std::memcpy(digest32_out, digest.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_hash160(const uint8_t* data, size_t len,
                              uint8_t digest20_out[20]) {
    if (!data || !digest20_out) return UFSECP_ERR_NULL_ARG;
    auto h = secp256k1::hash160(data, len);
    std::memcpy(digest20_out, h.data(), 20);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_tagged_hash(const char* tag,
                                  const uint8_t* data, size_t len,
                                  uint8_t digest32_out[32]) {
    if (!tag || !data || !digest32_out) return UFSECP_ERR_NULL_ARG;
    auto h = secp256k1::tagged_hash(tag, data, len);
    std::memcpy(digest32_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * Bitcoin addresses
 * =========================================================================== */

ufsecp_error_t ufsecp_addr_p2pkh(ufsecp_ctx* ctx,
                                 const uint8_t pubkey33[33], int network,
                                 char* addr_out, size_t* addr_len) {
    if (!ctx || !pubkey33 || !addr_out || !addr_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    auto pk = point_from_compressed(pubkey33);
    auto addr = secp256k1::address_p2pkh(pk, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2PKH generation failed");
}
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2PKH buffer too small");
}
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_addr_p2wpkh(ufsecp_ctx* ctx,
                                  const uint8_t pubkey33[33], int network,
                                  char* addr_out, size_t* addr_len) {
    if (!ctx || !pubkey33 || !addr_out || !addr_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    auto pk = point_from_compressed(pubkey33);
    auto addr = secp256k1::address_p2wpkh(pk, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2WPKH generation failed");
}
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2WPKH buffer too small");
}
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_addr_p2tr(ufsecp_ctx* ctx,
                                const uint8_t internal_key_x[32], int network,
                                char* addr_out, size_t* addr_len) {
    if (!ctx || !internal_key_x || !addr_out || !addr_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> key_x;
    std::memcpy(key_x.data(), internal_key_x, 32);
    auto addr = secp256k1::address_p2tr_raw(key_x, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2TR generation failed");
}
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2TR buffer too small");
}
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
}

/* ===========================================================================
 * WIF
 * =========================================================================== */

ufsecp_error_t ufsecp_wif_encode(ufsecp_ctx* ctx,
                                 const uint8_t privkey[32],
                                 int compressed, int network,
                                 char* wif_out, size_t* wif_len) {
    if (!ctx || !privkey || !wif_out || !wif_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto wif = secp256k1::wif_encode(sk, compressed != 0, to_network(network));
    if (wif.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "WIF encode failed");
}
    if (*wif_len < wif.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "WIF buffer too small");
}
    std::memcpy(wif_out, wif.c_str(), wif.size() + 1);
    *wif_len = wif.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_wif_decode(ufsecp_ctx* ctx,
                                 const char* wif,
                                 uint8_t privkey32_out[32],
                                 int* compressed_out,
                                 int* network_out) {
    if (!ctx || !wif || !privkey32_out || !compressed_out || !network_out) {
        return UFSECP_ERR_NULL_ARG;
}
    ctx_clear_err(ctx);

    auto result = secp256k1::wif_decode(std::string(wif));
    if (!result.valid) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid WIF string");
}

    scalar_to_bytes(result.key, privkey32_out);
    *compressed_out = result.compressed ? 1 : 0;
    *network_out = result.network == secp256k1::Network::Testnet
                       ? UFSECP_NET_TESTNET : UFSECP_NET_MAINNET;
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-32
 * =========================================================================== */

static void extkey_to_uf(const secp256k1::ExtendedKey& ek, ufsecp_bip32_key* out) {
    auto serialized = ek.serialize();
    std::memcpy(out->data, serialized.data(), 78);
    out->is_private = ek.is_private ? 1 : 0;
    std::memset(out->_pad, 0, sizeof(out->_pad));
}

static secp256k1::ExtendedKey extkey_from_uf(const ufsecp_bip32_key* k) {
    secp256k1::ExtendedKey ek{};
    ek.depth = k->data[4];
    std::memcpy(ek.parent_fingerprint.data(), k->data + 5, 4);
    ek.child_number = (uint32_t(k->data[9]) << 24)  | (uint32_t(k->data[10]) << 16) |
                      (uint32_t(k->data[11]) << 8)   | uint32_t(k->data[12]);
    std::memcpy(ek.chain_code.data(), k->data + 13, 32);
    if (k->is_private) {
        std::memcpy(ek.key.data(), k->data + 46, 32);
        ek.is_private = true;
    } else {
        std::memcpy(ek.key.data(), k->data + 46, 32);
        ek.is_private = false;
    }
    return ek;
}

ufsecp_error_t ufsecp_bip32_master(ufsecp_ctx* ctx,
                                   const uint8_t* seed, size_t seed_len,
                                   ufsecp_bip32_key* key_out) {
    if (!ctx || !seed || !key_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (seed_len < 16 || seed_len > 64) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "seed must be 16-64 bytes");
}

    auto [ek, ok] = secp256k1::bip32_master_key(seed, seed_len);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 master key failed");
}

    extkey_to_uf(ek, key_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_derive(ufsecp_ctx* ctx,
                                   const ufsecp_bip32_key* parent,
                                   uint32_t index,
                                   ufsecp_bip32_key* child_out) {
    if (!ctx || !parent || !child_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    auto ek = extkey_from_uf(parent);
    auto [child, ok] = ek.derive_child(index);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 derivation failed");
}

    extkey_to_uf(child, child_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_derive_path(ufsecp_ctx* ctx,
                                        const ufsecp_bip32_key* master,
                                        const char* path,
                                        ufsecp_bip32_key* key_out) {
    if (!ctx || !master || !path || !key_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    auto ek = extkey_from_uf(master);
    auto [derived, ok] = secp256k1::bip32_derive_path(ek, std::string(path));
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 path");
}

    extkey_to_uf(derived, key_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_privkey(ufsecp_ctx* ctx,
                                    const ufsecp_bip32_key* key,
                                    uint8_t privkey32_out[32]) {
    if (!ctx || !key || !privkey32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (!key->is_private) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "key is public, not private");
}

    auto ek = extkey_from_uf(key);
    auto sk = ek.private_key();
    scalar_to_bytes(sk, privkey32_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_pubkey(ufsecp_ctx* ctx,
                                   const ufsecp_bip32_key* key,
                                   uint8_t pubkey33_out[33]) {
    if (!ctx || !key || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    auto ek = extkey_from_uf(key);
    auto pk = ek.public_key();
    point_to_compressed(pk, pubkey33_out);
    return UFSECP_OK;
}

/* ===========================================================================
 * Taproot (BIP-341)
 * =========================================================================== */

ufsecp_error_t ufsecp_taproot_output_key(ufsecp_ctx* ctx,
                                         const uint8_t internal_x[32],
                                         const uint8_t* merkle_root,
                                         uint8_t output_x_out[32],
                                         int* parity_out) {
    if (!ctx || !internal_x || !output_x_out || !parity_out) {
        return UFSECP_ERR_NULL_ARG;
}
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> ik;
    std::memcpy(ik.data(), internal_x, 32);
    size_t const mr_len = merkle_root ? 32 : 0;

    auto [ok_x, parity] = secp256k1::taproot_output_key(ik, merkle_root, mr_len);
    std::memcpy(output_x_out, ok_x.data(), 32);
    *parity_out = parity;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_taproot_tweak_seckey(ufsecp_ctx* ctx,
                                           const uint8_t privkey[32],
                                           const uint8_t* merkle_root,
                                           uint8_t tweaked32_out[32]) {
    if (!ctx || !privkey || !tweaked32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    size_t const mr_len = merkle_root ? 32 : 0;

    auto tweaked = secp256k1::taproot_tweak_privkey(sk, merkle_root, mr_len);
    if (tweaked.is_zero()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "taproot tweak resulted in zero");
}

    scalar_to_bytes(tweaked, tweaked32_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_taproot_verify(ufsecp_ctx* ctx,
                                     const uint8_t output_x[32], int output_parity,
                                     const uint8_t internal_x[32],
                                     const uint8_t* merkle_root, size_t merkle_root_len) {
    if (!ctx || !output_x || !internal_x) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> ok_x, ik_x;
    std::memcpy(ok_x.data(), output_x, 32);
    std::memcpy(ik_x.data(), internal_x, 32);

    if (!secp256k1::taproot_verify_commitment(ok_x, output_parity, ik_x,
                                              merkle_root, merkle_root_len)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "taproot commitment invalid");
}

    return UFSECP_OK;
}
