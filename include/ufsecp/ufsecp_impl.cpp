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
#include <limits>
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
#include "secp256k1/bip143.hpp"
#include "secp256k1/bip144.hpp"
#include "secp256k1/segwit.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/bip39.hpp"
#include "secp256k1/batch_verify.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/frost.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/pedersen.hpp"
#include "secp256k1/zk.hpp"
#include "secp256k1/sha512.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/coins/coin_params.hpp"
#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/ecies.hpp"
#include "secp256k1/coins/coin_hd.hpp"
#include "secp256k1/coins/message_signing.hpp"

#if defined(SECP256K1_BIP324)
#include "secp256k1/chacha20_poly1305.hpp"
#include "secp256k1/hkdf.hpp"
#include "secp256k1/ellswift.hpp"
#include "secp256k1/bip324.hpp"
#endif

#if defined(SECP256K1_BUILD_ETHEREUM)
#include "secp256k1/coins/keccak256.hpp"
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/coins/eth_signing.hpp"
#endif

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

static inline Point point_from_compressed(const uint8_t pub[33]);

namespace {

constexpr std::size_t kMuSig2KeyAggHeaderLen = 38;
constexpr std::size_t kMuSig2KeyAggCoeffLen = 32;
constexpr std::size_t kMuSig2SessionSerializedLen = 98;
constexpr std::size_t kMuSig2SessionCountOffset = kMuSig2SessionSerializedLen;
constexpr std::size_t kMuSig2SessionCountLen = 4;
constexpr uint32_t kMuSig2MinParticipants = 2;
constexpr uint32_t kMuSig2MaxKeyAggParticipants =
    static_cast<uint32_t>((UFSECP_MUSIG2_KEYAGG_LEN - kMuSig2KeyAggHeaderLen) / kMuSig2KeyAggCoeffLen);

static_assert(kMuSig2MaxKeyAggParticipants >= kMuSig2MinParticipants,
              "MuSig2 keyagg blob must encode at least two participants");
static_assert(kMuSig2SessionCountOffset + kMuSig2SessionCountLen <= UFSECP_MUSIG2_SESSION_LEN,
              "MuSig2 session blob must have room for participant count metadata");

static ufsecp_error_t parse_musig2_keyagg(ufsecp_ctx* ctx,
                                          const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
                                          secp256k1::MuSig2KeyAggCtx& out) {
    uint32_t nk = 0;
    std::memcpy(&nk, keyagg, 4);
    if (nk < kMuSig2MinParticipants || nk > kMuSig2MaxKeyAggParticipants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid keyagg participant count");
    }

    out.Q_negated = (keyagg[4] != 0);
    out.Q = point_from_compressed(keyagg + 5);
    if (out.Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid aggregated key");
    }

    auto qc = out.Q.to_compressed();
    std::memcpy(out.Q_x.data(), qc.data() + 1, 32);
    out.key_coefficients.clear();
    out.key_coefficients.reserve(nk);
    for (uint32_t i = 0; i < nk; ++i) {
        Scalar coefficient;
        if (!scalar_parse_strict(keyagg + kMuSig2KeyAggHeaderLen + static_cast<std::size_t>(i) * kMuSig2KeyAggCoeffLen,
                                 coefficient)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid key coefficient in keyagg");
        }
        out.key_coefficients.push_back(coefficient);
    }
    return UFSECP_OK;
}

static ufsecp_error_t parse_musig2_session(ufsecp_ctx* ctx,
                                           const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
                                           secp256k1::MuSig2Session& out,
                                           uint32_t& participant_count_out) {
    out.R = point_from_compressed(session);
    if (out.R.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid session R point");
    }
    if (!scalar_parse_strict(session + 33, out.b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid session scalar b");
    }
    if (!scalar_parse_strict(session + 65, out.e)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid session scalar e");
    }
    out.R_negated = (session[97] != 0);

    std::memcpy(&participant_count_out, session + kMuSig2SessionCountOffset, sizeof(participant_count_out));
    if (participant_count_out < kMuSig2MinParticipants || participant_count_out > kMuSig2MaxKeyAggParticipants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid session participant count");
    }
    return UFSECP_OK;
}

static bool checked_mul_size(std::size_t left, std::size_t right, std::size_t& out) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    out = left * right;
    return true;
}

static bool checked_add_size(std::size_t left, std::size_t right, std::size_t& out) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    out = left + right;
    return true;
}

} // namespace

static inline Point point_from_compressed(const uint8_t pub[33]) {
    // Strict: only accept 0x02/0x03 prefix, reject x >= p
    if (pub[0] != 0x02 && pub[0] != 0x03) return Point::infinity();
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

template <typename T>
class SecureEraseGuard {
public:
    explicit SecureEraseGuard(T* value) noexcept : value_(value) {}
    SecureEraseGuard(const SecureEraseGuard&) = delete;
    SecureEraseGuard& operator=(const SecureEraseGuard&) = delete;

    ~SecureEraseGuard() {
        if (value_ != nullptr) {
            secp256k1::detail::secure_erase(value_, sizeof(T));
        }
    }

private:
    T* value_;
};

static inline void secure_erase_scalar_vector(std::vector<Scalar>& values) {
    for (auto& value : values) {
        secp256k1::detail::secure_erase(&value, sizeof(value));
    }
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
    if (!ctx || !privkey) return UFSECP_ERR_NULL_ARG;
    // BIP-340 strict: reject if privkey == 0 or privkey >= n (no reduction)
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        return UFSECP_ERR_BAD_KEY;
    }
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
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
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    // negate of valid nonzero scalar is always nonzero
    scalar_to_bytes(neg, privkey);
    secp256k1::detail::secure_erase(&neg, sizeof(neg));
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
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&tw, sizeof(tw));
    if (result.is_zero()) {
        secp256k1::detail::secure_erase(&result, sizeof(result));
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_add resulted in zero");
    }
    scalar_to_bytes(result, privkey);
    secp256k1::detail::secure_erase(&result, sizeof(result));
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
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&tw, sizeof(tw));
    if (result.is_zero()) {
        secp256k1::detail::secure_erase(&result, sizeof(result));
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_mul resulted in zero");
    }
    scalar_to_bytes(result, privkey);
    secp256k1::detail::secure_erase(&result, sizeof(result));
    return UFSECP_OK;
}

/* ===========================================================================
 * Public key
 * =========================================================================== */

static ufsecp_error_t pubkey_create_core(ufsecp_ctx* ctx,
                                         const uint8_t privkey[32],
                                         Point& pk_out) {
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    pk_out = secp256k1::ct::generator_mul(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (pk_out.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "pubkey at infinity");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_create(ufsecp_ctx* ctx,
                                    const uint8_t privkey[32],
                                    uint8_t pubkey33_out[33]) {
    if (!ctx || !privkey || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Point pk;
    const ufsecp_error_t err = pubkey_create_core(ctx, privkey, pk);
    if (err != UFSECP_OK) return err;
    point_to_compressed(pk, pubkey33_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_create_uncompressed(ufsecp_ctx* ctx,
                                                 const uint8_t privkey[32],
                                                 uint8_t pubkey65_out[65]) {
    if (!ctx || !privkey || !pubkey65_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Point pk;
    const ufsecp_error_t err = pubkey_create_core(ctx, privkey, pk);
    if (err != UFSECP_OK) return err;
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
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
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
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid public key");
    }

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

    // ARCH DECISION: No ct::ecdsa_sign_recoverable exists because recovery signing is
    // inherently non-constant-time — the recid value (0..3) depends on the R point's x
    // coordinate, leaking timing. We use the FAST path (secp256k1::ecdsa_sign_recoverable)
    // with explicit zeroization of the private-key scalar immediately after use.
    // If a future CT recovery path is needed, it must accept a fixed recid hint from the
    // caller and branch only on public data.
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

ufsecp_error_t ufsecp_ecdsa_sign_batch(
    ufsecp_ctx* ctx,
    size_t count,
    const uint8_t* msgs32,
    const uint8_t* privkeys32,
    uint8_t* sigs64_out)
{
    if (!ctx || !msgs32 || !privkeys32 || !sigs64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    for (size_t i = 0; i < count; ++i) {
        std::array<uint8_t, 32> msg;
        std::memcpy(msg.data(), msgs32 + i * 32, 32);
        Scalar sk;
        if (!scalar_parse_strict_nonzero(privkeys32 + i * 32, sk)) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY,
                               "privkey[i] is zero or >= n");
        }
        auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        auto compact = sig.to_compact();
        std::memcpy(sigs64_out + i * 64, compact.data(), 64);
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_sign_batch(
    ufsecp_ctx* ctx,
    size_t count,
    const uint8_t* msgs32,
    const uint8_t* privkeys32,
    const uint8_t* aux_rands32,
    uint8_t* sigs64_out)
{
    if (!ctx || !msgs32 || !privkeys32 || !sigs64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    static constexpr uint8_t kZeroAux[32] = {};

    for (size_t i = 0; i < count; ++i) {
        Scalar sk;
        if (!scalar_parse_strict_nonzero(privkeys32 + i * 32, sk)) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY,
                               "privkey[i] is zero or >= n");
        }

        std::array<uint8_t, 32> msg_arr, aux_arr;
        std::memcpy(msg_arr.data(), msgs32 + i * 32, 32);
        const uint8_t* aux_src = aux_rands32 ? aux_rands32 + i * 32 : kZeroAux;
        std::memcpy(aux_arr.data(), aux_src, 32);

        auto kp  = secp256k1::ct::schnorr_keypair_create(sk);
        auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));

        auto sig_bytes = sig.to_bytes();
        std::memcpy(sigs64_out + i * 64, sig_bytes.data(), 64);
    }
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

static ufsecp_error_t ecdh_parse_args(ufsecp_ctx* ctx,
                                      const uint8_t privkey[32],
                                      const uint8_t pubkey33[33],
                                      Scalar& sk, Point& pk) {
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid or infinity pubkey");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh(ufsecp_ctx* ctx,
                           const uint8_t privkey[32],
                           const uint8_t pubkey33[33],
                           uint8_t secret32_out[32]) {
    if (!ctx || !privkey || !pubkey33 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk; Point pk;
    const ufsecp_error_t err = ecdh_parse_args(ctx, privkey, pubkey33, sk, pk);
    if (err != UFSECP_OK) return err;
    auto secret = secp256k1::ecdh_compute(sk, pk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(secret32_out, secret.data(), 32);
    secp256k1::detail::secure_erase(secret.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh_xonly(ufsecp_ctx* ctx,
                                 const uint8_t privkey[32],
                                 const uint8_t pubkey33[33],
                                 uint8_t secret32_out[32]) {
    if (!ctx || !privkey || !pubkey33 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk; Point pk;
    const ufsecp_error_t err = ecdh_parse_args(ctx, privkey, pubkey33, sk, pk);
    if (err != UFSECP_OK) return err;
    auto secret = secp256k1::ecdh_compute_xonly(sk, pk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(secret32_out, secret.data(), 32);
    secp256k1::detail::secure_erase(secret.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh_raw(ufsecp_ctx* ctx,
                               const uint8_t privkey[32],
                               const uint8_t pubkey33[33],
                               uint8_t secret32_out[32]) {
    if (!ctx || !privkey || !pubkey33 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk; Point pk;
    const ufsecp_error_t err = ecdh_parse_args(ctx, privkey, pubkey33, sk, pk);
    if (err != UFSECP_OK) return err;
    auto secret = secp256k1::ecdh_compute_raw(sk, pk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(secret32_out, secret.data(), 32);
    secp256k1::detail::secure_erase(secret.data(), 32);
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
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
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
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
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

    // Reject all-zero x-only key (not a valid curve point)
    {
        uint8_t acc = 0;
        for (int i = 0; i < 32; ++i) acc |= internal_key_x[i];
        if (acc == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "zero x-only key");
    }

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
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
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
    secp256k1::detail::secure_erase(&result.key, sizeof(result.key));
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
    std::memcpy(ek.key.data(), k->data + 46, 32);
    if (k->is_private) {
        ek.is_private = true;
    } else {
        ek.is_private = false;
        ek.pub_prefix = k->data[45];
    }
    return ek;
}

static ufsecp_error_t parse_bip32_key(ufsecp_ctx* ctx,
                                      const ufsecp_bip32_key* key,
                                      secp256k1::ExtendedKey& out) {
    if (key->is_private > 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 key kind");
    }
    if (key->_pad[0] != 0 || key->_pad[1] != 0 || key->_pad[2] != 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 reserved bytes");
    }

    const uint32_t version = (uint32_t(key->data[0]) << 24) |
                             (uint32_t(key->data[1]) << 16) |
                             (uint32_t(key->data[2]) << 8) |
                             uint32_t(key->data[3]);
    const uint32_t expected_version = key->is_private ? 0x0488ADE4u : 0x0488B21Eu;
    if (version != expected_version) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 version");
    }

    if (key->is_private != 0) {
        if (key->data[45] != 0x00) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 private marker");
        }
        Scalar sk;
        if (!scalar_parse_strict_nonzero(key->data + 46, sk)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid BIP-32 private key");
        }
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
    } else {
        if (key->data[45] != 0x02 && key->data[45] != 0x03) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid BIP-32 public key prefix");
        }
        auto pk = point_from_compressed(key->data + 45);
        if (pk.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid BIP-32 public key");
        }
    }

    out = extkey_from_uf(key);
    return UFSECP_OK;
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
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_derive(ufsecp_ctx* ctx,
                                   const ufsecp_bip32_key* parent,
                                   uint32_t index,
                                   ufsecp_bip32_key* child_out) {
    if (!ctx || !parent || !child_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, parent, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto [child, ok] = ek.derive_child(index);
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 derivation failed");
}

    extkey_to_uf(child, child_out);
    secp256k1::detail::secure_erase(child.key.data(), child.key.size());
    secp256k1::detail::secure_erase(child.chain_code.data(), child.chain_code.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_derive_path(ufsecp_ctx* ctx,
                                        const ufsecp_bip32_key* master,
                                        const char* path,
                                        ufsecp_bip32_key* key_out) {
    if (!ctx || !master || !path || !key_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, master, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto [derived, ok] = secp256k1::bip32_derive_path(ek, std::string(path));
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 path");
}

    extkey_to_uf(derived, key_out);
    secp256k1::detail::secure_erase(derived.key.data(), derived.key.size());
    secp256k1::detail::secure_erase(derived.chain_code.data(), derived.chain_code.size());
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

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, key, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto sk = ek.private_key();
    scalar_to_bytes(sk, privkey32_out);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_pubkey(ufsecp_ctx* ctx,
                                   const ufsecp_bip32_key* key,
                                   uint8_t pubkey33_out[33]) {
    if (!ctx || !key || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, key, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto pk = ek.public_key();
    if (pk.is_infinity()) {
        secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
        secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid BIP-32 public key");
    }
    point_to_compressed(pk, pubkey33_out);
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
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

    // Reject all-zero x-only key (not a valid curve point)
    {
        uint8_t acc = 0;
        for (int i = 0; i < 32; ++i) acc |= internal_x[i];
        if (acc == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "zero x-only key");
    }

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
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (tweaked.is_zero()) {
        secp256k1::detail::secure_erase(&tweaked, sizeof(tweaked));
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "taproot tweak resulted in zero");
}

    scalar_to_bytes(tweaked, tweaked32_out);
    secp256k1::detail::secure_erase(&tweaked, sizeof(tweaked));
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

/* ===========================================================================
 * BIP-143: SegWit v0 Sighash
 * =========================================================================== */

ufsecp_error_t ufsecp_bip143_sighash(
    ufsecp_ctx* ctx,
    uint32_t version,
    const uint8_t hash_prevouts[32],
    const uint8_t hash_sequence[32],
    const uint8_t outpoint_txid[32], uint32_t outpoint_vout,
    const uint8_t* script_code, size_t script_code_len,
    uint64_t value,
    uint32_t sequence,
    const uint8_t hash_outputs[32],
    uint32_t locktime,
    uint32_t sighash_type,
    uint8_t sighash_out[32]) {
    if (!ctx || !hash_prevouts || !hash_sequence || !outpoint_txid ||
        !script_code || !hash_outputs || !sighash_out)
        return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::Bip143Preimage pre{};
    pre.version = version;
    std::memcpy(pre.hash_prevouts.data(), hash_prevouts, 32);
    std::memcpy(pre.hash_sequence.data(), hash_sequence, 32);
    std::memcpy(pre.hash_outputs.data(),  hash_outputs,  32);
    pre.locktime = locktime;

    secp256k1::Outpoint op{};
    std::memcpy(op.txid.data(), outpoint_txid, 32);
    op.vout = outpoint_vout;

    auto h = secp256k1::bip143_sighash(pre, op, script_code, script_code_len,
                                        value, sequence, sighash_type);
    std::memcpy(sighash_out, h.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip143_p2wpkh_script_code(
    const uint8_t pubkey_hash[20],
    uint8_t script_code_out[25]) {
    if (!pubkey_hash || !script_code_out) return UFSECP_ERR_NULL_ARG;

    auto sc = secp256k1::bip143_p2wpkh_script_code(pubkey_hash);
    std::memcpy(script_code_out, sc.data(), 25);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-144: Witness Transaction Serialization
 * =========================================================================== */

// Helper: read Bitcoin CompactSize from buffer; returns 0 on overflow
static size_t read_compact_size(const uint8_t* buf, size_t len,
                                size_t& offset, uint64_t& val) {
    if (offset >= len) return 0;
    uint8_t first = buf[offset++];
    if (first < 0xFD) { val = first; return 1; }
    if (first == 0xFD) {
        if (offset + 2 > len) return 0;
        val = uint64_t(buf[offset]) | (uint64_t(buf[offset+1]) << 8);
        offset += 2; return 3;
    }
    if (first == 0xFE) {
        if (offset + 4 > len) return 0;
        val = uint64_t(buf[offset]) | (uint64_t(buf[offset+1]) << 8) |
              (uint64_t(buf[offset+2]) << 16) | (uint64_t(buf[offset+3]) << 24);
        offset += 4; return 5;
    }
    // 0xFF
    if (offset + 8 > len) return 0;
    val = 0;
    for (int i = 0; i < 8; ++i) val |= uint64_t(buf[offset+i]) << (8*i);
    offset += 8; return 9;
}

// Helper: skip CompactSize-prefixed blob (e.g. scriptSig or scriptPubKey)
static bool skip_compact_bytes(const uint8_t* buf, size_t len, size_t& offset) {
    uint64_t sz = 0;
    if (!read_compact_size(buf, len, offset, sz)) return false;
    if (offset + sz > len) return false;
    offset += static_cast<size_t>(sz);
    return true;
}

ufsecp_error_t ufsecp_bip144_txid(
    ufsecp_ctx* ctx,
    const uint8_t* raw_tx, size_t raw_tx_len,
    uint8_t txid_out[32]) {
    if (!ctx || !raw_tx || !txid_out) return UFSECP_ERR_NULL_ARG;
    if (raw_tx_len < 10) return UFSECP_ERR_BAD_INPUT;

    // Detect witness flag: version(4) + marker(0x00) + flag(0x01)
    bool has_witness = (raw_tx_len > 6 && raw_tx[4] == 0x00 && raw_tx[5] == 0x01);

    if (!has_witness) {
        // Legacy tx: txid = double-SHA256 of the entire raw bytes
        auto h = secp256k1::SHA256::hash256(raw_tx, raw_tx_len);
        std::memcpy(txid_out, h.data(), 32);
        return UFSECP_OK;
    }

    // Witness tx: strip marker+flag and witness data
    // Legacy = version(4) | inputs | outputs | locktime(4)
    secp256k1::SHA256 h1;
    // version
    h1.update(raw_tx, 4);

    // Skip marker+flag, parse inputs+outputs from offset 6
    size_t off = 6;
    uint64_t n_in = 0;
    size_t cs_start = off;
    if (!read_compact_size(raw_tx, raw_tx_len, off, n_in)) return UFSECP_ERR_BAD_INPUT;

    // Record start of vin count for hashing
    size_t io_start = cs_start;

    // Skip all inputs (each: txid(32) + vout(4) + scriptSig + sequence(4))
    for (uint64_t i = 0; i < n_in; ++i) {
        if (off + 36 > raw_tx_len) return UFSECP_ERR_BAD_INPUT;
        off += 36; // txid + vout
        if (!skip_compact_bytes(raw_tx, raw_tx_len, off)) return UFSECP_ERR_BAD_INPUT;
        if (off + 4 > raw_tx_len) return UFSECP_ERR_BAD_INPUT;
        off += 4; // sequence
    }

    // Parse outputs count
    uint64_t n_out = 0;
    if (!read_compact_size(raw_tx, raw_tx_len, off, n_out)) return UFSECP_ERR_BAD_INPUT;

    // Skip all outputs (each: value(8) + scriptPubKey)
    for (uint64_t i = 0; i < n_out; ++i) {
        if (off + 8 > raw_tx_len) return UFSECP_ERR_BAD_INPUT;
        off += 8; // value
        if (!skip_compact_bytes(raw_tx, raw_tx_len, off)) return UFSECP_ERR_BAD_INPUT;
    }
    size_t io_end = off;

    // Hash inputs+outputs section (cs_start..io_end)
    h1.update(raw_tx + io_start, io_end - io_start);

    // locktime = last 4 bytes
    if (raw_tx_len < 4) return UFSECP_ERR_BAD_INPUT;
    h1.update(raw_tx + raw_tx_len - 4, 4);

    auto first = h1.finalize();
    secp256k1::SHA256 h2;
    h2.update(first.data(), 32);
    auto txid = h2.finalize();
    std::memcpy(txid_out, txid.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip144_wtxid(
    ufsecp_ctx* ctx,
    const uint8_t* raw_tx, size_t raw_tx_len,
    uint8_t wtxid_out[32]) {
    if (!ctx || !raw_tx || !wtxid_out) return UFSECP_ERR_NULL_ARG;
    if (raw_tx_len < 10) return UFSECP_ERR_BAD_INPUT;

    // wtxid = double-SHA256 of the full witness-serialized tx
    auto h = secp256k1::SHA256::hash256(raw_tx, raw_tx_len);
    std::memcpy(wtxid_out, h.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip144_witness_commitment(
    const uint8_t witness_root[32],
    const uint8_t witness_nonce[32],
    uint8_t commitment_out[32]) {
    if (!witness_root || !witness_nonce || !commitment_out)
        return UFSECP_ERR_NULL_ARG;

    std::array<uint8_t, 32> wr, wn;
    std::memcpy(wr.data(), witness_root, 32);
    std::memcpy(wn.data(), witness_nonce, 32);

    auto c = secp256k1::witness_commitment(wr, wn);
    std::memcpy(commitment_out, c.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-141: Segregated Witness — Witness Programs
 * =========================================================================== */

int ufsecp_segwit_is_witness_program(
    const uint8_t* script, size_t script_len) {
    if (!script) return 0;
    return secp256k1::is_witness_program(script, script_len) ? 1 : 0;
}

ufsecp_error_t ufsecp_segwit_parse_program(
    const uint8_t* script, size_t script_len,
    int* version_out,
    uint8_t* program_out, size_t* program_len_out) {
    if (!script || !version_out || !program_out || !program_len_out)
        return UFSECP_ERR_NULL_ARG;

    auto wp = secp256k1::parse_witness_program(script, script_len);
    if (wp.version < 0) {
        *version_out = -1;
        *program_len_out = 0;
        return UFSECP_ERR_BAD_INPUT;
    }
    if (wp.program.size() > 40) return UFSECP_ERR_INTERNAL;   /* BIP-141 cap */
    *version_out = wp.version;
    *program_len_out = wp.program.size();
    std::memcpy(program_out, wp.program.data(), wp.program.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_p2wpkh_spk(
    const uint8_t pubkey_hash[20],
    uint8_t spk_out[22]) {
    if (!pubkey_hash || !spk_out) return UFSECP_ERR_NULL_ARG;

    auto spk = secp256k1::segwit_scriptpubkey_p2wpkh(pubkey_hash);
    std::memcpy(spk_out, spk.data(), 22);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_p2wsh_spk(
    const uint8_t script_hash[32],
    uint8_t spk_out[34]) {
    if (!script_hash || !spk_out) return UFSECP_ERR_NULL_ARG;

    auto spk = secp256k1::segwit_scriptpubkey_p2wsh(script_hash);
    std::memcpy(spk_out, spk.data(), 34);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_p2tr_spk(
    const uint8_t output_key[32],
    uint8_t spk_out[34]) {
    if (!output_key || !spk_out) return UFSECP_ERR_NULL_ARG;

    auto spk = secp256k1::segwit_scriptpubkey_p2tr(output_key);
    std::memcpy(spk_out, spk.data(), 34);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_witness_script_hash(
    const uint8_t* script, size_t script_len,
    uint8_t hash_out[32]) {
    // Allow (nullptr, 0) as a valid empty-script input; only reject null when len > 0
    if ((!script && script_len > 0) || !hash_out) return UFSECP_ERR_NULL_ARG;

    auto h = secp256k1::witness_script_hash(script, script_len);
    std::memcpy(hash_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-342: Tapscript Sighash
 * =========================================================================== */

// Helper: build TapSighashTxData from flat arrays,
// converting flattened prevout_txids to array-of-array.
static secp256k1::TapSighashTxData build_tap_tx_data(
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids_flat,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    std::vector<std::array<uint8_t, 32>>& txid_storage) {

    // Convert flat txid array to array-of-array
    txid_storage.resize(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        std::memcpy(txid_storage[i].data(), prevout_txids_flat + i * 32, 32);
    }

    secp256k1::TapSighashTxData td{};
    td.version = version;
    td.locktime = locktime;
    td.input_count = input_count;
    td.prevout_txids = txid_storage.data();
    td.prevout_vouts = prevout_vouts;
    td.input_amounts = input_amounts;
    td.input_sequences = input_sequences;
    td.input_scriptpubkeys = input_spks;
    td.input_scriptpubkey_lens = input_spk_lens;
    td.output_count = output_count;
    td.output_values = output_values;
    td.output_scriptpubkeys = output_spks;
    td.output_scriptpubkey_lens = output_spk_lens;
    return td;
}

ufsecp_error_t ufsecp_taproot_keypath_sighash(
    ufsecp_ctx* ctx,
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    size_t input_index,
    uint8_t hash_type,
    const uint8_t* annex, size_t annex_len,
    uint8_t sighash_out[32]) {
    if (!ctx || !prevout_txids || !prevout_vouts || !input_amounts ||
        !input_sequences || !input_spks || !input_spk_lens ||
        !output_values || !output_spks || !output_spk_lens || !sighash_out)
        return UFSECP_ERR_NULL_ARG;
    if (input_index >= input_count)
        return UFSECP_ERR_BAD_INPUT;
    ctx_clear_err(ctx);

    std::vector<std::array<uint8_t, 32>> txid_storage;
    auto td = build_tap_tx_data(version, locktime, input_count,
        prevout_txids, prevout_vouts, input_amounts, input_sequences,
        input_spks, input_spk_lens, output_count, output_values,
        output_spks, output_spk_lens, txid_storage);

    auto h = secp256k1::taproot_keypath_sighash(td, input_index, hash_type,
                                                 annex, annex_len);
    std::memcpy(sighash_out, h.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_tapscript_sighash(
    ufsecp_ctx* ctx,
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    size_t input_index,
    uint8_t hash_type,
    const uint8_t tapleaf_hash[32],
    uint8_t key_version,
    uint32_t code_separator_pos,
    const uint8_t* annex, size_t annex_len,
    uint8_t sighash_out[32]) {
    if (!ctx || !prevout_txids || !prevout_vouts || !input_amounts ||
        !input_sequences || !input_spks || !input_spk_lens ||
        !output_values || !output_spks || !output_spk_lens ||
        !tapleaf_hash || !sighash_out)
        return UFSECP_ERR_NULL_ARG;
    if (input_index >= input_count)
        return UFSECP_ERR_BAD_INPUT;
    ctx_clear_err(ctx);

    std::vector<std::array<uint8_t, 32>> txid_storage;
    auto td = build_tap_tx_data(version, locktime, input_count,
        prevout_txids, prevout_vouts, input_amounts, input_sequences,
        input_spks, input_spk_lens, output_count, output_values,
        output_spks, output_spk_lens, txid_storage);

    std::array<uint8_t, 32> tlh;
    std::memcpy(tlh.data(), tapleaf_hash, 32);

    auto h = secp256k1::tapscript_sighash(td, input_index, hash_type,
                                           tlh, key_version, code_separator_pos,
                                           annex, annex_len);
    std::memcpy(sighash_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * Public key arithmetic
 * =========================================================================== */

ufsecp_error_t ufsecp_pubkey_add(ufsecp_ctx* ctx,
                                 const uint8_t a33[33],
                                 const uint8_t b33[33],
                                 uint8_t out33[33]) {
    if (!ctx || !a33 || !b33 || !out33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto pa = point_from_compressed(a33);
    if (pa.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey a");
    }
    auto pb = point_from_compressed(b33);
    if (pb.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey b");
    }
    auto sum = pa.add(pb);
    if (sum.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "sum is point at infinity");
    }
    point_to_compressed(sum, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_negate(ufsecp_ctx* ctx,
                                    const uint8_t pubkey33[33],
                                    uint8_t out33[33]) {
    if (!ctx || !pubkey33 || !out33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto p = point_from_compressed(pubkey33);
    if (p.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    auto neg = p.negate();
    point_to_compressed(neg, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_tweak_add(ufsecp_ctx* ctx,
                                       const uint8_t pubkey33[33],
                                       const uint8_t tweak[32],
                                       uint8_t out33[33]) {
    if (!ctx || !pubkey33 || !tweak || !out33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto p = point_from_compressed(pubkey33);
    if (p.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    Scalar tw;
    if (!scalar_parse_strict(tweak, tw)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak >= n");
    }
    auto tG = Point::generator().scalar_mul(tw);
    auto result = p.add(tG);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_add resulted in infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_tweak_mul(ufsecp_ctx* ctx,
                                       const uint8_t pubkey33[33],
                                       const uint8_t tweak[32],
                                       uint8_t out33[33]) {
    if (!ctx || !pubkey33 || !tweak || !out33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto p = point_from_compressed(pubkey33);
    if (p.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    Scalar tw;
    if (!scalar_parse_strict_nonzero(tweak, tw)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak is zero or >= n");
    }
    auto result = p.scalar_mul(tw);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_mul resulted in infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_combine(ufsecp_ctx* ctx,
                                     const uint8_t* pubkeys,
                                     size_t n,
                                     uint8_t out33[33]) {
    if (!ctx || !pubkeys || !out33) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "need >= 1 pubkey");
    ctx_clear_err(ctx);
    std::size_t total_pubkey_bytes = 0;
    if (!checked_mul_size(n, static_cast<std::size_t>(33), total_pubkey_bytes)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "pubkey array length too large");
    }
    auto acc = point_from_compressed(pubkeys);
    if (acc.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey[0]");
    }
    for (size_t i = 1; i < n; ++i) {
        auto pi = point_from_compressed(pubkeys + i * 33);
        if (pi.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey in array");
        }
        acc = acc.add(pi);
    }
    if (acc.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "combined pubkey is infinity");
    }
    point_to_compressed(acc, out33);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-39
 * =========================================================================== */

ufsecp_error_t ufsecp_bip39_generate(ufsecp_ctx* ctx,
                                     size_t entropy_bytes,
                                     const uint8_t* entropy_in,
                                     char* mnemonic_out,
                                     size_t* mnemonic_len) {
    if (!ctx || !mnemonic_out || !mnemonic_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (entropy_bytes != 16 && entropy_bytes != 20 && entropy_bytes != 24 &&
        entropy_bytes != 28 && entropy_bytes != 32) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "entropy must be 16/20/24/28/32");
    }
    auto [mnemonic, ok] = secp256k1::bip39_generate(entropy_bytes, entropy_in);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-39 generation failed");
    }
    if (*mnemonic_len < mnemonic.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "mnemonic buffer too small");
    }
    std::memcpy(mnemonic_out, mnemonic.c_str(), mnemonic.size() + 1);
    *mnemonic_len = mnemonic.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip39_validate(const ufsecp_ctx* ctx,
                                     const char* mnemonic) {
    if (!ctx || !mnemonic) return UFSECP_ERR_NULL_ARG;
    if (!secp256k1::bip39_validate(std::string(mnemonic))) {
        return UFSECP_ERR_BAD_INPUT;
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip39_to_seed(ufsecp_ctx* ctx,
                                    const char* mnemonic,
                                    const char* passphrase,
                                    uint8_t seed64_out[64]) {
    if (!ctx || !mnemonic || !seed64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    const std::string pass = passphrase ? passphrase : "";
    auto [seed, ok] = secp256k1::bip39_mnemonic_to_seed(std::string(mnemonic), pass);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid mnemonic");
    }
    std::memcpy(seed64_out, seed.data(), 64);
    secp256k1::detail::secure_erase(seed.data(), seed.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip39_to_entropy(ufsecp_ctx* ctx,
                                       const char* mnemonic,
                                       uint8_t* entropy_out,
                                       size_t* entropy_len) {
    if (!ctx || !mnemonic || !entropy_out || !entropy_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto [ent, ok] = secp256k1::bip39_mnemonic_to_entropy(std::string(mnemonic));
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid mnemonic");
    }
    if (*entropy_len < ent.length) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "entropy buffer too small");
    }
    std::memcpy(entropy_out, ent.data.data(), ent.length);
    *entropy_len = ent.length;
    return UFSECP_OK;
}

/* ===========================================================================
 * Batch verification
 * =========================================================================== */

ufsecp_error_t ufsecp_schnorr_batch_verify(ufsecp_ctx* ctx,
                                           const uint8_t* entries, size_t n) {
    if (!ctx || !entries) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    ctx_clear_err(ctx);
    /* Each entry: 32-byte xonly pubkey | 32-byte msg | 64-byte sig = 128 bytes */
    std::vector<secp256k1::SchnorrBatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 128;
        // Strict: reject x-only pubkey >= p at ABI gate
        FE pk_fe;
        if (!FE::parse_bytes_strict(e, pk_fe)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "non-canonical pubkey (x>=p) in batch");
        }
        std::memcpy(batch[i].pubkey_x.data(), e, 32);
        std::memcpy(batch[i].message.data(), e + 32, 32);
        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid Schnorr sig in batch");
        }
    }
    if (!secp256k1::schnorr_batch_verify(batch)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "batch verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_batch_verify(ufsecp_ctx* ctx,
                                         const uint8_t* entries, size_t n) {
    if (!ctx || !entries) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    ctx_clear_err(ctx);
    /* Each entry: 32-byte msg | 33-byte pubkey | 64-byte sig = 129 bytes */
    std::vector<secp256k1::ECDSABatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 129;
        std::memcpy(batch[i].msg_hash.data(), e, 32);
        batch[i].public_key = point_from_compressed(e + 32);
        if (batch[i].public_key.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey in batch");
        }
        std::array<uint8_t, 64> compact;
        std::memcpy(compact.data(), e + 65, 64);
        if (!secp256k1::ECDSASignature::parse_compact_strict(compact, batch[i].signature)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid ECDSA sig in batch");
        }
    }
    if (!secp256k1::ecdsa_batch_verify(batch)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "batch verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_batch_identify_invalid(
    ufsecp_ctx* ctx, const uint8_t* entries, size_t n,
    size_t* invalid_out, size_t* invalid_count) {
    if (!ctx || !entries || !invalid_out || !invalid_count) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    std::vector<secp256k1::SchnorrBatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 128;
        FE pk_fe;
        if (!FE::parse_bytes_strict(e, pk_fe)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "non-canonical pubkey (x>=p) in batch");
        }
        std::memcpy(batch[i].pubkey_x.data(), e, 32);
        std::memcpy(batch[i].message.data(), e + 32, 32);
        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid Schnorr sig in batch");
        }
    }
    auto invalids = secp256k1::schnorr_batch_identify_invalid(batch.data(), n);
    size_t const capacity = *invalid_count;
    size_t const count = invalids.size() < capacity ? invalids.size() : capacity;
    *invalid_count = invalids.size();
    for (size_t i = 0; i < count; ++i) {
        invalid_out[i] = invalids[i];
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_batch_identify_invalid(
    ufsecp_ctx* ctx, const uint8_t* entries, size_t n,
    size_t* invalid_out, size_t* invalid_count) {
    if (!ctx || !entries || !invalid_out || !invalid_count) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    std::vector<secp256k1::ECDSABatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 129;
        std::memcpy(batch[i].msg_hash.data(), e, 32);
        batch[i].public_key = point_from_compressed(e + 32);
        if (batch[i].public_key.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey in batch");
        }
        std::array<uint8_t, 64> compact;
        std::memcpy(compact.data(), e + 65, 64);
        if (!secp256k1::ECDSASignature::parse_compact_strict(compact, batch[i].signature)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid ECDSA sig in batch");
        }
    }
    auto invalids = secp256k1::ecdsa_batch_identify_invalid(batch.data(), n);
    size_t const capacity = *invalid_count;
    size_t const count = invalids.size() < capacity ? invalids.size() : capacity;
    *invalid_count = invalids.size();
    for (size_t i = 0; i < count; ++i) {
        invalid_out[i] = invalids[i];
    }
    return UFSECP_OK;
}

/* ===========================================================================
 * SHA-512
 * =========================================================================== */

ufsecp_error_t ufsecp_sha512(const uint8_t* data, size_t len,
                             uint8_t digest64_out[64]) {
    if (!data || !digest64_out) return UFSECP_ERR_NULL_ARG;
    auto hash = secp256k1::SHA512::hash(data, len);
    std::memcpy(digest64_out, hash.data(), 64);
    return UFSECP_OK;
}

/* ===========================================================================
 * Multi-scalar multiplication
 * =========================================================================== */

ufsecp_error_t ufsecp_shamir_trick(ufsecp_ctx* ctx,
                                   const uint8_t a[32], const uint8_t P33[33],
                                   const uint8_t b[32], const uint8_t Q33[33],
                                   uint8_t out33[33]) {
    if (!ctx || !a || !P33 || !b || !Q33 || !out33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sa, sb;
    if (!scalar_parse_strict(a, sa)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar a >= n");
    }
    if (!scalar_parse_strict(b, sb)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar b >= n");
    }
    auto P = point_from_compressed(P33);
    if (P.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid point P");
    }
    auto Q = point_from_compressed(Q33);
    if (Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid point Q");
    }
    auto result = secp256k1::shamir_trick(sa, P, sb, Q);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "result is infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_multi_scalar_mul(ufsecp_ctx* ctx,
                                       const uint8_t* scalars,
                                       const uint8_t* points,
                                       size_t n,
                                       uint8_t out33[33]) {
    if (!ctx || !scalars || !points || !out33) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n must be >= 1");
    ctx_clear_err(ctx);
    std::size_t total_scalar_bytes = 0;
    std::size_t total_point_bytes = 0;
    if (!checked_mul_size(n, static_cast<std::size_t>(32), total_scalar_bytes)
        || !checked_mul_size(n, static_cast<std::size_t>(33), total_point_bytes)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar/point array length too large");
    }
    std::vector<Scalar> svec(n);
    std::vector<Point> pvec(n);
    for (size_t i = 0; i < n; ++i) {
        if (!scalar_parse_strict(scalars + i * 32, svec[i])) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar >= n");
        }
        pvec[i] = point_from_compressed(points + i * 33);
        if (pvec[i].is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid point in array");
        }
    }
    auto result = secp256k1::multi_scalar_mul(svec, pvec);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "MSM result is infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

/* ===========================================================================
 * MuSig2 (BIP-327)
 * =========================================================================== */

ufsecp_error_t ufsecp_musig2_key_agg(ufsecp_ctx* ctx,
                                     const uint8_t* pubkeys, size_t n,
                                     uint8_t keyagg_out[UFSECP_MUSIG2_KEYAGG_LEN],
                                     uint8_t agg_pubkey32_out[32]) {
    if (!ctx || !pubkeys || !keyagg_out || !agg_pubkey32_out) return UFSECP_ERR_NULL_ARG;
    if (n < 2) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "need >= 2 pubkeys");
    if (n > kMuSig2MaxKeyAggParticipants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "too many pubkeys for keyagg blob");
    }
    ctx_clear_err(ctx);
    std::vector<std::array<uint8_t, 32>> pks(n);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(pks[i].data(), pubkeys + i * 32, 32);
    }
    auto kagg = secp256k1::musig2_key_agg(pks);
    std::memcpy(agg_pubkey32_out, kagg.Q_x.data(), 32);
    /* Serialize key agg ctx: n(4) | Q_negated(1) | Q_compressed(33) | coefficients(n*32) */
    std::memset(keyagg_out, 0, UFSECP_MUSIG2_KEYAGG_LEN);
    const auto nk = static_cast<uint32_t>(kagg.key_coefficients.size());
    std::memcpy(keyagg_out, &nk, 4);
    keyagg_out[4] = kagg.Q_negated ? 1 : 0;
    point_to_compressed(kagg.Q, keyagg_out + 5);
    for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {
        scalar_to_bytes(kagg.key_coefficients[i], keyagg_out + 38 + static_cast<size_t>(i) * 32);
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_nonce_gen(ufsecp_ctx* ctx,
                                       const uint8_t privkey[32],
                                       const uint8_t pubkey32[32],
                                       const uint8_t agg_pubkey32[32],
                                       const uint8_t msg32[32],
                                       const uint8_t extra_in[32],
                                       uint8_t secnonce_out[UFSECP_MUSIG2_SECNONCE_LEN],
                                       uint8_t pubnonce_out[UFSECP_MUSIG2_PUBNONCE_LEN]) {
    if (!ctx || !privkey || !pubkey32 || !agg_pubkey32 || !msg32 ||
        !secnonce_out || !pubnonce_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    std::array<uint8_t, 32> pk_arr, agg_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey32, 32);
    std::memcpy(agg_arr.data(), agg_pubkey32, 32);
    std::memcpy(msg_arr.data(), msg32, 32);
    auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pk_arr, agg_arr, msg_arr, extra_in);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    /* Secret nonce: k1 || k2 */
    auto k1_bytes = sec.k1.to_bytes();
    auto k2_bytes = sec.k2.to_bytes();
    std::memcpy(secnonce_out, k1_bytes.data(), 32);
    std::memcpy(secnonce_out + 32, k2_bytes.data(), 32);
    secp256k1::detail::secure_erase(k1_bytes.data(), k1_bytes.size());
    secp256k1::detail::secure_erase(k2_bytes.data(), k2_bytes.size());
    /* Public nonce: R1(33) || R2(33) */
    auto pn = pub.serialize();
    std::memcpy(pubnonce_out, pn.data(), 66);
    secp256k1::detail::secure_erase(&sec, sizeof(sec));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_nonce_agg(ufsecp_ctx* ctx,
                                       const uint8_t* pubnonces, size_t n,
                                       uint8_t aggnonce_out[UFSECP_MUSIG2_AGGNONCE_LEN]) {
    if (!ctx || !pubnonces || !aggnonce_out) return UFSECP_ERR_NULL_ARG;
    if (n < 2) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "need >= 2 nonces");
    ctx_clear_err(ctx);
    std::vector<secp256k1::MuSig2PubNonce> pns(n);
    for (size_t i = 0; i < n; ++i) {
        if (point_from_compressed(pubnonces + i * 66).is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid pubnonce R1");
        }
        if (point_from_compressed(pubnonces + i * 66 + 33).is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid pubnonce R2");
        }
        std::array<uint8_t, 66> buf;
        std::memcpy(buf.data(), pubnonces + i * 66, 66);
        pns[i] = secp256k1::MuSig2PubNonce::deserialize(buf);
    }
    auto agg = secp256k1::musig2_nonce_agg(pns);
    /* Serialize: R1(33) || R2(33) */
    auto r1 = agg.R1.to_compressed();
    auto r2 = agg.R2.to_compressed();
    std::memcpy(aggnonce_out, r1.data(), 33);
    std::memcpy(aggnonce_out + 33, r2.data(), 33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_start_sign_session(
    ufsecp_ctx* ctx,
    const uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t msg32[32],
    uint8_t session_out[UFSECP_MUSIG2_SESSION_LEN]) {
    if (!ctx || !aggnonce || !keyagg || !msg32 || !session_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    /* Deserialize agg nonce */
    secp256k1::MuSig2AggNonce an;
    an.R1 = point_from_compressed(aggnonce);
    if (an.R1.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid agg nonce R1");
    }
    an.R2 = point_from_compressed(aggnonce + 33);
    if (an.R2.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid agg nonce R2");
    }
    /* Deserialize key agg context */
    secp256k1::MuSig2KeyAggCtx kagg;
    {
        const ufsecp_error_t rc = parse_musig2_keyagg(ctx, keyagg, kagg);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto sess = secp256k1::musig2_start_sign_session(an, kagg, msg_arr);
    /* Serialize session: R(33) | b(32) | e(32) | R_negated(1) = 98 bytes */
    std::memset(session_out, 0, UFSECP_MUSIG2_SESSION_LEN);
    point_to_compressed(sess.R, session_out);
    scalar_to_bytes(sess.b, session_out + 33);
    scalar_to_bytes(sess.e, session_out + 65);
    session_out[97] = sess.R_negated ? 1 : 0;
    const uint32_t participant_count = static_cast<uint32_t>(kagg.key_coefficients.size());
    std::memcpy(session_out + kMuSig2SessionCountOffset, &participant_count, sizeof(participant_count));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_partial_sign(
    ufsecp_ctx* ctx,
    uint8_t secnonce[UFSECP_MUSIG2_SECNONCE_LEN],
    const uint8_t privkey[32],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    size_t signer_index,
    uint8_t partial_sig32_out[32]) {
    if (!ctx || !secnonce || !privkey || !keyagg || !session || !partial_sig32_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    secp256k1::MuSig2SecNonce sn;
    Scalar k1, k2;
    if (!scalar_parse_strict_nonzero(secnonce, k1)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid secnonce k1");
    }
    if (!scalar_parse_strict_nonzero(secnonce + 32, k2)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid secnonce k2");
    }
    sn.k1 = k1;
    sn.k2 = k2;
    secp256k1::MuSig2KeyAggCtx kagg;
    {
        const ufsecp_error_t rc = parse_musig2_keyagg(ctx, keyagg, kagg);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (signer_index >= kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "signer_index out of range");
    }
    secp256k1::MuSig2Session sess;
    uint32_t session_participant_count = 0;
    {
        const ufsecp_error_t rc = parse_musig2_session(ctx, session, sess, session_participant_count);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (session_participant_count != kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "session participant count does not match keyagg");
    }
    auto psig = secp256k1::musig2_partial_sign(sn, sk, kagg, sess, signer_index);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&sn, sizeof(sn));
    // Consume caller's secnonce to prevent catastrophic nonce reuse
    secp256k1::detail::secure_erase(secnonce, UFSECP_MUSIG2_SECNONCE_LEN);
    scalar_to_bytes(psig, partial_sig32_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_partial_verify(
    ufsecp_ctx* ctx,
    const uint8_t partial_sig32[32],
    const uint8_t pubnonce[UFSECP_MUSIG2_PUBNONCE_LEN],
    const uint8_t pubkey32[32],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    size_t signer_index) {
    if (!ctx || !partial_sig32 || !pubnonce || !pubkey32 || !keyagg || !session) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar psig;
    if (!scalar_parse_strict(partial_sig32, psig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "partial sig >= n");
    }
    std::array<uint8_t, 66> pn_buf;
    std::memcpy(pn_buf.data(), pubnonce, 66);
    auto pn = secp256k1::MuSig2PubNonce::deserialize(pn_buf);
    std::array<uint8_t, 32> pk_arr;
    std::memcpy(pk_arr.data(), pubkey32, 32);
    secp256k1::MuSig2KeyAggCtx kagg;
    {
        const ufsecp_error_t rc = parse_musig2_keyagg(ctx, keyagg, kagg);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (signer_index >= kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "signer_index out of range");
    }
    secp256k1::MuSig2Session sess;
    uint32_t session_participant_count = 0;
    {
        const ufsecp_error_t rc = parse_musig2_session(ctx, session, sess, session_participant_count);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (session_participant_count != kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "session participant count does not match keyagg");
    }
    if (!secp256k1::musig2_partial_verify(psig, pn, pk_arr, kagg, sess, signer_index)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "partial sig verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_partial_sig_agg(
    ufsecp_ctx* ctx,
    const uint8_t* partial_sigs, size_t n,
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    uint8_t sig64_out[64]) {
    if (!ctx || !partial_sigs || !session || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (n == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sigs must be non-empty");
    }
    std::vector<Scalar> psigs(n);
    for (size_t i = 0; i < n; ++i) {
        if (!scalar_parse_strict(partial_sigs + i * 32, psigs[i])) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "partial sig >= n");
        }
    }
    secp256k1::MuSig2Session sess;
    uint32_t session_participant_count = 0;
    {
        const ufsecp_error_t rc = parse_musig2_session(ctx, session, sess, session_participant_count);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (n != session_participant_count) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sigs count does not match session participant count");
    }
    auto final_sig = secp256k1::musig2_partial_sig_agg(psigs, sess);
    std::memcpy(sig64_out, final_sig.data(), 64);
    return UFSECP_OK;
}

/* ===========================================================================
 * FROST (threshold signatures)
 * =========================================================================== */

ufsecp_error_t ufsecp_frost_keygen_begin(
    ufsecp_ctx* ctx,
    uint32_t participant_id, uint32_t threshold, uint32_t num_participants,
    const uint8_t seed[32],
    uint8_t* commits_out, size_t* commits_len,
    uint8_t* shares_out, size_t* shares_len) {
    if (!ctx || !seed || !commits_out || !commits_len || !shares_out || !shares_len) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (threshold < 2 || threshold > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid threshold");
    }
    if (participant_id == 0 || participant_id > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid participant_id");
    }
    std::size_t required_commit_coeff_bytes = 0;
    std::size_t required_commits = 0;
    std::size_t required_shares = 0;
    if (!checked_mul_size(static_cast<std::size_t>(threshold), static_cast<std::size_t>(33), required_commit_coeff_bytes)
        || !checked_add_size(static_cast<std::size_t>(8), required_commit_coeff_bytes, required_commits)
        || !checked_mul_size(static_cast<std::size_t>(num_participants), static_cast<std::size_t>(UFSECP_FROST_SHARE_LEN), required_shares)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "FROST cardinality too large");
    }
    if (*commits_len < required_commits) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "commits buffer too small");
    }
    if (*shares_len < required_shares) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "shares buffer too small");
    }
    std::array<uint8_t, 32> seed_arr;
    std::memcpy(seed_arr.data(), seed, 32);
    auto [commit, shares] = secp256k1::frost_keygen_begin(
        participant_id, threshold, num_participants, seed_arr);
    secp256k1::detail::secure_erase(seed_arr.data(), 32);
    auto erase_shares = [&]() {
        for (auto& share : shares) {
            secp256k1::detail::secure_erase(&share.value, sizeof(share.value));
        }
    };
    /* Serialize commitment: coeff count(4) + from(4) + coeffs(33 each) */
    const size_t coeff_count = commit.coeffs.size();
    const size_t needed_commits = 8 + coeff_count * 33;
    if (*commits_len < needed_commits) {
        erase_shares();
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "commits buffer too small");
    }
    const auto cc32 = static_cast<uint32_t>(coeff_count);
    std::memcpy(commits_out, &cc32, 4);
    std::memcpy(commits_out + 4, &commit.from, 4);
    for (size_t i = 0; i < coeff_count; ++i) {
        point_to_compressed(commit.coeffs[i], commits_out + 8 + i * 33);

    }
    *commits_len = 8 + coeff_count * 33;
    /* Serialize shares */
    const size_t needed_shares = shares.size() * UFSECP_FROST_SHARE_LEN;
    if (*shares_len < needed_shares) {
        erase_shares();
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "shares buffer too small");
    }
    for (size_t i = 0; i < shares.size(); ++i) {
        uint8_t* s = shares_out + i * UFSECP_FROST_SHARE_LEN;
        std::memcpy(s, &shares[i].from, 4);
        scalar_to_bytes(shares[i].value, s + 4);
    }
    *shares_len = needed_shares;
    erase_shares();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_frost_keygen_finalize(
    ufsecp_ctx* ctx,
    uint32_t participant_id,
    const uint8_t* all_commits, size_t commits_len,
    const uint8_t* received_shares, size_t shares_len,
    uint32_t threshold, uint32_t num_participants,
    uint8_t keypkg_out[UFSECP_FROST_KEYPKG_LEN]) {
    if (!ctx || !all_commits || !received_shares || !keypkg_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (threshold < 2 || threshold > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid threshold");
    }
    if (participant_id == 0 || participant_id > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid participant_id");
    }
    std::size_t expected_commit_coeff_bytes = 0;
    std::size_t expected_commit_record_len = 0;
    std::size_t expected_commits_len = 0;
    std::size_t expected_shares_len = 0;
    if (!checked_mul_size(static_cast<std::size_t>(threshold), static_cast<std::size_t>(33), expected_commit_coeff_bytes)
        || !checked_add_size(static_cast<std::size_t>(8), expected_commit_coeff_bytes, expected_commit_record_len)
        || !checked_mul_size(static_cast<std::size_t>(num_participants), expected_commit_record_len, expected_commits_len)
        || !checked_mul_size(static_cast<std::size_t>(num_participants), static_cast<std::size_t>(UFSECP_FROST_SHARE_LEN), expected_shares_len)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "FROST cardinality too large");
    }
    if (commits_len != expected_commits_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "all_commits length does not match threshold and num_participants");
    }
    if (shares_len != expected_shares_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "received_shares length does not match num_participants");
    }
    /* Deserialize commitments */
    std::vector<secp256k1::FrostCommitment> commits;
    std::vector<uint8_t> seen_commit_from(static_cast<size_t>(num_participants) + 1, 0);
    size_t pos = 0;
    while (pos < commits_len) {
        secp256k1::FrostCommitment fc;
        uint32_t cc = 0;
        if (pos + 8 > commits_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "truncated commit header");
        }
        std::memcpy(&cc, all_commits + pos, 4); pos += 4;
        std::memcpy(&fc.from, all_commits + pos, 4); pos += 4;
        if (cc != threshold) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment coefficient count");
        }
        if (fc.from == 0 || fc.from > num_participants) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment sender");
        }
        if (seen_commit_from[fc.from] != 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate commitment sender");
        }
        seen_commit_from[fc.from] = 1;
        if (pos + static_cast<size_t>(cc) * 33 > commits_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "truncated commit coefficients");
        }
        for (uint32_t j = 0; j < cc; ++j) {
            auto pt = point_from_compressed(all_commits + pos);
            if (pt.is_infinity()) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment coefficient");
            }
            fc.coeffs.push_back(pt);
            pos += 33;
        }
        commits.push_back(std::move(fc));
    }
    if (commits.size() != num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment count");
    }
    /* Deserialize shares */
    if (shares_len == 0 || (shares_len % UFSECP_FROST_SHARE_LEN) != 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share blob length");
    }
    const size_t n_shares = shares_len / UFSECP_FROST_SHARE_LEN;
    if (n_shares != num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share count");
    }
    std::vector<secp256k1::FrostShare> shares(n_shares);
    auto erase_shares = [&]() {
        for (auto& share : shares) {
            secp256k1::detail::secure_erase(&share.value, sizeof(share.value));
        }
    };
    std::vector<uint8_t> seen_share_from(static_cast<size_t>(num_participants) + 1, 0);
    for (size_t i = 0; i < n_shares; ++i) {
        const uint8_t* s = received_shares + i * UFSECP_FROST_SHARE_LEN;
        std::memcpy(&shares[i].from, s, 4);
        if (shares[i].from == 0 || shares[i].from > num_participants) {
            erase_shares();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share sender");
        }
        if (seen_share_from[shares[i].from] != 0) {
            erase_shares();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate share sender");
        }
        seen_share_from[shares[i].from] = 1;
        shares[i].id = participant_id;
        Scalar v;
        if (!scalar_parse_strict(s + 4, v)) {
            erase_shares();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share scalar");
        }
        shares[i].value = v;
    }
    auto [kp, ok] = secp256k1::frost_keygen_finalize(
        participant_id, commits, shares, threshold, num_participants);
    if (!ok) {
        erase_shares();
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "FROST keygen finalize failed");
    }
    erase_shares();
    /* Serialize FrostKeyPackage: id(4) | threshold(4) | num_participants(4) |
       signing_share(32) | verification_share(33) | group_public_key(33) = 110 bytes */
    std::memset(keypkg_out, 0, UFSECP_FROST_KEYPKG_LEN);
    std::memcpy(keypkg_out, &kp.id, 4);
    std::memcpy(keypkg_out + 4, &kp.threshold, 4);
    std::memcpy(keypkg_out + 8, &kp.num_participants, 4);
    scalar_to_bytes(kp.signing_share, keypkg_out + 12);
    point_to_compressed(kp.verification_share, keypkg_out + 44);
    point_to_compressed(kp.group_public_key, keypkg_out + 77);
    secp256k1::detail::secure_erase(&kp.signing_share, sizeof(kp.signing_share));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_frost_sign_nonce_gen(
    ufsecp_ctx* ctx,
    uint32_t participant_id,
    const uint8_t nonce_seed[32],
    uint8_t nonce_out[UFSECP_FROST_NONCE_LEN],
    uint8_t nonce_commit_out[UFSECP_FROST_NONCE_COMMIT_LEN]) {
    if (!ctx || !nonce_seed || !nonce_out || !nonce_commit_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (participant_id == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid participant_id");
    }
    std::array<uint8_t, 32> seed_arr;
    std::memcpy(seed_arr.data(), nonce_seed, 32);
    auto [nonce, commit] = secp256k1::frost_sign_nonce_gen(participant_id, seed_arr);
    auto h_bytes = nonce.hiding_nonce.to_bytes();
    auto b_bytes = nonce.binding_nonce.to_bytes();
    std::memcpy(nonce_out, h_bytes.data(), 32);
    std::memcpy(nonce_out + 32, b_bytes.data(), 32);
    secp256k1::detail::secure_erase(seed_arr.data(), 32);
    secp256k1::detail::secure_erase(&nonce.hiding_nonce, sizeof(nonce.hiding_nonce));
    secp256k1::detail::secure_erase(&nonce.binding_nonce, sizeof(nonce.binding_nonce));
    secp256k1::detail::secure_erase(h_bytes.data(), 32);
    secp256k1::detail::secure_erase(b_bytes.data(), 32);
    std::memcpy(nonce_commit_out, &commit.id, 4);
    auto hp = commit.hiding_point.to_compressed();
    auto bp = commit.binding_point.to_compressed();
    std::memcpy(nonce_commit_out + 4, hp.data(), 33);
    std::memcpy(nonce_commit_out + 37, bp.data(), 33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_frost_sign(
    ufsecp_ctx* ctx,
    const uint8_t keypkg[UFSECP_FROST_KEYPKG_LEN],
    const uint8_t nonce[UFSECP_FROST_NONCE_LEN],
    const uint8_t msg32[32],
    const uint8_t* nonce_commits, size_t n_signers,
    uint8_t partial_sig_out[36]) {
    if (!ctx || !keypkg || !nonce || !msg32 || !nonce_commits || !partial_sig_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n_signers == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers must be non-zero");
    }
    secp256k1::FrostKeyPackage kp;
    std::memcpy(&kp.id, keypkg, 4);
    std::memcpy(&kp.threshold, keypkg + 4, 4);
    std::memcpy(&kp.num_participants, keypkg + 8, 4);
    if (kp.num_participants == 0 || kp.id == 0 || kp.id > kp.num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid key package participant metadata");
    }
    if (kp.threshold < 2 || kp.threshold > kp.num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid key package threshold");
    }
    if (n_signers > kp.num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid signer count");
    }
    if (!scalar_parse_strict(keypkg + 12, kp.signing_share)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid signing share in keypkg");
    }
    kp.verification_share = point_from_compressed(keypkg + 44);
    if (kp.verification_share.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid verification share");
    }
    kp.group_public_key = point_from_compressed(keypkg + 77);
    if (kp.group_public_key.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid group public key");
    }
    secp256k1::FrostNonce fn;
    Scalar h, b;
    if (!scalar_parse_strict(nonce, h)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce");
    }
    if (!scalar_parse_strict(nonce + 32, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce");
    }
    fn.hiding_nonce = h;
    fn.binding_nonce = b;
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::vector<secp256k1::FrostNonceCommitment> ncs(n_signers);
    size_t self_commitment_count = 0;
    ufsecp_error_t nc_err = UFSECP_OK;
    for (size_t i = 0; i < n_signers; ++i) {
        const uint8_t* nc = nonce_commits + i * UFSECP_FROST_NONCE_COMMIT_LEN;
        std::memcpy(&ncs[i].id, nc, 4);
        if (ncs[i].id == 0 || ncs[i].id > kp.num_participants) {
            nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid nonce commitment signer");
            break;
        }
        for (size_t j = 0; j < i; ++j) {
            if (ncs[j].id == ncs[i].id) {
                nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate nonce commitment signer");
                break;
            }
        }
        if (nc_err != UFSECP_OK) break;
        if (ncs[i].id == kp.id) {
            ++self_commitment_count;
        }
        ncs[i].hiding_point = point_from_compressed(nc + 4);
        if (ncs[i].hiding_point.is_infinity()) {
            nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce point");
            break;
        }
        ncs[i].binding_point = point_from_compressed(nc + 37);
        if (ncs[i].binding_point.is_infinity()) {
            nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce point");
            break;
        }
    }
    if (nc_err == UFSECP_OK && self_commitment_count != 1) {
        nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "missing signer nonce commitment");
    }
    if (nc_err != UFSECP_OK) {
        secp256k1::detail::secure_erase(&kp.signing_share, sizeof(kp.signing_share));
        secp256k1::detail::secure_erase(&fn.hiding_nonce, sizeof(fn.hiding_nonce));
        secp256k1::detail::secure_erase(&fn.binding_nonce, sizeof(fn.binding_nonce));
        secp256k1::detail::secure_erase(&h, sizeof(h));
        secp256k1::detail::secure_erase(&b, sizeof(b));
        return nc_err;
    }
    auto psig = secp256k1::frost_sign(kp, fn, msg_arr, ncs);
    secp256k1::detail::secure_erase(&kp.signing_share, sizeof(kp.signing_share));
    secp256k1::detail::secure_erase(&fn.hiding_nonce, sizeof(fn.hiding_nonce));
    secp256k1::detail::secure_erase(&fn.binding_nonce, sizeof(fn.binding_nonce));
    secp256k1::detail::secure_erase(&h, sizeof(h));
    secp256k1::detail::secure_erase(&b, sizeof(b));
    std::memcpy(partial_sig_out, &psig.id, 4);
    scalar_to_bytes(psig.z_i, partial_sig_out + 4);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_frost_verify_partial(
    ufsecp_ctx* ctx,
    const uint8_t partial_sig[36],
    const uint8_t verification_share33[33],
    const uint8_t* nonce_commits, size_t n_signers,
    const uint8_t msg32[32],
    const uint8_t group_pubkey33[33]) {
    if (!ctx || !partial_sig || !verification_share33 || !nonce_commits || !msg32 || !group_pubkey33) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n_signers == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers must be non-zero");
    }
    secp256k1::FrostPartialSig psig;
    std::memcpy(&psig.id, partial_sig, 4);
    if (psig.id == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sig.id must be non-zero");
    }
    Scalar z;
    if (!scalar_parse_strict(partial_sig + 4, z)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid partial sig scalar");
    }
    psig.z_i = z;
    auto vs = point_from_compressed(verification_share33);
    if (vs.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid verification share");
    }
    std::vector<secp256k1::FrostNonceCommitment> ncs(n_signers);
    secp256k1::FrostNonceCommitment signer_commit{};
    size_t signer_matches = 0;
    for (size_t i = 0; i < n_signers; ++i) {
        const uint8_t* nc = nonce_commits + i * UFSECP_FROST_NONCE_COMMIT_LEN;
        std::memcpy(&ncs[i].id, nc, 4);
        if (ncs[i].id == 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "nonce commitment signer IDs must be non-zero");
        }
        for (size_t j = 0; j < i; ++j) {
            if (ncs[j].id == ncs[i].id) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate nonce commitment signer IDs");
            }
        }
        ncs[i].hiding_point = point_from_compressed(nc + 4);
        if (ncs[i].hiding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce point");
        }
        ncs[i].binding_point = point_from_compressed(nc + 37);
        if (ncs[i].binding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce point");
        }
        if (ncs[i].id == psig.id) {
            signer_commit = ncs[i];
            ++signer_matches;
        }
    }
    if (signer_matches != 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT,
            signer_matches == 0 ? "partial_sig.id not found in nonce_commits"
                                : "partial_sig.id must appear exactly once in nonce_commits");
    }
    auto gp = point_from_compressed(group_pubkey33);
    if (gp.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid group public key");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    const bool ok = secp256k1::frost_verify_partial(psig, signer_commit, vs, msg_arr, ncs, gp);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "FROST partial signature verification failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_frost_aggregate(
    ufsecp_ctx* ctx,
    const uint8_t* partial_sigs, size_t n,
    const uint8_t* nonce_commits, size_t n_signers,
    const uint8_t group_pubkey33[33],
    const uint8_t msg32[32],
    uint8_t sig64_out[64]) {
    if (!ctx || !partial_sigs || !nonce_commits || !group_pubkey33 || !msg32 || !sig64_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sigs must be non-empty");
    }
    if (n_signers == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers must be non-zero");
    }
    if (n != n_signers) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial/nonces signer count mismatch");
    }
    std::vector<secp256k1::FrostPartialSig> psigs(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* ps = partial_sigs + i * 36;
        std::memcpy(&psigs[i].id, ps, 4);
        if (psigs[i].id == 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid partial sig signer");
        }
        for (size_t j = 0; j < i; ++j) {
            if (psigs[j].id == psigs[i].id) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate partial sig signer");
            }
        }
        Scalar z;
        if (!scalar_parse_strict(ps + 4, z)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid partial sig scalar");
        }
        psigs[i].z_i = z;
    }
    std::vector<secp256k1::FrostNonceCommitment> ncs(n_signers);
    for (size_t i = 0; i < n_signers; ++i) {
        const uint8_t* nc = nonce_commits + i * UFSECP_FROST_NONCE_COMMIT_LEN;
        std::memcpy(&ncs[i].id, nc, 4);
        if (ncs[i].id == 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid nonce commitment signer");
        }
        for (size_t j = 0; j < i; ++j) {
            if (ncs[j].id == ncs[i].id) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate nonce commitment signer");
            }
        }
        ncs[i].hiding_point = point_from_compressed(nc + 4);
        if (ncs[i].hiding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce point");
        }
        ncs[i].binding_point = point_from_compressed(nc + 37);
        if (ncs[i].binding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce point");
        }
    }
    for (const auto& psig : psigs) {
        bool found = false;
        for (const auto& nc : ncs) {
            if (nc.id == psig.id) {
                found = true;
                break;
            }
        }
        if (!found) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial sig signer missing from nonce commitments");
        }
    }
    auto gp = point_from_compressed(group_pubkey33);
    if (gp.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid group public key");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto sig = secp256k1::frost_aggregate(psigs, ncs, gp, msg_arr);
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

/* ===========================================================================
 * Adaptor signatures
 * =========================================================================== */

ufsecp_error_t ufsecp_schnorr_adaptor_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33],
    const uint8_t aux_rand[32],
    uint8_t pre_sig_out[UFSECP_SCHNORR_ADAPTOR_SIG_LEN]) {
    if (!ctx || !privkey || !msg32 || !adaptor_point33 || !aux_rand || !pre_sig_out) {
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
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    auto pre = secp256k1::schnorr_adaptor_sign(sk, msg_arr, ap, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto rhat = pre.R_hat.to_compressed();
    auto shat = pre.s_hat.to_bytes();
    std::memcpy(pre_sig_out, rhat.data(), 33);
    std::memcpy(pre_sig_out + 33, shat.data(), 32);
    /* Serialize needs_negation as a 32-byte flag for completeness */
    std::memset(pre_sig_out + 65, 0, 32);
    pre_sig_out[65] = pre.needs_negation ? 1 : 0;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_adaptor_verify(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t pubkey_x[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33]) {
    if (!ctx || !pre_sig || !pubkey_x || !msg32 || !adaptor_point33) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    secp256k1::SchnorrAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    as.needs_negation = (pre_sig[65] != 0);
    // Strict: reject x-only pubkey >= p at ABI gate
    FE pk_fe;
    if (!FE::parse_bytes_strict(pubkey_x, pk_fe)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "non-canonical pubkey (x>=p)");
    }
    std::array<uint8_t, 32> pk_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey_x, 32);
    std::memcpy(msg_arr.data(), msg32, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    if (!secp256k1::schnorr_adaptor_verify(as, pk_arr, msg_arr, ap)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "adaptor verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_adaptor_adapt(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t adaptor_secret[32],
    uint8_t sig64_out[64]) {
    if (!ctx || !pre_sig || !adaptor_secret || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::SchnorrAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    as.needs_negation = (pre_sig[65] != 0);
    Scalar secret;
    if (!scalar_parse_strict_nonzero(adaptor_secret, secret)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "adaptor secret is zero or >= n");
    }
    auto sig = secp256k1::schnorr_adaptor_adapt(as, secret);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_adaptor_extract(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t sig64[64],
    uint8_t secret32_out[32]) {
    if (!ctx || !pre_sig || !sig64 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::SchnorrAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    as.needs_negation = (pre_sig[65] != 0);
    secp256k1::SchnorrSignature sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid schnorr signature");
    }
    auto [secret, ok] = secp256k1::schnorr_adaptor_extract(as, sig);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "adaptor extract failed");
    }
    scalar_to_bytes(secret, secret32_out);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33],
    uint8_t pre_sig_out[UFSECP_ECDSA_ADAPTOR_SIG_LEN]) {
    if (!ctx || !privkey || !msg32 || !adaptor_point33 || !pre_sig_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    auto pre = secp256k1::ecdsa_adaptor_sign(sk, msg_arr, ap);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto rhat = pre.R_hat.to_compressed();
    auto shat = pre.s_hat.to_bytes();
    auto r_bytes = pre.r.to_bytes();
    std::memcpy(pre_sig_out, rhat.data(), 33);
    std::memcpy(pre_sig_out + 33, shat.data(), 32);
    std::memcpy(pre_sig_out + 65, r_bytes.data(), 32);
    /* zero-pad remainder */
    std::memset(pre_sig_out + 97, 0, UFSECP_ECDSA_ADAPTOR_SIG_LEN - 97);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_verify(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33]) {
    if (!ctx || !pre_sig || !pubkey33 || !msg32 || !adaptor_point33) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    secp256k1::ECDSAAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    if (!scalar_parse_strict(pre_sig + 65, as.r)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig r");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    if (!secp256k1::ecdsa_adaptor_verify(as, pk, msg_arr, ap)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECDSA adaptor verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_adapt(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t adaptor_secret[32],
    uint8_t sig64_out[64]) {
    if (!ctx || !pre_sig || !adaptor_secret || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::ECDSAAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    if (!scalar_parse_strict(pre_sig + 65, as.r)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig r");
    }
    Scalar secret;
    if (!scalar_parse_strict_nonzero(adaptor_secret, secret)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "adaptor secret is zero or >= n");
    }
    auto sig = secp256k1::ecdsa_adaptor_adapt(as, secret);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_extract(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t sig64[64],
    uint8_t secret32_out[32]) {
    if (!ctx || !pre_sig || !sig64 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::ECDSAAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    if (!scalar_parse_strict(pre_sig + 65, as.r)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig r");
    }
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);
    secp256k1::ECDSASignature ecdsasig;
    if (!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid ECDSA sig");
    }
    auto [secret, ok] = secp256k1::ecdsa_adaptor_extract(as, ecdsasig);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "ECDSA adaptor extract failed");
    }
    scalar_to_bytes(secret, secret32_out);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    return UFSECP_OK;
}

/* ===========================================================================
 * Pedersen commitments
 * =========================================================================== */

ufsecp_error_t ufsecp_pedersen_commit(ufsecp_ctx* ctx,
                                      const uint8_t value[32],
                                      const uint8_t blinding[32],
                                      uint8_t commitment33_out[33]) {
    if (!ctx || !value || !blinding || !commitment33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar v, b;
    if (!scalar_parse_strict(value, v)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "value >= n");
    }
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    auto c = secp256k1::pedersen_commit(v, b);
    auto comp = c.point.to_compressed();
    std::memcpy(commitment33_out, comp.data(), 33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pedersen_verify(ufsecp_ctx* ctx,
                                      const uint8_t commitment33[33],
                                      const uint8_t value[32],
                                      const uint8_t blinding[32]) {
    if (!ctx || !commitment33 || !value || !blinding) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar v, b;
    if (!scalar_parse_strict(value, v)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "value >= n");
    }
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    auto commit_pt = point_from_compressed(commitment33);
    if (commit_pt.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment point");
    }
    if (!secp256k1::pedersen_verify(secp256k1::PedersenCommitment{commit_pt}, v, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "Pedersen verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pedersen_verify_sum(ufsecp_ctx* ctx,
                                          const uint8_t* pos, size_t n_pos,
                                          const uint8_t* neg, size_t n_neg) {
    if (!ctx || (!pos && n_pos > 0) || (!neg && n_neg > 0)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    std::vector<secp256k1::PedersenCommitment> pcs(n_pos), ncs(n_neg);
    for (size_t i = 0; i < n_pos; ++i) {
        auto p = point_from_compressed(pos + i * 33);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid positive commitment");
        }
        pcs[i] = secp256k1::PedersenCommitment{p};
    }
    for (size_t i = 0; i < n_neg; ++i) {
        auto p = point_from_compressed(neg + i * 33);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid negative commitment");
        }
        ncs[i] = secp256k1::PedersenCommitment{p};
    }
    if (!secp256k1::pedersen_verify_sum(pcs.data(), n_pos, ncs.data(), n_neg)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "Pedersen sum verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pedersen_blind_sum(ufsecp_ctx* ctx,
                                         const uint8_t* blinds_in, size_t n_in,
                                         const uint8_t* blinds_out, size_t n_out,
                                         uint8_t sum32_out[32]) {
    if (!ctx || (!blinds_in && n_in > 0) || (!blinds_out && n_out > 0) || !sum32_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    std::vector<Scalar> ins(n_in), outs(n_out);
    for (size_t i = 0; i < n_in; ++i) {
        if (!scalar_parse_strict(blinds_in + i * 32, ins[i])) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid input blind");
        }
    }
    for (size_t i = 0; i < n_out; ++i) {
        if (!scalar_parse_strict(blinds_out + i * 32, outs[i])) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid output blind");
        }
    }
    auto sum = secp256k1::pedersen_blind_sum(ins.data(), n_in, outs.data(), n_out);
    scalar_to_bytes(sum, sum32_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pedersen_switch_commit(ufsecp_ctx* ctx,
                                             const uint8_t value[32],
                                             const uint8_t blinding[32],
                                             const uint8_t switch_blind[32],
                                             uint8_t commitment33_out[33]) {
    if (!ctx || !value || !blinding || !switch_blind || !commitment33_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar v, b, sb;
    if (!scalar_parse_strict(value, v)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "value >= n");
    }
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    if (!scalar_parse_strict(switch_blind, sb)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "switch_blind >= n");
    }
    auto c = secp256k1::pedersen_switch_commit(v, b, sb);
    auto comp = c.point.to_compressed();
    std::memcpy(commitment33_out, comp.data(), 33);
    return UFSECP_OK;
}

/* ===========================================================================
 * Zero-knowledge proofs
 * =========================================================================== */

ufsecp_error_t ufsecp_zk_knowledge_prove(
    ufsecp_ctx* ctx,
    const uint8_t secret[32],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32],
    const uint8_t aux_rand[32],
    uint8_t proof_out[UFSECP_ZK_KNOWLEDGE_PROOF_LEN]) {
    if (!ctx || !secret || !pubkey33 || !msg32 || !aux_rand || !proof_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar s;
    if (!scalar_parse_strict_nonzero(secret, s)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "secret is zero or >= n");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto proof = secp256k1::zk::knowledge_prove(s, pk, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&s, sizeof(s));
    auto ser = proof.serialize();
    std::memcpy(proof_out, ser.data(), UFSECP_ZK_KNOWLEDGE_PROOF_LEN);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_knowledge_verify(
    ufsecp_ctx* ctx,
    const uint8_t proof[UFSECP_ZK_KNOWLEDGE_PROOF_LEN],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32]) {
    if (!ctx || !proof || !pubkey33 || !msg32) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    secp256k1::zk::KnowledgeProof kp;
    if (!secp256k1::zk::KnowledgeProof::deserialize(proof, kp)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid knowledge proof");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    if (!secp256k1::zk::knowledge_verify(kp, pk, msg_arr)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "knowledge proof failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_dleq_prove(
    ufsecp_ctx* ctx,
    const uint8_t secret[32],
    const uint8_t G33[33], const uint8_t H33[33],
    const uint8_t P33[33], const uint8_t Q33[33],
    const uint8_t aux_rand[32],
    uint8_t proof_out[UFSECP_ZK_DLEQ_PROOF_LEN]) {
    if (!ctx || !secret || !G33 || !H33 || !P33 || !Q33 || !aux_rand || !proof_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar s;
    if (!scalar_parse_strict_nonzero(secret, s)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "secret is zero or >= n");
    }
    auto G = point_from_compressed(G33);
    auto H = point_from_compressed(H33);
    auto P = point_from_compressed(P33);
    auto Q = point_from_compressed(Q33);
    if (G.is_infinity() || H.is_infinity() || P.is_infinity() || Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid DLEQ point");
    }
    std::array<uint8_t, 32> aux_arr;
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto proof = secp256k1::zk::dleq_prove(s, G, H, P, Q, aux_arr);
    secp256k1::detail::secure_erase(&s, sizeof(s));
    auto ser = proof.serialize();
    std::memcpy(proof_out, ser.data(), UFSECP_ZK_DLEQ_PROOF_LEN);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_dleq_verify(
    ufsecp_ctx* ctx,
    const uint8_t proof[UFSECP_ZK_DLEQ_PROOF_LEN],
    const uint8_t G33[33], const uint8_t H33[33],
    const uint8_t P33[33], const uint8_t Q33[33]) {
    if (!ctx || !proof || !G33 || !H33 || !P33 || !Q33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto G = point_from_compressed(G33);
    auto H = point_from_compressed(H33);
    auto P = point_from_compressed(P33);
    auto Q = point_from_compressed(Q33);
    if (G.is_infinity() || H.is_infinity() || P.is_infinity() || Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid DLEQ point");
    }
    secp256k1::zk::DLEQProof dp;
    if (!secp256k1::zk::DLEQProof::deserialize(proof, dp)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid DLEQ proof");
    }
    if (!secp256k1::zk::dleq_verify(dp, G, H, P, Q)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "DLEQ proof failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_range_prove(
    ufsecp_ctx* ctx,
    uint64_t value,
    const uint8_t blinding[32],
    const uint8_t commitment33[33],
    const uint8_t aux_rand[32],
    uint8_t* proof_out, size_t* proof_len) {
    if (!ctx || !blinding || !commitment33 || !aux_rand || !proof_out || !proof_len) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar b;
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    auto commit_pt = point_from_compressed(commitment33);
    if (commit_pt.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment point");
    }
    auto commit = secp256k1::PedersenCommitment{commit_pt};
    std::array<uint8_t, 32> aux_arr;
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto rp = secp256k1::zk::range_prove(value, b, commit, aux_arr);
    /* Serialize range proof: A(33)+S(33)+T1(33)+T2(33)+tau_x(32)+mu(32)+t_hat(32)+L[6]*33+R[6]*33+a(32)+b(32) */
    const size_t needed = 33*4 + 32*3 + 6*33 + 6*33 + 32*2;
    if (*proof_len < needed) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "range proof buffer too small");
    }
    size_t off = 0;
    auto write_point = [&](const Point& p) {
        auto c = p.to_compressed();
        std::memcpy(proof_out + off, c.data(), 33);
        off += 33;
    };
    auto write_scalar = [&](const Scalar& s) {
        scalar_to_bytes(s, proof_out + off);
        off += 32;
    };
    write_point(rp.A); write_point(rp.S);
    write_point(rp.T1); write_point(rp.T2);
    write_scalar(rp.tau_x); write_scalar(rp.mu); write_scalar(rp.t_hat);
    for (int i = 0; i < 6; ++i) write_point(rp.L[i]);
    for (int i = 0; i < 6; ++i) write_point(rp.R[i]);
    write_scalar(rp.a); write_scalar(rp.b);
    *proof_len = off;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_range_verify(
    ufsecp_ctx* ctx,
    const uint8_t commitment33[33],
    const uint8_t* proof, size_t proof_len) {
    if (!ctx || !commitment33 || !proof) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    /* Deserialize range proof */
    const size_t expected = 33*4 + 32*3 + 6*33 + 6*33 + 32*2;
    if (proof_len < expected) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "range proof too short");
    }
    if (proof_len != expected) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "range proof length mismatch");
    }
    secp256k1::zk::RangeProof rp;
    size_t off = 0;
    bool point_ok = true;
    auto read_point = [&]() -> Point {
        auto p = point_from_compressed(proof + off);
        if (p.is_infinity()) point_ok = false;
        off += 33;
        return p;
    };
    bool scalar_ok = true;
    auto read_scalar = [&]() -> Scalar {
        Scalar s;
        if (!scalar_parse_strict(proof + off, s)) {
            scalar_ok = false;
        }
        off += 32;
        return s;
    };
    rp.A = read_point(); rp.S = read_point();
    rp.T1 = read_point(); rp.T2 = read_point();
    rp.tau_x = read_scalar(); rp.mu = read_scalar(); rp.t_hat = read_scalar();
    for (int i = 0; i < 6; ++i) rp.L[i] = read_point();
    for (int i = 0; i < 6; ++i) rp.R[i] = read_point();
    rp.a = read_scalar(); rp.b = read_scalar();
    if (!point_ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid point in range proof");
    }
    if (!scalar_ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid scalar in range proof");
    }
    auto commit_pt = point_from_compressed(commitment33);
    if (commit_pt.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment point");
    }
    auto commit = secp256k1::PedersenCommitment{commit_pt};
    if (!secp256k1::zk::range_verify(commit, rp)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "range proof failed");
    }
    return UFSECP_OK;
}

/* ===========================================================================
 * Multi-coin wallet infrastructure
 * =========================================================================== */

static const secp256k1::coins::CoinParams* find_coin(uint32_t coin_type) {
    return secp256k1::coins::find_by_coin_type(coin_type);
}

ufsecp_error_t ufsecp_coin_address(ufsecp_ctx* ctx,
                                   const uint8_t pubkey33[33],
                                   uint32_t coin_type, int testnet,
                                   char* addr_out, size_t* addr_len) {
    if (!ctx || !pubkey33 || !addr_out || !addr_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto coin = find_coin(coin_type);
    if (!coin) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown coin type");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    auto addr = secp256k1::coins::coin_address(pk, *coin, testnet != 0);
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "address generation failed");
    }
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "address buffer too small");
    }
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_coin_derive_from_seed(
    ufsecp_ctx* ctx,
    const uint8_t* seed, size_t seed_len,
    uint32_t coin_type, uint32_t account, int change, uint32_t index,
    int testnet,
    uint8_t* privkey32_out,
    uint8_t* pubkey33_out,
    char* addr_out, size_t* addr_len) {
    if (!ctx || !seed) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if ((addr_out == nullptr) != (addr_len == nullptr)) {
        return ctx_set_err(ctx, UFSECP_ERR_NULL_ARG,
            "addr_out and addr_len must both be null or both be non-null");
    }
    if (seed_len < 16 || seed_len > 64) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "seed must be 16-64 bytes");
    }
    auto coin = find_coin(coin_type);
    if (!coin) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown coin type");
    }
    /* BIP-32 master */
    auto bip32_result = secp256k1::bip32_master_key(seed, seed_len);
    if (!bip32_result.second) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 master key failed");
    }
    auto master = bip32_result.first;
    const auto cleanup_keys = [&]() {
        secp256k1::detail::secure_erase(master.key.data(), master.key.size());
        secp256k1::detail::secure_erase(master.chain_code.data(), master.chain_code.size());
    };
    /* Derive coin key */
    auto derived = secp256k1::coins::coin_derive_key(
        master, *coin, account, change != 0, index);
    auto key = derived.first;
    bool const d_ok = derived.second;
    if (!d_ok) {
        cleanup_keys();
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "coin key derivation failed");
    }
    const auto cleanup_derived_key = [&]() {
        secp256k1::detail::secure_erase(key.key.data(), key.key.size());
        secp256k1::detail::secure_erase(key.chain_code.data(), key.chain_code.size());
    };
    if (privkey32_out) {
        auto sk = key.private_key();
        scalar_to_bytes(sk, privkey32_out);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
    }
    auto pk = key.public_key();
    cleanup_keys();
    cleanup_derived_key();
    if (pubkey33_out) {
        point_to_compressed(pk, pubkey33_out);
    }
    if (addr_out && addr_len) {
        auto addr = secp256k1::coins::coin_address(pk, *coin, testnet != 0);
        if (*addr_len < addr.size() + 1) {
            return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "address buffer too small");
        }
        std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
        *addr_len = addr.size();
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_coin_wif_encode(ufsecp_ctx* ctx,
                                      const uint8_t privkey[32],
                                      uint32_t coin_type, int testnet,
                                      char* wif_out, size_t* wif_len) {
    if (!ctx || !privkey || !wif_out || !wif_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto coin = find_coin(coin_type);
    if (!coin) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown coin type");
    }
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto wif = secp256k1::coins::coin_wif_encode(sk, *coin, true, testnet != 0);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
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

ufsecp_error_t ufsecp_btc_message_sign(ufsecp_ctx* ctx,
                                       const uint8_t* msg, size_t msg_len,
                                       const uint8_t privkey[32],
                                       char* base64_out, size_t* base64_len) {
    if (!ctx || !msg || !privkey || !base64_out || !base64_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto rsig = secp256k1::coins::bitcoin_sign_message(msg, msg_len, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto b64 = secp256k1::coins::bitcoin_sig_to_base64(rsig);
    if (*base64_len < b64.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "base64 buffer too small");
    }
    std::memcpy(base64_out, b64.c_str(), b64.size() + 1);
    *base64_len = b64.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_btc_message_verify(ufsecp_ctx* ctx,
                                         const uint8_t* msg, size_t msg_len,
                                         const uint8_t pubkey33[33],
                                         const char* base64_sig) {
    if (!ctx || !msg || !pubkey33 || !base64_sig) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    auto dec = secp256k1::coins::bitcoin_sig_from_base64(std::string(base64_sig));
    if (!dec.valid) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid base64 signature");
    }
    if (!secp256k1::coins::bitcoin_verify_message(msg, msg_len, pk, dec.sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "BTC message verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_btc_message_hash(const uint8_t* msg, size_t msg_len,
                                       uint8_t digest32_out[32]) {
    if (!msg || !digest32_out) return UFSECP_ERR_NULL_ARG;
    auto h = secp256k1::coins::bitcoin_message_hash(msg, msg_len);
    std::memcpy(digest32_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-352 Silent Payments
 * =========================================================================== */

ufsecp_error_t ufsecp_silent_payment_address(
    ufsecp_ctx* ctx,
    const uint8_t scan_privkey[32],
    const uint8_t spend_privkey[32],
    uint8_t scan_pubkey33_out[33],
    uint8_t spend_pubkey33_out[33],
    char* addr_out, size_t* addr_len) {
    if (!ctx || !scan_privkey || !spend_privkey || !scan_pubkey33_out ||
        !spend_pubkey33_out || !addr_out || !addr_len) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar scan_sk, spend_sk;
    auto cleanup = [&]() {
        secp256k1::detail::secure_erase(&scan_sk, sizeof(scan_sk));
        secp256k1::detail::secure_erase(&spend_sk, sizeof(spend_sk));
    };
    if (!scalar_parse_strict_nonzero(scan_privkey, scan_sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "scan privkey is zero or >= n");
    }
    if (!scalar_parse_strict_nonzero(spend_privkey, spend_sk)) {
        cleanup();
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "spend privkey is zero or >= n");
    }

    auto spa = secp256k1::silent_payment_address(scan_sk, spend_sk);
    auto scan_comp  = spa.scan_pubkey.to_compressed();
    auto spend_comp = spa.spend_pubkey.to_compressed();
    std::memcpy(scan_pubkey33_out, scan_comp.data(), 33);
    std::memcpy(spend_pubkey33_out, spend_comp.data(), 33);

    auto addr_str = spa.encode();
    if (addr_str.size() >= *addr_len) {
        cleanup();
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "address buffer too small");
    }
    std::memcpy(addr_out, addr_str.c_str(), addr_str.size() + 1);
    *addr_len = addr_str.size();

    cleanup();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_silent_payment_create_output(
    ufsecp_ctx* ctx,
    const uint8_t* input_privkeys, size_t n_inputs,
    const uint8_t scan_pubkey33[33],
    const uint8_t spend_pubkey33[33],
    uint32_t k,
    uint8_t output_pubkey33_out[33],
    uint8_t* tweak32_out) {
    if (!ctx || !input_privkeys || n_inputs == 0 || !scan_pubkey33 ||
        !spend_pubkey33 || !output_pubkey33_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    // Parse input private keys
    std::vector<Scalar> privkeys;
    auto cleanup_privkeys = [&]() {
        for (auto& sk : privkeys) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
        }
    };
    privkeys.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        Scalar sk;
        if (!scalar_parse_strict_nonzero(input_privkeys + i * 32, sk)) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            cleanup_privkeys();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "input privkey is zero or >= n");
        }
        privkeys.push_back(sk);
    }

    // Parse recipient address
    secp256k1::SilentPaymentAddress recipient;
    recipient.scan_pubkey = point_from_compressed(scan_pubkey33);
    recipient.spend_pubkey = point_from_compressed(spend_pubkey33);
    if (recipient.scan_pubkey.is_infinity() || recipient.spend_pubkey.is_infinity()) {
        cleanup_privkeys();
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid recipient pubkey");
    }

    auto [output_point, tweak] = secp256k1::silent_payment_create_output(privkeys, recipient, k);
    if (output_point.is_infinity()) {
        cleanup_privkeys();
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "output point is infinity");
    }

    auto out_comp = output_point.to_compressed();
    std::memcpy(output_pubkey33_out, out_comp.data(), 33);

    if (tweak32_out) {
        auto tweak_bytes = tweak.to_bytes();
        std::memcpy(tweak32_out, tweak_bytes.data(), 32);
    }

    cleanup_privkeys();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_silent_payment_scan(
    ufsecp_ctx* ctx,
    const uint8_t scan_privkey[32],
    const uint8_t spend_privkey[32],
    const uint8_t* input_pubkeys33, size_t n_input_pubkeys,
    const uint8_t* output_xonly32, size_t n_outputs,
    uint32_t* found_indices_out,
    uint8_t* found_privkeys_out,
    size_t* n_found) {
    if (!ctx || !scan_privkey || !spend_privkey || !input_pubkeys33 ||
        !output_xonly32 || !n_found) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (n_input_pubkeys == 0 || n_outputs == 0) {
        return UFSECP_ERR_BAD_INPUT;
    }
    ctx_clear_err(ctx);

    Scalar scan_sk, spend_sk;
    auto cleanup = [&]() {
        secp256k1::detail::secure_erase(&scan_sk, sizeof(scan_sk));
        secp256k1::detail::secure_erase(&spend_sk, sizeof(spend_sk));
    };
    if (!scalar_parse_strict_nonzero(scan_privkey, scan_sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "scan privkey is zero or >= n");
    }
    if (!scalar_parse_strict_nonzero(spend_privkey, spend_sk)) {
        cleanup();
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "spend privkey is zero or >= n");
    }

    // Parse input pubkeys
    std::vector<Point> input_pks;
    input_pks.reserve(n_input_pubkeys);
    for (size_t i = 0; i < n_input_pubkeys; ++i) {
        auto pk = point_from_compressed(input_pubkeys33 + i * 33);
        if (pk.is_infinity()) {
            cleanup();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid input pubkey");
        }
        input_pks.push_back(pk);
    }

    // Parse output x-only pubkeys
    std::vector<std::array<uint8_t, 32>> outputs;
    outputs.reserve(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        std::array<uint8_t, 32> x;
        std::memcpy(x.data(), output_xonly32 + i * 32, 32);
        outputs.push_back(x);
    }

    auto results = secp256k1::silent_payment_scan(scan_sk, spend_sk, input_pks, outputs);

    size_t const capacity = *n_found;
    size_t const count = results.size() < capacity ? results.size() : capacity;
    *n_found = results.size();

    for (size_t i = 0; i < count; ++i) {
        if (found_indices_out) found_indices_out[i] = results[i].first;
        if (found_privkeys_out) {
            auto key_bytes = results[i].second.to_bytes();
            std::memcpy(found_privkeys_out + i * 32, key_bytes.data(), 32);
            secp256k1::detail::secure_erase(key_bytes.data(), key_bytes.size());
        }
    }

    // Erase result private keys from heap before vector destruction
    for (auto& r : results) {
        secp256k1::detail::secure_erase(&r.second, sizeof(r.second));
    }

    cleanup();
    return UFSECP_OK;
}

/* ===========================================================================
 * ECIES (Elliptic Curve Integrated Encryption Scheme)
 * =========================================================================== */

ufsecp_error_t ufsecp_ecies_encrypt(
    ufsecp_ctx* ctx,
    const uint8_t recipient_pubkey33[33],
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* envelope_out, size_t* envelope_len) {
    if (!ctx || !recipient_pubkey33 || !plaintext || !envelope_out || !envelope_len) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (plaintext_len == 0) {
        return UFSECP_ERR_BAD_INPUT;
    }
    ctx_clear_err(ctx);

    if (plaintext_len > SIZE_MAX - UFSECP_ECIES_OVERHEAD) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "plaintext_len too large");
    }
    size_t const needed = plaintext_len + UFSECP_ECIES_OVERHEAD;
    if (*envelope_len < needed) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "envelope buffer too small");
    }

    auto pk = point_from_compressed(recipient_pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid recipient pubkey");
    }

    auto envelope = secp256k1::ecies_encrypt(pk, plaintext, plaintext_len);
    if (envelope.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "ECIES encryption failed");
    }

    std::memcpy(envelope_out, envelope.data(), envelope.size());
    *envelope_len = envelope.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecies_decrypt(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t* envelope, size_t envelope_len,
    uint8_t* plaintext_out, size_t* plaintext_len) {
    if (!ctx || !privkey || !envelope || !plaintext_out || !plaintext_len) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (envelope_len < 82) { // min: 33 + 16 + 1 + 32
        return UFSECP_ERR_BAD_INPUT;
    }
    ctx_clear_err(ctx);

    size_t const expected_pt_len = envelope_len - UFSECP_ECIES_OVERHEAD;
    if (*plaintext_len < expected_pt_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "plaintext buffer too small");
    }

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto pt = secp256k1::ecies_decrypt(sk, envelope, envelope_len);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));

    if (pt.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECIES decryption failed (bad key or tampered)");
    }

    std::memcpy(plaintext_out, pt.data(), pt.size());
    *plaintext_len = pt.size();
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-324: Version 2 P2P Encrypted Transport (conditional: SECP256K1_BIP324)
 * =========================================================================== */

#if defined(SECP256K1_BIP324)

struct ufsecp_bip324_session {
    secp256k1::Bip324Session* cpp_session;
};

ufsecp_error_t ufsecp_bip324_create(
    ufsecp_ctx* ctx,
    int initiator,
    ufsecp_bip324_session** session_out,
    uint8_t ellswift64_out[64]) {
    if (!ctx || !session_out || !ellswift64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    *session_out = nullptr;
    if (initiator != 0 && initiator != 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "initiator must be 0 or 1");
    }

    auto* sess = new (std::nothrow) ufsecp_bip324_session{};
    if (!sess) return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "allocation failed");

    sess->cpp_session = new (std::nothrow) secp256k1::Bip324Session(initiator == 1);
    if (!sess->cpp_session) {
        delete sess;
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "allocation failed");
    }

    auto& enc = sess->cpp_session->our_ellswift_encoding();
    std::memcpy(ellswift64_out, enc.data(), 64);

    *session_out = sess;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip324_handshake(
    ufsecp_bip324_session* session,
    const uint8_t peer_ellswift64[64],
    uint8_t session_id32_out[32]) {
    if (!session || !session->cpp_session || !peer_ellswift64) return UFSECP_ERR_NULL_ARG;

    if (!session->cpp_session->complete_handshake(peer_ellswift64)) {
        return UFSECP_ERR_INTERNAL;
    }

    if (session_id32_out) {
        auto& sid = session->cpp_session->session_id();
        std::memcpy(session_id32_out, sid.data(), 32);
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip324_encrypt(
    ufsecp_bip324_session* session,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* out, size_t* out_len) {
    if (!session || !session->cpp_session || !out || !out_len) return UFSECP_ERR_NULL_ARG;
    if (!plaintext && plaintext_len > 0) return UFSECP_ERR_NULL_ARG;
    if (plaintext_len > SIZE_MAX - 19) return UFSECP_ERR_BAD_INPUT;

    size_t const needed = plaintext_len + 19; // 3 (length) + payload + 16 (tag)
    if (*out_len < needed) return UFSECP_ERR_BUF_TOO_SMALL;

    auto enc = session->cpp_session->encrypt(plaintext, plaintext_len);
    if (enc.empty()) return UFSECP_ERR_INTERNAL;

    std::memcpy(out, enc.data(), enc.size());
    *out_len = enc.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip324_decrypt(
    ufsecp_bip324_session* session,
    const uint8_t* encrypted, size_t encrypted_len,
    uint8_t* plaintext_out, size_t* plaintext_len) {
    if (!session || !session->cpp_session || !encrypted || !plaintext_out || !plaintext_len)
        return UFSECP_ERR_NULL_ARG;

    // encrypted = [3B header][payload][16B tag], minimum length 19
    if (encrypted_len < 19) return UFSECP_ERR_BUF_TOO_SMALL;

    const uint8_t* header = encrypted;
    const uint8_t* payload_tag = encrypted + 3;
    const size_t payload_tag_len = encrypted_len - 3;

    std::vector<uint8_t> dec;
    if (!session->cpp_session->decrypt(header, payload_tag, payload_tag_len, dec)) {
        return UFSECP_ERR_VERIFY_FAIL;
    }

    if (*plaintext_len < dec.size()) return UFSECP_ERR_BUF_TOO_SMALL;
    std::memcpy(plaintext_out, dec.data(), dec.size());
    *plaintext_len = dec.size();
    return UFSECP_OK;
}

void ufsecp_bip324_destroy(ufsecp_bip324_session* session) {
    if (session) {
        if (session->cpp_session) {
            delete session->cpp_session;
        }
        delete session;
    }
}

ufsecp_error_t ufsecp_aead_chacha20_poly1305_encrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const uint8_t* aad, size_t aad_len,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* out, uint8_t tag[16]) {
    if (!key || !nonce || !out || !tag) return UFSECP_ERR_NULL_ARG;
    if (!plaintext && plaintext_len > 0) return UFSECP_ERR_NULL_ARG;
    if (!aad && aad_len > 0) return UFSECP_ERR_NULL_ARG;

    secp256k1::aead_chacha20_poly1305_encrypt(
        key, nonce, aad, aad_len, plaintext, plaintext_len, out, tag);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_aead_chacha20_poly1305_decrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const uint8_t* aad, size_t aad_len,
    const uint8_t* ciphertext, size_t ciphertext_len,
    const uint8_t tag[16], uint8_t* out) {
    if (!key || !nonce || !tag || !out) return UFSECP_ERR_NULL_ARG;
    if (!ciphertext && ciphertext_len > 0) return UFSECP_ERR_NULL_ARG;
    if (!aad && aad_len > 0) return UFSECP_ERR_NULL_ARG;

    bool ok = secp256k1::aead_chacha20_poly1305_decrypt(
        key, nonce, aad, aad_len, ciphertext, ciphertext_len, tag, out);
    return ok ? UFSECP_OK : UFSECP_ERR_VERIFY_FAIL;
}

ufsecp_error_t ufsecp_ellswift_create(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    uint8_t encoding64_out[64]) {
    if (!ctx || !privkey || !encoding64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto enc = secp256k1::ellswift_create(sk);
    std::memcpy(encoding64_out, enc.data(), 64);

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ellswift_xdh(
    ufsecp_ctx* ctx,
    const uint8_t ell_a64[64],
    const uint8_t ell_b64[64],
    const uint8_t our_privkey[32],
    int initiating,
    uint8_t secret32_out[32]) {
    if (!ctx || !ell_a64 || !ell_b64 || !our_privkey || !secret32_out)
        return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(our_privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto secret = secp256k1::ellswift_xdh(ell_a64, ell_b64, sk, initiating != 0);
    std::memcpy(secret32_out, secret.data(), 32);

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

#endif /* SECP256K1_BIP324 */

/* ===========================================================================
 * Ethereum (conditional: SECP256K1_BUILD_ETHEREUM)
 * =========================================================================== */

#if defined(SECP256K1_BUILD_ETHEREUM)

ufsecp_error_t ufsecp_keccak256(const uint8_t* data, size_t len,
                                uint8_t digest32_out[32]) {
    if (!data && len > 0) return UFSECP_ERR_NULL_ARG;
    if (!digest32_out) return UFSECP_ERR_NULL_ARG;

    auto hash = secp256k1::coins::keccak256(data, len);
    std::memcpy(digest32_out, hash.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_address(ufsecp_ctx* ctx,
                                  const uint8_t pubkey33[33],
                                  uint8_t addr20_out[20]) {
    if (!ctx || !pubkey33 || !addr20_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    const Point pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid compressed pubkey");
    }

    auto addr = secp256k1::coins::ethereum_address_bytes(pk);
    std::memcpy(addr20_out, addr.data(), 20);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_address_checksummed(ufsecp_ctx* ctx,
                                              const uint8_t pubkey33[33],
                                              char* addr_out, size_t* addr_len) {
    if (!ctx || !pubkey33 || !addr_out || !addr_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (*addr_len < 43) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "need >= 43 bytes for ETH address");
    }

    const Point pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid compressed pubkey");
    }

    const std::string addr_str = secp256k1::coins::ethereum_address(pk);
    std::memcpy(addr_out, addr_str.c_str(), addr_str.size());
    addr_out[addr_str.size()] = '\0';
    *addr_len = addr_str.size();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_personal_hash(const uint8_t* msg, size_t msg_len,
                                        uint8_t digest32_out[32]) {
    if (!msg && msg_len > 0) return UFSECP_ERR_NULL_ARG;
    if (!digest32_out) return UFSECP_ERR_NULL_ARG;

    auto hash = secp256k1::coins::eip191_hash(msg, msg_len);
    std::memcpy(digest32_out, hash.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_sign(ufsecp_ctx* ctx,
                               const uint8_t msg32[32],
                               const uint8_t privkey[32],
                               uint8_t r_out[32],
                               uint8_t s_out[32],
                               uint64_t* v_out,
                               uint64_t chain_id) {
    if (!ctx || !msg32 || !privkey || !r_out || !s_out || !v_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    std::array<uint8_t, 32> hash;
    std::memcpy(hash.data(), msg32, 32);

    auto esig = secp256k1::coins::eth_sign_hash(hash, sk, chain_id);
    std::memcpy(r_out, esig.r.data(), 32);
    std::memcpy(s_out, esig.s.data(), 32);
    *v_out = esig.v;

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_ecrecover(ufsecp_ctx* ctx,
                                    const uint8_t msg32[32],
                                    const uint8_t r[32],
                                    const uint8_t s[32],
                                    uint64_t v,
                                    uint8_t addr20_out[20]) {
    if (!ctx || !msg32 || !r || !s || !addr20_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> hash, r_arr, s_arr;
    std::memcpy(hash.data(), msg32, 32);
    std::memcpy(r_arr.data(), r, 32);
    std::memcpy(s_arr.data(), s, 32);

    auto [addr, ok] = secp256k1::coins::ecrecover(hash, r_arr, s_arr, v);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "ecrecover failed");
    }

    std::memcpy(addr20_out, addr.data(), 20);
    return UFSECP_OK;
}

#endif /* SECP256K1_BUILD_ETHEREUM */
