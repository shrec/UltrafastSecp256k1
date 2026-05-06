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

#include <atomic>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <new>
#include <vector>

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
#include "secp256k1/detail/arith64.hpp"
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
    // Atomic: ufsecp_context_randomize is safe to call concurrently on distinct
    // contexts (blinding state is thread_local). last_err is shared per-ctx and
    // written by ctx_clear_err/ctx_set_err on every API call, so it must be
    // atomic to avoid a TSan data race when two threads share one context.
    std::atomic<int>  last_err;
    // BUG-4 FIX: last_msg is no longer written here — it is now thread_local
    // (see tl_last_msg below).  The field is kept for ABI stability (sizeof
    // ufsecp_ctx must not change across minor versions), but is zero-initialized
    // and never read by the library.
    char              last_msg[128];
    bool              selftest_ok;
};

// BUG-4 FIX: per-thread error message storage.
// ctx_set_err writes here; ufsecp_last_error_msg reads here.
// Eliminates the TSan data race when two threads share one ufsecp_ctx and
// both encounter errors concurrently.  Each thread sees its own most-recent
// error message, which is the only meaningful semantic for a shared context.
static thread_local char tl_last_msg[128] = {};

static void ctx_clear_err(ufsecp_ctx* ctx) {
    ctx->last_err.store(UFSECP_OK, std::memory_order_relaxed);
    // Do NOT write last_msg here — concurrent API calls on a shared ctx (e.g.
    // TLB-2/TLB-3 thread-local-blinding tests) would race on this char array.
    // last_msg is only meaningful when last_err != OK; ufsecp_last_error_msg
    // guards on err != UFSECP_OK before consulting it.
}

static ufsecp_error_t ctx_set_err(ufsecp_ctx* ctx, ufsecp_error_t err, const char* msg) {
    ctx->last_err.store(err, std::memory_order_relaxed);
    // BUG-4 FIX: write to thread_local tl_last_msg, not ctx->last_msg.
    // ctx->last_msg is non-atomic and was a TSan data race when two threads
    // shared a context and both hit errors simultaneously.
    if (msg) {
        size_t i = 0;
        for (; i < sizeof(tl_last_msg) - 1 && msg[i]; ++i) {
            tl_last_msg[i] = msg[i];
        }
        tl_last_msg[i] = '\0';
    } else {
        tl_last_msg[0] = '\0';
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

// BUG-6 FIX: deterministic little-endian uint32_t serialization helpers.
// std::memcpy(&u32, buf, 4) uses native endianness and produces wrong results
// on big-endian platforms (s390x, PowerPC).  These helpers guarantee LE layout
// on all platforms, matching secp256k1 conventions.
static inline uint32_t read_le32(const uint8_t* p) noexcept {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}
static inline void write_le32(uint8_t* p, uint32_t v) noexcept {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}

static ufsecp_error_t parse_musig2_keyagg(ufsecp_ctx* ctx,
                                          const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
                                          secp256k1::MuSig2KeyAggCtx& out) {
    uint32_t nk = read_le32(keyagg);
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

    participant_count_out = read_le32(session + kMuSig2SessionCountOffset);
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

/* Hard upper bound on user-supplied batch/array counts.
   1 << 20 (~1M) is generous for any legitimate use case and prevents
   hostile callers from triggering multi-GB allocations.               */
static constexpr std::size_t kMaxBatchN = std::size_t{1} << 20;

} // namespace

/* ---------------------------------------------------------------------------
 * Exception-safety macro for extern "C" functions.
 *
 * C++ exceptions propagating through an extern "C" boundary are undefined
 * behaviour.  Every function that touches STL containers, std::string, or
 * any other throwing code must be wrapped.
 * ---------------------------------------------------------------------------*/
#define UFSECP_CATCH_RETURN(ctx_ptr)                                          \
    catch (const std::bad_alloc&) {                                           \
        return (ctx_ptr) ? ctx_set_err(ctx_ptr, UFSECP_ERR_INTERNAL,          \
                                       "allocation failed")                   \
                         : UFSECP_ERR_INTERNAL;                               \
    } catch (...) {                                                           \
        return (ctx_ptr) ? ctx_set_err(ctx_ptr, UFSECP_ERR_INTERNAL,          \
                                       "internal error")                      \
                         : UFSECP_ERR_INTERNAL;                               \
    }

// RAII guard: calls secure_erase on the pointed-to object when destroyed.
// Guarantees erasure on ALL exit paths (normal return, early return, exception).
template<typename T>
struct ScopeSecureErase {
    T*          ptr_;
    std::size_t sz_;
    ScopeSecureErase(T* p, std::size_t s) noexcept : ptr_(p), sz_(s) {}
    ~ScopeSecureErase() noexcept { secp256k1::detail::secure_erase(ptr_, sz_); }
    ScopeSecureErase(const ScopeSecureErase&)            = delete;
    ScopeSecureErase& operator=(const ScopeSecureErase&) = delete;
};

// RAII scope-exit guard: calls a callable on destruction.
// Used for multi-field or container cleanup on all exit paths.
template<typename F>
struct ScopeExit {
    F fn_;
    explicit ScopeExit(F fn) noexcept : fn_(std::move(fn)) {}
    ~ScopeExit() noexcept { fn_(); }
    ScopeExit(const ScopeExit&)            = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
};

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

static bool valid_network(int n) {
    return n == UFSECP_NET_MAINNET || n == UFSECP_NET_TESTNET;
}

static secp256k1::Network to_network(int n) {
    return n == UFSECP_NET_TESTNET ? secp256k1::Network::Testnet
                                   : secp256k1::Network::Mainnet;
}

/* ===========================================================================
 * Version / error (stateless, no ctx needed)
 * =========================================================================== */


/* ============================================================================
 * Domain implementation files (unity build)
 * Each file covers one functional domain; see include/ufsecp/impl/
 * ============================================================================ */

#include "impl/ufsecp_core.cpp"
#include "impl/ufsecp_ecdsa.cpp"
#include "impl/ufsecp_address.cpp"
#include "impl/ufsecp_taproot.cpp"
#include "impl/ufsecp_musig2.cpp"
#include "impl/ufsecp_zk.cpp"
#include "impl/ufsecp_coins.cpp"
#include "impl/ufsecp_bip322.cpp"

