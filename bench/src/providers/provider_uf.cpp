// provider_uf.cpp -- IProvider implementation for UltrafastSecp256k1
//
// Stateless API: no context object needed.
// Uses the C++ API from cpu/include/*.hpp
//
// Build: link against fastsecp256k1 (CMake target)
// Include dirs: cpu/include, include (via target_include_directories)

#include "bench_api.h"
#include <cstring>

// UF headers
#include <UltrafastSecp256k1.hpp>
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/schnorr.hpp>
#include <secp256k1/ecdh.hpp>
#include <secp256k1/precompute.hpp>

namespace bench {
namespace {

// ------------------------------------------------------------------
// Internal helpers: reinterpret opaque ParsedPubkey/ParsedSig buffers
// ------------------------------------------------------------------

// Store a fast::Point in a ParsedPubkey buffer
inline void store_point(ParsedPubkey* dst, const secp256k1::fast::Point& p) {
    static_assert(sizeof(secp256k1::fast::Point) <= sizeof(dst->data),
                  "ParsedPubkey buffer too small for fast::Point");
    new (dst->data) secp256k1::fast::Point(p);
}

inline const secp256k1::fast::Point& load_point(const ParsedPubkey* src) {
    return *reinterpret_cast<const secp256k1::fast::Point*>(src->data);
}

// Store an ECDSASignature in a ParsedSig buffer
inline void store_ecdsa_sig(ParsedSig* dst, const secp256k1::ECDSASignature& sig) {
    static_assert(sizeof(secp256k1::ECDSASignature) <= sizeof(dst->data),
                  "ParsedSig buffer too small for ECDSASignature");
    new (dst->data) secp256k1::ECDSASignature(sig);
}

inline const secp256k1::ECDSASignature& load_ecdsa_sig(const ParsedSig* src) {
    return *reinterpret_cast<const secp256k1::ECDSASignature*>(src->data);
}

// Store SchnorrXonlyPubkey in ParsedXonlyPubkey buffer
inline void store_xonly(ParsedXonlyPubkey* dst, const secp256k1::SchnorrXonlyPubkey& xpk) {
    static_assert(sizeof(secp256k1::SchnorrXonlyPubkey) <= sizeof(dst->data),
                  "ParsedXonlyPubkey buffer too small for SchnorrXonlyPubkey");
    new (dst->data) secp256k1::SchnorrXonlyPubkey(xpk);
}

inline const secp256k1::SchnorrXonlyPubkey& load_xonly(const ParsedXonlyPubkey* src) {
    return *reinterpret_cast<const secp256k1::SchnorrXonlyPubkey*>(src->data);
}

// Decompress a 33-byte compressed pubkey to a fast::Point
// Returns false on invalid pubkey.
// Uses FE52 (5x52) arithmetic for sqrt -- ~40% faster than 4x64 FieldElement.
bool decompress_pubkey(secp256k1::fast::Point& out, const uint8_t* pubkey33, size_t len) {
    if (len != 33) return false;
    uint8_t prefix = pubkey33[0];
    if (prefix != 0x02 && prefix != 0x03) return false;

    using FE52 = secp256k1::fast::FieldElement52;

    // Parse x directly from big-endian bytes into 5x52 representation
    auto x = FE52::from_bytes(pubkey33 + 1);

    // y^2 = x^3 + 7  (all in native FE52 -- no 4x64 conversion)
    auto x3 = x.square() * x;
    static constexpr FE52 SEVEN{{7, 0, 0, 0, 0}};
    auto rhs = x3 + SEVEN;
    auto y = rhs.sqrt();

    // Check parity from normalized FE52 limbs.
    // After normalize(), n[0] bit 0 == parity of the canonical value.
    y.normalize();
    bool y_is_odd = (y.n[0] & 1) != 0;
    bool want_odd = (prefix == 0x03);
    if (y_is_odd != want_odd) {
        y.negate_assign(1);
    }

    // Zero-conversion construction: directly from FE52 coords
    out = secp256k1::fast::Point::from_affine52(x, y);
    return true;
}

// Convert 64-byte compact sig to ECDSASignature
bool parse_compact_sig(secp256k1::ECDSASignature& out, const uint8_t* sig64) {
    out = secp256k1::ECDSASignature::from_compact(sig64);
    return true;
}

// Parse DER-encoded ECDSA signature
// UF has no from_der() in C++ API. We implement a minimal DER parser.
bool parse_der_sig(secp256k1::ECDSASignature& out, const uint8_t* sig, size_t len) {
    // DER: 0x30 <len> 0x02 <rlen> <r> 0x02 <slen> <s>
    if (len < 8 || sig[0] != 0x30) return false;

    size_t total_len = sig[1];
    if (total_len + 2 != len) return false;

    size_t pos = 2;
    if (sig[pos] != 0x02) return false;
    pos++;
    size_t rlen = sig[pos]; pos++;
    if (pos + rlen + 2 > len) return false;

    // Extract r (strip leading zero padding)
    const uint8_t* r_ptr = sig + pos;
    size_t r_actual_len = rlen;
    while (r_actual_len > 32 && *r_ptr == 0) { r_ptr++; r_actual_len--; }
    if (r_actual_len > 32) return false;

    std::array<uint8_t, 32> r_bytes{};
    std::memcpy(r_bytes.data() + (32 - r_actual_len), r_ptr, r_actual_len);

    pos += rlen;
    if (sig[pos] != 0x02) return false;
    pos++;
    size_t slen = sig[pos]; pos++;
    if (pos + slen > len) return false;

    // Extract s
    const uint8_t* s_ptr = sig + pos;
    size_t s_actual_len = slen;
    while (s_actual_len > 32 && *s_ptr == 0) { s_ptr++; s_actual_len--; }
    if (s_actual_len > 32) return false;

    std::array<uint8_t, 32> s_bytes{};
    std::memcpy(s_bytes.data() + (32 - s_actual_len), s_ptr, s_actual_len);

    auto r_scalar = secp256k1::fast::Scalar::from_bytes(r_bytes);
    auto s_scalar = secp256k1::fast::Scalar::from_bytes(s_bytes);

    out.r = r_scalar;
    out.s = s_scalar;
    return true;
}

// ------------------------------------------------------------------
class ProviderUF final : public IProvider {
public:
    const char* name() const override { return "UltrafastSecp256k1"; }
    const char* version() const override { return "3.14.0"; }

    ProviderCaps caps() const override {
        return {true, true, true, true};
    }

    bool init(bool /*randomize_ctx*/) override {
        // UF is stateless -- no context to randomize.
        // Ensure precomputed tables are ready for scalar_mul_generator.
        secp256k1::fast::ensure_fixed_base_ready();
        return true;
    }

    void shutdown() override {
        // No-op; UF is stateless.
    }

    // -- ECDSA verify (bytes) -----------------------------------------------
    bool ecdsa_verify_bytes(
        const uint8_t* pubkey33, size_t pubkey_len,
        const uint8_t* sig, size_t sig_len,
        const uint8_t* msg32,
        bool normalize_low_s) override
    {
        secp256k1::fast::Point pk;
        if (!decompress_pubkey(pk, pubkey33, pubkey_len)) return false;

        secp256k1::ECDSASignature ecdsa_sig;
        if (sig_len == 64) {
            if (!parse_compact_sig(ecdsa_sig, sig)) return false;
        } else {
            if (!parse_der_sig(ecdsa_sig, sig, sig_len)) return false;
        }

        if (normalize_low_s && !ecdsa_sig.is_low_s()) {
            ecdsa_sig = ecdsa_sig.normalize();
        }

        return secp256k1::ecdsa_verify(msg32, pk, ecdsa_sig);
    }

    // -- ECDSA parse pubkey -------------------------------------------------
    bool ecdsa_parse_pubkey(ParsedPubkey* out,
        const uint8_t* pubkey33, size_t len) override
    {
        secp256k1::fast::Point pk;
        if (!decompress_pubkey(pk, pubkey33, len)) return false;
        store_point(out, pk);
        return true;
    }

    // -- ECDSA parse sig ----------------------------------------------------
    bool ecdsa_parse_sig(ParsedSig* out,
        const uint8_t* sig, size_t sig_len, bool is_der) override
    {
        secp256k1::ECDSASignature ecdsa_sig;
        if (is_der) {
            if (!parse_der_sig(ecdsa_sig, sig, sig_len)) return false;
        } else {
            if (sig_len != 64) return false;
            if (!parse_compact_sig(ecdsa_sig, sig)) return false;
        }
        store_ecdsa_sig(out, ecdsa_sig);
        return true;
    }

    // -- ECDSA verify (preparsed) -------------------------------------------
    bool ecdsa_verify_preparsed(
        const ParsedPubkey* pubkey,
        const ParsedSig* sig,
        const uint8_t* msg32,
        bool normalize_low_s) override
    {
        const auto& pk = load_point(pubkey);
        auto ecdsa_sig = load_ecdsa_sig(sig);

        if (normalize_low_s && !ecdsa_sig.is_low_s()) {
            ecdsa_sig = ecdsa_sig.normalize();
        }

        return secp256k1::ecdsa_verify(msg32, pk, ecdsa_sig);
    }

    // -- Schnorr verify (bytes) ---------------------------------------------
    bool schnorr_verify_bytes(
        const uint8_t* xonly_pubkey32,
        const uint8_t* sig64,
        const uint8_t* msg32) override
    {
        auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig64);

        return secp256k1::schnorr_verify(xonly_pubkey32, msg32, schnorr_sig);
    }

    // -- Schnorr parse xonly ------------------------------------------------
    bool schnorr_parse_xonly(ParsedXonlyPubkey* out,
        const uint8_t* xonly32) override
    {
        secp256k1::SchnorrXonlyPubkey xpk;
        if (!secp256k1::schnorr_xonly_pubkey_parse(xpk, xonly32)) return false;
        store_xonly(out, xpk);
        return true;
    }

    // -- Schnorr verify (preparsed) -----------------------------------------
    bool schnorr_verify_preparsed(
        const ParsedXonlyPubkey* pubkey,
        const uint8_t* sig64,
        const uint8_t* msg32) override
    {
        const auto& xpk = load_xonly(pubkey);

        auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig64);

        return secp256k1::schnorr_verify(xpk, msg32, schnorr_sig);
    }

    // -- pubkey_create (k * G) ----------------------------------------------
    bool pubkey_create(uint8_t* out33,
        const uint8_t* seckey32) override
    {
        auto scalar = secp256k1::fast::Scalar::from_bytes(seckey32);

        if (scalar.is_zero()) return false;

        auto point = secp256k1::fast::scalar_mul_generator(scalar);
        auto compressed = point.to_compressed();
        std::memcpy(out33, compressed.data(), 33);
        return true;
    }

    // -- ECDH ---------------------------------------------------------------
    bool ecdh(uint8_t* out32,
        const uint8_t* seckey32,
        const uint8_t* pubkey33, size_t pubkey_len) override
    {
        secp256k1::fast::Point pk;
        if (!decompress_pubkey(pk, pubkey33, pubkey_len)) return false;

        auto scalar = secp256k1::fast::Scalar::from_bytes(seckey32);

        if (scalar.is_zero()) return false;

        auto secret = secp256k1::ecdh_compute_xonly(scalar, pk);
        std::memcpy(out32, secret.data(), 32);
        return true;
    }
};

} // anon namespace

std::unique_ptr<IProvider> create_provider_uf() {
    return std::make_unique<ProviderUF>();
}

} // namespace bench
