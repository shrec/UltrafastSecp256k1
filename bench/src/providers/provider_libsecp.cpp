// provider_libsecp.cpp -- IProvider implementation for bitcoin-core/libsecp256k1
//
// Uses the standard C API from secp256k1.h + modules.
// Build: link against secp256k1 target (via FetchContent).

#include "bench_api.h"
#include <cstring>
#include <cstdio>

// libsecp256k1 C headers
#include <secp256k1.h>
#include <secp256k1_ecdh.h>
#include <secp256k1_schnorrsig.h>
#include <secp256k1_extrakeys.h>

namespace bench {
namespace {

// ------------------------------------------------------------------
class ProviderLibsecp final : public IProvider {
    secp256k1_context* ctx_ = nullptr;

public:
    const char* name() const override { return "libsecp256k1"; }
    const char* version() const override { return "0.6.0"; }

    ProviderCaps caps() const override {
        return {true, true, true, true};
    }

    bool init(bool randomize_ctx) override {
        ctx_ = secp256k1_context_create(
            SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
        if (!ctx_) return false;

        if (randomize_ctx) {
            // Use a fixed seed for reproducibility
            uint8_t seed[32];
            std::memset(seed, 0x42, sizeof(seed));
            if (!secp256k1_context_randomize(ctx_, seed)) {
                std::fprintf(stderr,
                    "[libsecp] WARNING: context randomization failed\n");
            }
        }
        return true;
    }

    void shutdown() override {
        if (ctx_) {
            secp256k1_context_destroy(ctx_);
            ctx_ = nullptr;
        }
    }

    // -- ECDSA verify (bytes) -----------------------------------------------
    bool ecdsa_verify_bytes(
        const uint8_t* pubkey_data, size_t pubkey_len,
        const uint8_t* sig, size_t sig_len,
        const uint8_t* msg32,
        bool normalize_low_s) override
    {
        secp256k1_pubkey pk;
        if (!secp256k1_ec_pubkey_parse(ctx_, &pk, pubkey_data, pubkey_len))
            return false;

        secp256k1_ecdsa_signature ecdsa_sig;
        if (sig_len == 64) {
            if (!secp256k1_ecdsa_signature_parse_compact(ctx_, &ecdsa_sig, sig))
                return false;
        } else {
            if (!secp256k1_ecdsa_signature_parse_der(ctx_, &ecdsa_sig, sig, sig_len))
                return false;
        }

        if (normalize_low_s) {
            secp256k1_ecdsa_signature_normalize(ctx_, &ecdsa_sig, &ecdsa_sig);
        }

        return secp256k1_ecdsa_verify(ctx_, &ecdsa_sig, msg32, &pk) == 1;
    }

    // -- ECDSA parse pubkey -------------------------------------------------
    bool ecdsa_parse_pubkey(ParsedPubkey* out,
        const uint8_t* pubkey_data, size_t len) override
    {
        static_assert(sizeof(secp256k1_pubkey) <= sizeof(out->data),
                      "ParsedPubkey too small for secp256k1_pubkey");
        auto* pk = reinterpret_cast<secp256k1_pubkey*>(out->data);
        return secp256k1_ec_pubkey_parse(ctx_, pk, pubkey_data, len) == 1;
    }

    // -- ECDSA parse sig ----------------------------------------------------
    bool ecdsa_parse_sig(ParsedSig* out,
        const uint8_t* sig, size_t sig_len, bool is_der) override
    {
        static_assert(sizeof(secp256k1_ecdsa_signature) <= sizeof(out->data),
                      "ParsedSig too small for secp256k1_ecdsa_signature");
        auto* s = reinterpret_cast<secp256k1_ecdsa_signature*>(out->data);
        if (is_der) {
            return secp256k1_ecdsa_signature_parse_der(ctx_, s, sig, sig_len) == 1;
        } else {
            if (sig_len != 64) return false;
            return secp256k1_ecdsa_signature_parse_compact(ctx_, s, sig) == 1;
        }
    }

    // -- ECDSA verify (preparsed) -------------------------------------------
    bool ecdsa_verify_preparsed(
        const ParsedPubkey* pubkey,
        const ParsedSig* sig,
        const uint8_t* msg32,
        bool normalize_low_s) override
    {
        const auto* pk = reinterpret_cast<const secp256k1_pubkey*>(pubkey->data);
        // Copy because normalize is in-place
        secp256k1_ecdsa_signature ecdsa_sig;
        std::memcpy(&ecdsa_sig,
                    reinterpret_cast<const secp256k1_ecdsa_signature*>(sig->data),
                    sizeof(ecdsa_sig));

        if (normalize_low_s) {
            secp256k1_ecdsa_signature_normalize(ctx_, &ecdsa_sig, &ecdsa_sig);
        }

        return secp256k1_ecdsa_verify(ctx_, &ecdsa_sig, msg32, pk) == 1;
    }

    // -- Schnorr verify (bytes) ---------------------------------------------
    bool schnorr_verify_bytes(
        const uint8_t* xonly_pubkey32,
        const uint8_t* sig64,
        const uint8_t* msg32) override
    {
        secp256k1_xonly_pubkey xpk;
        if (!secp256k1_xonly_pubkey_parse(ctx_, &xpk, xonly_pubkey32))
            return false;

        return secp256k1_schnorrsig_verify(ctx_, sig64, msg32, 32, &xpk) == 1;
    }

    // -- Schnorr parse xonly ------------------------------------------------
    bool schnorr_parse_xonly(ParsedXonlyPubkey* out,
        const uint8_t* xonly32) override
    {
        static_assert(sizeof(secp256k1_xonly_pubkey) <= sizeof(out->data),
                      "ParsedXonlyPubkey too small for secp256k1_xonly_pubkey");
        auto* xpk = reinterpret_cast<secp256k1_xonly_pubkey*>(out->data);
        return secp256k1_xonly_pubkey_parse(ctx_, xpk, xonly32) == 1;
    }

    // -- Schnorr verify (preparsed) -----------------------------------------
    bool schnorr_verify_preparsed(
        const ParsedXonlyPubkey* pubkey,
        const uint8_t* sig64,
        const uint8_t* msg32) override
    {
        const auto* xpk = reinterpret_cast<const secp256k1_xonly_pubkey*>(pubkey->data);
        return secp256k1_schnorrsig_verify(ctx_, sig64, msg32, 32, xpk) == 1;
    }

    // -- pubkey_create (k * G) ----------------------------------------------
    bool pubkey_create(uint8_t* out33,
        const uint8_t* seckey32) override
    {
        secp256k1_pubkey pk;
        if (!secp256k1_ec_pubkey_create(ctx_, &pk, seckey32)) return false;

        size_t outlen = 33;
        return secp256k1_ec_pubkey_serialize(ctx_, out33, &outlen, &pk,
                                             SECP256K1_EC_COMPRESSED) == 1;
    }

    // -- ECDH ---------------------------------------------------------------
    bool ecdh(uint8_t* out32,
        const uint8_t* seckey32,
        const uint8_t* pubkey_data, size_t pubkey_len) override
    {
        secp256k1_pubkey pk;
        if (!secp256k1_ec_pubkey_parse(ctx_, &pk, pubkey_data, pubkey_len))
            return false;

        return secp256k1_ecdh(ctx_, out32, &pk, seckey32, nullptr, nullptr) == 1;
    }
};

} // anon namespace

std::unique_ptr<IProvider> create_provider_libsecp() {
    return std::make_unique<ProviderLibsecp>();
}

} // namespace bench
