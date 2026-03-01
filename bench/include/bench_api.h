// bench_api.h -- Unified provider interface for apples-to-apples benchmarking
// UltrafastSecp256k1 vs libsecp256k1
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bench {

// ---------------------------------------------------------------------------
// Provider capability flags
// ---------------------------------------------------------------------------
struct ProviderCaps {
    bool ecdsa_verify    = false;
    bool schnorr_verify  = false;
    bool pubkey_create   = false;
    bool ecdh            = false;
};

// ---------------------------------------------------------------------------
// Opaque data holders -- providers store pre-parsed keys/sigs here
// ---------------------------------------------------------------------------
struct ParsedPubkey {
    alignas(64) uint8_t data[128];  // large enough for any provider
};

struct ParsedSig {
    alignas(64) uint8_t data[128];
};

struct ParsedXonlyPubkey {
    alignas(64) uint8_t data[256];
};

struct ParsedSchnorrSig {
    alignas(64) uint8_t data[64];
};

// ---------------------------------------------------------------------------
// IProvider -- unified benchmark interface
// ---------------------------------------------------------------------------
class IProvider {
public:
    virtual ~IProvider() = default;

    // -- identity -----------------------------------------------------------
    virtual const char* name()    const = 0;
    virtual const char* version() const = 0;
    virtual ProviderCaps caps()   const = 0;

    // -- lifecycle ----------------------------------------------------------
    virtual bool init(bool randomize_ctx) = 0;
    virtual void shutdown() = 0;

    // -- ECDSA verify -------------------------------------------------------
    // bytes path: raw DER/compact sig + compressed pubkey + msg32
    virtual bool ecdsa_verify_bytes(
        const uint8_t* pubkey33, size_t pubkey_len,
        const uint8_t* sig, size_t sig_len,
        const uint8_t* msg32,
        bool normalize_low_s) = 0;

    // preparsed path: parse once, verify many
    virtual bool ecdsa_parse_pubkey(ParsedPubkey* out,
        const uint8_t* pubkey33, size_t len) = 0;
    virtual bool ecdsa_parse_sig(ParsedSig* out,
        const uint8_t* sig, size_t sig_len, bool is_der) = 0;
    virtual bool ecdsa_verify_preparsed(
        const ParsedPubkey* pubkey,
        const ParsedSig* sig,
        const uint8_t* msg32,
        bool normalize_low_s) = 0;

    // -- Schnorr verify (BIP-340) -------------------------------------------
    virtual bool schnorr_verify_bytes(
        const uint8_t* xonly_pubkey32,
        const uint8_t* sig64,
        const uint8_t* msg32) = 0;

    virtual bool schnorr_parse_xonly(ParsedXonlyPubkey* out,
        const uint8_t* xonly32) = 0;
    virtual bool schnorr_verify_preparsed(
        const ParsedXonlyPubkey* pubkey,
        const uint8_t* sig64,
        const uint8_t* msg32) = 0;

    // -- pubkey_create (k * G) ----------------------------------------------
    // returns 33-byte compressed pubkey
    virtual bool pubkey_create(uint8_t* out33,
        const uint8_t* seckey32) = 0;

    // -- ECDH ---------------------------------------------------------------
    // returns 32-byte shared secret (SHA256 of x-coordinate)
    virtual bool ecdh(uint8_t* out32,
        const uint8_t* seckey32,
        const uint8_t* pubkey33, size_t pubkey_len) = 0;
};

// -- Factory functions (implemented in provider_*.cpp) ----------------------
std::unique_ptr<IProvider> create_provider_uf();
std::unique_ptr<IProvider> create_provider_libsecp();

} // namespace bench
