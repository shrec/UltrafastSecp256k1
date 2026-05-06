// ============================================================================
// shim_ecdsa.cpp -- ECDSA sign/verify, signature parse/serialize
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>
#include <cstdint>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/sign.hpp"

using namespace secp256k1::fast;

// -- Thread-local pubkey GLV cache -------------------------------------------
// Eliminates ~1,954 ns build_glv52_table_zr on repeated verify of same pubkey.
// Direct-mapped, 16 slots, keyed by XOR of first 8 bytes of pubkey->data.
// Even on cache miss, parsing uncompressed x||y is cheaper than the full
// rebuild path in ecdsa_verify(Point): ~21 µs vs ~23 µs.
namespace {
struct ShimPkCache {
    static constexpr std::size_t SLOTS = 256;
    struct Slot {
        std::uint64_t             fingerprint{0};  // FNV-1a over 8×uint64; replaces raw[64]+memcmp
        secp256k1::EcdsaPublicKey epk{};
        bool                      valid = false;
    };
    Slot slots[SLOTS]{};

    // Hash 64 bytes as 8 × uint64 words — 8× fewer iterations than byte-loop.
    // Returns slot index AND full fingerprint in one pass.
    static void hash64(const unsigned char data[64],
                       std::size_t& idx_out, std::uint64_t& fp_out) noexcept {
        std::uint64_t h = 14695981039346656037ULL, w;
        for (int i = 0; i < 8; ++i) {
            __builtin_memcpy(&w, data + i * 8, 8);
            h = (h ^ w) * 1099511628211ULL;
        }
        fp_out  = h;
        idx_out = static_cast<std::size_t>(h & (SLOTS - 1));
    }

    const secp256k1::EcdsaPublicKey* get(const unsigned char data[64]) const noexcept {
        std::size_t idx; std::uint64_t fp;
        hash64(data, idx, fp);
        const Slot& s = slots[idx];
        if (s.valid && s.fingerprint == fp) return &s.epk;
        return nullptr;
    }

    const secp256k1::EcdsaPublicKey* put(const unsigned char data[64]) noexcept {
        std::size_t idx; std::uint64_t fp;
        hash64(data, idx, fp);
        Slot& s = slots[idx];
        unsigned char unc[65]; unc[0] = 0x04;
        std::memcpy(unc + 1, data, 64);
        s.valid = secp256k1::ecdsa_pubkey_parse(s.epk, unc, 65);
        if (s.valid) s.fingerprint = fp;
        return s.valid ? &s.epk : nullptr;
    }
};
static thread_local ShimPkCache s_pk_cache;
} // namespace

// Context flag helpers — use the canonical implementations from shim_internal.hpp.
// NULL ctx returns false (matches libsecp256k1: triggers illegal callback, returns 0).
using secp256k1_shim_internal::ctx_flags;
using secp256k1_shim_internal::ctx_can_sign;
using secp256k1_shim_internal::ctx_can_verify;

// -- Internal: opaque sig stores r (32 BE) || s (32 BE) -------------------
static void ecdsa_sig_to_data(const secp256k1::ECDSASignature& sig, unsigned char data[64]) {
    auto rb = sig.r.to_bytes();
    auto sb = sig.s.to_bytes();
    std::memcpy(data, rb.data(), 32);
    std::memcpy(data + 32, sb.data(), 32);
}

static secp256k1::ECDSASignature ecdsa_sig_from_data(const unsigned char data[64]) {
    std::array<uint8_t, 32> rb{}, sb{};
    std::memcpy(rb.data(), data, 32);
    std::memcpy(sb.data(), data + 32, 32);
    return { Scalar::from_bytes(rb), Scalar::from_bytes(sb) };
}

// -- Internal: reconstruct Point from opaque pubkey ----------------------
static Point pubkey_data_to_point(const unsigned char data[64]) {
    std::array<uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), data, 32);
    std::memcpy(yb.data(), data + 32, 32);
    auto x = FieldElement::from_bytes(xb);
    auto y = FieldElement::from_bytes(yb);
    return Point::from_affine(x, y);
}

extern "C" {

// -- Compact parse/serialize ----------------------------------------------

int secp256k1_ecdsa_signature_parse_compact(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input64)
{
    (void)ctx;
    if (!sig || !input64) return 0;
    // libsecp accepts r==0 and s==0 at parse time (verify will reject them).
    // Only r>=n or s>=n triggers a parse failure, matching upstream behaviour.
    Scalar r, s;
    if (!Scalar::parse_bytes_strict(input64,      r)) return 0;
    if (!Scalar::parse_bytes_strict(input64 + 32, s)) return 0;
    std::memcpy(sig->data, input64, 64);
    return 1;
}

int secp256k1_ecdsa_signature_serialize_compact(
    const secp256k1_context *ctx, unsigned char *output64,
    const secp256k1_ecdsa_signature *sig)
{
    (void)ctx;
    if (!output64 || !sig) return 0;
    std::memcpy(output64, sig->data, 64);
    return 1;
}

// -- DER parse/serialize --------------------------------------------------

static int parse_der_int(const unsigned char *&p, const unsigned char *end,
                         unsigned char out[32])
{
    if (p >= end || *p != 0x02) return 0;
    ++p;
    if (p >= end) return 0;
    size_t len = *p++;
    if (len == 0 || p + len > end) return 0;

    // BIP-66 strict DER: reject negative integers (high bit set without 0x00 prefix).
    // In DER encoding a set high bit means a negative number — always invalid for r,s.
    if (*p & 0x80) return 0;

    // BIP-66 strict DER: reject unnecessary leading zeros.
    // A leading 0x00 is only valid when the NEXT byte has its high bit set
    // (to distinguish a positive integer from a negative one in DER encoding).
    // Any other leading 0x00 is non-canonical and must be rejected.
    if (*p == 0x00) {
        if (len < 2 || (p[1] & 0x80) == 0) return 0;  // unnecessary leading zero
        ++p; --len;  // consume the required padding byte
    }

    if (len > 32) return 0;
    std::memset(out, 0, 32);
    std::memcpy(out + (32 - len), p, len);
    p += len;
    return 1;
}

int secp256k1_ecdsa_signature_parse_der(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input, size_t inputlen)
{
    (void)ctx;
    if (!sig || !input || inputlen < 8) return 0;

    const unsigned char *p = input;
    const unsigned char *end = input + inputlen;

    if (*p++ != 0x30) return 0;
    size_t seqlen = *p++;
    // BIP-66: reject if SEQUENCE content overflows OR if there are trailing
    // bytes after the SEQUENCE (libsecp256k1 enforces exact boundary match).
    if (seqlen > 70 || p + seqlen != end) return 0;

    unsigned char r[32]{}, s[32]{};
    if (!parse_der_int(p, end, r)) return 0;
    if (!parse_der_int(p, end, s)) return 0;

    // Validate r, s are in (0, n-1] — matches libsecp strict contract.
    Scalar rs, ss;
    if (!Scalar::parse_bytes_strict_nonzero(
            reinterpret_cast<const uint8_t*>(r), rs)) return 0;
    if (!Scalar::parse_bytes_strict_nonzero(
            reinterpret_cast<const uint8_t*>(s), ss)) return 0;

    std::memcpy(sig->data, r, 32);
    std::memcpy(sig->data + 32, s, 32);
    return 1;
}

static int der_encode_int(unsigned char *out, size_t *len, const unsigned char val[32]) {
    // Find first non-zero
    int start = 0;
    while (start < 32 && val[start] == 0) ++start;
    if (start == 32) start = 31; // encode zero as 0x00

    bool need_pad = (val[start] & 0x80) != 0;
    size_t int_len = 32 - start + (need_pad ? 1 : 0);

    out[0] = 0x02;
    out[1] = static_cast<unsigned char>(int_len);
    size_t pos = 2;
    if (need_pad) out[pos++] = 0x00;
    std::memcpy(out + pos, val + start, 32 - start);
    *len = 2 + int_len;
    return 1;
}

int secp256k1_ecdsa_signature_serialize_der(
    const secp256k1_context *ctx, unsigned char *output, size_t *outputlen,
    const secp256k1_ecdsa_signature *sig)
{
    (void)ctx;
    if (!output || !outputlen || !sig) return 0;

    // Max DER integer: 0x02 + len(1) + 0x00 pad(1) + 32 data = 35 bytes
    unsigned char r_der[35]{}, s_der[35]{};
    size_t r_len = 0, s_len = 0;
    der_encode_int(r_der, &r_len, sig->data);
    der_encode_int(s_der, &s_len, sig->data + 32);

    size_t total = 2 + r_len + s_len;
    if (*outputlen < total) { *outputlen = total; return 0; }

    output[0] = 0x30;
    output[1] = static_cast<unsigned char>(r_len + s_len);
    std::memcpy(output + 2, r_der, r_len);
    std::memcpy(output + 2 + r_len, s_der, s_len);
    *outputlen = total;
    return 1;
}

// -- Normalize ------------------------------------------------------------

int secp256k1_ecdsa_signature_normalize(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sigout,
    const secp256k1_ecdsa_signature *sigin)
{
    (void)ctx;
    if (!sigin) return 0;

    auto sig = ecdsa_sig_from_data(sigin->data);
    bool was_high = !sig.is_low_s();
    auto norm = sig.normalize();
    if (sigout) ecdsa_sig_to_data(norm, sigout->data);
    return was_high ? 1 : 0;
}

// -- Verify ---------------------------------------------------------------

int secp256k1_ecdsa_verify(
    const secp256k1_context *ctx, const secp256k1_ecdsa_signature *sig,
    const unsigned char *msghash32, const secp256k1_pubkey *pubkey)
{
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_VERIFY
    // (or a context created with CONTEXT_SIGN which is a superset).
    // SECP256K1_CONTEXT_NONE contexts are rejected to match upstream contract.
    if (!ctx_can_verify(ctx)) return 0;
    if (!sig || !msghash32 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_verify: NULL argument");
        return 0;
    }

    auto internal_sig = ecdsa_sig_from_data(sig->data);

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msghash32, 32);

    // Use cached GLV tables when available; build on miss.
    const secp256k1::EcdsaPublicKey* epk = s_pk_cache.get(pubkey->data);
    if (!epk) epk = s_pk_cache.put(pubkey->data);
    if (!epk) return 0;  // degenerate pubkey (not on curve)
    return secp256k1::ecdsa_verify(msg, *epk, internal_sig) ? 1 : 0;
}

// -- Sign -----------------------------------------------------------------
// ndata: when non-null, 32 bytes of caller-supplied extra entropy.
// Bitcoin Core's R-grinding loop (grind=true default in CKey::Sign) calls
// secp256k1_ecdsa_sign repeatedly with increasing counter bytes in ndata.
// Ignoring ndata makes every call return the same signature → infinite loop.
// Fix: use ecdsa_sign_hedged() when ndata is provided.

int secp256k1_ecdsa_sign(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *msghash32, const unsigned char *seckey,
    secp256k1_nonce_function noncefp, const void *ndata)
{
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_SIGN.
    // A context created with only CONTEXT_VERIFY is rejected for signing.
    if (!ctx_can_sign(ctx)) return 0;
    if (!sig || !msghash32 || !seckey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_sign: NULL argument");
        return 0;
    }
    // Reject custom nonce functions: this shim uses RFC 6979 internally and
    // cannot forward an arbitrary noncefp callback. Fail-closed so callers
    // that rely on a specific nonce function are not silently given RFC 6979.
    // The two standard constants (rfc6979 / default) are treated as NULL.
    if (noncefp != nullptr &&
        noncefp != secp256k1_nonce_function_rfc6979 &&
        noncefp != secp256k1_nonce_function_default) {
        return 0;
    }

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msghash32, 32);
    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) return 0;

    secp256k1::ECDSASignature result;
    if (ndata) {
        std::array<uint8_t, 32> aux{};
        std::memcpy(aux.data(), ndata, 32);
        result = secp256k1::ct::ecdsa_sign_hedged(msg, k, aux);
    } else {
        result = secp256k1::ct::ecdsa_sign(msg, k);
    }
    // C5: explicit error propagation via ECDSASignature::is_valid() rather than
    // ad-hoc zero-checks. is_valid() ↔ (r ∈ [1,n-1] ∧ s ∈ [1,n-1]).
    // CT signing returns zero (r,s) on any degenerate case (k≡0 mod n, etc.).
    if (!result.is_valid()) return 0;
    ecdsa_sig_to_data(result, sig->data);
    return 1;
}

// -- RFC 6979 nonce function pointers -----------------------------------------
// The shim's ecdsa_sign uses its own internal RFC6979 nonce derivation and does
// NOT call these function pointers. They are exported for ABI compatibility only.
// Returning 0 (failure) is correct: a direct call to these pointers means no
// nonce bytes were written, so the caller MUST treat it as a failure.
// (Returning 1 from a stub that writes nothing is a silent ABI trap — SC-08.)

static int nonce_function_rfc6979_stub(unsigned char *, const unsigned char *,
    const unsigned char *, const unsigned char *, void *, unsigned int)
{ return 0; }

const secp256k1_nonce_function secp256k1_nonce_function_rfc6979 = nonce_function_rfc6979_stub;
const secp256k1_nonce_function secp256k1_nonce_function_default = nonce_function_rfc6979_stub;

} // extern "C"
