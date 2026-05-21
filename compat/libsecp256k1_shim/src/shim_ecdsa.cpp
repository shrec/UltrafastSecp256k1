// ============================================================================
// shim_ecdsa.cpp -- ECDSA sign/verify, signature parse/serialize
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"
#include "shim_pubkey_helpers.hpp"

#include <cstring>
#include <array>
#include <cstdint>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"

using namespace secp256k1::fast;

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
    // T-07: strict parse — rejects r,s >= n by zeroing (downstream verify then fails).
    // PERF-OPT: when data was written by secp256k1_ecdsa_signature_parse_der or
    // secp256k1_ecdsa_signature_parse_compact, the valid_scalar check already ran.
    // For API correctness we still use parse_bytes_strict (not unchecked) because
    // callers can write arbitrary bytes into secp256k1_ecdsa_signature.data directly.
    // The strict parse is ~5-8 ns; removing it risks silent mod-n reduction (T-07).
    Scalar r_scalar, s_scalar;
    if (!Scalar::parse_bytes_strict(data,      r_scalar)) r_scalar = Scalar::zero();
    if (!Scalar::parse_bytes_strict(data + 32, s_scalar)) s_scalar = Scalar::zero();
    return { r_scalar, s_scalar };
}

// pubkey_data_to_point and point_to_pubkey_data are in shim_pubkey_helpers.hpp
using secp256k1_shim_internal::pubkey_data_to_point;

extern "C" {

// -- Compact parse/serialize ----------------------------------------------

int secp256k1_ecdsa_signature_parse_compact(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input64)
{
    SHIM_REQUIRE_CTX(ctx);
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
    SHIM_REQUIRE_CTX(ctx);
    if (!output64 || !sig) return 0;
    std::memcpy(output64, sig->data, 64);
    return 1;
}

// -- DER parse/serialize --------------------------------------------------
// DER parse optimizations (vs prior version):
//   1. Lambda instead of static function → compiler inlines, no call overhead
//   2. Write directly into sig->data → eliminates r[32], s[32] stack buffers
//      and the two final memcpy(sig->data, r, 32) calls
//   3. __builtin_memset/memcpy → stronger optimization hints than std::mem*
//   4. Only zero the leading prefix bytes (32 - len), not full 32 bytes

int secp256k1_ecdsa_signature_parse_der(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input, size_t inputlen)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!sig || !input || inputlen < 8) return 0;

    const unsigned char *p = input;
    const unsigned char *end = input + inputlen;

    if (*p++ != 0x30) return 0;
    unsigned char seqlen = *p++;
    // BIP-66: reject BER long-form (high bit = multi-byte length) and
    // SEQUENCE content that doesn't exactly fill the input buffer.
    if (seqlen & 0x80) return 0;
    if (seqlen > 70 || p + seqlen != end) return 0;

    // Parse one DER INTEGER into a 32-byte big-endian buffer.
    // Writes directly to *out (must point into sig->data).
    // Returns false on any BIP-66 violation.
    auto parse_int = [](const unsigned char*& p,
                        const unsigned char* end,
                        unsigned char* out) -> bool {
        if (p >= end || *p++ != 0x02) return false;
        if (p >= end) return false;
        unsigned char len = *p++;
        if (len == 0 || p + len > end) return false;
        if (*p & 0x80) return false;          // negative integer — invalid
        if (*p == 0x00) {
            // Leading 0x00 is only valid before a high-bit byte.
            if (len < 2 || !(p[1] & 0x80)) return false;
            ++p; --len;                        // skip required padding byte
        }
        if (len > 32) return false;
        __builtin_memset(out, 0, 32u - len);   // zero only the prefix
        __builtin_memcpy(out + 32u - len, p, len);
        p += len;
        return true;
    };

    // Parse r and s directly into sig->data — no intermediate stack buffers.
    if (!parse_int(p, end, sig->data))      return 0;  // r → data[0..31]
    if (!parse_int(p, end, sig->data + 32)) return 0;  // s → data[32..63]

    // Validate r, s ∈ [1, n-1]: reject overflow (>= n) AND zero.
    //
    // Fast path: load 4 big-endian uint64_t limbs and compare against the
    // curve order n in one branchless pass — avoids constructing a Scalar
    // object and the associated stack allocation / copy.
    //
    // Secp256k1 order n (big-endian):
    //   0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    static constexpr uint64_t N0 = 0xFFFFFFFFFFFFFFFFULL;  // bytes [ 0.. 7]
    static constexpr uint64_t N1 = 0xFFFFFFFFFFFFFFFEULL;  // bytes [ 8..15]
    static constexpr uint64_t N2 = 0xBAAEDCE6AF48A03BULL;  // bytes [16..23]
    static constexpr uint64_t N3 = 0xBFD25E8CD0364141ULL;  // bytes [24..31]

    // Load 32 bytes as 4 big-endian uint64_t (avoids Scalar ctor overhead).
    auto load64be = [](const unsigned char* p) noexcept -> uint64_t {
        uint64_t v;
        __builtin_memcpy(&v, p, 8);
        return __builtin_bswap64(v);
    };

    // Returns true if the 32-byte big-endian value is in [1, n-1].
    auto valid_scalar = [&](const unsigned char* b) noexcept -> bool {
        uint64_t a0 = load64be(b);       // most significant
        uint64_t a1 = load64be(b + 8);
        uint64_t a2 = load64be(b + 16);
        uint64_t a3 = load64be(b + 24);  // least significant

        // Check non-zero (at least one limb non-zero).
        if ((a0 | a1 | a2 | a3) == 0) return false;

        // Check < n: lexicographic comparison from most-significant limb.
        if (a0 < N0) return true;   if (a0 > N0) return false;
        if (a1 < N1) return true;   if (a1 > N1) return false;
        if (a2 < N2) return true;   if (a2 > N2) return false;
        return a3 < N3;
    };

    if (!valid_scalar(sig->data))      return 0;  // r
    if (!valid_scalar(sig->data + 32)) return 0;  // s

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
    SHIM_REQUIRE_CTX(ctx);
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
    SHIM_REQUIRE_CTX(ctx);
    if (!sigin) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_normalize: sigin is NULL");
        return 0;
    }

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
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_ecdsa_verify: NULL context");
        return 0;
    }
    if (!ctx_can_verify(ctx)) return 0;
    if (!sig || !msghash32 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_verify: NULL argument");
        return 0;
    }

    auto internal_sig = ecdsa_sig_from_data(sig->data);

    // ZERO-WRITE VERIFY PATH (libsecp-style):
    // Callgrind profiling showed Ultra had 7× libsecp's L3 write misses due to
    // ShimPkCache (256 × ~1.4 KB = 365 KB) thrashing L2 on unique-pubkey workloads.
    // Each cache access loaded 23 cache lines even for the fingerprint-only fast path.
    //
    // Fix: remove the cache from the hot verify path entirely.
    // pubkey->data = X_be[32] || Y_be[32], already parsed by ec_pubkey_parse.
    // Parse directly to Point → dual_scalar_mul_gen_point (integrates table build).
    // Matches libsecp's pattern: read 64-byte struct → compute → done, zero writes.
    //
    // For repeated-pubkey workloads: use secp256k1_ec_pubkey_precomp() + verify_precomp().
    {
        // PERF-003: bind directly to pubkey->data — avoids two 32-byte stack copies.
        // std::array<uint8_t,32> has identical layout to uint8_t[32] (§23.3.2.1).
        const auto& xb = *reinterpret_cast<const std::array<uint8_t,32>*>(pubkey->data);
        const auto& yb = *reinterpret_cast<const std::array<uint8_t,32>*>(pubkey->data + 32);
        auto x = secp256k1::fast::FieldElement::from_bytes(xb);
        auto y = secp256k1::fast::FieldElement::from_bytes(yb);
        // P1-SEC-RED-TEAM-008: curve membership check y²=x³+7.
        // Defends against hostile callers who write off-curve bytes into pubkey->data
        // directly (bypassing ec_pubkey_parse). Consistent with batch-verify path.
        // PERF-OPT: secp256k1 b-constant (7) as static const to avoid per-call ctor.
        {
            static const secp256k1::fast::FieldElement B7 =
                secp256k1::fast::FieldElement::from_uint64(7);
            auto lhs = y.square();
            auto rhs = x.square() * x + B7;
            if (!(lhs == rhs)) return 0;
        }
        auto pt = secp256k1::fast::Point::from_affine(x, y);
        if (pt.is_infinity()) return 0;
        return secp256k1::ecdsa_verify(msghash32, pt, internal_sig) ? 1 : 0;
    }
}

// -- Pre-computed pubkey API -----------------------------------------------

// Safety check: the secp256k1_pubkey_precomp opaque buffer must be large enough
// to hold EcdsaPublicKey with its GLV tables.
static_assert(sizeof(secp256k1_pubkey_precomp) >= sizeof(secp256k1::EcdsaPublicKey),
    "SECP256K1_PUBKEY_PRECOMP_SIZE too small — update the #define in secp256k1.h");
static_assert(alignof(secp256k1_pubkey_precomp) >= alignof(secp256k1::EcdsaPublicKey),
    "secp256k1_pubkey_precomp alignment insufficient for EcdsaPublicKey");

int secp256k1_ec_pubkey_precomp(
    const secp256k1_context* /*ctx*/,
    secp256k1_pubkey_precomp* out,
    const secp256k1_pubkey* pubkey)
{
    if (!out || !pubkey) return 0;
    auto* epk = reinterpret_cast<secp256k1::EcdsaPublicKey*>(out);
    unsigned char unc[65]; unc[0] = 0x04;
    std::memcpy(unc + 1, pubkey->data, 64);
    return secp256k1::ecdsa_pubkey_parse(*epk, unc, 65) ? 1 : 0;
}

int secp256k1_ec_pubkey_parse_precomp(
    const secp256k1_context* /*ctx*/,
    secp256k1_pubkey_precomp* out,
    const unsigned char* input, size_t inputlen)
{
    if (!out || !input) return 0;
    auto* epk = reinterpret_cast<secp256k1::EcdsaPublicKey*>(out);
    return secp256k1::ecdsa_pubkey_parse(*epk, input, inputlen) ? 1 : 0;
}

int secp256k1_ecdsa_verify_precomp(
    const secp256k1_context* ctx,
    const secp256k1_ecdsa_signature* sig,
    const unsigned char* msghash32,
    const secp256k1_pubkey_precomp* pubkey)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (!sig || !msghash32 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_verify_precomp: NULL argument");
        return 0;
    }
    const auto* epk = reinterpret_cast<const secp256k1::EcdsaPublicKey*>(pubkey);
    auto internal_sig = ecdsa_sig_from_data(sig->data);
    // PERF-005: use raw-pointer overload — avoids 32-byte stack copy.
    // Direct use of pre-built GLV tables — zero table rebuild overhead.
    return secp256k1::ecdsa_verify(msghash32, *epk, internal_sig) ? 1 : 0;
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
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_ecdsa_sign: NULL context");
        return 0;
    }
    if (!ctx_can_sign(ctx)) return 0;
    if (!sig || !msghash32 || !seckey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_sign: NULL argument");
        return 0;
    }
    // Apply per-context blinding for the duration of this signing call (SHIM-001 fix).
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);
    // Reject custom nonce functions: this shim uses RFC 6979 internally and
    // cannot forward an arbitrary noncefp callback. Fail-closed so callers
    // that rely on a specific nonce function are not silently given RFC 6979.
    // The two standard constants (rfc6979 / default) are treated as NULL.
    // Fire the illegal callback so callers with handlers know exactly why (PASS3-001 fix).
    if (noncefp != nullptr &&
        noncefp != secp256k1_nonce_function_rfc6979 &&
        noncefp != secp256k1_nonce_function_default) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_ecdsa_sign: custom nonce functions are not supported; "
            "pass NULL or secp256k1_nonce_function_rfc6979");
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
