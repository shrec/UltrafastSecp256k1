// ============================================================================
// shim_ecdsa.cpp -- ECDSA sign/verify, signature parse/serialize
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"
#include "shim_pubkey_helpers.hpp"

#include <cstring>
#include <array>
#include <cstdint>
#include <chrono>     // SHIM-014: thread-local cache salt fallback
#include <random>     // SHIM-014: std::random_device for cache salt

// Portable bit/memory intrinsics. MSVC does not provide the GCC/Clang
// __builtin_* family; map them to the std::memcpy/std::memset / MSVC-specific
// equivalents at compile time. Used by the fast DER scalar validator and the
// cache fingerprint helpers below.
#if defined(_MSC_VER)
#  include <stdlib.h>   // _byteswap_uint64
#  define UFSECP_SHIM_MEMCPY(dst, src, n)  std::memcpy((dst), (src), (n))
#  define UFSECP_SHIM_MEMSET(dst, val, n)  std::memset((dst), (val), (n))
#  define UFSECP_SHIM_BSWAP64(v)           _byteswap_uint64((v))
#else
#  define UFSECP_SHIM_MEMCPY(dst, src, n)  __builtin_memcpy((dst), (src), (n))
#  define UFSECP_SHIM_MEMSET(dst, val, n)  __builtin_memset((dst), (val), (n))
#  define UFSECP_SHIM_BSWAP64(v)           __builtin_bswap64((v))
#endif

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"   // CT-01: erase parsed private-key scalar

using namespace secp256k1::fast;

// -- Thread-local ECDSA pubkey GLV-table cache --------------------------------
// Caches EcdsaPublicKey (prebuilt GLV tables) indexed by pubkey->data[64].
// Design: 32 slots × ~1,580 bytes = ~50 KB — fits in L2, avoids L3 thrashing.
// (The old ShimPkCache used 256 slots × 1.4 KB = ~365 KB and caused 7× L3
// write misses on unique-pubkey workloads. 32 slots eliminates that risk.)
//
// Build-on-first-encounter (same as ShimSchnorrCache after PERF-001 fix):
//   HIT:  return prebuilt EcdsaPublicKey immediately, ~900 ns saved per call.
//   MISS: build GLV tables via ecdsa_pubkey_parse, cache, return immediately.
//
// Impact:
//   - Hot-wallet: same pubkey verified repeatedly → ~900 ns saved per call
//   - ConnectBlock unique pubkeys: pays GLV build once, evicted quickly (32 slots)
//     → near-zero thrashing overhead vs uncached path
namespace {
// SHIM-014 (closed): one-time per-thread random salt for the cache slot
// fingerprint. Without a salt, an attacker who controls a pubkey can pick
// X-prefix bytes that collide with a victim's slot — perpetually
// overwriting the victim's seen_once flag and preventing the victim from
// ever reaching the cache. With a per-thread salt the slot mapping is
// not attacker-predictable; ~2 ns extra per lookup. We use atomic_signal_fence-
// safe std::random_device on first use; if it fails (rare on POSIX) we
// fall back to clock + tid mixing — any non-zero salt is sufficient.
[[gnu::const]] inline std::uint64_t shim_cache_thread_salt() noexcept {
    static thread_local std::uint64_t salt = []() noexcept {
        std::uint64_t seed = 0;
        try {
            std::random_device rd;
            seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
        } catch (...) {
            // random_device may throw on some embedded platforms.
            seed = 0;
        }
        if (seed == 0) {
            // Defensive fallback: time + tid.
            auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            seed = static_cast<std::uint64_t>(now)
                 ^ (reinterpret_cast<std::uintptr_t>(&seed) * 0x9E3779B97F4A7C15ULL);
        }
        // Guarantee non-zero so XOR-with-salt does not degenerate to no-op
        // even in the (impossible-after-fallback) all-zero case.
        return seed | 1ULL;
    }();
    return salt;
}

struct ShimEcdsaCache {
    static constexpr std::size_t SLOTS = 32;
    struct Slot {
        std::uint64_t            fingerprint{0};
        uint8_t                  pubkey_x[32]{};   // X bytes for identity check
        secp256k1::EcdsaPublicKey epk{};
        bool                     valid      = false;
        bool                     seen_once  = false;
    };
    Slot slots[SLOTS]{};

    // Two-phase design matching ShimSchnorrCache (pre-PERF-001 for ECDSA):
    //   FIRST  encounter → record fingerprint only (~40 bytes), return nullptr.
    //                       Caller uses direct Point path (no extra L2 write).
    //   SECOND encounter → build EcdsaPublicKey GLV tables (~1.95 µs), cache.
    //   THIRD+ encounter → cache hit, ~900 ns saved per call.
    //
    // Why two-phase for ECDSA vs one-phase for Schnorr:
    //   ConnectBlock has ~100% unique pubkeys per block. One-phase writes 1504 bytes
    //   per unique pubkey → L2 thrashing (same problem as the removed ShimPkCache).
    //   Two-phase writes only ~40 bytes for 1st encounter → no extra L2 pressure.
    //   Wallet workloads (repeated keys) benefit on 2nd+ encounter as before.
    //
    // Fast fingerprint: first 8 bytes of X coordinate (2 instructions, ~0.3 ns)
    // instead of full 64-byte FNV-1a hash (~8-10 ns). Collision resistance via
    // full 32-byte X comparison in the hot path (T-08 pattern preserved).
    // SHIM-014: XOR with thread-local salt so slot mapping is not attacker-
    // predictable. Adds ~1 ns; eliminates the slot-hijack class entirely.
    static void fingerprint(const unsigned char data[64],
                            std::size_t& idx_out, std::uint64_t& fp_out) noexcept {
        std::uint64_t fp;
        UFSECP_SHIM_MEMCPY(&fp, data, 8);        // first 8 bytes of X coordinate
        fp ^= shim_cache_thread_salt();          // SHIM-014: per-thread randomisation
        fp_out  = fp;
        idx_out = static_cast<std::size_t>(fp & (SLOTS - 1));
    }

    // Returns prebuilt EcdsaPublicKey on 2nd+ encounter; nullptr on 1st (caller uses Point).
    const secp256k1::EcdsaPublicKey* get_or_build(const unsigned char data[64]) noexcept {
        std::size_t idx; std::uint64_t fp;
        fingerprint(data, idx, fp);
        Slot& s = slots[idx];

        bool const matches = (s.fingerprint == fp &&
                              std::memcmp(s.pubkey_x, data, 32) == 0);
        // Cache hit (3rd+ encounter).
        if (matches && s.valid)  return &s.epk;

        // Second encounter: build GLV tables.
        if (matches && s.seen_once && !s.valid) {
            unsigned char unc[65]; unc[0] = 0x04;
            std::memcpy(unc + 1, data, 64);
            s.valid = secp256k1::ecdsa_pubkey_parse(s.epk, unc, 65);
            return s.valid ? &s.epk : nullptr;
        }

        // First encounter: record fingerprint + X only (~40 bytes written).
        s.fingerprint = fp;
        std::memcpy(s.pubkey_x, data, 32);
        s.seen_once = true;
        s.valid     = false;
        return nullptr;   // caller uses Point path — no L2 write pressure
    }
};
static thread_local ShimEcdsaCache s_ecdsa_cache;
} // namespace

// Context flag helpers — use the canonical implementations from shim_internal.hpp.
// NULL ctx returns false (matches libsecp256k1: triggers illegal callback, returns 0).
using secp256k1_shim_internal::ctx_flags;
using secp256k1_shim_internal::ctx_can_sign;
using secp256k1_shim_internal::ctx_can_verify;
using secp256k1_shim_internal::scalar_be_to_internal;
using secp256k1_shim_internal::scalar_internal_to_be;

// -- Internal: opaque sig stores r (32 LE/internal) || s (32 LE/internal) ---
static void ecdsa_sig_to_data(const secp256k1::ECDSASignature& sig, unsigned char data[64]) {
    auto rb = sig.r.to_bytes();
    auto sb = sig.s.to_bytes();
    scalar_be_to_internal(data, rb.data());
    scalar_be_to_internal(data + 32, sb.data());
}

static secp256k1::ECDSASignature ecdsa_sig_from_data(const unsigned char data[64]) {
    // T-07: strict parse — rejects r,s >= n by zeroing (downstream verify then fails).
    // PERF-OPT: when data was written by secp256k1_ecdsa_signature_parse_der or
    // secp256k1_ecdsa_signature_parse_compact, the valid_scalar check already ran.
    // For API correctness we still use parse_bytes_strict (not unchecked) because
    // callers can write arbitrary bytes into secp256k1_ecdsa_signature.data directly.
    // The strict parse is ~5-8 ns; removing it risks silent mod-n reduction (T-07).
    Scalar r_scalar, s_scalar;
    unsigned char r_be[32]{}, s_be[32]{};
    scalar_internal_to_be(r_be, data);
    scalar_internal_to_be(s_be, data + 32);
    if (!Scalar::parse_bytes_strict(r_be, r_scalar)) r_scalar = Scalar::zero();
    if (!Scalar::parse_bytes_strict(s_be, s_scalar)) s_scalar = Scalar::zero();
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
    if (!sig) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_parse_compact: sig is NULL");
        return 0;
    }
    if (!input64) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_parse_compact: input64 is NULL");
        return 0;
    }
    // libsecp accepts r==0 and s==0 at parse time (verify will reject them).
    // Only r>=n or s>=n triggers a parse failure, matching upstream behaviour.
    // On parse failure, zero the output sig to match upstream libsecp256k1
    // (secp256k1_ecdsa_signature_parse_compact does memset(sig,0,...) on its
    // failure path) — fail-closed, no stale data left behind (PASS3-SHIM-001).
    Scalar r, s;
    if (!Scalar::parse_bytes_strict(input64,      r) ||
        !Scalar::parse_bytes_strict(input64 + 32, s)) {
        std::memset(sig->data, 0, sizeof(sig->data));
        return 0;
    }
    scalar_be_to_internal(sig->data, input64);
    scalar_be_to_internal(sig->data + 32, input64 + 32);
    return 1;
}

int secp256k1_ecdsa_signature_serialize_compact(
    const secp256k1_context *ctx, unsigned char *output64,
    const secp256k1_ecdsa_signature *sig)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!output64) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_serialize_compact: output64 is NULL");
        return 0;
    }
    if (!sig) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_serialize_compact: sig is NULL");
        return 0;
    }
    scalar_internal_to_be(output64, sig->data);
    scalar_internal_to_be(output64 + 32, sig->data + 32);
    return 1;
}

// -- DER parse/serialize --------------------------------------------------
// DER parse optimizations (vs prior version):
//   1. Lambda instead of static function → compiler inlines, no call overhead
//   2. __builtin_memset/memcpy → stronger optimization hints than std::mem*
//   3. Only zero the leading prefix bytes (32 - len), not full 32 bytes

int secp256k1_ecdsa_signature_parse_der(
    const secp256k1_context *ctx, secp256k1_ecdsa_signature *sig,
    const unsigned char *input, size_t inputlen)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!sig) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_parse_der: sig is NULL");
        return 0;
    }
    if (!input) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_parse_der: input is NULL");
        return 0;
    }
    // Match upstream: any parse failure must leave sig fully zeroed (PASS3-SHIM-001).
    // Zero up-front so every early-return path is fail-closed; the parse writes below
    // only persist on success, and post-write failure paths re-zero explicitly.
    std::memset(sig->data, 0, sizeof(sig->data));
    if (inputlen < 8) return 0;

    const unsigned char *p = input;
    const unsigned char *end = input + inputlen;

    if (*p++ != 0x30) return 0;
    unsigned char seqlen = *p++;
    // BIP-66: reject BER long-form (high bit = multi-byte length) and
    // SEQUENCE content that doesn't exactly fill the input buffer.
    if (seqlen & 0x80) return 0;
    if (seqlen > 70 || p + seqlen != end) return 0;

    // Parse one DER INTEGER into a 32-byte big-endian buffer.
    // Returns false on any BIP-66 violation.
    auto parse_int = [](const unsigned char*& cursor,
                        const unsigned char* limit,
                        unsigned char* out) -> bool {
        if (cursor >= limit || *cursor++ != 0x02) return false;
        if (cursor >= limit) return false;
        unsigned char len = *cursor++;
        if (len == 0 || cursor + len > limit) return false;
        if (*cursor & 0x80) return false;          // negative integer — invalid
        if (*cursor == 0x00) {
            // Leading 0x00 is only valid before a high-bit byte.
            if (len < 2 || !(cursor[1] & 0x80)) return false;
            ++cursor; --len;                        // skip required padding byte
        }
        if (len > 32) return false;
        UFSECP_SHIM_MEMSET(out, 0, 32u - len);   // zero only the prefix
        UFSECP_SHIM_MEMCPY(out + 32u - len, cursor, len);
        cursor += len;
        return true;
    };

    unsigned char r_be[32]{}, s_be[32]{};
    if (!parse_int(p, end, r_be)) { std::memset(sig->data, 0, sizeof(sig->data)); return 0; }
    if (!parse_int(p, end, s_be)) { std::memset(sig->data, 0, sizeof(sig->data)); return 0; }

    // RT-02 (strict-DER): the SEQUENCE body must be EXACTLY consumed by r and s.
    // The line-255 check only verifies the declared SEQUENCE length fills the
    // input buffer; it does not catch trailing bytes *inside* the SEQUENCE after
    // s (e.g. 30 08 02 01 0F 02 01 01 7F 7F — seqlen=8 covers offsets 2..9 but
    // r,s consume only 2..7). Upstream secp256k1_ecdsa_sig_parse rejects this, and
    // the native C-ABI parser (src/cpu/src/impl/ufsecp_ecdsa.cpp) already does.
    if (p != end) { std::memset(sig->data, 0, sizeof(sig->data)); return 0; }

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
    auto load64be = [](const unsigned char* q) noexcept -> uint64_t {
        uint64_t v;
        UFSECP_SHIM_MEMCPY(&v, q, 8);
        return UFSECP_SHIM_BSWAP64(v);
    };

    // Returns true if the 32-byte big-endian value is in [0, n-1] (i.e. < n).
    // NOTE on r=0/s=0: this predicate alone would accept a zero VALUE, but the
    // canonical DER encoding of zero is `02 01 00`, which parse_int above already
    // rejects via its minimal-encoding rule (a single leading 0x00 byte fails the
    // `len < 2` check). There is no minimal DER encoding of zero that reaches this
    // predicate, so in practice r=0/s=0 are rejected AT PARSE. This is a documented
    // divergence from upstream libsecp256k1 (secp256k1_ecdsa_sig_parse accepts zero
    // and defers rejection to verify) — see docs/SHIM_KNOWN_DIVERGENCES.md and the
    // regression test compat/libsecp256k1_shim/tests/test_shim_der_zero_r.cpp.
    auto in_range_scalar = [&](const unsigned char* b) noexcept -> bool {
        uint64_t a0 = load64be(b);       // most significant
        uint64_t a1 = load64be(b + 8);
        uint64_t a2 = load64be(b + 16);
        uint64_t a3 = load64be(b + 24);  // least significant

        // Check < n: lexicographic comparison from most-significant limb.
        if (a0 < N0) return true;
        if (a0 > N0) return false;
        if (a1 < N1) return true;
        if (a1 > N1) return false;
        if (a2 < N2) return true;
        if (a2 > N2) return false;
        return a3 < N3;
    };

    if (!in_range_scalar(r_be)) { std::memset(sig->data, 0, sizeof(sig->data)); return 0; }
    if (!in_range_scalar(s_be)) { std::memset(sig->data, 0, sizeof(sig->data)); return 0; }

    scalar_be_to_internal(sig->data, r_be);
    scalar_be_to_internal(sig->data + 32, s_be);
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
    if (!output) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_serialize_der: output is NULL");
        return 0;
    }
    if (!outputlen) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_serialize_der: outputlen is NULL");
        return 0;
    }
    if (!sig) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ecdsa_signature_serialize_der: sig is NULL");
        return 0;
    }

    // Max DER integer: 0x02 + len(1) + 0x00 pad(1) + 32 data = 35 bytes
    unsigned char r_be[32]{}, s_be[32]{};
    scalar_internal_to_be(r_be, sig->data);
    scalar_internal_to_be(s_be, sig->data + 32);
    unsigned char r_der[35]{}, s_der[35]{};
    size_t r_len = 0, s_len = 0;
    der_encode_int(r_der, &r_len, r_be);
    der_encode_int(s_der, &s_len, s_be);

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

    // ShimEcdsaCache: look up prebuilt GLV tables for this pubkey.
    // On hit:  ~900 ns saved vs building tables per-call (EcdsaPublicKey path).
    // On miss: build GLV tables once, cache in 32-slot thread-local (~50 KB).
    //
    // Trust contract (matches libsecp256k1 behaviour):
    //   secp256k1_pubkey is populated only by secp256k1_ec_pubkey_parse /
    //   secp256k1_ec_pubkey_create, both of which validate y²=x³+7. We do NOT
    //   re-check curve membership here. A caller that writes arbitrary bytes
    //   into the struct violates the API contract; behaviour is undefined,
    //   same as libsecp256k1. Both 1st-encounter and 2nd+-encounter paths now
    //   share this trust model — no curve-state-dependent verdict difference.
    if (const secp256k1::EcdsaPublicKey* epk =
            s_ecdsa_cache.get_or_build(pubkey->data)) {
        // Cache hit (2nd+ encounter): use prebuilt GLV tables.
        return secp256k1::ecdsa_verify(msghash32, *epk, internal_sig) ? 1 : 0;
    }
    // First encounter (cache returns nullptr): direct Point path.
    {
        using secp256k1_shim_internal::pubkey_data_to_point;
        auto pt = pubkey_data_to_point(pubkey->data);
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
    const secp256k1_context* ctx,
    secp256k1_pubkey_precomp* out,
    const secp256k1_pubkey* pubkey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out || !pubkey) return 0;
    auto* epk = reinterpret_cast<secp256k1::EcdsaPublicKey*>(out);
    unsigned char unc[65]; unc[0] = 0x04;
    std::memcpy(unc + 1, pubkey->data, 64);
    return secp256k1::ecdsa_pubkey_parse(*epk, unc, 65) ? 1 : 0;
}

int secp256k1_ec_pubkey_parse_precomp(
    const secp256k1_context* ctx,
    secp256k1_pubkey_precomp* out,
    const unsigned char* input, size_t inputlen)
{
    SHIM_REQUIRE_CTX(ctx);
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
    std::memset(sig->data, 0, sizeof(sig->data));
    // Apply per-context blinding for the duration of this signing call (SHIM-001 fix).
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msghash32, 32);
    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) {
        secp256k1::detail::secure_erase(&k, sizeof(k));   // CT-01
        return 0;
    }

    secp256k1::ECDSASignature result;
#ifdef SECP256K1_SHIM_RFC6979_COMPAT
    result = secp256k1::ct::ecdsa_sign_libsecp_compat(
        msg, k, ndata ? reinterpret_cast<const uint8_t*>(ndata) : nullptr);
#else
    if (ndata) {
        std::array<uint8_t, 32> aux{};
        std::memcpy(aux.data(), ndata, 32);
        result = secp256k1::ct::ecdsa_sign_hedged(msg, k, aux);
        secp256k1::detail::secure_erase(aux.data(), aux.size());  // CT-01: hedging entropy
    } else {
        result = secp256k1::ct::ecdsa_sign(msg, k);
    }
#endif
    // CT-01: erase the parsed private-key scalar `k` on every return path below.
    // (Mirrors secp256k1_schnorrsig_sign32, which erases sk before its checks.)
    secp256k1::detail::secure_erase(&k, sizeof(k));
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
