#include "secp256k1_schnorrsig.h"
#include "secp256k1_extrakeys.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>
#include <memory>
#include <vector>
#include <cstdint>
#include <chrono>     // SHIM-014: thread-local cache salt fallback
#include <random>     // SHIM-014: std::random_device for cache salt

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using namespace secp256k1::fast;

// -- Thread-local Schnorr xonly pubkey cache ----------------------------------
// One-phase design: build GLV tables on FIRST encounter (TASK-011 fix).
//
// HOT PATH:
//   HIT  → return cached SchnorrXonlyPubkey (tables pre-built, skip lift_x).
//   MISS → parse + build GLV tables immediately; return for use in same call.
//          put() returns nullptr only when x-coordinate is invalid (x >= p).
//
// Trade-off: unique-pubkey workloads (ConnectBlock ~19K unique P2TR outputs)
// write 1.5 KB × N into cache on first verify. Repeated-pubkey workloads
// (wallet, P2PK coinbases) amortize the table-build cost fully on cache hits.
namespace {
struct ShimSchnorrCache {
    static constexpr std::size_t SLOTS = 256;
    struct Slot {
        std::uint64_t             fingerprint{0};
        // T-08: store full 32-byte x-coordinate so get() verifies via memcmp, not
        // fingerprint alone. A 64-bit FNV-1a fingerprint is ~2^32-birthday-attackable
        // when the attacker controls pubkey bytes — full memcmp eliminates this.
        uint8_t                   x_bytes[32]{};
        secp256k1::SchnorrXonlyPubkey epk{};
        bool                      valid = false;  // full struct built + usable
    };
    Slot slots[SLOTS]{};

    // SHIM-014: see shim_ecdsa.cpp's shim_cache_thread_salt() block for the
    // full rationale. Schnorr's 256-slot FNV cache needs the same per-thread
    // salt so the slot mapping is not attacker-predictable (otherwise a single
    // attacker pubkey whose FNV1a maps to a victim's slot can perpetually
    // evict the victim's cached entry, preventing cache hits).
    static std::uint64_t thread_salt() noexcept {
        static thread_local std::uint64_t salt = []() noexcept {
            std::uint64_t seed = 0;
            try {
                std::random_device rd;
                seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
            } catch (...) {
                seed = 0;
            }
            if (seed == 0) {
                auto now = std::chrono::steady_clock::now().time_since_epoch().count();
                seed = static_cast<std::uint64_t>(now)
                     ^ (reinterpret_cast<std::uintptr_t>(&seed) * 0x9E3779B97F4A7C15ULL);
            }
            return seed | 1ULL;
        }();
        return salt;
    }

    static void hash32(const unsigned char data[32],
                       std::size_t& idx_out, std::uint64_t& fp_out) noexcept {
        std::uint64_t h = 14695981039346656037ULL ^ thread_salt(), w;
        for (int i = 0; i < 4; ++i) {
            // std::memcpy compiles to the same MOV on every supported toolchain
            // (GCC/Clang/MSVC); __builtin_memcpy is GCC/Clang-only and breaks
            // the MSVC build with C3861 (identifier not found).
            std::memcpy(&w, data + i * 8, 8);
            h = (h ^ w) * 1099511628211ULL;
        }
        fp_out  = h;
        idx_out = static_cast<std::size_t>(h & (SLOTS - 1));
    }

    const secp256k1::SchnorrXonlyPubkey* get(const unsigned char data[32]) const noexcept {
        std::size_t idx; std::uint64_t fp;
        hash32(data, idx, fp);
        const Slot& s = slots[idx];
        // T-08: require full 32-byte match — fingerprint alone is ~2^32-birthday-collidable.
        if (s.valid && s.fingerprint == fp && std::memcmp(s.x_bytes, data, 32) == 0) return &s.epk;
        return nullptr;
    }

    // One-phase design (PERF-001 fix): build GLV tables on FIRST encounter.
    // Eliminates the 2-call warm-up where the first verify call used the slow
    // x32 path (lift_x sqrt per call) instead of prebuilt GLV tables.
    // For workloads where each pubkey appears exactly once (unique P2TR outputs),
    // this saves ~6,285 ns (lift_x cost) per call at the cost of ~1.5 KB writes
    // per unique pubkey. For blocks with many unique pubkeys the L2 write cost
    // may trade against lift_x savings — benchmark required to confirm net effect.
    //
    // PERF-OPT: takes fp/idx pre-computed by the caller (from get()) to avoid
    // recomputing the FNV-1a hash on cache miss.
    const secp256k1::SchnorrXonlyPubkey* put(const unsigned char data[32],
                                              std::size_t idx,
                                              std::uint64_t fp) noexcept {
        Slot& s = slots[idx];
        // Cache hit: already built.
        if (s.valid && s.fingerprint == fp && std::memcmp(s.x_bytes, data, 32) == 0)
            return &s.epk;
        // Miss (first or eviction): build GLV tables immediately.
        s.fingerprint = fp;
        std::memcpy(s.x_bytes, data, 32);
        s.valid = secp256k1::schnorr_xonly_pubkey_parse(s.epk, data);
        return s.valid ? &s.epk : nullptr;
    }

    // Convenience overload: computes hash internally (for callers without cached fp/idx).
    const secp256k1::SchnorrXonlyPubkey* put(const unsigned char data[32]) noexcept {
        std::size_t idx; std::uint64_t fp;
        hash32(data, idx, fp);
        return put(data, idx, fp);
    }

    // get() variant that also returns fp/idx for reuse in put() on miss.
    const secp256k1::SchnorrXonlyPubkey* get(const unsigned char data[32],
                                              std::size_t& idx_out,
                                              std::uint64_t& fp_out) const noexcept {
        hash32(data, idx_out, fp_out);
        const Slot& s = slots[idx_out];
        if (s.valid && s.fingerprint == fp_out && std::memcmp(s.x_bytes, data, 32) == 0) return &s.epk;
        return nullptr;
    }
};
static thread_local ShimSchnorrCache s_schnorr_cache;
} // namespace

// Use the canonical context flag helpers from shim_internal.hpp.
// These replace the previous local reinterpret_cast copies — a single
// implementation prevents silent divergence if the struct layout changes.
using secp256k1_shim_internal::ctx_can_sign;
using secp256k1_shim_internal::ctx_can_verify;

extern "C" {

// SHIM-002 fix: return 0 (failure). This stub is exported for ABI compatibility
// only — the shim never calls this pointer; it uses its own BIP-340 aux_rand nonce
// derivation internally. Returning 1 from a function that writes nothing would be
// a silent ABI trap: any caller using the pointer as an independent hash primitive
// would receive "success" with zero output bytes.
static int nonce_function_bip340_stub(unsigned char *, const unsigned char *,
    size_t, const unsigned char *, const unsigned char *,
    const unsigned char *, size_t, void *)
{ return 0; }

const secp256k1_nonce_function_hardened secp256k1_nonce_function_bip340 =
    nonce_function_bip340_stub;

// -- Sign -----------------------------------------------------------------

int secp256k1_schnorrsig_sign32(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg32,
    const secp256k1_keypair *keypair,
    const unsigned char *aux_rand32)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_sign32: NULL context");
        return 0;
    }
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_SIGN.
    if (!ctx_can_sign(ctx)) return 0;
    if (!sig64 || !msg32 || !keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_sign32: NULL argument");
        return 0;
    }
    std::memset(sig64, 0, 64);
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);

    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk)) return 0;

    std::array<uint8_t, 32> msg{};
    std::memcpy(msg.data(), msg32, 32);

    std::array<uint8_t, 32> aux{};
    if (aux_rand32) std::memcpy(aux.data(), aux_rand32, 32);

    // Fast path: reuse the pubkey X already stored in keypair->data[32..63]
    // by secp256k1_keypair_create / secp256k1_keypair_xonly_tweak_add.
    // Avoids ct::generator_mul_blinded (~9-10 µs) on every signing call.
    //
    // Layout: data[0..31]=sk, data[32..63]=pub_X, data[64..95]=pub_Y
    // BIP-340: signing key d must yield even-Y pubkey. Y-parity from data[95].
    secp256k1::SchnorrKeypair kp;
    {
        bool const y_odd = (keypair->data[95] & 1u) != 0u;
        // NEW-006: use ct::scalar_cneg instead of ternary to avoid variable-time
        // branch on y_odd that is correlated with the secret signing key d.
        // Y-parity of a pubkey is public, but the compiler may emit a branch
        // that leaks via timing. ct::scalar_cneg is branchless.
        kp.d = secp256k1::ct::scalar_cneg(sk, secp256k1::ct::bool_to_mask(y_odd));
        std::memcpy(kp.px.data(), keypair->data + 32, 32);
    }
    auto sig = secp256k1::ct::schnorr_sign(kp, msg, aux);
    secp256k1::detail::secure_erase(&sk,   sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    if (sig.s.is_zero()) return 0;  // fail-closed: degenerate nonce (≈2^-256)
    // SEC-006: CT OR accumulator replaces variable-time for+break loop.
    // All 32 bytes of r are visited unconditionally — no early exit on signing output.
    {
        std::uint32_t r_nonzero = 0;
        for (int i = 0; i < 32; i++) r_nonzero |= sig.r[i];
        if (r_nonzero == 0) return 0;
    }
    auto sig_bytes = sig.to_bytes();
    std::memcpy(sig64, sig_bytes.data(), 64);
    return 1;
}

int secp256k1_schnorrsig_sign_custom(
    const secp256k1_context *ctx,
    unsigned char *sig64,
    const unsigned char *msg,
    size_t msglen,
    const secp256k1_keypair *keypair,
    secp256k1_schnorrsig_extraparams *extraparams)
{
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_SIGN.
    // NULL ctx must fire the illegal callback (PASS3-008 divergence fix).
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_schnorrsig_sign_custom: NULL context");
        return 0;
    }
    if (!ctx_can_sign(ctx)) return 0;
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);

    // NULL output-arg check fires before extraparams parsing (matches upstream ordering:
    // libsecp256k1 calls the illegal callback for NULL sig64/keypair before inspecting
    // extraparams or rejecting non-canonical nonce functions).
    if (!sig64 || !keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_sign_custom: NULL argument");
        return 0;
    }
    std::memset(sig64, 0, 64);

    // Unpack extraparams (upstream libsecp256k1 v0.4+ API).
    secp256k1_nonce_function_hardened noncefp = nullptr;
    void *ndata = nullptr;
    if (extraparams != nullptr) {
        // Validate magic bytes { 0xda, 0x6f, 0xb3, 0x8c }
        static const unsigned char magic[4] = SECP256K1_SCHNORRSIG_EXTRAPARAMS_MAGIC;
        if (std::memcmp(extraparams->magic, magic, 4) != 0) return 0;
        noncefp = extraparams->noncefp;
        ndata   = extraparams->ndata;
    }
    // Fail-closed: reject non-canonical nonce functions. Fire illegal callback (PASS3-001 fix).
    if (noncefp != nullptr && noncefp != secp256k1_nonce_function_bip340) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_schnorrsig_sign_custom: custom nonce functions are not supported; "
            "pass NULL or secp256k1_nonce_function_bip340");
        return 0;
    }
    if (msglen > 0 && !msg) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_sign_custom: NULL msg with nonzero msglen");
        return 0;
    }
    // msglen == 32: forward to the optimized sign32 fast path. This keeps
    // sign_custom(msglen=32) byte-identical to sign32 (guarded by VCS-6) and
    // reuses sign32's keypair-from-stored-pubkey fast reconstruction.
    // ndata is the aux_rand32 per the libsecp contract.
    if (msglen == 32) {
        return secp256k1_schnorrsig_sign32(ctx, sig64, msg, keypair,
                                           static_cast<const unsigned char*>(ndata));
    }

    // Variable-length path (msglen != 32): full BIP-340 with the message folded
    // verbatim into H_BIP0340/nonce(t‖P_x‖msg) and H_BIP0340/challenge(R_x‖P_x‖msg),
    // matching upstream libsecp256k1 secp256k1_schnorrsig_sign_custom. The CT
    // construction (blinded nonce, branchless conditional negate, secure_erase of
    // every secret-derived buffer) lives in the audited library overload
    // secp256k1::ct::schnorr_sign(kp, msg, msglen, aux). secp256k1_schnorrsig_verify
    // accepts any msglen (SHIM-004), so the sign/verify pair is symmetric again —
    // the asymmetry that motivated removing this path (AUDIT-003) no longer exists.
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk)) return 0;

    // Reconstruct the BIP-340 keypair from the strict-parsed secret key. Using
    // ct::schnorr_keypair_create (one ct::generator_mul) rather than the stored
    // pubkey is intentional on this rarely-hot varlen path: it does not trust the
    // cached pubkey bytes in keypair->data and matches the historical impl.
    secp256k1::SchnorrKeypair kp = secp256k1::ct::schnorr_keypair_create(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));

    std::array<uint8_t, 32> aux{};
    if (ndata) std::memcpy(aux.data(), ndata, 32);

    auto sig = secp256k1::ct::schnorr_sign(
        kp, static_cast<const uint8_t*>(msg), msglen, aux);
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    secp256k1::detail::secure_erase(aux.data(), aux.size());

    if (sig.s.is_zero()) return 0;  // fail-closed: degenerate nonce (≈2^-256)
    // SEC-006: CT OR accumulator — visit all 32 bytes, no early exit on output.
    {
        std::uint32_t r_nonzero = 0;
        for (int i = 0; i < 32; ++i) r_nonzero |= sig.r[i];
        if (r_nonzero == 0) return 0;
    }
    auto sig_bytes = sig.to_bytes();
    std::memcpy(sig64, sig_bytes.data(), 64);
    return 1;
}

// -- Verify ---------------------------------------------------------------

int secp256k1_schnorrsig_verify(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg, size_t msglen,
    const secp256k1_xonly_pubkey *pubkey)
{
    // NULL context: fire illegal callback (matches upstream libsecp256k1 behavior)
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(NULL, "secp256k1_schnorrsig_verify: NULL context");
        return 0;
    }
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_VERIFY
    // (or a context created with CONTEXT_SIGN which is a superset).
    if (!ctx_can_verify(ctx)) return 0;
    if (!sig64 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_verify: NULL argument");
        return 0;
    }
    // SHIM-003 fix: upstream libsecp256k1 allows NULL msg when msglen == 0 (zero-length
    // message is a valid BIP-340 construct). Only fire illegal callback when msg is NULL
    // with a non-zero msglen, which would cause a null dereference in schnorr_verify.
    // (Main's varlen-verify commit a3b77fde used a shorter callback string; the SHIM-003
    // form on dev is more diagnostic — keep it.)
    if (!msg && msglen > 0) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_verify: NULL msg with non-zero msglen");
        return 0;
    }

    // PERF-007: use raw-pointer parse_strict overload — avoids 64-byte stack copy.
    secp256k1::SchnorrSignature sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, sig)) return 0;

    // PERF-005: pass pubkey->data raw pointer directly — eliminates two 32-byte
    // stack copies (xb/yb arrays) that were an avoidable memcpy per verify call.
    // P1-PERF-001: use stored Y from data[32..63] to skip lift_x sqrt per call.
    // T-11: check ShimSchnorrCache first — on cache hit, use SchnorrXonlyPubkey
    //   overload with prebuilt GLV tables, saving ~1,954 ns per repeated-pubkey call.
    {
        const uint8_t* xb = pubkey->data;
        const uint8_t* yb = pubkey->data + 32;

        // SHIM-004 fix: when msglen != 32, use the varlen overload which forwards the
        // full message length to the BIP-340 challenge hash. All optimized code paths
        // below use 32-byte fixed overloads and are only correct for msglen == 32.
        if (msglen != 32) {
            return secp256k1::schnorr_verify(xb, msg, msglen, sig) ? 1 : 0;
        }

        // msglen == 32: use optimized paths with caching and prebuilt GLV tables.
        // PERF-OPT: compute hash once (get() returns fp/idx), pass to put() on miss.
        // Eliminates the double FNV-1a hash that get()+put() previously caused.
        std::size_t cache_idx; std::uint64_t cache_fp;
        if (const secp256k1::SchnorrXonlyPubkey* cached =
                s_schnorr_cache.get(xb, cache_idx, cache_fp)) {
            return secp256k1::schnorr_verify(*cached, msg, sig) ? 1 : 0;
        }

        // Cache miss: put() builds GLV tables on first encounter (1-phase design).
        // put() uses the pre-computed hash (cache_idx, cache_fp) — no re-hash.
        if (const secp256k1::SchnorrXonlyPubkey* built =
                s_schnorr_cache.put(xb, cache_idx, cache_fp)) {
            return secp256k1::schnorr_verify(*built, msg, sig) ? 1 : 0;
        }
        // put() returned nullptr: x-coordinate failed schnorr_xonly_pubkey_parse
        // (e.g. x >= p). Fall back to the raw verify path.
        return secp256k1::schnorr_verify(xb, msg, sig) ? 1 : 0;
    }
}

// -- Pre-computed xonly pubkey API -----------------------------------------

static_assert(sizeof(secp256k1_xonly_pubkey_precomp) >= sizeof(secp256k1::SchnorrXonlyPubkey),
    "SECP256K1_XONLY_PUBKEY_PRECOMP_SIZE too small — update the #define in secp256k1_schnorrsig.h");
static_assert(alignof(secp256k1_xonly_pubkey_precomp) >= alignof(secp256k1::SchnorrXonlyPubkey),
    "secp256k1_xonly_pubkey_precomp alignment insufficient for SchnorrXonlyPubkey");

int secp256k1_xonly_ec_pubkey_precomp(
    const secp256k1_context* ctx,
    secp256k1_xonly_pubkey_precomp* out,
    const secp256k1_xonly_pubkey* pubkey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out || !pubkey) return 0;
    auto* epk = reinterpret_cast<secp256k1::SchnorrXonlyPubkey*>(out);
    // pubkey->data[0..31] = x-only key bytes
    return secp256k1::schnorr_xonly_pubkey_parse(*epk, pubkey->data) ? 1 : 0;
}

int secp256k1_xonly_pubkey_parse_precomp(
    const secp256k1_context* ctx,
    secp256k1_xonly_pubkey_precomp* out,
    const unsigned char* pubkey_x32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out || !pubkey_x32) return 0;
    auto* epk = reinterpret_cast<secp256k1::SchnorrXonlyPubkey*>(out);
    // Single call: schnorr_xonly_pubkey_parse now builds GLV tables on first call.
    // The old two-call protocol was needed when tables were built on the second call;
    // that has been superseded — the first call is sufficient.
    return secp256k1::schnorr_xonly_pubkey_parse(*epk, pubkey_x32) ? 1 : 0;
}

int secp256k1_schnorrsig_verify_precomp(
    const secp256k1_context* ctx,
    const unsigned char* sig64,
    const unsigned char* msg32,
    const secp256k1_xonly_pubkey_precomp* pubkey)
{
    if (!ctx_can_verify(ctx)) return 0;
    if (!sig64 || !msg32 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_verify_precomp: NULL argument");
        return 0;
    }
    // PERF-001: use raw-pointer parse_strict overload — avoids 64-byte stack copy.
    secp256k1::SchnorrSignature sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, sig)) return 0;

    const auto* epk = reinterpret_cast<const secp256k1::SchnorrXonlyPubkey*>(pubkey);
    // PERF-001: pass msg32 directly — avoids 32-byte stack copy.
    // Direct use of pre-built tables — zero lift_x and zero GLV rebuild overhead.
    return secp256k1::schnorr_verify(*epk, msg32, sig) ? 1 : 0;
}

} // extern "C"
