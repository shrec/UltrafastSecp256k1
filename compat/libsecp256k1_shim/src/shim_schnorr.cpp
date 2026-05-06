#include "secp256k1_schnorrsig.h"
#include "secp256k1_extrakeys.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>
#include <vector>
#include <cstdint>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using namespace secp256k1::fast;

// -- Thread-local Schnorr xonly pubkey cache ----------------------------------
// Eliminates ~2,600 ns lift_x (sqrt) + build_glv52_table_zr per verify on hit.
// Direct-mapped 16 slots, keyed by first 8 bytes of pubkey->data (x-only 32 B).
namespace {
struct ShimSchnorrCache {
    static constexpr std::size_t SLOTS = 256;
    struct Slot {
        std::uint8_t              raw[32]{};
        secp256k1::SchnorrXonlyPubkey epk{};
        bool                      valid = false;
    };
    Slot slots[SLOTS]{};

    static std::size_t slot_of(const unsigned char data[32]) noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < 8; ++i) h = (h ^ data[i]) * 1099511628211ULL;
        return static_cast<std::size_t>(h & (SLOTS - 1));
    }

    const secp256k1::SchnorrXonlyPubkey* get(const unsigned char data[32]) const noexcept {
        const Slot& s = slots[slot_of(data)];
        if (s.valid && std::memcmp(s.raw, data, 32) == 0) return &s.epk;
        return nullptr;
    }

    const secp256k1::SchnorrXonlyPubkey* put(const unsigned char data[32]) noexcept {
        Slot& s = slots[slot_of(data)];
        s.valid = secp256k1::schnorr_xonly_pubkey_parse(s.epk, data);
        if (s.valid) std::memcpy(s.raw, data, 32);
        return s.valid ? &s.epk : nullptr;
    }
};
static thread_local ShimSchnorrCache s_schnorr_cache;
} // namespace

// Context flag helpers (mirrors shim_ecdsa.cpp; flags is first field in struct).
namespace {
    inline unsigned int schnorr_ctx_flags(const secp256k1_context *ctx) {
        if (!ctx) return 0;
        return *reinterpret_cast<const unsigned int *>(ctx);
    }
    inline bool schnorr_ctx_can_sign(const secp256k1_context *ctx) {
        if (!ctx) return false;
        unsigned int f = schnorr_ctx_flags(ctx);
        if (!(f & SECP256K1_FLAGS_TYPE_CONTEXT)) return false;
        // libsecp v0.6+: CONTEXT_NONE accepted for signing (Bitcoin Core uses it since v26)
        return (f & SECP256K1_FLAGS_BIT_CONTEXT_SIGN) ||
               ((f & ~SECP256K1_FLAGS_TYPE_MASK) == 0);
    }
    inline bool schnorr_ctx_can_verify(const secp256k1_context *ctx) {
        if (!ctx) return false;
        unsigned int f = schnorr_ctx_flags(ctx);
        if (!(f & SECP256K1_FLAGS_TYPE_CONTEXT)) return false;
        // CONTEXT_NONE, CONTEXT_VERIFY, or CONTEXT_SIGN all allow verify
        return (f & SECP256K1_FLAGS_BIT_CONTEXT_VERIFY) ||
               (f & SECP256K1_FLAGS_BIT_CONTEXT_SIGN) ||
               ((f & ~SECP256K1_FLAGS_TYPE_MASK) == 0);
    }
}

extern "C" {

static int nonce_function_bip340_stub(unsigned char *, const unsigned char *,
    size_t, const unsigned char *, const unsigned char *,
    const unsigned char *, size_t, void *)
{ return 1; }

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
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_SIGN.
    if (!schnorr_ctx_can_sign(ctx)) return 0;
    if (!sig64 || !msg32 || !keypair) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_sign32: NULL argument");
        return 0;
    }

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
        kp.d = y_odd ? sk.negate() : sk;
        std::memcpy(kp.px.data(), keypair->data + 32, 32);
    }
    auto sig = secp256k1::ct::schnorr_sign(kp, msg, aux);
    secp256k1::detail::secure_erase(&sk,   sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    if (sig.s.is_zero()) return 0;  // fail-closed: degenerate nonce (≈2^-256)
    {
        bool r_all_zero = true;
        for (int i = 0; i < 32; i++) { if (sig.r[i] != 0) { r_all_zero = false; break; } }
        if (r_all_zero) return 0;
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
    if (!schnorr_ctx_can_sign(ctx)) return 0;

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
    (void)ndata;  // ndata accepted for API compat but not forwarded (see header doc)

    // Fail-closed: reject non-canonical nonce functions.
    if (noncefp != nullptr && noncefp != secp256k1_nonce_function_bip340) return 0;

    if (!sig64 || !keypair) return 0;
    if (msglen > 0 && !msg) return 0;

    // Fast path: 32-byte message uses optimised internal sign32 directly.
    // sign32 accepts aux_rand32 == nullptr → deterministic zero-aux nonce.
    if (msglen == 32)
        return secp256k1_schnorrsig_sign32(ctx, sig64, msg, keypair, nullptr);

    // Variable-length path: full BIP-340 with arbitrary-length msg in hashes.
    // Upstream libsecp256k1 sign_custom includes msg verbatim in:
    //   H_BIP0340/nonce(t || P_x || msg)  and  H_BIP0340/challenge(R_x || P_x || msg)
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk)) return 0;

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));

    // t = d XOR H_BIP0340/aux(zero_aux)
    // NULL / missing aux_rand path: aux is always 32 zero bytes here.
    // DIVERGENCE from upstream sign_custom: upstream passes extraparams->ndata
    // as aux to the nonce function. This shim uses zero aux for the variable-
    // length path because ndata is not forwarded (see above).
    static constexpr uint8_t kZeroAux[32] = {};
    auto t_hash = secp256k1::tagged_hash("BIP0340/aux", kZeroAux, 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // k' = H_BIP0340/nonce(t || P_x || msg)
    std::vector<uint8_t> nonce_input(64 + msglen);
    std::memcpy(nonce_input.data(),      t,             32);
    std::memcpy(nonce_input.data() + 32, kp.px.data(), 32);
    if (msglen > 0) std::memcpy(nonce_input.data() + 64, msg, msglen);
    auto rand_hash = secp256k1::tagged_hash("BIP0340/nonce", nonce_input.data(), nonce_input.size());
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) {
        secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
        return 0;
    }

    // R = k' * G (CT path)
    auto R = secp256k1::ct::generator_mul(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();
    // CT: branchless conditional negate — r_y_odd is derived from secret nonce.
    auto k = secp256k1::ct::scalar_cneg(k_prime, secp256k1::ct::bool_to_mask(r_y_odd));
    secp256k1::detail::secure_erase(&k_prime, sizeof(k_prime));

    // e = H_BIP0340/challenge(R_x || P_x || msg)
    std::vector<uint8_t> challenge_input(64 + msglen);
    std::memcpy(challenge_input.data(),      rx.data(),     32);
    std::memcpy(challenge_input.data() + 32, kp.px.data(), 32);
    if (msglen > 0) std::memcpy(challenge_input.data() + 64, msg, msglen);
    auto e_hash = secp256k1::tagged_hash("BIP0340/challenge", challenge_input.data(), challenge_input.size());
    auto e = Scalar::from_bytes(e_hash);

    // s = k + e * d  — CT: both k (nonce) and kp.d (secret key) are secret
    auto s = secp256k1::ct::scalar_add(k, secp256k1::ct::scalar_mul(e, kp.d));
    secp256k1::detail::secure_erase(&k,      sizeof(k));
    secp256k1::detail::secure_erase(&kp.d,   sizeof(kp.d));
    secp256k1::detail::secure_erase(d_bytes.data(), d_bytes.size());
    secp256k1::detail::secure_erase(t, sizeof(t));

    if (s.is_zero()) return 0;
    {
        bool r_all_zero = true;
        for (int i = 0; i < 32; i++) { if (rx[i] != 0) { r_all_zero = false; break; } }
        if (r_all_zero) return 0;
    }

    std::memcpy(sig64,      rx.data(),         32);
    auto s_bytes = s.to_bytes();
    std::memcpy(sig64 + 32, s_bytes.data(),    32);
    return 1;
}

// -- Verify ---------------------------------------------------------------
// NOTE on sign/verify asymmetry:
// secp256k1_schnorrsig_sign_custom accepts arbitrary msglen (full BIP-340
// construction), but secp256k1_schnorrsig_verify ONLY accepts msglen == 32.
// This matches the default BIP-340 profile in upstream libsecp256k1 where
// verify always uses a 32-byte message hash. Callers that use sign_custom with
// msglen != 32 CANNOT verify with this function — they need a custom verifier.

int secp256k1_schnorrsig_verify(
    const secp256k1_context *ctx,
    const unsigned char *sig64,
    const unsigned char *msg, size_t msglen,
    const secp256k1_xonly_pubkey *pubkey)
{
    // Context flag enforcement: upstream libsecp256k1 requires CONTEXT_VERIFY
    // (or a context created with CONTEXT_SIGN which is a superset).
    if (!schnorr_ctx_can_verify(ctx)) return 0;
    if (!sig64 || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_schnorrsig_verify: NULL argument");
        return 0;
    }
    // Only 32-byte messages supported — see asymmetry note above.
    if (!msg || msglen != 32) return 0;

    secp256k1::SchnorrSignature sig;
    std::array<uint8_t, 64> sig_buf{};
    std::memcpy(sig_buf.data(), sig64, 64);
    if (!secp256k1::SchnorrSignature::parse_strict(sig_buf, sig)) return 0;

    std::array<uint8_t, 32> msg32{};
    std::memcpy(msg32.data(), msg, 32);

    // x-only key is stored in the first 32 bytes of the opaque 64-byte struct.
    // Use cached SchnorrXonlyPubkey to skip lift_x + build_glv52_table_zr (~2,600ns).
    const secp256k1::SchnorrXonlyPubkey* epk = s_schnorr_cache.get(pubkey->data);
    if (!epk) epk = s_schnorr_cache.put(pubkey->data);
    if (!epk) return 0;  // invalid x-coordinate
    return secp256k1::schnorr_verify(*epk, msg32, sig) ? 1 : 0;
}

} // extern "C"
