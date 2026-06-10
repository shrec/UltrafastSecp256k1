// ============================================================================
// ct_sign.cpp -- Constant-Time Signing Functions
// ============================================================================
// Drop-in CT replacements for ecdsa_sign() and schnorr_sign().
// Nonce multiplications (R = k*G) use ct::generator_mul_blinded() for DPA
// defense when secp256k1_context_randomize() has been called.
// Pubkey derivations in fault-check verify steps use ct::generator_mul()
// (public output, DPA blinding not required there).
// ============================================================================

#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/config.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>
#include <memory>

namespace {
using secp256k1::detail::secure_erase;
} // anonymous namespace

namespace secp256k1::ct {

// ============================================================================
// CT ECDSA Sign
// ============================================================================
// Pure CT sign: no sign-then-verify countermeasure.
// Use ct::ecdsa_sign_verified() if fault attack resistance is needed.

ECDSASignature ecdsa_sign(const std::array<uint8_t, 32>& msg_hash,
                          const Scalar& private_key) {
    if (private_key.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);  // NOT a secret scalar — public message hash

    // Deterministic nonce (RFC 6979)
    auto k = rfc6979_nonce(private_key, msg_hash);
    // k.is_zero_ct() guards against RFC 6979 100-attempt exhaustion (≈2^−8000
    // probability); not a timing concern since k is never zero in practice.
    if (k.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G  -- blinded CT path (DPA defense via secp256k1_context_randomize)
    auto R = ct::generator_mul_blinded(k);
    // k != 0 on secp256k1's prime-order group guarantees R ≠ ∞; no check needed.

    // r = R.x mod n (R.x is public after ct::generator_mul_blinded)
    // from_limbs() skips the byte-swap round-trip; ge(ORDER) check is still
    // performed because secp256k1 p > n (rare values in [n,p) need reduction).
    auto r = Scalar::from_limbs(R.x().limbs());  // NOT a secret scalar — public curve point x-coord
    // r is derived from k*G.x (secret nonce): use is_zero_ct() per IS-ZERO-CT constraint.
    // Probability of r==0 is ≈2^−128 (requires R.x == curve order exactly).
    if (r.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    // s = k^{-1} * (z + r * d) mod n
    // CT inverse: SafeGCD Bernstein-Yang divsteps-59, constant-time.
    // ct::scalar_mul / ct::scalar_add: branchless reduction — no secret-dependent branch
    // unlike fast::Scalar::operator* which has branchy ge() in final reduction (V-01).
    auto k_inv = ct::scalar_inverse(k);
    auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
    // s.is_zero_ct() is astronomically unlikely for valid inputs.
    if (s.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    // CT low-S normalization: branchless comparison with n/2 + conditional negate.
    ECDSASignature const sig = ct::ct_normalize_low_s(ECDSASignature{r, s});

    // Erase secret nonce material from stack.
    secure_erase(&k,     sizeof(k));
    secure_erase(&k_inv, sizeof(k_inv));
    secure_erase(&z,     sizeof(z));
    secure_erase(&s,     sizeof(s));

    return sig;
}

// ============================================================================
// CT ECDSA Sign + Verify (fault attack countermeasure)
// ============================================================================
// Signs and then verifies (FIPS 186-4 fault countermeasure).
// Verify uses fast path -- public key and signature are not secret.

ECDSASignature ecdsa_sign_verified(const std::array<uint8_t, 32>& msg_hash,
                                   const Scalar& private_key) {
    auto sig = ecdsa_sign(msg_hash, private_key);

    // CT: use is_zero_ct() for consistency with hedged_verified (belt-and-suspenders).
    if (!sig.r.is_zero_ct()) {
        auto pk = ct::generator_mul(private_key);
        if (!ecdsa_verify(msg_hash.data(), pk, sig)) {
            return {Scalar::zero(), Scalar::zero()};
        }
    }

    return sig;
}

// ============================================================================
// CT ECDSA Sign (hedged, with extra entropy)
// ============================================================================
// RFC 6979 Section 3.6: aux_rand mixed into HMAC-DRBG for defense-in-depth.
// Uses ct::generator_mul() for R = k*G (constant-time).

// Forward declaration (defined in ecdsa.cpp, declared in ecdsa.hpp)
// rfc6979_nonce_hedged is in namespace secp256k1 (already accessible via ecdsa.hpp include)

ECDSASignature ecdsa_sign_hedged(const std::array<uint8_t, 32>& msg_hash,
                                  const Scalar& private_key,
                                  const std::array<uint8_t, 32>& aux_rand) {
    if (private_key.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = secp256k1::rfc6979_nonce_hedged(private_key, msg_hash, aux_rand);
    // k.is_zero_ct() guards against RFC 6979 exhaustion (≈2^−8000 probability).
    if (k.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G  -- blinded CT path (DPA defense via secp256k1_context_randomize)
    auto R = ct::generator_mul_blinded(k);
    // k != 0 on secp256k1's prime-order group guarantees R ≠ ∞; no check needed.

    // from_limbs: direct 4×64 copy + ge(ORDER) check — same as ct::ecdsa_sign.
    // Avoids to_bytes()+from_bytes() round-trip (~15 ns of byte-swap overhead).
    auto r = Scalar::from_limbs(R.x().limbs());
    // r is derived from k*G.x (secret nonce): use is_zero_ct() per IS-ZERO-CT constraint.
    if (r.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    // CT inverse + CT scalar multiplication (V-01 fix: operator* is variable-time).
    auto k_inv = ct::scalar_inverse(k);
    auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
    if (s.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    // CT low-S normalization (branchless)
    ECDSASignature const sig = ct::ct_normalize_low_s(ECDSASignature{r, s});

    // Erase secret nonce material from stack.
    secure_erase(&k,     sizeof(k));
    secure_erase(&k_inv, sizeof(k_inv));
    secure_erase(&z,     sizeof(z));
    secure_erase(&s,     sizeof(s));

    return sig;
}

// ============================================================================
// CT ECDSA Sign Hedged + Verify (fault attack countermeasure)
// ============================================================================

ECDSASignature ecdsa_sign_hedged_verified(const std::array<uint8_t, 32>& msg_hash,
                                          const Scalar& private_key,
                                          const std::array<uint8_t, 32>& aux_rand) {
    auto sig = ecdsa_sign_hedged(msg_hash, private_key, aux_rand);

    // CT: r = kG.x mod n is nonce-derived — use is_zero_ct() not is_zero().
    if (!sig.r.is_zero_ct()) {
        auto pk = ct::generator_mul(private_key);
        if (!ecdsa_verify(msg_hash.data(), pk, sig)) {
            return {Scalar::zero(), Scalar::zero()};
        }
    }

    return sig;
}

// ============================================================================
// CT ECDSA Sign libsecp256k1-compatible (SECP256K1_SHIM_RFC6979_COMPAT)
// ============================================================================
// Identical to ecdsa_sign_hedged / ecdsa_sign_hedged_recoverable but calls
// rfc6979_nonce_libsecp_compat instead of rfc6979_nonce_hedged.
// ndata32 is passed directly — no OS CSPRNG mixing.

ECDSASignature ecdsa_sign_libsecp_compat(const std::array<uint8_t, 32>& msg_hash,
                                          const Scalar& private_key,
                                          const uint8_t* ndata32) {
    if (private_key.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = secp256k1::rfc6979_nonce_libsecp_compat(private_key, msg_hash, ndata32);
    if (k.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    auto R = ct::generator_mul_blinded(k);

    auto r = Scalar::from_limbs(R.x().limbs());
    // r is derived from k*G.x (secret nonce): use is_zero_ct() per IS-ZERO-CT constraint.
    if (r.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    auto k_inv = ct::scalar_inverse(k);
    auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
    if (s.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    ECDSASignature const sig = ct::ct_normalize_low_s(ECDSASignature{r, s});

    secure_erase(&k,     sizeof(k));
    secure_erase(&k_inv, sizeof(k_inv));
    secure_erase(&z,     sizeof(z));
    secure_erase(&s,     sizeof(s));

    return sig;
}

RecoverableSignature ecdsa_sign_libsecp_compat_recoverable(
    const std::array<uint8_t, 32>& msg_hash,
    const Scalar& private_key,
    const uint8_t* ndata32) {

    static const std::array<uint8_t, 32> ORDER_BYTES = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };

    if (private_key.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = secp256k1::rfc6979_nonce_libsecp_compat(private_key, msg_hash, ndata32);
    if (k.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    auto R = ct::generator_mul_blinded(k);

    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    // r is derived from k*G.x (secret nonce): use is_zero_ct() per IS-ZERO-CT constraint.
    if (r.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Recovery ID bit 0: parity of R.y (branchless)
    int recid = static_cast<int>(R.y().limbs()[0] & 1u);

    // Recovery ID bit 1: R.x >= n overflow (branchless byte comparison)
    {
        unsigned gt = 0u, eq_run = 1u;
        for (int i = 0; i < 32; ++i) {
            unsigned const rb = static_cast<unsigned>(r_bytes[i]);
            unsigned const ob = static_cast<unsigned>(ORDER_BYTES[i]);
            unsigned const byte_gt = ((ob - rb) >> 31) & 1u;
            unsigned const byte_lt = ((rb - ob) >> 31) & 1u;
            gt     = gt | (eq_run & byte_gt);
            eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
        }
        recid |= static_cast<int>(gt) << 1;
    }

    auto k_inv = ct::scalar_inverse(k);
    auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
    if (s.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    ECDSASignature pre_sig{r, s};
    std::uint64_t const high_mask = ct::scalar_is_high(pre_sig.s);
    const ECDSASignature sig = ct::ct_normalize_low_s(pre_sig);
    recid ^= static_cast<int>(high_mask & 1);

    secure_erase(&k,       sizeof(k));
    secure_erase(&k_inv,   sizeof(k_inv));
    secure_erase(&z,       sizeof(z));
    secure_erase(&s,       sizeof(s));
    secure_erase(&pre_sig, sizeof(pre_sig));

    return {sig, recid};
}

// ============================================================================
// CT Schnorr helpers
// ============================================================================

// -- Shared BIP-340 tagged-hash midstates (from tagged_hash.hpp) ---------------
using detail::g_aux_midstate;
using detail::g_nonce_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

// ============================================================================
// CT Schnorr Pubkey
// ============================================================================

std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    auto P = ct::generator_mul(private_key);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    (void)p_y_odd;
    return px;
}

// ============================================================================
// CT Schnorr Keypair Create
// ============================================================================

SchnorrKeypair schnorr_keypair_create(const Scalar& private_key) {
    SchnorrKeypair kp{};
    auto d_prime = private_key;
    if (d_prime.is_zero_ct()) return kp;

    auto P = ct::generator_mul(d_prime);
    auto [px, p_y_odd] = P.x_bytes_and_parity();

    // CT: conditional negate based on parity. p_y_odd is derived from the
    // secret key, so the ternary branch would leak via timing.
    kp.d = ct::scalar_cneg(d_prime, ct::bool_to_mask(p_y_odd));
    kp.px = px;
    // d_prime is a private-key copy — scrub it from the stack (kp.d, the public
    // x-only signing key, is the intended output and is returned by value).
    secure_erase(&d_prime, sizeof(d_prime));
    return kp;
}

// ============================================================================
// CT Schnorr Sign (BIP-340, keypair variant)
// ============================================================================

SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    if (kp.d.is_zero_ct()) return SchnorrSignature{};

    // Step 1: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 2: k' = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, kp.px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, 96);
    // CT note: from_bytes uses a constant-time mod-n reduction (branchless subtraction).
    // parse_bytes_strict_nonzero + for loop creates a data-dependent branch that
    // dudect detects as a timing leak. from_bytes passes dudect with |t| < 4.5.
    // The theoretical 2^-128 branch in from_bytes is negligible in practice.
    auto k_prime = Scalar::from_bytes(rand_hash);

    // Step 3: R = k' * G -- blinded CT path (DPA defense via secp256k1_context_randomize)
    auto R = ct::generator_mul_blinded(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 4: k = k' if even_y(R), else n - k'
    // CT: branchless conditional negate. r_y_odd is secret-derived (from k').
    auto k = ct::scalar_cneg(k_prime, ct::bool_to_mask(r_y_odd));

    // Step 5: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg)
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, rx.data(), 32);
    std::memcpy(challenge_input + 32, kp.px.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 6: sig = (R.x, k + e * d)  — CT scalar ops on secret k and kp.d (V-01 fix)
    SchnorrSignature sig{};
    sig.r = rx;
    sig.s = ct::scalar_add(k, ct::scalar_mul(e, kp.d));

    // Erase ALL stack buffers that held secret-derived material:
    //   d_bytes[32]          -- private key serialized
    //   t_hash[32]           -- tagged_hash output (XOR'd with d_bytes)
    //   t[32]                -- d XOR t_hash (derived from private key)
    //   nonce_input[96]      -- t || pubkey_x || msg (contains t)
    //   rand_hash[32]        -- nonce hash output (determines k')
    //   challenge_input[96]  -- R.x || pubkey_x || msg (public but erase for hygiene)
    //   k_prime, k           -- secret nonce scalars
    secure_erase(d_bytes.data(), d_bytes.size());
    secure_erase(t_hash.data(), t_hash.size());
    secure_erase(t, sizeof(t));
    secure_erase(nonce_input, sizeof(nonce_input));
    secure_erase(rand_hash.data(), rand_hash.size());
    secure_erase(challenge_input, sizeof(challenge_input));
    secure_erase(e_hash.data(), e_hash.size());
    secure_erase(&e, sizeof(e));
    // Erase secret nonce scalars. Scalar is a POD-like 32-byte struct (4x uint64_t).
    secure_erase(&k_prime, sizeof(k_prime));
    secure_erase(&k, sizeof(k));

    return sig;
}

// ============================================================================
// CT Schnorr Sign — variable-length message (BIP-340, sign_custom)
// ============================================================================
// Identical construction to the fixed-32 overload above, but the message is an
// arbitrary-length byte string folded verbatim into the BIP-340 nonce and
// challenge tagged hashes — matching upstream libsecp256k1
// secp256k1_schnorrsig_sign_custom. For msglen == 32 the output is byte-identical
// to schnorr_sign(kp, msg32, aux). All secret-bearing buffers are erased on every
// return path exactly as in the fixed-32 path. msg may be nullptr only when
// msglen == 0.
//
// History: this varlen signing path previously lived inline in the libsecp256k1
// shim (shim_schnorr.cpp). It was removed (AUDIT-003) only because the shim's
// verify was 32-byte-only at the time, creating a sign/verify asymmetry. Verify
// is now varlen (matches upstream), so the symmetric varlen signing path is
// restored here in the audited CT library rather than re-duplicated in the shim.
SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const uint8_t* msg, std::size_t msglen,
                              const std::array<uint8_t, 32>& aux_rand) {
    if (kp.d.is_zero_ct()) return SchnorrSignature{};

    // SBO: avoid heap alloc for messages up to 512 bytes (mirrors
    // schnorr_challenge_scalar_varlen in schnorr.cpp). Heap only for larger.
    static constexpr std::size_t kSBOMax = 512;
    const std::size_t hbuf_len = 64 + msglen;

    // Step 1: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 2: k' = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_stack[64 + kSBOMax];
    std::unique_ptr<uint8_t[]> nonce_heap;
    uint8_t* nonce_input = (msglen <= kSBOMax)
        ? nonce_stack
        : (nonce_heap.reset(new uint8_t[hbuf_len]), nonce_heap.get());
    std::memcpy(nonce_input,      t,            32);
    std::memcpy(nonce_input + 32, kp.px.data(), 32);
    if (msglen > 0) std::memcpy(nonce_input + 64, msg, msglen);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, hbuf_len);
    auto k_prime = Scalar::from_bytes(rand_hash);

    // Step 3: R = k' * G -- blinded CT path (DPA defense via context_randomize)
    auto R = ct::generator_mul_blinded(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 4: k = k' if even_y(R), else n - k'  (branchless; r_y_odd is secret-derived)
    auto k = ct::scalar_cneg(k_prime, ct::bool_to_mask(r_y_odd));

    // Step 5: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg)
    uint8_t challenge_stack[64 + kSBOMax];
    std::unique_ptr<uint8_t[]> challenge_heap;
    uint8_t* challenge_input = (msglen <= kSBOMax)
        ? challenge_stack
        : (challenge_heap.reset(new uint8_t[hbuf_len]), challenge_heap.get());
    std::memcpy(challenge_input,      rx.data(),    32);
    std::memcpy(challenge_input + 32, kp.px.data(), 32);
    if (msglen > 0) std::memcpy(challenge_input + 64, msg, msglen);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, hbuf_len);
    auto e = Scalar::from_bytes(e_hash);

    // Step 6: sig = (R.x, k + e * d)  — CT scalar ops on secret k and kp.d
    SchnorrSignature sig{};
    sig.r = rx;
    sig.s = ct::scalar_add(k, ct::scalar_mul(e, kp.d));

    // Erase ALL stack/heap buffers that held secret-derived material. The
    // nonce_input/challenge_input erasures cover the full 64+msglen span whether
    // it landed in the SBO stack buffer or the heap allocation.
    secure_erase(d_bytes.data(), d_bytes.size());
    secure_erase(t_hash.data(), t_hash.size());
    secure_erase(t, sizeof(t));
    secure_erase(nonce_input, hbuf_len);
    secure_erase(rand_hash.data(), rand_hash.size());
    secure_erase(challenge_input, hbuf_len);
    secure_erase(e_hash.data(), e_hash.size());
    secure_erase(&e, sizeof(e));
    secure_erase(&k_prime, sizeof(k_prime));
    secure_erase(&k, sizeof(k));

    return sig;
}

// ============================================================================
// CT Schnorr Sign + Verify (fault attack countermeasure)
// ============================================================================
// Signs and then verifies (FIPS 186-4 fault countermeasure).
// Public key and signature are not secret -- fast verify is safe.

SchnorrSignature schnorr_sign_verified(const SchnorrKeypair& kp,
                                       const std::array<uint8_t, 32>& msg,
                                       const std::array<uint8_t, 32>& aux_rand) {
    auto sig = ct::schnorr_sign(kp, msg, aux_rand);

    // Rule 14: check both s==0 AND R.x all-zeros before returning success.
    const auto* rw = reinterpret_cast<const std::uint64_t*>(sig.r.data());
    const bool r_zero = ((rw[0] | rw[1] | rw[2] | rw[3]) == 0);
    if (sig.s.is_zero_ct() || r_zero) return SchnorrSignature{};

    // Fast (non-CT) verify: timing variation is over the public sig/key only —
    // d and k are both erased inside schnorr_sign before this call.
    if (!schnorr_verify(kp.px, msg, sig)) {
        return SchnorrSignature{};
    }

    return sig;
}

// ============================================================================
// CT ECDSA Sign with Recovery ID
// ============================================================================
// Uses ct::generator_mul() for R=k*G and ct::scalar_inverse() for k^{-1}.
// Replaces the variable-time ::ecdsa_sign_recoverable() for all secret-key
// signing paths (bitcoin_sign_message, Ethereum personal_sign, shim, ECIES).
//
// Recovery ID computation:
//   bit 0 -- R.y parity via FieldElement::limbs()[0]&1 (no secret branch)
//   bit 1 -- R.x >= n overflow via constant-time byte comparison

RecoverableSignature ecdsa_sign_recoverable(
    const std::array<uint8_t, 32>& msg_hash,
    const Scalar& private_key) {

    static const std::array<uint8_t, 32> ORDER_BYTES = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };

    if (private_key.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    auto z = Scalar::from_bytes(msg_hash);

    // Deterministic nonce (RFC 6979)
    auto k = rfc6979_nonce(private_key, msg_hash);
    // k.is_zero_ct() guards against RFC 6979 exhaustion (≈2^−8000 probability).
    if (k.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // R = k * G  -- blinded CT path (DPA defense via secp256k1_context_randomize)
    auto R = ct::generator_mul_blinded(k);
    // k != 0 on secp256k1's prime-order group guarantees R ≠ ∞; no check needed.

    // r = R.x mod n (R.x is public after ct::generator_mul_blinded)
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    // r is derived from k*G.x (secret nonce): use is_zero_ct() per IS-ZERO-CT constraint.
    // r == 0 requires R.x = n exactly — negligible (≈2^−128) probability.
    if (r.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Recovery ID bit 0: parity of R.y
    // Branchless: mask LSB directly -- no conditional branch on the secret nonce.
    int recid = static_cast<int>(R.y().limbs()[0] & 1u);

    // Recovery ID bit 1: whether R.x >= n (R.x overflowed the curve order).
    // CT: compare r_bytes vs ORDER_BYTES without early-exit branches that would
    // leak the nonce via timing.  Uses branchless byte-by-byte MSB detection;
    // the final OR is also branchless (no conditional on the secret-derived gt).
    {
        unsigned gt = 0u, eq_run = 1u;
        for (int i = 0; i < 32; ++i) {
            unsigned const rb = static_cast<unsigned>(r_bytes[i]);
            unsigned const ob = static_cast<unsigned>(ORDER_BYTES[i]);
            unsigned const byte_gt = ((ob - rb) >> 31) & 1u;  // 1 iff rb > ob
            unsigned const byte_lt = ((rb - ob) >> 31) & 1u;  // 1 iff rb < ob
            gt     = gt | (eq_run & byte_gt);
            eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
        }
        recid |= static_cast<int>(gt) << 1;  // branchless set of overflow bit
    }

    // s = k^{-1} * (z + r * d) mod n
    // CT inverse + CT scalar multiplication (V-01 fix: operator* is variable-time).
    auto k_inv = ct::scalar_inverse(k);
    auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
    if (s.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // CT low-S normalization (branchless throughout).
    ECDSASignature pre_sig{r, s};
    // TIMING NOTE: high_mask is derived from s (which depends on nonce k).
    // k is already consumed and erased below before this function returns.
    // The only observable timing variation here is whether s required low-S
    // normalization — this is a function of the nonce, not the signing key.
    // No signing key material is leaked by this branch outcome.
    std::uint64_t const high_mask = ct::scalar_is_high(pre_sig.s);
    const ECDSASignature sig = ct::ct_normalize_low_s(pre_sig);
    // Negating s flips the R.y parity bit in the recovery ID.
    // CT: branchless XOR -- high_mask is 0 or ~0, mask to bit 0.
    recid ^= static_cast<int>(high_mask & 1);

    // Erase all stack buffers that held secret-derived material.
    secure_erase(&k,     sizeof(k));
    secure_erase(&k_inv, sizeof(k_inv));
    secure_erase(&z,     sizeof(z));
    secure_erase(&s,     sizeof(s));
    secure_erase(&pre_sig, sizeof(pre_sig));

    return {sig, recid};
}

// ============================================================================
// CT ECDSA Sign Hedged with Recovery ID
// ============================================================================
// Combines ecdsa_sign_hedged() with recovery-ID derivation from R's y-parity
// during signing. Avoids the 4x ecdsa_recover loop used by the shim when
// ndata != NULL (Bitcoin Core's CKey::Sign R-grinding path).
//
// This is the correct path for secp256k1_ecdsa_sign_recoverable when ndata
// is provided. The recid bits are derived from K the same way as
// ecdsa_sign_recoverable() above.

RecoverableSignature ecdsa_sign_hedged_recoverable(
    const std::array<uint8_t, 32>& msg_hash,
    const Scalar& private_key,
    const std::array<uint8_t, 32>& aux_rand) {

    static const std::array<uint8_t, 32> ORDER_BYTES = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };

    if (private_key.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    auto z = Scalar::from_bytes(msg_hash);

    // RFC 6979 + aux_rand hedging (extra entropy defense-in-depth)
    auto k = secp256k1::rfc6979_nonce_hedged(private_key, msg_hash, aux_rand);
    if (k.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // R = k * G  -- blinded CT path (DPA defense via secp256k1_context_randomize)
    auto R = ct::generator_mul_blinded(k);

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    // r is derived from k*G.x (secret nonce): use is_zero_ct() per IS-ZERO-CT constraint.
    if (r.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Recovery ID bit 0: parity of R.y (branchless, no branch on secret nonce).
    int recid = static_cast<int>(R.y().limbs()[0] & 1u);

    // Recovery ID bit 1: R.x >= n (overflow) — branchless byte comparison.
    {
        unsigned gt = 0u, eq_run = 1u;
        for (int i = 0; i < 32; ++i) {
            unsigned const rb = static_cast<unsigned>(r_bytes[i]);
            unsigned const ob = static_cast<unsigned>(ORDER_BYTES[i]);
            unsigned const byte_gt = ((ob - rb) >> 31) & 1u;
            unsigned const byte_lt = ((rb - ob) >> 31) & 1u;
            gt     = gt | (eq_run & byte_gt);
            eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
        }
        recid |= static_cast<int>(gt) << 1;
    }

    // s = k^{-1} * (z + r * d) mod n  (CT arithmetic throughout)
    auto k_inv = ct::scalar_inverse(k);
    auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
    if (s.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // CT low-S normalization; if s was negated, flip recid bit 0.
    ECDSASignature pre_sig{r, s};
    std::uint64_t const high_mask = ct::scalar_is_high(pre_sig.s);
    const ECDSASignature sig = ct::ct_normalize_low_s(pre_sig);
    recid ^= static_cast<int>(high_mask & 1);

    // Erase all secret-derived stack material.
    secure_erase(&k,       sizeof(k));
    secure_erase(&k_inv,   sizeof(k_inv));
    secure_erase(&z,       sizeof(z));
    secure_erase(&s,       sizeof(s));
    secure_erase(&pre_sig, sizeof(pre_sig));

    return {sig, recid};
}

} // namespace secp256k1::ct
