#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/config.hpp"  // SECP256K1_FAST_52BIT
#include "secp256k1/field_52.hpp"
#include "secp256k1/ct/point.hpp"  // ct::generator_mul for sign-then-verify
#include "secp256k1/ct/scalar.hpp" // ct::ct_normalize_low_s
#include "secp256k1/detail/secure_erase.hpp"
#include "secp256k1/debug_invariants.hpp"
#include <cassert>
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement; // NOLINT(misc-unused-using-decls) -- used in #else path

// -- Half-order for low-S check -----------------------------------------------
// n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
// Stored as 4x64 LE limbs for direct comparison (no to_bytes() needed).
static constexpr std::uint64_t HALF_ORDER_LIMBS[4] = {
    0xDFE92F46681B20A0ULL,  // limb 0 (least significant)
    0x5D576E7357A4501DULL,  // limb 1
    0xFFFFFFFFFFFFFFFFULL,  // limb 2
    0x7FFFFFFFFFFFFFFFULL   // limb 3 (most significant)
};

// -- Signature Methods --------------------------------------------------------

std::pair<std::array<std::uint8_t, 72>, std::size_t> ECDSASignature::to_der() const {
    auto r_bytes = r.to_bytes();
    auto s_bytes = s.to_bytes();

    // Find actual length (skip leading zeros, add 0x00 pad if high bit set)
    auto encode_int = [](const std::array<uint8_t, 32>& val,
                         uint8_t* out) -> size_t {
        size_t start = 0;
        while (start < 31 && val[start] == 0) ++start;
        bool const need_pad = (val[start] & 0x80) != 0;
        size_t const len = 32 - start + (need_pad ? 1 : 0);

        out[0] = 0x02; // INTEGER tag
        out[1] = static_cast<uint8_t>(len);
        size_t pos = 2;
        if (need_pad) out[pos++] = 0x00;
        std::memcpy(out + pos, val.data() + start, 32 - start);
        return 2 + len;
    };

    std::array<uint8_t, 72> buf{};
    uint8_t r_enc[35]{}, s_enc[35]{};
    size_t const r_len = encode_int(r_bytes, r_enc);
    size_t const s_len = encode_int(s_bytes, s_enc);

    buf[0] = 0x30; // SEQUENCE tag
    buf[1] = static_cast<uint8_t>(r_len + s_len);
    std::memcpy(buf.data() + 2, r_enc, r_len);
    std::memcpy(buf.data() + 2 + r_len, s_enc, s_len);

    return {buf, 2 + r_len + s_len};
}

std::array<std::uint8_t, 64> ECDSASignature::to_compact() const {
    std::array<uint8_t, 64> out{};
    auto rb = r.to_bytes();
    auto sb = s.to_bytes();
    std::memcpy(out.data(), rb.data(), 32);
    std::memcpy(out.data() + 32, sb.data(), 32);
    return out;
}

ECDSASignature ECDSASignature::from_compact(const uint8_t* data64) {
    return {Scalar::from_bytes(data64), Scalar::from_bytes(data64 + 32)};
}

ECDSASignature ECDSASignature::from_compact(const std::array<uint8_t, 64>& data) {
    return from_compact(data.data());
}

bool ECDSASignature::parse_compact_strict(const uint8_t* data64,
                                           ECDSASignature& out) noexcept {
    // Strict: reject r >= n or r == 0; reject s >= n or s == 0
    Scalar r_val, s_val;
    if (!Scalar::parse_bytes_strict_nonzero(data64, r_val)) return false;
    if (!Scalar::parse_bytes_strict_nonzero(data64 + 32, s_val)) return false;
    out.r = r_val;
    out.s = s_val;
    return true;
}

bool ECDSASignature::parse_compact_strict(const std::array<uint8_t, 64>& data,
                                           ECDSASignature& out) noexcept {
    return parse_compact_strict(data.data(), out);
}

ECDSASignature ECDSASignature::normalize() const {
    return ct::ct_normalize_low_s(*this);
}

bool ECDSASignature::is_low_s() const {
    // Variable-time: s is PUBLIC at the call sites of this function (verification,
    // post-signing ABI checks on already-serialized sig). For secret s, use
    // normalize() which calls ct::ct_normalize_low_s().
    const auto& sl = s.limbs();
    for (int i = 3; i >= 0; --i) {
        if (sl[static_cast<unsigned>(i)] < HALF_ORDER_LIMBS[static_cast<unsigned>(i)]) return true;
        if (sl[static_cast<unsigned>(i)] > HALF_ORDER_LIMBS[static_cast<unsigned>(i)]) return false;
    }
    return true;
}

bool ECDSASignature::is_low_s_ct() const {
    // CT comparison: s <= n/2 using bitmask logic, fully unrolled (no loop overhead).
    const auto& sl = s.limbs();
    uint64_t above = 0, below = 0, undecided;
    // Limb 3 (most significant)
    undecided = ~0ULL;
    below  = undecided & ct::bool_to_mask(sl[3] < HALF_ORDER_LIMBS[3]);
    above  = undecided & ct::bool_to_mask(sl[3] > HALF_ORDER_LIMBS[3]);
    // Limb 2
    undecided = ~(above | below);
    below |= undecided & ct::bool_to_mask(sl[2] < HALF_ORDER_LIMBS[2]);
    above |= undecided & ct::bool_to_mask(sl[2] > HALF_ORDER_LIMBS[2]);
    // Limb 1
    undecided = ~(above | below);
    below |= undecided & ct::bool_to_mask(sl[1] < HALF_ORDER_LIMBS[1]);
    above |= undecided & ct::bool_to_mask(sl[1] > HALF_ORDER_LIMBS[1]);
    // Limb 0 (least significant)
    undecided = ~(above | below);
    below |= undecided & ct::bool_to_mask(sl[0] < HALF_ORDER_LIMBS[0]);
    above |= undecided & ct::bool_to_mask(sl[0] > HALF_ORDER_LIMBS[0]);
    // low-s if below OR (neither above nor below → equal)
    return (below != 0) || (above == 0);
}

// -- RFC 6979 Deterministic Nonce ---------------------------------------------
// Optimized RFC 6979 using HMAC-SHA256 with precomputed ipad/opad midstates.
// Calls sha256_compress_dispatch() directly -- no SHA256 object overhead,
// no per-byte finalize() padding. Saves ~4 compress calls via midstate reuse.

namespace {

using secp256k1::detail::secure_erase;

static inline Point signing_generator_mul(const Scalar& scalar) {
    return ct::generator_mul_blinded(scalar);
}

// -- SHA-256 IV ---------------------------------------------------------------
static constexpr std::uint32_t SHA256_IV[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

// -- Serialize 8xuint32 state -> 32 bytes (big-endian) -------------------------
static inline void state_to_bytes(const std::uint32_t st[8], std::uint8_t out[32]) {
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = static_cast<uint8_t>(st[i] >> 24);
        out[i*4+1] = static_cast<uint8_t>(st[i] >> 16);
        out[i*4+2] = static_cast<uint8_t>(st[i] >> 8);
        out[i*4+3] = static_cast<uint8_t>(st[i]);
    }
}

// -- Write big-endian 64-bit length at block[56..63] --------------------------
static inline void write_be_len(std::uint8_t block[64], std::uint64_t bits) {
    block[56] = static_cast<uint8_t>(bits >> 56);
    block[57] = static_cast<uint8_t>(bits >> 48);
    block[58] = static_cast<uint8_t>(bits >> 40);
    block[59] = static_cast<uint8_t>(bits >> 32);
    block[60] = static_cast<uint8_t>(bits >> 24);
    block[61] = static_cast<uint8_t>(bits >> 16);
    block[62] = static_cast<uint8_t>(bits >> 8);
    block[63] = static_cast<uint8_t>(bits);
}

// -- HMAC-SHA256 with precomputed ipad/opad midstates -------------------------
// Key is always 32 bytes in RFC-6979.
// Midstates are computed once per key and can be reused across multiple
// HMAC calls with the same key (e.g. steps e+f share K, steps g+h share K).

struct HMAC_Ctx {
    std::uint32_t inner_mid[8];  // SHA256(IV, ipad_block) midstate
    std::uint32_t outer_mid[8];  // SHA256(IV, opad_block) midstate

    void init_key32(const std::uint8_t key[32]) noexcept {
        alignas(16) std::uint8_t pad[64];
        // ipad = key ^ 0x36 (padded to 64 bytes)
        for (int i = 0; i < 32; i++) pad[i] = key[i] ^ 0x36;
        std::memset(pad + 32, 0x36, 32);
        std::memcpy(inner_mid, SHA256_IV, 32);
        detail::sha256_compress_dispatch(pad, inner_mid);
        // opad = key ^ 0x5c (padded to 64 bytes)
        for (int i = 0; i < 32; i++) pad[i] = key[i] ^ 0x5c;
        std::memset(pad + 32, 0x5c, 32);
        std::memcpy(outer_mid, SHA256_IV, 32);
        detail::sha256_compress_dispatch(pad, outer_mid);
        // Erase stack buffer: it held key XOR ipad/opad derivations (secret material).
        detail::secure_erase(pad, sizeof(pad));
    }

    // HMAC for short messages (msg_len <= 55): 1 inner compress + 1 outer
    // Total inner input: 64 (ipad midstate) + msg_len
    void compute_short(const std::uint8_t* msg, std::size_t msg_len,
                       std::uint8_t out[32]) const noexcept {
        // Safety: msg_len > 55 causes `55 - msg_len` to wrap around as size_t
        // (which is unsigned), producing ~0ULL and smashing the stack.
        // This is a hard precondition violation — assert in debug, guard in release.
        assert(msg_len <= 55 && "compute_short: msg_len must not exceed 55");
        if (msg_len > 55) return;  // release-mode safeguard against size_t wrap

        std::uint32_t st[8];
        alignas(16) std::uint8_t block[64];

        // Inner: compress(midstate, [msg | 0x80 | zeros | len])
        std::memcpy(st, inner_mid, 32);
        std::memcpy(block, msg, msg_len);
        block[msg_len] = 0x80;
        std::memset(block + msg_len + 1, 0, 55 - msg_len);
        write_be_len(block, static_cast<uint64_t>(64 + msg_len) * 8);
        detail::sha256_compress_dispatch(block, st);

        // Serialize inner result
        std::uint8_t ihash[32];
        state_to_bytes(st, ihash);

        // Outer: compress(outer_mid, [ihash | 0x80 | zeros | 0x0300])
        std::memcpy(st, outer_mid, 32);
        std::memcpy(block, ihash, 32);
        block[32] = 0x80;
        std::memset(block + 33, 0, 23);
        // length = (64 + 32) * 8 = 768 = 0x0300
        block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
        block[60] = 0; block[61] = 0; block[62] = 0x03; block[63] = 0x00;
        detail::sha256_compress_dispatch(block, st);

        state_to_bytes(st, out);
    }

    // HMAC for medium messages (55 < msg_len <= 119): 2 inner compress + 1 outer
    // Precondition: msg_len in (64, 119] so that rem = msg_len-64 is in [1,55].
    // If violated, 55-rem wraps as size_t (unsigned), producing a huge memset count.
    void compute_two_block(const std::uint8_t* msg, std::size_t msg_len,
                           std::uint8_t out[32]) const noexcept {
        if (msg_len <= 64 || msg_len > 119) return;  // precondition guard

        std::uint32_t st[8];
        alignas(16) std::uint8_t block[64];

        // Inner block 1: msg[0..63]
        std::memcpy(st, inner_mid, 32);
        detail::sha256_compress_dispatch(msg, st);

        // Inner block 2: msg[64..] + padding
        std::size_t const rem = msg_len - 64;
        std::memcpy(block, msg + 64, rem);
        block[rem] = 0x80;
        std::memset(block + rem + 1, 0, 55 - rem);
        write_be_len(block, static_cast<uint64_t>(64 + msg_len) * 8);
        detail::sha256_compress_dispatch(block, st);

        // Serialize inner
        std::uint8_t ihash[32];
        state_to_bytes(st, ihash);

        // Outer
        std::memcpy(st, outer_mid, 32);
        std::memcpy(block, ihash, 32);
        block[32] = 0x80;
        std::memset(block + 33, 0, 23);
        block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
        block[60] = 0; block[61] = 0; block[62] = 0x03; block[63] = 0x00;
        detail::sha256_compress_dispatch(block, st);

        state_to_bytes(st, out);
    }

    // HMAC for longer messages (119 < msg_len <= 183): 3 inner compress + 1 outer
    // Used by hedged RFC 6979: V(32) + byte(1) + x(32) + h1(32) + extra(32) = 129
    void compute_three_block(const std::uint8_t* msg, std::size_t msg_len,
                             std::uint8_t out[32]) const noexcept {
        std::uint32_t st[8];
        alignas(16) std::uint8_t block[64];

        // Inner block 1: msg[0..63]
        std::memcpy(st, inner_mid, 32);
        detail::sha256_compress_dispatch(msg, st);

        // Inner block 2: msg[64..127]
        detail::sha256_compress_dispatch(msg + 64, st);

        // Inner block 3: msg[128..] + padding
        std::size_t const rem = msg_len - 128;
        std::memcpy(block, msg + 128, rem);
        block[rem] = 0x80;
        std::memset(block + rem + 1, 0, 55 - rem);
        write_be_len(block, static_cast<uint64_t>(64 + msg_len) * 8);
        detail::sha256_compress_dispatch(block, st);

        // Serialize inner
        std::uint8_t ihash[32];
        state_to_bytes(st, ihash);

        // Outer
        std::memcpy(st, outer_mid, 32);
        std::memcpy(block, ihash, 32);
        block[32] = 0x80;
        std::memset(block + 33, 0, 23);
        block[56] = 0; block[57] = 0; block[58] = 0; block[59] = 0;
        block[60] = 0; block[61] = 0; block[62] = 0x03; block[63] = 0x00;
        detail::sha256_compress_dispatch(block, st);

        state_to_bytes(st, out);
    }
};

} // namespace

Scalar rfc6979_nonce(const Scalar& private_key,
                     const std::array<uint8_t, 32>& msg_hash) {
    // RFC 6979 Section 3.2 -- optimized with HMAC midstate caching.
    auto x_bytes = private_key.to_bytes();

    // Step b: V = 0x01 * 32
    alignas(16) uint8_t V[32];
    std::memset(V, 0x01, 32);

    // Step c: K = 0x00 * 32
    alignas(16) uint8_t K[32];
    std::memset(K, 0x00, 32);

    // Reusable message buffer for 97-byte messages (V||byte||x||h1)
    alignas(16) uint8_t buf97[97];

    // Steps d+e: K1 = HMAC(K0, V||0x00||x||h1), V = HMAC(K1, V)
    HMAC_Ctx hmac;
    hmac.init_key32(K);

    std::memcpy(buf97, V, 32);
    buf97[32] = 0x00;
    std::memcpy(buf97 + 33, x_bytes.data(), 32);
    std::memcpy(buf97 + 65, msg_hash.data(), 32);
    hmac.compute_two_block(buf97, 97, K);  // K = HMAC(K0, V||0||x||h1)

    // Steps e+f share the same K -- precompute midstate once
    hmac.init_key32(K);
    hmac.compute_short(V, 32, V);          // V = HMAC(K1, V)

    // Step f: K = HMAC(K1, V||0x01||x||h1) -- reuses K1 midstate!
    std::memcpy(buf97, V, 32);
    buf97[32] = 0x01;
    std::memcpy(buf97 + 33, x_bytes.data(), 32);
    std::memcpy(buf97 + 65, msg_hash.data(), 32);
    hmac.compute_two_block(buf97, 97, K);  // K = HMAC(K1, V||1||x||h1)

    // Steps g+h share the same K -- precompute midstate once
    hmac.init_key32(K);
    hmac.compute_short(V, 32, V);          // V = HMAC(K2, V)

    // Step h: generate candidate -- reuses K2 midstate!
    // Hoist t/buf33 outside loop: stack alloc once, reuse each iteration.
    std::array<uint8_t, 32> t;
    uint8_t buf33[33];
    for (int attempt = 0; attempt < 100; ++attempt) {
        hmac.compute_short(V, 32, V);      // V = HMAC(K2, V)

        std::memcpy(t.data(), V, 32);
        // RFC 6979 §3.2(h): k = bits2int(T); retry if k == 0 or k >= n.
        Scalar candidate;
        if (Scalar::parse_bytes_strict_nonzero(t.data(), candidate)) {
            secure_erase(t.data(), t.size());
            secure_erase(V, sizeof(V));
            secure_erase(K, sizeof(K));
            secure_erase(x_bytes.data(), x_bytes.size());
            secure_erase(buf97, sizeof(buf97));
            secure_erase(&hmac, sizeof(hmac));
            return candidate;
        }

        // Retry: K = HMAC(K, V||0x00), V = HMAC(K, V)
        std::memcpy(buf33, V, 32);
        buf33[32] = 0x00;
        hmac.compute_short(buf33, 33, K);
        secure_erase(buf33, sizeof(buf33));
        hmac.init_key32(K);
        hmac.compute_short(V, 32, V);
    }
    secure_erase(t.data(), t.size());
    secure_erase(buf33, sizeof(buf33));

    // Should never reach -- zeroize anyway
    secure_erase(V, sizeof(V));
    secure_erase(K, sizeof(K));
    secure_erase(x_bytes.data(), x_bytes.size());
    secure_erase(buf97, sizeof(buf97));
    secure_erase(&hmac, sizeof(hmac));
    return Scalar::zero();
}

// -- Hedged RFC 6979 ----------------------------------------------------------
// RFC 6979 Section 3.6 "Additional Data" variant: appends extra entropy to the
// HMAC-DRBG steps d and f. This hedges against HMAC-SHA256 weakness / fault
// injection while maintaining RFC 6979 determinism as fallback.
// When aux_rand is all-zeros, behavior differs from standard rfc6979_nonce
// (different HMAC input length), but nonce is still safe and deterministic.

Scalar rfc6979_nonce_hedged(const Scalar& private_key,
                            const std::array<uint8_t, 32>& msg_hash,
                            const std::array<uint8_t, 32>& aux_rand) {
    auto x_bytes = private_key.to_bytes();

    alignas(16) uint8_t V[32];
    std::memset(V, 0x01, 32);
    alignas(16) uint8_t K[32];
    std::memset(K, 0x00, 32);

    // Buffer for 129-byte messages: V(32) + byte(1) + x(32) + h1(32) + extra(32)
    alignas(16) uint8_t buf129[129];

    HMAC_Ctx hmac;
    hmac.init_key32(K);

    // Step d: K = HMAC(K0, V || 0x00 || x || h1 || aux_rand)
    std::memcpy(buf129, V, 32);
    buf129[32] = 0x00;
    std::memcpy(buf129 + 33, x_bytes.data(), 32);
    std::memcpy(buf129 + 65, msg_hash.data(), 32);
    std::memcpy(buf129 + 97, aux_rand.data(), 32);
    hmac.compute_three_block(buf129, 129, K);

    // Step e: V = HMAC(K1, V)
    hmac.init_key32(K);
    hmac.compute_short(V, 32, V);

    // Step f: K = HMAC(K1, V || 0x01 || x || h1 || aux_rand)
    std::memcpy(buf129, V, 32);
    buf129[32] = 0x01;
    std::memcpy(buf129 + 33, x_bytes.data(), 32);
    std::memcpy(buf129 + 65, msg_hash.data(), 32);
    std::memcpy(buf129 + 97, aux_rand.data(), 32);
    hmac.compute_three_block(buf129, 129, K);

    // Steps g+h: V = HMAC(K2, V), generate candidates
    hmac.init_key32(K);
    hmac.compute_short(V, 32, V);

    // Hoist t/buf33 outside loop: stack alloc once, reuse each iteration.
    std::array<uint8_t, 32> t;
    uint8_t buf33[33];
    for (int attempt = 0; attempt < 100; ++attempt) {
        hmac.compute_short(V, 32, V);
        std::memcpy(t.data(), V, 32);
        // RFC 6979 §3.2(h): reject k == 0 or k >= n (no implicit mod-n reduction).
        // Use parse_bytes_strict_nonzero, matching the standard nonce path above.
        Scalar candidate;
        if (Scalar::parse_bytes_strict_nonzero(t.data(), candidate)) {
            secure_erase(t.data(), t.size());
            secure_erase(V, sizeof(V));
            secure_erase(K, sizeof(K));
            secure_erase(x_bytes.data(), x_bytes.size());
            secure_erase(buf129, sizeof(buf129));
            secure_erase(&hmac, sizeof(hmac));
            return candidate;
        }
        std::memcpy(buf33, V, 32);
        buf33[32] = 0x00;
        hmac.compute_short(buf33, 33, K);
        secure_erase(buf33, sizeof(buf33));
        hmac.init_key32(K);
        hmac.compute_short(V, 32, V);
    }
    secure_erase(t.data(), t.size());
    secure_erase(buf33, sizeof(buf33));

    secure_erase(V, sizeof(V));
    secure_erase(K, sizeof(K));
    secure_erase(x_bytes.data(), x_bytes.size());
    secure_erase(buf129, sizeof(buf129));
    secure_erase(&hmac, sizeof(hmac));
    return Scalar::zero();
}

// -- ECDSA Sign (FAST PATH — NOT FOR PRODUCTION SIGNING WITH REAL PRIVATE KEYS) --------
// This function uses fast::Scalar::operator* for s = k^{-1}*(z+r*d), which has
// data-dependent branches in the modular reduction (ge() in scalar.cpp).
//
// INTENTIONAL DESIGN: secp256k1::ecdsa_sign() is the benchmark / CT-equivalence-test
// fast path only. Production signing MUST go through secp256k1::ct::ecdsa_sign().
//
// All public signing APIs (C ABI ufsecp_*, WASM, shim, Android JNI, Rust FFI) MUST
// route through ct::ecdsa_sign(). A regression test (test_exploit_ecdsa_fast_path_isolation)
// enforces this: it scans public interface source files and fails if ecdsa_sign() appears
// without the ct:: namespace prefix.
//
// Pure sign: no sign-then-verify countermeasure. Use ecdsa_sign_verified() for fault resistance.

ECDSASignature ecdsa_sign(const std::array<uint8_t, 32>& msg_hash,
                          const Scalar& private_key) {
    if (private_key.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};
    SECP_ASSERT_SCALAR_VALID(private_key);

    // z = message hash interpreted as scalar
    auto z = Scalar::from_bytes(msg_hash);

    // Generate deterministic nonce (V/K buffers zeroed inside rfc6979_nonce)
    auto k = rfc6979_nonce(private_key, msg_hash);
    ECDSASignature result{Scalar::zero(), Scalar::zero()};

    if (!k.is_zero_ct()) {
        // R = k * G
        auto R = signing_generator_mul(k);
        if (!R.is_infinity()) {
            // Normalize to affine before reading x-coordinate limbs: from_limbs()
            // reads the affine X directly and skips the byte-swap round-trip.
            // Normalization is required — Jacobian X = affine_x * Z^2 (wrong if not Z=1).
            R.normalize();
            auto r = Scalar::from_limbs(R.x().limbs());
            if (!r.is_zero()) {
                // s = k^{-1} * (z + r * d) mod n  (CT scalar arithmetic)
                auto k_inv = ct::scalar_inverse(k);
                auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
                if (!s.is_zero_ct()) {
                    // Normalize to low-S (BIP-62)
                    result = ECDSASignature{r, s}.normalize();
                }
                secure_erase(&k_inv, sizeof(k_inv));
            }
        }
    }

    // Zeroize sensitive scalar temporaries before returning
    secure_erase(&k, sizeof(k));
    secure_erase(&z, sizeof(z));
    return result;
}

// -- ECDSA Sign + Verify (fault attack countermeasure) ------------------------
// Signs and then verifies the signature (FIPS 186-4 fault countermeasure).
// A transient fault during signing could produce a corrupted (r,s) from which
// the private key is recoverable via lattice attack. This variant verifies
// the signature before releasing it.

ECDSASignature ecdsa_sign_verified(const std::array<uint8_t, 32>& msg_hash,
                                   const Scalar& private_key) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    auto result = ecdsa_sign(msg_hash, private_key);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

    if (!result.r.is_zero()) {
        auto pk = ct::generator_mul(private_key);
        if (!ecdsa_verify(msg_hash.data(), pk, result)) {
            result = {Scalar::zero(), Scalar::zero()};
        }
    }

    return result;
}

// -- ECDSA Sign (hedged, with extra entropy) ----------------------------------
// RFC 6979 Section 3.6: aux_rand is mixed into the HMAC-DRBG as additional
// data. The nonce is deterministic for a given (key, msg, aux_rand) triple.
// Use 32 bytes of fresh CSPRNG randomness for maximum defense-in-depth.

ECDSASignature ecdsa_sign_hedged(const std::array<uint8_t, 32>& msg_hash,
                                  const Scalar& private_key,
                                  const std::array<uint8_t, 32>& aux_rand) {
    if (private_key.is_zero_ct()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = rfc6979_nonce_hedged(private_key, msg_hash, aux_rand);
    ECDSASignature result{Scalar::zero(), Scalar::zero()};

    if (!k.is_zero_ct()) {
        auto R = signing_generator_mul(k);
        if (!R.is_infinity()) {
            auto r = Scalar::from_limbs(R.x().limbs());
            if (!r.is_zero()) {
                auto k_inv = ct::scalar_inverse(k);
                auto s = ct::scalar_mul(k_inv, ct::scalar_add(z, ct::scalar_mul(r, private_key)));
                if (!s.is_zero_ct()) {
                    result = ECDSASignature{r, s}.normalize();
                }
                secure_erase(&k_inv, sizeof(k_inv));
            }
        }
    }

    secure_erase(&k, sizeof(k));
    secure_erase(&z, sizeof(z));
    return result;
}

// -- ECDSA Sign Hedged + Verify (fault attack countermeasure) -----------------

ECDSASignature ecdsa_sign_hedged_verified(const std::array<uint8_t, 32>& msg_hash,
                                          const Scalar& private_key,
                                          const std::array<uint8_t, 32>& aux_rand) {
    auto result = ecdsa_sign_hedged(msg_hash, private_key, aux_rand);

    if (!result.r.is_zero()) {
        auto pk = ct::generator_mul(private_key);
        if (!ecdsa_verify(msg_hash.data(), pk, result)) {
            result = {Scalar::zero(), Scalar::zero()};
        }
    }

    return result;
}

// -- ECDSA Verify -------------------------------------------------------------

// Shared x-coordinate check: R_prime.x / R_prime.z^2 mod n == sig.r
// Used by both ecdsa_verify(Point) and ecdsa_verify(EcdsaPublicKey).
static bool ecdsa_check_xcoord(const Point& R_prime, const ECDSASignature& sig) {
    if (R_prime.is_infinity()) return false;
#if defined(SECP256K1_FAST_52BIT)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    using FE52 = fast::FieldElement52;
    FE52 const r52 = FE52::from_4x64_limbs(sig.r.limbs().data());
    FE52 const z2 = R_prime.Z52().square();
    FE52 const r_z2 = r52 * z2;
    {
        FE52 diff = R_prime.X52();
        diff.negate_assign(23);
        diff.add_assign(r_z2);
        if (diff.normalizes_to_zero_var()) return true;
    }
    static constexpr std::uint64_t PMN_0 = 0x402da1722fc9baeeULL;
    static constexpr std::uint64_t PMN_1 = 0x4551231950b75fc4ULL;
    const auto& rl = sig.r.limbs();
    bool r_less_than_pmn = false;
    if (rl[3] != 0 || rl[2] > 1) {
        r_less_than_pmn = false;
    } else if (rl[2] == 0) {
        r_less_than_pmn = true;
    } else {
        if (rl[1] != PMN_1) { r_less_than_pmn = (rl[1] < PMN_1); }
        else                 { r_less_than_pmn = (rl[0] < PMN_0); }
    }
    if (r_less_than_pmn) {
        static constexpr std::uint64_t N_LIMBS[4] = {
            0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
        };
        alignas(32) std::uint64_t rn[4];
        unsigned __int128 acc = static_cast<unsigned __int128>(rl[0]) + N_LIMBS[0];
        rn[0] = static_cast<std::uint64_t>(acc);
        acc = static_cast<unsigned __int128>(rl[1]) + N_LIMBS[1] + static_cast<std::uint64_t>(acc >> 64);
        rn[1] = static_cast<std::uint64_t>(acc);
        acc = static_cast<unsigned __int128>(rl[2]) + N_LIMBS[2] + static_cast<std::uint64_t>(acc >> 64);
        rn[2] = static_cast<std::uint64_t>(acc);
        rn[3] = rl[3] + N_LIMBS[3] + static_cast<uint64_t>(acc >> 64);
        FE52 const r2_z2 = FE52::from_4x64_limbs(rn) * z2;
        FE52 diff2 = R_prime.X52();
        diff2.negate_assign(23);
        diff2.add_assign(r2_z2);
        if (diff2.normalizes_to_zero_var()) return true;
    }
    return false;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else
    auto r_fe = FieldElement::from_limbs(sig.r.limbs());
    auto z2   = R_prime.z_raw().square();
    auto lhs  = r_fe * z2;
    auto rx   = R_prime.x_raw();
    if (lhs == rx) return true;
    static constexpr std::uint64_t PMN_0 = 0x402da1722fc9baeeULL;
    static constexpr std::uint64_t PMN_1 = 0x4551231950b75fc4ULL;
    const auto& rl = sig.r.limbs();
    bool r_less_than_pmn;
    if (rl[3] != 0 || rl[2] > 1) { r_less_than_pmn = false; }
    else if (rl[2] == 0)          { r_less_than_pmn = true;  }
    else {
        if (rl[1] != PMN_1) { r_less_than_pmn = (rl[1] < PMN_1); }
        else                 { r_less_than_pmn = (rl[0] < PMN_0); }
    }
    if (r_less_than_pmn) {
        static constexpr std::uint64_t N_LIMBS[4] = {
            0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
        };
        std::uint64_t rn[4], carry = 0;
        rn[0] = rl[0] + N_LIMBS[0]; carry = (rn[0] < rl[0]) ? 1u : 0u;
        std::uint64_t t1 = rl[1]+N_LIMBS[1], c1 = (t1<rl[1])?1u:0u;
        rn[1] = t1 + carry; carry = c1 + ((rn[1]<t1)?1u:0u);
        std::uint64_t t2 = rl[2]+N_LIMBS[2], c2 = (t2<rl[2])?1u:0u;
        rn[2] = t2 + carry; carry = c2 + ((rn[2]<t2)?1u:0u);
        rn[3] = rl[3] + N_LIMBS[3] + carry;
        FieldElement::limbs_type rn_arr = {rn[0], rn[1], rn[2], rn[3]};
        if (FieldElement::from_limbs(rn_arr) * z2 == rx) return true;
    }
    return false;
#endif
}

// Primary implementation: raw pointer, no copies.
// NOTE: Accepts both low-S and high-S signatures by design. ECDSA verification
// is mathematically symmetric in s (both s and n-s satisfy the equation).
// Callers requiring BIP-62 low-S enforcement must call sig.is_low_s() first,
// or use ecdsa_batch_verify() which enforces low-S as a pre-validation step.
bool ecdsa_verify(const uint8_t* msg_hash32,
                  const Point& public_key,
                  const ECDSASignature& sig) noexcept {
    SECP_ASSERT_ON_CURVE(public_key);
    // Reject degenerate inputs early
    if (public_key.is_infinity()) return false;
    if (sig.r.is_zero() || sig.s.is_zero()) return false;

    // z = message hash as scalar (direct from raw pointer)
    auto z = Scalar::from_bytes(msg_hash32);

    // w = s^{-1} mod n
    auto w = sig.s.inverse();

    // u1 = z * w mod n
    auto u1 = z * w;

    // u2 = r * w mod n
    auto u2 = sig.r * w;

    // R' = u1 * G + u2 * Q  (4-stream GLV Strauss -- single interleaved loop)
    auto R_prime = Point::dual_scalar_mul_gen_point(u1, u2, public_key);
    return ecdsa_check_xcoord(R_prime, sig);
}


// Array wrapper: delegates to raw-pointer implementation.
bool ecdsa_verify(const std::array<uint8_t, 32>& msg_hash,
                  const Point& public_key,
                  const ECDSASignature& sig) noexcept {
    return ecdsa_verify(msg_hash.data(), public_key, sig);
}

// -- EcdsaPublicKey: cached GLV tables ----------------------------------------

bool ecdsa_pubkey_parse(EcdsaPublicKey& out,
                        const std::uint8_t* bytes, std::size_t len) noexcept {
    using fast::FieldElement;
    Point P = Point::infinity();

    if (len == 33) {
        if (bytes[0] != 0x02 && bytes[0] != 0x03) return false;
        FieldElement x;
        if (!FieldElement::parse_bytes_strict(bytes + 1, x)) return false;
        auto y2 = x * x * x + FieldElement::from_uint64(7);
        auto y  = y2.sqrt();
        if (!(y * y == y2)) return false;
        bool y_is_odd = (y.limbs()[0] & 1u) != 0;  // LSB of limbs_[0] == LSB of the field element
        bool want_odd = (bytes[0] == 0x03);
        if (y_is_odd != want_odd) y = FieldElement::zero() - y;
        P = Point::from_affine(x, y);
    } else if (len == 65) {
        if (bytes[0] != 0x04) return false;
        FieldElement x, y;
        if (!FieldElement::parse_bytes_strict(bytes + 1,  x)) return false;
        if (!FieldElement::parse_bytes_strict(bytes + 33, y)) return false;
        auto rhs = x * x * x + FieldElement::from_uint64(7);
        if (!(y * y == rhs)) return false;
        P = Point::from_affine(x, y);
    } else {
        return false;
    }

    if (P.is_infinity()) return false;
    out.point = P;

#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    out.tables_valid = Point::build_schnorr_verify_tables(
        P, out.tbl_P, out.tbl_phi, out.Z_shared);
#endif
    return true;
}

bool ecdsa_verify(const std::uint8_t* msg_hash32,
                  const EcdsaPublicKey& pubkey,
                  const ECDSASignature& sig) noexcept {
    if (pubkey.point.is_infinity()) return false;
    if (sig.r.is_zero() || sig.s.is_zero()) return false;

    auto z  = Scalar::from_bytes(msg_hash32);
    auto w  = sig.s.inverse();
    auto u1 = z * w;
    auto u2 = sig.r * w;

    Point R_prime;
#if defined(SECP256K1_FAST_52BIT) && !defined(SECP256K1_USE_4X64_POINT_OPS)
    if (pubkey.tables_valid) {
        R_prime = Point::dual_scalar_mul_gen_prebuilt(
            u1, u2, pubkey.tbl_P, pubkey.tbl_phi, pubkey.Z_shared);
    } else {
        R_prime = Point::dual_scalar_mul_gen_point(u1, u2, pubkey.point);
    }
#else
    R_prime = Point::dual_scalar_mul_gen_point(u1, u2, pubkey.point);
#endif
    return ecdsa_check_xcoord(R_prime, sig);
}

bool ecdsa_verify(const std::array<std::uint8_t, 32>& msg_hash,
                  const EcdsaPublicKey& pubkey,
                  const ECDSASignature& sig) noexcept {
    return ecdsa_verify(msg_hash.data(), pubkey, sig);
}

} // namespace secp256k1
