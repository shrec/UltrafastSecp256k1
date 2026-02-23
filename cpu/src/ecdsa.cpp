#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/config.hpp"  // SECP256K1_FAST_52BIT
#include "secp256k1/field_52.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- Half-order for low-S check -----------------------------------------------
// n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
static const Scalar HALF_ORDER = Scalar::from_hex(
    "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0");

// -- Signature Methods --------------------------------------------------------

std::pair<std::array<std::uint8_t, 72>, std::size_t> ECDSASignature::to_der() const {
    auto r_bytes = r.to_bytes();
    auto s_bytes = s.to_bytes();

    // Find actual length (skip leading zeros, add 0x00 pad if high bit set)
    auto encode_int = [](const std::array<uint8_t, 32>& val,
                         uint8_t* out) -> size_t {
        size_t start = 0;
        while (start < 31 && val[start] == 0) ++start;
        bool need_pad = (val[start] & 0x80) != 0;
        size_t len = 32 - start + (need_pad ? 1 : 0);

        out[0] = 0x02; // INTEGER tag
        out[1] = static_cast<uint8_t>(len);
        size_t pos = 2;
        if (need_pad) out[pos++] = 0x00;
        std::memcpy(out + pos, val.data() + start, 32 - start);
        return 2 + len;
    };

    std::array<uint8_t, 72> buf{};
    uint8_t r_enc[35]{}, s_enc[35]{};
    size_t r_len = encode_int(r_bytes, r_enc);
    size_t s_len = encode_int(s_bytes, s_enc);

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

ECDSASignature ECDSASignature::from_compact(const std::array<std::uint8_t, 64>& data) {
    std::array<uint8_t, 32> rb{}, sb{};
    std::memcpy(rb.data(), data.data(), 32);
    std::memcpy(sb.data(), data.data() + 32, 32);
    return {Scalar::from_bytes(rb), Scalar::from_bytes(sb)};
}

ECDSASignature ECDSASignature::normalize() const {
    if (is_low_s()) return *this;
    return {r, s.negate()};
}

bool ECDSASignature::is_low_s() const {
    // s <= n/2 ?
    auto s_bytes = s.to_bytes();
    auto half_bytes = HALF_ORDER.to_bytes();
    for (size_t i = 0; i < 32; ++i) {
        if (s_bytes[i] < half_bytes[i]) return true;
        if (s_bytes[i] > half_bytes[i]) return false;
    }
    return true; // equal
}

// -- RFC 6979 Deterministic Nonce ---------------------------------------------
// Optimized RFC 6979 using HMAC-SHA256 with precomputed ipad/opad midstates.
// Calls sha256_compress_dispatch() directly -- no SHA256 object overhead,
// no per-byte finalize() padding. Saves ~4 compress calls via midstate reuse.

namespace {

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
    }

    // HMAC for short messages (msg_len <= 55): 1 inner compress + 1 outer
    // Total inner input: 64 (ipad midstate) + msg_len
    void compute_short(const std::uint8_t* msg, std::size_t msg_len,
                       std::uint8_t out[32]) const noexcept {
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
    void compute_two_block(const std::uint8_t* msg, std::size_t msg_len,
                           std::uint8_t out[32]) const noexcept {
        std::uint32_t st[8];
        alignas(16) std::uint8_t block[64];

        // Inner block 1: msg[0..63]
        std::memcpy(st, inner_mid, 32);
        detail::sha256_compress_dispatch(msg, st);

        // Inner block 2: msg[64..] + padding
        std::size_t rem = msg_len - 64;
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
    for (int attempt = 0; attempt < 100; ++attempt) {
        hmac.compute_short(V, 32, V);      // V = HMAC(K2, V)

        std::array<uint8_t, 32> t;
        std::memcpy(t.data(), V, 32);
        auto candidate = Scalar::from_bytes(t);
        if (!candidate.is_zero()) {
            return candidate;
        }

        // Retry: K = HMAC(K, V||0x00), V = HMAC(K, V)
        uint8_t buf33[33];
        std::memcpy(buf33, V, 32);
        buf33[32] = 0x00;
        hmac.compute_short(buf33, 33, K);
        hmac.init_key32(K);
        hmac.compute_short(V, 32, V);
    }

    return Scalar::zero(); // should never reach
}

// -- ECDSA Sign ---------------------------------------------------------------

ECDSASignature ecdsa_sign(const std::array<uint8_t, 32>& msg_hash,
                          const Scalar& private_key) {
    if (private_key.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // z = message hash interpreted as scalar
    auto z = Scalar::from_bytes(msg_hash);

    // Generate deterministic nonce
    auto k = rfc6979_nonce(private_key, msg_hash);
    if (k.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G
    auto R = Point::generator().scalar_mul(k);
    if (R.is_infinity()) return {Scalar::zero(), Scalar::zero()};

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // s = k^{-1} * (z + r * d) mod n
    auto k_inv = k.inverse();
    auto s = k_inv * (z + r * private_key);
    if (s.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // Normalize to low-S (BIP-62)
    ECDSASignature sig{r, s};
    return sig.normalize();
}

// -- ECDSA Verify -------------------------------------------------------------

bool ecdsa_verify(const std::array<uint8_t, 32>& msg_hash,
                  const Point& public_key,
                  const ECDSASignature& sig) {
    // Check r, s in [1, n-1]
    if (sig.r.is_zero() || sig.s.is_zero()) return false;

    // z = message hash as scalar
    auto z = Scalar::from_bytes(msg_hash);

    // w = s^{-1} mod n
    auto w = sig.s.inverse();

    // u1 = z * w mod n
    auto u1 = z * w;

    // u2 = r * w mod n
    auto u2 = sig.r * w;

    // R' = u1 * G + u2 * Q  (4-stream GLV Strauss -- single interleaved loop)
    auto R_prime = Point::dual_scalar_mul_gen_point(u1, u2, public_key);

    if (R_prime.is_infinity()) return false;

    // -- Fast Z^2-based x-coordinate check (avoids field inverse) ------
    // Check: R'.x/R'.z^2 mod n == sig.r
    // Equivalent: sig.r * R'.z^2 == R'.x (mod p)
    // This saves ~3us by avoiding the field inversion in Point::x().
#if defined(SECP256K1_FAST_52BIT)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
    using FE52 = fast::FieldElement52;

    // Direct Scalar->FE52: sig.r limbs are 4x64 LE, same layout as FieldElement.
    // Since sig.r < n < p, the raw limbs are a valid field element -- no reduction needed.
    FE52 r52 = FE52::from_4x64_limbs(sig.r.limbs().data());
    FE52 z2 = R_prime.Z52().square();    // Z^2  [1S] mag=1
    FE52 lhs = r52 * z2;                 // r*Z^2 [1M] mag=1
    lhs.normalize();

    FE52 rx = R_prime.X52();
    rx.normalize();

    if (lhs == rx) return true;

    // Rare case: x_R mod p in [n, p), so x_R mod n = x_R - n = sig.r
    // -> need to check (sig.r + n) * Z^2 == X.  Probability ~2^-128.
    // n = order, p - n ~= 2^129.  sig.r < n, so sig.r + n < p iff sig.r < p - n.
    //
    // p - n as 4x64 LE limbs:
    // 0x000000000000000145512319_50b75fc4_402da173_2fc9bebf
    static constexpr std::uint64_t PMN_0 = 0x402da1732fc9bebfULL;
    static constexpr std::uint64_t PMN_1 = 0x14551231950b75fcULL;  // top nibble = 1
    // PMN limbs [2] = 0, [3] = 0  -> sig.r < p-n iff sig.r fits in ~129 bits
    // Since sig.r < n ~= 2^256, upper limbs are always >= PMN upper limbs (0).
    // Quick check: r[3]==0 && r[2]==0 -> r < 2^128, definitely < p-n.
    // Otherwise compare lexicographically.
    const auto& rl = sig.r.limbs();
    bool r_less_than_pmn;
    if (rl[3] != 0 || rl[2] != 0) {
        r_less_than_pmn = false;  // r >= 2^128 > p-n
    } else if (rl[1] != PMN_1) {
        r_less_than_pmn = (rl[1] < PMN_1);
    } else {
        r_less_than_pmn = (rl[0] < PMN_0);
    }

    if (r_less_than_pmn) {
        // sig.r < p - n, so (sig.r + n) is a valid field element < p.
        // 256-bit addition: r + n (no modular reduction needed).
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
        rn[3] = rl[3] + N_LIMBS[3] + static_cast<std::uint64_t>(acc >> 64);

        FE52 r2_52 = FE52::from_4x64_limbs(rn);
        FE52 lhs2 = r2_52 * z2;
        lhs2.normalize();
        if (lhs2 == rx) return true;
    }

    return false;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#else
    // -- Z^2-based x-coordinate check for 4x64 path (ESP32/MSVC/generic) --
    // Avoids field inverse: sig.r * Z^2 == X (mod p)
    // Since sig.r < n < p, raw scalar limbs are a valid field element.
    auto r_fe = FieldElement::from_limbs(sig.r.limbs());
    auto z2   = R_prime.z_raw().square();     // Z^2
    auto lhs  = r_fe * z2;                    // r*Z^2
    auto rx   = R_prime.x_raw();              // Jacobian X

    if (lhs == rx) return true;

    // Rare case (probability ~2^-128): x_R mod p in [n, p),
    // so x_R mod n == sig.r means we need to check (sig.r + n)*Z^2 == X.
    // sig.r + n < p  iff  sig.r < p - n.
    // p - n ~= 2^129:  0x14551231950b75fc4402da1732fc9bebf
    static constexpr std::uint64_t PMN_0 = 0x402da1732fc9bebfULL;
    static constexpr std::uint64_t PMN_1 = 0x14551231950b75fcULL;
    const auto& rl = sig.r.limbs();

    bool r_less_than_pmn;
    if (rl[3] != 0 || rl[2] != 0) {
        r_less_than_pmn = false;
    } else if (rl[1] != PMN_1) {
        r_less_than_pmn = (rl[1] < PMN_1);
    } else {
        r_less_than_pmn = (rl[0] < PMN_0);
    }

    if (r_less_than_pmn) {
        // 256-bit addition r + n without __int128 (ESP32/MSVC safe)
        static constexpr std::uint64_t N_LIMBS[4] = {
            0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
        };
        std::uint64_t rn[4];
        std::uint64_t carry = 0;
        // limb 0
        rn[0] = rl[0] + N_LIMBS[0];
        carry = (rn[0] < rl[0]) ? 1u : 0u;
        // limb 1
        std::uint64_t tmp1 = rl[1] + N_LIMBS[1];
        std::uint64_t c1   = (tmp1 < rl[1]) ? 1u : 0u;
        rn[1] = tmp1 + carry;
        carry = c1 + ((rn[1] < tmp1) ? 1u : 0u);
        // limb 2
        std::uint64_t tmp2 = rl[2] + N_LIMBS[2];
        std::uint64_t c2   = (tmp2 < rl[2]) ? 1u : 0u;
        rn[2] = tmp2 + carry;
        carry = c2 + ((rn[2] < tmp2) ? 1u : 0u);
        // limb 3
        rn[3] = rl[3] + N_LIMBS[3] + carry;

        FieldElement::limbs_type rn_arr = {rn[0], rn[1], rn[2], rn[3]};
        auto r2_fe = FieldElement::from_limbs(rn_arr);
        auto lhs2 = r2_fe * z2;
        if (lhs2 == rx) return true;
    }

    return false;
#endif
}

} // namespace secp256k1
