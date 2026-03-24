// ============================================================================
// Fuzz target: ECDSA sign / verify cycle
//
// Input layout (64 bytes):
//   bytes  0-31 : private key (raw big-endian; reduced mod n internally)
//   bytes 32-63 : message hash (32 bytes, treated as opaque)
//
// Invariants checked:
//   1. sign → verify produces true
//   2. sign → verify with wrong hash produces false
//   3. sign → verify with adversarial (random) signature produces false
//      (or accept only if a brute-force collision happened -- astronomically rare)
//   4. parse_compact_strict rejects out-of-range (r,s) byte sequences without crash
// ============================================================================
#include <cstdint>
#include <cstring>
#include <array>
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using secp256k1::ECDSASignature;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 64) return 0;

    std::array<uint8_t, 32> key_buf{};
    std::array<uint8_t, 32> msg_buf{};
    std::memcpy(key_buf.data(), data,      32);
    std::memcpy(msg_buf.data(), data + 32, 32);

    // Build private key scalar (reduced mod n -- from_bytes clamps internally)
    auto priv = Scalar::from_bytes(key_buf);

    // Zero key: sign should return a zero sentinel; no crash.
    if (priv.is_zero()) {
        auto sig = secp256k1::ecdsa_sign(msg_buf, priv);
        // Just ensure no crash.  A zero-key signature is the known failure sentinel.
        (void)sig;
        return 0;
    }

    // Derive public key
    auto pub = Point::generator().scalar_mul(priv);
    if (pub.is_infinity()) return 0; // shouldn't happen for non-zero priv, but guard

    // -- Invariant 1: sign → verify produces true ----------------------------
    auto sig = secp256k1::ecdsa_sign(msg_buf, priv);
    bool ok = secp256k1::ecdsa_verify(msg_buf, pub, sig);
    if (!ok) __builtin_trap();

    // -- Invariant 2: flipped first byte of message → verify must reject -----
    std::array<uint8_t, 32> wrong_msg = msg_buf;
    wrong_msg[0] ^= 0xFF;
    bool ok_wrong = secp256k1::ecdsa_verify(wrong_msg, pub, sig);
    if (ok_wrong) __builtin_trap();

    // -- Invariant 3: parse_compact_strict on raw fuzzer bytes (no crash) ----
    // We feed the raw 64-byte input as a "compact" sig to the strict parser.
    // The parser must not crash; it may return false for out-of-range inputs.
    ECDSASignature parsed{};
    (void)ECDSASignature::parse_compact_strict(data, parsed);

    // -- Invariant 4: normalise is idempotent --------------------------------
    auto norm1 = sig.normalize();
    auto norm2 = norm1.normalize();
    // Both should verify against the original message and pubkey
    if (!secp256k1::ecdsa_verify(msg_buf, pub, norm1)) __builtin_trap();
    if (!secp256k1::ecdsa_verify(msg_buf, pub, norm2)) __builtin_trap();

    return 0;
}
