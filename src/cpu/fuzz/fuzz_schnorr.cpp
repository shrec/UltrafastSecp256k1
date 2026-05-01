// ============================================================================
// Fuzz target: BIP-340 Schnorr sign / verify cycle
//
// Input layout (96 bytes):
//   bytes  0-31 : private key (raw big-endian; reduced mod n internally)
//   bytes 32-63 : message (32 bytes)
//   bytes 64-95 : aux_rand (32 bytes of "randomness" from the fuzzer)
//
// Invariants checked:
//   1. sign → verify produces true
//   2. sign → verify with wrong message produces false
//   3. schnorr_xonly_pubkey_parse from the same x-coordinate round-trips
//   4. adversarial schnorr_verify with random sig+key bytes does not crash
// ============================================================================
#include <cstdint>
#include <cstring>
#include <array>
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::SchnorrKeypair;
using secp256k1::SchnorrSignature;
using secp256k1::SchnorrXonlyPubkey;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 96) return 0;

    std::array<uint8_t, 32> key_buf{};
    std::array<uint8_t, 32> msg_buf{};
    std::array<uint8_t, 32> aux_buf{};
    std::memcpy(key_buf.data(), data,      32);
    std::memcpy(msg_buf.data(), data + 32, 32);
    std::memcpy(aux_buf.data(), data + 64, 32);

    auto priv = Scalar::from_bytes(key_buf);

    // Zero key: no crash required, no invariant to enforce
    if (priv.is_zero()) return 0;

    // -- Create keypair -------------------------------------------------------
    SchnorrKeypair kp = secp256k1::schnorr_keypair_create(priv);

    // -- Invariant 1: sign → verify = true -----------------------------------
    auto sig = secp256k1::schnorr_sign(kp, msg_buf, aux_buf);
    bool ok = secp256k1::schnorr_verify(kp.px, msg_buf, sig);
    if (!ok) __builtin_trap();

    // -- Invariant 2: wrong message → verify = false -------------------------
    std::array<uint8_t, 32> wrong_msg = msg_buf;
    wrong_msg[0] ^= 0xFF;
    bool ok_wrong = secp256k1::schnorr_verify(kp.px, wrong_msg, sig);
    if (ok_wrong) __builtin_trap();

    // -- Invariant 3: xonly_pubkey_parse round-trips -------------------------
    SchnorrXonlyPubkey xpk{};
    if (secp256k1::schnorr_xonly_pubkey_parse(xpk, kp.px)) {
        // The parsed pubkey's x_bytes must equal kp.px
        if (xpk.x_bytes != kp.px) __builtin_trap();
        // Verify using cached pubkey must agree
        bool ok2 = secp256k1::schnorr_verify(xpk, msg_buf, sig);
        if (!ok2) __builtin_trap();
    }

    // -- Invariant 4: adversarial verify with fuzzer sig bytes ---------------
    // Feed raw bytes as a "signature" to verify against our real pubkey.
    // We do NOT assert the result -- it might be true by coincidence.
    // This only checks for crashes or undefined behaviour.
    if (size >= 96 + 64) {
        auto adv_sig = SchnorrSignature::from_bytes(data + 96);
        (void)secp256k1::schnorr_verify(kp.px, msg_buf, adv_sig);
    }

    return 0;
}
