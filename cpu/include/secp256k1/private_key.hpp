#ifndef SECP256K1_PRIVATE_KEY_HPP
#define SECP256K1_PRIVATE_KEY_HPP
#pragma once

// ============================================================================
// PrivateKey -- Strong Type for Secret Key Material
// ============================================================================
// Wraps fast::Scalar with:
//   - No implicit conversion to Scalar (prevents accidental fast:: usage)
//   - Explicit .scalar() accessor (code-review signal for CT correctness)
//   - Destructor securely erases key material (volatile memset)
//   - Factory methods with strict validation (rejects 0 and >= n)
//
// Use PrivateKey instead of raw Scalar when handling long-lived secret keys.
// Pass to ct::ecdsa_sign(), ct::schnorr_sign() etc. for safe signing.
//
// Example:
//   secp256k1::PrivateKey pk;
//   if (!secp256k1::PrivateKey::from_bytes(raw_key, pk)) { /* invalid */ }
//   auto sig = secp256k1::ct::ecdsa_sign(msg_hash, pk);
//   auto kp  = secp256k1::ct::schnorr_keypair_create(pk);
// ============================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

class PrivateKey {
public:
    // -- Construction (explicit only) -----------------------------------------

    // Default: zero (invalid key, must be initialized via from_bytes).
    PrivateKey() noexcept = default;

    // Parse from 32-byte big-endian. Rejects values >= n or == 0.
    // Returns false on invalid input (out is left zeroed).
    [[nodiscard]] static bool from_bytes(const std::uint8_t* bytes32,
                                         PrivateKey& out) noexcept {
        return fast::Scalar::parse_bytes_strict_nonzero(bytes32, out.scalar_);
    }
    [[nodiscard]] static bool from_bytes(const std::array<std::uint8_t, 32>& bytes,
                                         PrivateKey& out) noexcept {
        return fast::Scalar::parse_bytes_strict_nonzero(bytes, out.scalar_);
    }

    // Wrap an already-validated scalar. Caller must ensure 0 < scalar < n.
    // Named "wrap" to emphasize this bypasses validation.
    [[nodiscard]] static PrivateKey wrap(const fast::Scalar& s) noexcept {
        PrivateKey pk;
        pk.scalar_ = s;
        return pk;
    }

    // -- Access (explicit only) -----------------------------------------------

    // Returns the underlying scalar for use in signing operations.
    // WARNING: any call site using this MUST use ct:: operations for
    // secret-dependent computation. Variable-time fast:: paths leak timing.
    [[nodiscard]] const fast::Scalar& scalar() const noexcept { return scalar_; }

    // Serialize to 32-byte big-endian.
    [[nodiscard]] std::array<std::uint8_t, 32> to_bytes() const {
        return scalar_.to_bytes();
    }

    // Check if the key is valid (nonzero).
    [[nodiscard]] bool is_valid() const noexcept { return !scalar_.is_zero(); }

    // -- No implicit conversion -----------------------------------------------
    // Intentionally no operator Scalar() or operator const Scalar&().
    // This forces callers to write .scalar() explicitly -- a code-review signal.

    // -- Lifecycle ------------------------------------------------------------

    ~PrivateKey() { secure_erase(); }

    // Copy: allowed but zeroes are copied (no hidden state).
    PrivateKey(const PrivateKey& other) noexcept : scalar_(other.scalar_) {}
    PrivateKey& operator=(const PrivateKey& other) noexcept {
        if (this != &other) {
            secure_erase();
            scalar_ = other.scalar_;
        }
        return *this;
    }

    // Move: source is zeroed after transfer.
    PrivateKey(PrivateKey&& other) noexcept : scalar_(other.scalar_) {
        other.secure_erase();
    }
    PrivateKey& operator=(PrivateKey&& other) noexcept {
        if (this != &other) {
            secure_erase();
            scalar_ = other.scalar_;
            other.secure_erase();
        }
        return *this;
    }

private:
    void secure_erase() noexcept {
        // volatile prevents compiler from optimizing away the memset
        volatile std::uint8_t* p =
            reinterpret_cast<volatile std::uint8_t*>(&scalar_);
        for (std::size_t i = 0; i < sizeof(scalar_); ++i) {
            p[i] = 0;
        }
    }

    fast::Scalar scalar_{};
};

// Comparison (for testing; compares underlying scalars)
inline bool operator==(const PrivateKey& a, const PrivateKey& b) noexcept {
    return a.scalar() == b.scalar();
}

} // namespace secp256k1

#endif // SECP256K1_PRIVATE_KEY_HPP
