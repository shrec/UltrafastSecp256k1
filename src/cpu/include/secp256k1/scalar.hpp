#ifndef B4F0584E_3733_4C2F_BB01_D4BDC40E1760
#define B4F0584E_3733_4C2F_BB01_D4BDC40E1760
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "secp256k1/types.hpp"

namespace secp256k1::fast {

class Scalar {
public:
    using limbs_type = std::array<std::uint64_t, 4>;

    Scalar();
    static Scalar zero();
    static Scalar one();
    static Scalar from_uint64(std::uint64_t value);
    static Scalar from_limbs(const limbs_type& limbs);
    static Scalar from_bytes(const std::array<std::uint8_t, 32>& bytes);
    static Scalar from_bytes(const std::uint8_t* bytes32);

    // BIP-340 strict parsing: rejects values >= curve order n (no reduction).
    // Returns false if bytes represent a value >= n.
    // Use for signature/key parsing where canonical encoding is required.
    static bool parse_bytes_strict(const std::uint8_t* bytes32, Scalar& out) noexcept;
    static bool parse_bytes_strict(const std::array<std::uint8_t, 32>& bytes, Scalar& out) noexcept;

    // BIP-340 strict parsing + nonzero: rejects values >= n OR == 0.
    // Use for secret key validation (BIP-340: 0 < d' < n).
    static bool parse_bytes_strict_nonzero(const std::uint8_t* bytes32, Scalar& out) noexcept;
    static bool parse_bytes_strict_nonzero(const std::array<std::uint8_t, 32>& bytes, Scalar& out) noexcept;

    // Developer-friendly: Create from hex string (64 hex chars)
    // Example: Scalar::from_hex("fffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140")
    static Scalar from_hex(const std::string& hex);

    std::array<std::uint8_t, 32> to_bytes() const;
    void write_bytes(std::uint8_t* out32) const noexcept;
    std::string to_hex() const;
    const limbs_type& limbs() const noexcept { return limbs_; }

    Scalar operator+(const Scalar& rhs) const noexcept;
    Scalar operator-(const Scalar& rhs) const noexcept;
    Scalar operator*(const Scalar& rhs) const noexcept;

    Scalar& operator+=(const Scalar& rhs) noexcept;
    Scalar& operator-=(const Scalar& rhs) noexcept;
    Scalar& operator*=(const Scalar& rhs) noexcept;

    bool is_zero() const noexcept;
    bool is_zero_ct() const noexcept;  // CT variant: reads all limbs, no early-return
    bool operator==(const Scalar& rhs) const noexcept;

    // Modular inverse: a^{-1} mod n  (Fermat's little theorem: a^{n-2} mod n)
    // Returns zero if this is zero.
    Scalar inverse() const;

    // Modular negation: -a mod n  (= n - a, or 0 if a == 0)
    // CT variant: always runs full computation — use for SECRET scalars (signing, nonces).
    Scalar negate() const noexcept;
    // Variable-time variant: branches on is_zero() — use ONLY on PUBLIC scalars (verify paths).
    Scalar negate_var() const noexcept;

    // Parity check: returns true if lowest bit is 0
    bool is_even() const noexcept;

    // Safe conversion to shared data type (memcpy, compiler-optimized to noop).
    // Returns by value -- no strict-aliasing UB.
    ::secp256k1::ScalarData data() const noexcept {
        ::secp256k1::ScalarData d;
        std::memcpy(&d, &limbs_, sizeof(d));
        return d;
    }
    static Scalar from_data(const ::secp256k1::ScalarData& d) {
        return from_limbs({d.limbs[0], d.limbs[1], d.limbs[2], d.limbs[3]});
    }

    std::uint8_t bit(std::size_t index) const;

    // Phase 5.6: NAF (Non-Adjacent Form) encoding
    // Zero-allocation variant: writes into caller-supplied buffer, returns digit count.
    // out must point to at least 257 bytes.
    std::size_t to_naf_into(std::int8_t* out) const noexcept;
    // Heap-allocating convenience wrapper (backward-compatible public API).
    std::vector<int8_t> to_naf() const;

    // Phase 5.7: wNAF (width-w Non-Adjacent Form) encoding
    // Zero-allocation variant: writes into caller-supplied buffer, returns digit count.
    // out must point to at least 257 bytes. width must be in [2,8].
    std::size_t to_wnaf_into(std::int8_t* out, unsigned width) const noexcept;
    // Heap-allocating convenience wrapper (backward-compatible public API).
    std::vector<int8_t> to_wnaf(unsigned width) const;

private:
    explicit Scalar(const limbs_type& limbs, bool normalized);

    limbs_type limbs_{};
};

// Cross-backend layout compatibility (shared types contract)
static_assert(sizeof(Scalar) == sizeof(::secp256k1::ScalarData),
              "CPU Scalar must match shared data layout size");
static_assert(sizeof(Scalar) == 32, "Scalar must be 256 bits");

} // namespace secp256k1::fast


#endif /* B4F0584E_3733_4C2F_BB01_D4BDC40E1760 */
