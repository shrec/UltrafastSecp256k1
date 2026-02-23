#ifndef B4F0584E_3733_4C2F_BB01_D4BDC40E1760
#define B4F0584E_3733_4C2F_BB01_D4BDC40E1760
#pragma once

#include <array>
#include <cstdint>
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
    
    // Developer-friendly: Create from hex string (64 hex chars)
    // Example: Scalar::from_hex("fffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140")
    static Scalar from_hex(const std::string& hex);

    std::array<std::uint8_t, 32> to_bytes() const;
    std::string to_hex() const;
    const limbs_type& limbs() const noexcept { return limbs_; }

    Scalar operator+(const Scalar& rhs) const;
    Scalar operator-(const Scalar& rhs) const;
    Scalar operator*(const Scalar& rhs) const;

    Scalar& operator+=(const Scalar& rhs);
    Scalar& operator-=(const Scalar& rhs);
    Scalar& operator*=(const Scalar& rhs);

    bool is_zero() const noexcept;
    bool operator==(const Scalar& rhs) const noexcept;
    bool operator!=(const Scalar& rhs) const noexcept { return !(*this == rhs); }

    // Modular inverse: a^{-1} mod n  (Fermat's little theorem: a^{n-2} mod n)
    // Returns zero if this is zero.
    Scalar inverse() const;

    // Modular negation: -a mod n  (= n - a, or 0 if a == 0)
    Scalar negate() const;

    // Parity check: returns true if lowest bit is 0
    bool is_even() const noexcept;

    // Zero-cost conversion to/from shared data type (for cross-backend interop)
#if defined(__GNUC__)
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wstrict-aliasing\"")
#endif
    const ::secp256k1::ScalarData& data() const noexcept {
        return *reinterpret_cast<const ::secp256k1::ScalarData*>(&limbs_);
    }
    ::secp256k1::ScalarData& data() noexcept {
        return *reinterpret_cast<::secp256k1::ScalarData*>(&limbs_);
    }
#if defined(__GNUC__)
    _Pragma("GCC diagnostic pop")
#endif
    static Scalar from_data(const ::secp256k1::ScalarData& d) {
        return from_limbs({d.limbs[0], d.limbs[1], d.limbs[2], d.limbs[3]});
    }

    std::uint8_t bit(std::size_t index) const;

    // Phase 5.6: NAF (Non-Adjacent Form) encoding
    // Converts scalar to signed digit representation {-1, 0, 1}
    // Returns vector where naf[i] represents digit at bit position i
    // NAF has ~33% fewer non-zero digits than binary
    std::vector<int8_t> to_naf() const;

    // Phase 5.7: wNAF (width-w Non-Adjacent Form) encoding
    // Converts scalar to signed odd-digit representation {+/-1, +/-3, +/-5, ..., +/-(2^w-1)}
    // width: window width (typically 4-6 bits)
    // Returns vector where wnaf[i] represents digit at position i
    // Only odd values are used, reducing precompute table size by 50%
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
