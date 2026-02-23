#ifndef B9C07A97_9853_412C_BCB7_F2FB19B8C1A7
#define B9C07A97_9853_412C_BCB7_F2FB19B8C1A7

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "config.hpp"
#include "secp256k1/types.hpp"

namespace secp256k1::fast {

// Forward declaration
class FieldElement;

// ============================================================================
// HYBRID 32/64-bit support: Zero-cost view for optimized operations
// ============================================================================
// MidFieldElement provides 32-bit limb view of FieldElement for operations
// where 32-bit multiplication is faster (benchmarked 1.10x faster than 64-bit)
// Memory layout is IDENTICAL - just different interpretation!
struct MidFieldElement {
    std::uint32_t limbs[8];  // Same 256 bits, different view
    
    // Zero-cost conversion back to 64-bit representation
    inline FieldElement* ToFieldElement() noexcept {
        return reinterpret_cast<FieldElement*>(this);
    }
    inline const FieldElement* ToFieldElement() const noexcept {
        return reinterpret_cast<const FieldElement*>(this);
    }
    
    // Legacy lowercase (compatibility)
    inline FieldElement* toFieldElement() noexcept {
        return reinterpret_cast<FieldElement*>(this);
    }
    inline const FieldElement* toFieldElement() const noexcept {
        return reinterpret_cast<const FieldElement*>(this);
    }
};

class FieldElement {
public:
    using limbs_type = std::array<std::uint64_t, 4>;

    FieldElement();
    static FieldElement zero();
    static FieldElement one();
    static FieldElement from_uint64(std::uint64_t value);
    static FieldElement from_limbs(const limbs_type& limbs);
    static FieldElement from_bytes(const std::array<std::uint8_t, 32>& bytes);

    // Convert from Montgomery domain (a*R) to Standard domain (a)
    static FieldElement from_mont(const FieldElement& a);
    
    // Developer-friendly: Create from hex string (64 hex chars)
    // Example: FieldElement::from_hex("09af57f4f5c1d64c6bea6d4193c5d9130421f4f078868e5ec00a56e68001136c")
    static FieldElement from_hex(const std::string& hex);

    std::array<std::uint8_t, 32> to_bytes() const;
    // Write 32 big-endian bytes into provided buffer (avoids temporary array)
    void to_bytes_into(std::uint8_t* out) const noexcept;
    std::string to_hex() const;
    const limbs_type& limbs() const noexcept { return limbs_; }

    // Raw limb setter (no normalization) -- for use when caller guarantees canonical form.
    // Used by FieldElement52::to_fe() which already normalizes via fe52_normalize_inline.
    static FieldElement from_limbs_raw(const limbs_type& limbs) noexcept {
        FieldElement fe;
        fe.limbs_ = limbs;
        return fe;
    }

    FieldElement operator+(const FieldElement& rhs) const;
    FieldElement operator-(const FieldElement& rhs) const;
    FieldElement operator*(const FieldElement& rhs) const;
    FieldElement square() const;
    FieldElement inverse() const;

    // Square root: a^((p+1)/4) mod p.  Valid since p == 3 (mod 4).
    // Uses an optimized addition chain (~255 sqr + 13 mul).
    // Returns a root r such that r^2 == a (mod p). Caller must verify r^2==a.
    FieldElement sqrt() const;

    FieldElement& operator+=(const FieldElement& rhs);
    FieldElement& operator-=(const FieldElement& rhs);
    FieldElement& operator*=(const FieldElement& rhs);
    
    // Modular negation: returns p - this (mod p).
    // `magnitude` parameter exists for API compatibility with FieldElement52/26
    // (lazy-reduction magnitude tracking); ignored here since FE64 is always normalized.
    FieldElement negate(unsigned /*magnitude*/ = 1) const {
        return FieldElement::zero() - *this;
    }
    void negate_assign(unsigned /*magnitude*/ = 1) {
        *this = FieldElement::zero() - *this;
    }

    // In-place mutable versions (modify this object directly)
    // ~10-15% faster than immutable versions due to no memory allocation
    void square_inplace();    // this = this^2 (modifies this)
    void inverse_inplace();   // this = this^-^1 (modifies this)

    bool operator==(const FieldElement& rhs) const noexcept;
    bool operator!=(const FieldElement& rhs) const noexcept { return !(*this == rhs); }

    // Zero-cost conversion to/from shared data type (for cross-backend interop)
#if defined(__GNUC__)
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wstrict-aliasing\"")
#endif
    const ::secp256k1::FieldElementData& data() const noexcept {
        return *reinterpret_cast<const ::secp256k1::FieldElementData*>(&limbs_);
    }
    ::secp256k1::FieldElementData& data() noexcept {
        return *reinterpret_cast<::secp256k1::FieldElementData*>(&limbs_);
    }
#if defined(__GNUC__)
    _Pragma("GCC diagnostic pop")
#endif
    static FieldElement from_data(const ::secp256k1::FieldElementData& d) {
        return from_limbs({d.limbs[0], d.limbs[1], d.limbs[2], d.limbs[3]});
    }

private:
    explicit FieldElement(const limbs_type& limbs, bool normalized);

    limbs_type limbs_{};
};

// Zero-cost conversions for FieldElement <-> MidFieldElement
inline MidFieldElement* toMid(FieldElement* fe) noexcept {
    return reinterpret_cast<MidFieldElement*>(fe);
}

inline const MidFieldElement* toMid(const FieldElement* fe) noexcept {
    return reinterpret_cast<const MidFieldElement*>(fe);
}

// Compile-time verification
static_assert(sizeof(FieldElement) == sizeof(MidFieldElement), 
              "FieldElement and MidFieldElement must be same size");
static_assert(sizeof(FieldElement) == 32, "Must be 256 bits");

// Cross-backend layout compatibility (shared types contract)
static_assert(sizeof(FieldElement) == sizeof(::secp256k1::FieldElementData),
              "CPU FieldElement must match shared data layout size");
static_assert(sizeof(MidFieldElement) == sizeof(::secp256k1::MidFieldElementData),
              "CPU MidFieldElement must match shared data layout size");
static_assert(sizeof(FieldElement) == 32, "Must be 256 bits");

// ============================================================================
// Montgomery Domain Constants
// ============================================================================
// Montgomery domain: R = 2^256 mod p
// For secp256k1, p = 2^256 - 0x1000003D1, so R = 0x1000003D1
//
// Usage:
//   - to_mont(a) = (a * R^2) mod p  // Convert a -> a*R
//   - from_mont(aR) = (aR * R^-^1) mod p  // Convert a*R -> a
//   - mont_mul(aR, bR) = (aR * bR * R^-^1) mod p = (ab)R  // Stays in Montgomery
//
// Benefits:
//   - Modular reduction uses cheap multiplication instead of division
//   - 10-15% faster in batch operations
//   - Required for GPU H-based inversion pipeline

namespace montgomery {
    // R = 2^256 mod p = 0x1000003D1
    inline const FieldElement& R() {
        static const FieldElement r = FieldElement::from_uint64(0x1000003D1ULL);
        return r;
    }
    
    // R^2 mod p = (2^256)^2 mod p = 0x000007A2000E90A1
    inline const FieldElement& R2() {
        static const FieldElement r2 = FieldElement::from_limbs(
            {0x000007A2000E90A1ULL, 0x0000000000000001ULL, 0ULL, 0ULL});
        return r2;
    }
    
    // R^3 mod p = (2^256)^3 mod p
    inline const FieldElement& R3() {
        static const FieldElement r3 = FieldElement::from_limbs(
            {0x002BB1E33795F671ULL, 0x0000000100000B73ULL, 0ULL, 0ULL});
        return r3;
    }
    
    // R^-^1 mod p = (2^256)^-^1 mod p
    inline const FieldElement& R_inv() {
        static const FieldElement r_inv = FieldElement::from_limbs(
            {0xD838091D0868192AULL, 0xBCB223FEDC24A059ULL, 
             0x9C46C2C295F2B761ULL, 0xC9BD190515538399ULL});
        return r_inv;
    }
    
    // K = 2^32 + 977 = 0x1000003D1 (secp256k1 reduction constant)
    constexpr std::uint64_t K_MOD = 0x1000003D1ULL;
}

// ============================================================================
// Field Element Arithmetic Functions
// ============================================================================

FieldElement fe_inverse_binary(const FieldElement& value);
FieldElement fe_inverse_window4(const FieldElement& value);
FieldElement fe_inverse_addchain(const FieldElement& value);
FieldElement fe_inverse_eea(const FieldElement& value);

// Direct access to internal pow_p_minus_2 implementations for benchmarking
FieldElement pow_p_minus_2_binary(FieldElement base);
FieldElement pow_p_minus_2_window4(FieldElement base);
FieldElement pow_p_minus_2_addchain(FieldElement base);
FieldElement pow_p_minus_2_eea(FieldElement base);
FieldElement pow_p_minus_2_window_naf_v2(FieldElement base);
FieldElement pow_p_minus_2_hybrid_eea(FieldElement base);
FieldElement pow_p_minus_2_yao(FieldElement base);
FieldElement pow_p_minus_2_bos_coster(FieldElement base);
FieldElement pow_p_minus_2_ltr_precomp(FieldElement base);
FieldElement pow_p_minus_2_pippenger(FieldElement base);
FieldElement pow_p_minus_2_karatsuba(FieldElement base);
FieldElement pow_p_minus_2_booth(FieldElement base);
FieldElement pow_p_minus_2_strauss(FieldElement base);

// New optimized methods - Round 2
FieldElement pow_p_minus_2_kary16(FieldElement base);
FieldElement pow_p_minus_2_fixed_window5(FieldElement base);
FieldElement pow_p_minus_2_rtl_binary(FieldElement base);
FieldElement pow_p_minus_2_addchain_unrolled(FieldElement base);
FieldElement pow_p_minus_2_binary_opt(FieldElement base);
FieldElement pow_p_minus_2_sliding_dynamic(FieldElement base);

// Round 3 - GPU-optimized and ECC-specific
FieldElement pow_p_minus_2_fermat_gpu(FieldElement base);
FieldElement pow_p_minus_2_montgomery_redc(FieldElement base);
FieldElement pow_p_minus_2_branchless(FieldElement base);
FieldElement pow_p_minus_2_parallel_window(FieldElement base);
FieldElement pow_p_minus_2_binary_euclidean(FieldElement base);
FieldElement pow_p_minus_2_lehmer(FieldElement base);
FieldElement pow_p_minus_2_stein(FieldElement base);
FieldElement pow_p_minus_2_secp256k1_special(FieldElement base);
FieldElement pow_p_minus_2_warp_optimized(FieldElement base);
FieldElement pow_p_minus_2_double_base(FieldElement base);
FieldElement pow_p_minus_2_compact_table(FieldElement base);

// Wrapper functions
FieldElement fe_inverse_window_naf_v2(const FieldElement& value);
FieldElement fe_inverse_hybrid_eea(const FieldElement& value);
FieldElement fe_inverse_safegcd(const FieldElement& value);
FieldElement fe_inverse_yao(const FieldElement& value);
FieldElement fe_inverse_bos_coster(const FieldElement& value);
FieldElement fe_inverse_ltr_precomp(const FieldElement& value);
FieldElement fe_inverse_pippenger(const FieldElement& value);
FieldElement fe_inverse_karatsuba(const FieldElement& value);
FieldElement fe_inverse_booth(const FieldElement& value);
FieldElement fe_inverse_strauss(const FieldElement& value);

// New optimized wrappers
FieldElement fe_inverse_kary16(const FieldElement& value);
FieldElement fe_inverse_fixed_window5(const FieldElement& value);
FieldElement fe_inverse_rtl_binary(const FieldElement& value);
FieldElement fe_inverse_addchain_unrolled(const FieldElement& value);
FieldElement fe_inverse_binary_opt(const FieldElement& value);
FieldElement fe_inverse_sliding_dynamic(const FieldElement& value);

// Round 3 wrappers
FieldElement fe_inverse_fermat_gpu(const FieldElement& value);
FieldElement fe_inverse_montgomery_redc(const FieldElement& value);
FieldElement fe_inverse_branchless(const FieldElement& value);
FieldElement fe_inverse_parallel_window(const FieldElement& value);
FieldElement fe_inverse_binary_euclidean(const FieldElement& value);
FieldElement fe_inverse_lehmer(const FieldElement& value);
FieldElement fe_inverse_stein(const FieldElement& value);
FieldElement fe_inverse_secp256k1_special(const FieldElement& value);
FieldElement fe_inverse_warp_optimized(const FieldElement& value);
FieldElement fe_inverse_double_base(const FieldElement& value);
FieldElement fe_inverse_compact_table(const FieldElement& value);

// Montgomery batch inversion - invert N field elements simultaneously
// Uses Montgomery's trick: (a*b*c)^-1 -> compute individual a^-1, b^-1, c^-1
// Cost: 1 inverse + 3*(N-1) multiplications instead of N inverses
// For N=8: ~8 us instead of 28 us (3.5x speedup!)
void fe_batch_inverse(FieldElement* elements, size_t count);

// Zero-allocation version: uses provided scratch buffer
// scratch buffer must be at least size 'count'
void fe_batch_inverse(FieldElement* elements, size_t count, std::vector<FieldElement>& scratch);

} // namespace secp256k1::fast


#endif /* B9C07A97_9853_412C_BCB7_F2FB19B8C1A7 */
