#include "secp256k1/scalar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>


namespace secp256k1::fast {
namespace {

using limbs4 = std::array<std::uint64_t, 4>;

constexpr limbs4 ORDER{
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

constexpr limbs4 ONE{1ULL, 0ULL, 0ULL, 0ULL};

// Barrett constant: mu = floor(2^512 / ORDER), 5 non-zero limbs
// mu = 0x1_00000000_00000001_4551231950B75FC4_402DA1732FC9BEC0
constexpr std::array<std::uint64_t, 5> BARRETT_MU{
    0x402DA1732FC9BEC0ULL,
    0x4551231950B75FC4ULL,
    0x0000000000000001ULL,
    0x0000000000000000ULL,
    0x0000000000000001ULL
};

// 8-limb wide integer
using wide8 = std::array<std::uint64_t, 8>;

#if defined(_MSC_VER) && !defined(__clang__)

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int64 out;
    carry = _addcarry_u64(carry, a, b, &out);
    return out;
}

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    unsigned __int64 out;
    borrow = _subborrow_u64(borrow, a, b, &out);
    return out;
}

#else

// 32-bit safe implementation (no __int128)
#ifdef SECP256K1_NO_INT128

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    std::uint64_t result = a + b;
    unsigned char new_carry = (result < a) ? 1 : 0;
    if (carry) {
        std::uint64_t temp = result + 1;
        new_carry |= (temp < result) ? 1 : 0;
        result = temp;
    }
    carry = new_carry;
    return result;
}

#else

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + carry;
    carry = static_cast<unsigned char>(sum >> 64);
    return static_cast<std::uint64_t>(sum);
}

#endif // SECP256K1_NO_INT128

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    uint64_t temp = a - borrow;
    unsigned char borrow1 = (a < borrow);
    uint64_t result = temp - b;
    unsigned char borrow2 = (temp < b);
    borrow = borrow1 | borrow2;
    return result;
}

#endif

[[nodiscard]] bool ge(const limbs4& a, const limbs4& b) {
    for (std::size_t i = 4; i-- > 0;) {
        if (a[i] > b[i]) {
            return true;
        }
        if (a[i] < b[i]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] limbs4 sub_impl(const limbs4& a, const limbs4& b);

[[nodiscard]] limbs4 add_impl(const limbs4& a, const limbs4& b) {
    // Compute raw 256-bit sum with carry
    limbs4 sum{};
    unsigned char carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        sum[i] = add64(a[i], b[i], carry);
    }

    // Compute sum - ORDER without modular wrap to decide reduction
    limbs4 sum_minus_order{};
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        sum_minus_order[i] = sub64(sum[i], ORDER[i], borrow);
    }

    // If carry from addition (sum >= 2^256) OR no borrow in subtraction (sum >= ORDER),
    // then result = sum - ORDER; otherwise result = sum.
    if (carry || (borrow == 0)) {
        return sum_minus_order;
    }
    return sum;
}

[[nodiscard]] limbs4 sub_impl(const limbs4& a, const limbs4& b) {
    limbs4 out{};
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        out[i] = sub64(a[i], b[i], borrow);
    }
    if (borrow) {
        unsigned char carry = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            out[i] = add64(out[i], ORDER[i], carry);
        }
    }
    return out;
}

} // namespace

Scalar::Scalar() = default;

Scalar::Scalar(const limbs_type& limbs, bool normalized) : limbs_(limbs) {
    if (!normalized && ge(limbs_, ORDER)) {
        limbs_ = sub_impl(limbs_, ORDER);
    }
}

Scalar Scalar::zero() {
    return Scalar();
}

Scalar Scalar::one() {
    return Scalar(ONE, true);
}

Scalar Scalar::from_uint64(std::uint64_t value) {
    limbs_type limbs{};
    limbs[0] = value;
    return Scalar(limbs, true);
}

Scalar Scalar::from_limbs(const limbs_type& limbs) {
    Scalar s;
    s.limbs_ = limbs;
    if (ge(s.limbs_, ORDER)) {
        s.limbs_ = sub_impl(s.limbs_, ORDER);
    }
    return s;
}

Scalar Scalar::from_bytes(const std::array<std::uint8_t, 32>& bytes) {
    limbs4 limbs{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t limb = 0;
        for (std::size_t j = 0; j < 8; ++j) {
            limb = (limb << 8) | bytes[i * 8 + j];
        }
        limbs[3 - i] = limb;
    }
    if (ge(limbs, ORDER)) {
        limbs = sub_impl(limbs, ORDER);
    }
    Scalar s;
    s.limbs_ = limbs;
    return s;
}

std::array<std::uint8_t, 32> Scalar::to_bytes() const {
    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t limb = limbs_[3 - i];
        for (std::size_t j = 0; j < 8; ++j) {
            out[i * 8 + j] = static_cast<std::uint8_t>(limb >> (56 - 8 * j));
        }
    }
    return out;
}

std::string Scalar::to_hex() const {
    auto bytes = to_bytes();
    std::string hex;
    hex.reserve(64);
    static const char hex_chars[] = "0123456789abcdef";
    for (auto b : bytes) {
        hex += hex_chars[(b >> 4) & 0xF];
        hex += hex_chars[b & 0xF];
    }
    return hex;
}

Scalar Scalar::from_hex(const std::string& hex) {
    if (hex.length() != 64) {
        #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
            return Scalar::zero(); // Embedded: no exceptions, return zero
        #else
            throw std::invalid_argument("Hex string must be exactly 64 characters (32 bytes)");
        #endif
    }
    
    std::array<std::uint8_t, 32> bytes{};
    for (size_t i = 0; i < 32; i++) {
        char c1 = hex[i * 2];
        char c2 = hex[i * 2 + 1];
        
        auto hex_to_nibble = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
                return 0; // Embedded: no exceptions, return 0
            #else
                throw std::invalid_argument("Invalid hex character");
            #endif
        };
        
        bytes[i] = (hex_to_nibble(c1) << 4) | hex_to_nibble(c2);
    }
    
    return from_bytes(bytes);
}

Scalar Scalar::operator+(const Scalar& rhs) const {
    return Scalar(add_impl(limbs_, rhs.limbs_), true);
}

Scalar Scalar::operator-(const Scalar& rhs) const {
    return Scalar(sub_impl(limbs_, rhs.limbs_), true);
}

Scalar Scalar::operator*(const Scalar& rhs) const {
    // Schoolbook 4×4 limb multiplication → 8-limb wide result
    // Then Barrett reduction mod ORDER
    // ~25-50× faster than double-and-add

    // Step 1: Schoolbook 4×4 → 512-bit product
    wide8 prod{};

#ifdef SECP256K1_NO_INT128
    // 32-bit fallback: use add64 chains for accumulation
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t carry_hi = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            // Compute a[i] * b[j] using 32-bit pieces
            std::uint64_t a_lo = limbs_[i] & 0xFFFFFFFFULL;
            std::uint64_t a_hi = limbs_[i] >> 32;
            std::uint64_t b_lo = rhs.limbs_[j] & 0xFFFFFFFFULL;
            std::uint64_t b_hi = rhs.limbs_[j] >> 32;

            std::uint64_t p0 = a_lo * b_lo;
            std::uint64_t p1 = a_lo * b_hi;
            std::uint64_t p2 = a_hi * b_lo;
            std::uint64_t p3 = a_hi * b_hi;

            // Combine cross terms
            std::uint64_t mid = p1 + p2;
            std::uint64_t mid_carry = (mid < p1) ? (1ULL << 32) : 0;

            std::uint64_t lo = p0 + (mid << 32);
            std::uint64_t lo_carry = (lo < p0) ? 1ULL : 0;
            std::uint64_t hi = p3 + (mid >> 32) + mid_carry + lo_carry;

            // Accumulate into prod[i+j] and prod[i+j+1]
            unsigned char c = 0;
            prod[i + j] = add64(prod[i + j], lo, c);
            prod[i + j + 1] = add64(prod[i + j + 1], hi, c);
            // Propagate carry
            for (std::size_t k = i + j + 2; c && k < 8; ++k) {
                prod[k] = add64(prod[k], 0ULL, c);
            }
        }
    }
#else
    // 64-bit path: use __int128
    for (std::size_t i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            unsigned __int128 t = static_cast<unsigned __int128>(limbs_[i]) * rhs.limbs_[j]
                                  + prod[i + j] + carry;
            prod[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        prod[i + 4] = static_cast<std::uint64_t>(carry);
    }
#endif

    // Step 2: Barrett reduction
    // q = floor(prod / 2^256) (high 4 limbs = prod[4..7])
    // q_approx = q * mu >> 256
    // Then result = prod - q_approx * ORDER, with at most 2 conditional subtracts

    // Compute q * mu (5-limb mu × 4-limb q → we need high limbs)
    // mu has non-zero limbs at [0],[1],[2],[4]
    // q = prod[4..7]
    const auto& q = prod; // use prod[4], prod[5], prod[6], prod[7]

    // We need floor((q * mu) / 2^256) which means the high part of q * mu
    // q is 4 limbs (prod[4..7]), mu is 5 limbs
    // Full product is 9 limbs; we need limbs [4..8]

#ifdef SECP256K1_NO_INT128
    // 32-bit fallback for Barrett
    std::array<std::uint64_t, 9> qmu{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            if (BARRETT_MU[j] == 0) continue;
            std::uint64_t a_val = q[4 + i];
            std::uint64_t b_val = BARRETT_MU[j];

            std::uint64_t a_lo = a_val & 0xFFFFFFFFULL;
            std::uint64_t a_hi = a_val >> 32;
            std::uint64_t b_lo = b_val & 0xFFFFFFFFULL;
            std::uint64_t b_hi = b_val >> 32;

            std::uint64_t p0 = a_lo * b_lo;
            std::uint64_t p1 = a_lo * b_hi;
            std::uint64_t p2 = a_hi * b_lo;
            std::uint64_t p3 = a_hi * b_hi;

            std::uint64_t mid = p1 + p2;
            std::uint64_t mid_carry = (mid < p1) ? (1ULL << 32) : 0;

            std::uint64_t lo = p0 + (mid << 32);
            std::uint64_t lo_carry = (lo < p0) ? 1ULL : 0;
            std::uint64_t hi = p3 + (mid >> 32) + mid_carry + lo_carry;

            unsigned char c = 0;
            qmu[i + j] = add64(qmu[i + j], lo, c);
            qmu[i + j + 1] = add64(qmu[i + j + 1], hi, c);
            for (std::size_t k = i + j + 2; c && k < 9; ++k) {
                qmu[k] = add64(qmu[k], 0ULL, c);
            }
        }
    }
#else
    // 64-bit path
    std::array<std::uint64_t, 9> qmu{};
    for (std::size_t i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < 5; ++j) {
            unsigned __int128 t = static_cast<unsigned __int128>(q[4 + i]) * BARRETT_MU[j]
                                  + qmu[i + j] + carry;
            qmu[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        qmu[i + 5] = static_cast<std::uint64_t>(carry);
    }
#endif

    // q_approx = high part of qmu = qmu[4..8]
    // r = prod mod 2^256 - q_approx * ORDER mod 2^256
    // Compute q_approx * ORDER (only need low 5 limbs since prod < 2^512)
    limbs4 q_approx{qmu[4], qmu[5], qmu[6], qmu[7]};

#ifdef SECP256K1_NO_INT128
    // 32-bit fallback for q_approx * ORDER
    std::array<std::uint64_t, 5> qn{};
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (i + j >= 5) break;
            std::uint64_t a_val = q_approx[i];
            std::uint64_t b_val = ORDER[j];

            std::uint64_t a_lo = a_val & 0xFFFFFFFFULL;
            std::uint64_t a_hi = a_val >> 32;
            std::uint64_t b_lo = b_val & 0xFFFFFFFFULL;
            std::uint64_t b_hi = b_val >> 32;

            std::uint64_t p0 = a_lo * b_lo;
            std::uint64_t p1 = a_lo * b_hi;
            std::uint64_t p2 = a_hi * b_lo;
            std::uint64_t p3 = a_hi * b_hi;

            std::uint64_t mid = p1 + p2;
            std::uint64_t mid_carry = (mid < p1) ? (1ULL << 32) : 0;

            std::uint64_t lo = p0 + (mid << 32);
            std::uint64_t lo_carry = (lo < p0) ? 1ULL : 0;
            std::uint64_t hi = p3 + (mid >> 32) + mid_carry + lo_carry;

            unsigned char c = 0;
            qn[i + j] = add64(qn[i + j], lo, c);
            if (i + j + 1 < 5) {
                qn[i + j + 1] = add64(qn[i + j + 1], hi, c);
                for (std::size_t k = i + j + 2; c && k < 5; ++k) {
                    qn[k] = add64(qn[k], 0ULL, c);
                }
            }
        }
    }
#else
    // 64-bit path: q_approx * ORDER, only low 5 limbs
    std::array<std::uint64_t, 5> qn{};
    for (std::size_t i = 0; i < 4; ++i) {
        unsigned __int128 carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            if (i + j >= 5) break;
            unsigned __int128 t = static_cast<unsigned __int128>(q_approx[i]) * ORDER[j]
                                  + qn[i + j] + carry;
            qn[i + j] = static_cast<std::uint64_t>(t);
            carry = t >> 64;
        }
        if (i + 4 < 5) {
            qn[i + 4] = static_cast<std::uint64_t>(carry);
        }
    }
#endif

    // r = prod[0..3] - qn[0..3], tracking overflow into r4
    // R = prod - q_approx*ORDER can be up to ~2*ORDER ≈ 2^257, so we need
    // the 5th limb (r4) to detect values >= 2^256.
    limbs4 r;
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        r[i] = sub64(prod[i], qn[i], borrow);
    }
    // r4 captures the overflow: prod[4] - qn[4] - borrow_from_low256
    std::uint64_t r4 = prod[4] - qn[4] - borrow;

    // At most 2 conditional subtracts to bring into [0, ORDER)
    if (r4 > 0 || ge(r, ORDER)) {
        borrow = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r[i] = sub64(r[i], ORDER[i], borrow);
        }
        r4 -= borrow;
    }
    if (r4 > 0 || ge(r, ORDER)) {
        borrow = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            r[i] = sub64(r[i], ORDER[i], borrow);
        }
    }

    return Scalar(r, true);
}

Scalar& Scalar::operator+=(const Scalar& rhs) {
    limbs_ = add_impl(limbs_, rhs.limbs_);
    return *this;
}

Scalar& Scalar::operator-=(const Scalar& rhs) {
    limbs_ = sub_impl(limbs_, rhs.limbs_);
    return *this;
}

Scalar& Scalar::operator*=(const Scalar& rhs) {
    *this = *this * rhs;
    return *this;
}

bool Scalar::is_zero() const noexcept {
    for (auto limb : limbs_) {
        if (limb != 0) {
            return false;
        }
    }
    return true;
}

bool Scalar::operator==(const Scalar& rhs) const noexcept {
    return limbs_ == rhs.limbs_;
}

std::uint8_t Scalar::bit(std::size_t index) const {
    if (index >= 256) {
        return 0;
    }
    std::size_t limb_idx = index / 64;
    std::size_t bit_idx = index % 64;
    return static_cast<std::uint8_t>((limbs_[limb_idx] >> bit_idx) & 0x1u);
}

// Phase 5.6: NAF (Non-Adjacent Form) encoding
// Converts scalar to signed representation {-1, 0, 1}
// NAF property: no two adjacent non-zero digits
// This reduces the number of non-zero digits by ~33%
// Algorithm: scan from LSB, if odd → take ±1, adjust remaining
std::vector<int8_t> Scalar::to_naf() const {
    std::vector<int8_t> naf;
    naf.reserve(257);  // Maximum NAF length is n+1 for n-bit number
    
    // Work with a mutable copy
    Scalar k = *this;
    
    while (!k.is_zero()) {
        if (k.bit(0) == 1) {  // k is odd
            // Get lowest 2 bits to determine sign
            std::uint8_t low_bits = static_cast<std::uint8_t>(k.limbs_[0] & 0x3);
            int8_t digit;
            
            if (low_bits == 1 || low_bits == 2) {
                // k ≡ 1 or 2 (mod 4) → use +1
                digit = 1;
                k -= Scalar::one();
            } else {
                // k ≡ 3 (mod 4) → use -1 (equivalent to k-1 being even)
                digit = -1;
                k += Scalar::one();
            }
            naf.push_back(digit);
        } else {
            // k is even → digit is 0
            naf.push_back(0);
        }
        
        // Divide k by 2 (right shift)
        std::uint64_t carry = 0;
        for (int i = 3; i >= 0; --i) {
            std::uint64_t limb = k.limbs_[i];
            k.limbs_[i] = (limb >> 1) | (carry << 63);
            carry = limb & 1;
        }
    }
    
    // NAF can be one bit longer than the original number
    // but we're done when k becomes zero
    return naf;
}

// Phase 5.7: wNAF (width-w Non-Adjacent Form)
// Converts scalar to signed odd-digit representation
// Window width w → digits in range {±1, ±3, ±5, ..., ±(2^w - 1)}
// Property: At most one non-zero digit in any w consecutive positions
// This reduces precompute table size by ~50% (only odd multiples needed)
std::vector<int8_t> Scalar::to_wnaf(unsigned width) const {
    if (width < 2 || width > 8) {
        #if defined(SECP256K1_ESP32) || defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
            return std::vector<int8_t>(); // Embedded: no exceptions, return empty
        #else
            throw std::invalid_argument("wNAF width must be between 2 and 8");
        #endif
    }
    
    std::vector<int8_t> wnaf;
    wnaf.reserve(257);  // Maximum length
    
    Scalar k = *this;
    const int window_size = 1 << width;          // 2^w
    const int window_mask = window_size - 1;      // 2^w - 1
    const int window_half = window_size >> 1;     // 2^(w-1)
    
    while (!k.is_zero()) {
        if (k.bit(0) == 1) {  // k is odd
            // Extract w bits
            int digit = static_cast<int>(k.limbs_[0] & window_mask);
            
            // If digit >= 2^(w-1), use negative representation
            if (digit >= window_half) {
                digit -= window_size;  // Make negative
                k += Scalar::from_uint64(static_cast<std::uint64_t>(-digit));
            } else {
                k -= Scalar::from_uint64(static_cast<std::uint64_t>(digit));
            }
            
            wnaf.push_back(static_cast<int8_t>(digit));
        } else {
            // k is even → digit is 0
            wnaf.push_back(0);
        }
        
        // Divide k by 2 (right shift)
        std::uint64_t carry = 0;
        for (int i = 3; i >= 0; --i) {
            std::uint64_t limb = k.limbs_[i];
            k.limbs_[i] = (limb >> 1) | (carry << 63);
            carry = limb & 1;
        }
    }
    
    return wnaf;
}

} // namespace secp256k1::fast
