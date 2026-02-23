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

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + carry;
    carry = static_cast<unsigned char>(sum >> 64);
    return static_cast<std::uint64_t>(sum);
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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

// Generic scalar add/sub mod N using 64-bit limbs

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
            if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(c - 'A' + 10);
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
    // Schoolbook 4x4 limb multiplication -> 8-limb wide result
    // Then Barrett reduction mod ORDER
    // ~25-50x faster than double-and-add

    // Step 1: Schoolbook 4x4 -> 512-bit product
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
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
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
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

    // Step 2: Barrett reduction
    // q = floor(prod / 2^256) (high 4 limbs = prod[4..7])
    // q_approx = q * mu >> 256
    // Then result = prod - q_approx * ORDER, with at most 2 conditional subtracts

    // Compute q * mu (5-limb mu x 4-limb q -> we need high limbs)
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
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
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
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
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
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
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
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

    // r = prod[0..3] - qn[0..3], tracking overflow into r4
    // R = prod - q_approx*ORDER can be up to ~2*ORDER ~= 2^257, so we need
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

// ============================================================================
// SafeGCD scalar modular inverse -- Bernstein-Yang divsteps algorithm (mod n)
// Variable-time.  ~10x faster than Fermat square-and-multiply.
// Ref: "Fast constant-time gcd computation and modular inversion" (2019)
// Direct port of bitcoin-core/secp256k1 secp256k1_modinv64_var.
// ============================================================================
#if defined(__SIZEOF_INT128__)
namespace scalar_safegcd {

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
using i128 = __int128;

struct S62  { int64_t v[5]; };
struct T2x2 { int64_t u, v, q, r; };
struct ModInfo { S62 modulus; uint64_t modulus_inv62; };

// secp256k1 order n in signed-62 form, plus modular inverse
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Matches bitcoin-core secp256k1_const_modinfo_scalar exactly.
static constexpr ModInfo NINFO = {
    {{0x3FD25E8CD0364141LL, 0x2ABB739ABD2280EELL, -0x15LL, 0LL, 256LL}},
    0x34F20099AA774EC1ULL
};

static inline int ctz64_var(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
#else
    int c = 0; while (!(x & 1)) { x >>= 1; ++c; } return c;
#endif
}

// Exactly matches secp256k1_modinv64_divsteps_62_var
static int64_t divsteps_62_var(int64_t eta, uint64_t f0, uint64_t g0, T2x2& t) {
    uint64_t u = 1, v = 0, q = 0, r = 1;
    uint64_t f = f0, g = g0, m;
    uint32_t w;
    int i = 62, limit, zeros;

    for (;;) {
        zeros = ctz64_var(g | (UINT64_MAX << i));
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros;
        i -= zeros;
        if (i == 0) break;

        if (eta < 0) {
            uint64_t tmp;
            eta = -eta;
            tmp = f; f = g; g = (uint64_t)(-(int64_t)tmp);
            tmp = u; u = q; q = (uint64_t)(-(int64_t)tmp);
            tmp = v; v = r; r = (uint64_t)(-(int64_t)tmp);
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (UINT64_MAX >> (64 - limit)) & 63U;
            w = (uint32_t)((f * g * (f * f - 2)) & m);
        } else {
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (UINT64_MAX >> (64 - limit)) & 15U;
            w = (uint32_t)(f + (((f + 1) & 4) << 1));
            w = (uint32_t)((-(uint64_t)w * g) & m);
        }
        g += f * (uint64_t)w;
        q += u * (uint64_t)w;
        r += v * (uint64_t)w;
    }

    t.u = (int64_t)u; t.v = (int64_t)v;
    t.q = (int64_t)q; t.r = (int64_t)r;
    return eta;
}

// Exactly matches secp256k1_modinv64_update_de_62
static void update_de_62(S62& d, S62& e, const T2x2& t, const ModInfo& mod) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t d0 = d.v[0], d1 = d.v[1], d2 = d.v[2], d3 = d.v[3], d4 = d.v[4];
    const int64_t e0 = e.v[0], e1 = e.v[1], e2 = e.v[2], e3 = e.v[3], e4 = e.v[4];
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    int64_t md, me, sd, se;
    i128 cd, ce;

    sd = d4 >> 63;
    se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    cd = (i128)u * d0 + (i128)v * e0;
    ce = (i128)q * d0 + (i128)r * e0;

    md -= (int64_t)((mod.modulus_inv62 * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((mod.modulus_inv62 * (uint64_t)ce + (uint64_t)me) & M62);

    cd += (i128)mod.modulus.v[0] * md;
    ce += (i128)mod.modulus.v[0] * me;
    cd >>= 62; ce >>= 62;

    cd += (i128)u * d1 + (i128)v * e1;
    ce += (i128)q * d1 + (i128)r * e1;
    if (mod.modulus.v[1]) { cd += (i128)mod.modulus.v[1] * md; ce += (i128)mod.modulus.v[1] * me; }
    d.v[0] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[0] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (i128)u * d2 + (i128)v * e2;
    ce += (i128)q * d2 + (i128)r * e2;
    if (mod.modulus.v[2]) { cd += (i128)mod.modulus.v[2] * md; ce += (i128)mod.modulus.v[2] * me; }
    d.v[1] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[1] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (i128)u * d3 + (i128)v * e3;
    ce += (i128)q * d3 + (i128)r * e3;
    if (mod.modulus.v[3]) { cd += (i128)mod.modulus.v[3] * md; ce += (i128)mod.modulus.v[3] * me; }
    d.v[2] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[2] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (i128)u * d4 + (i128)v * e4;
    ce += (i128)q * d4 + (i128)r * e4;
    cd += (i128)mod.modulus.v[4] * md;
    ce += (i128)mod.modulus.v[4] * me;
    d.v[3] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[3] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    d.v[4] = (int64_t)cd;
    e.v[4] = (int64_t)ce;
}

// Exactly matches secp256k1_modinv64_update_fg_62_var
static void update_fg_62_var(int len, S62& f, S62& g, const T2x2& t) {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t u = t.u, v = t.v, q = t.q, r = t.r;
    int64_t fi, gi;
    i128 cf, cg;

    fi = f.v[0]; gi = g.v[0];
    cf = (i128)u * fi + (i128)v * gi;
    cg = (i128)q * fi + (i128)r * gi;
    cf >>= 62; cg >>= 62;

    for (int j = 1; j < len; ++j) {
        fi = f.v[j]; gi = g.v[j];
        cf += (i128)u * fi + (i128)v * gi;
        cg += (i128)q * fi + (i128)r * gi;
        f.v[j - 1] = (int64_t)((uint64_t)cf & M62); cf >>= 62;
        g.v[j - 1] = (int64_t)((uint64_t)cg & M62); cg >>= 62;
    }
    f.v[len - 1] = (int64_t)cf;
    g.v[len - 1] = (int64_t)cg;
    for (int j = len; j < 5; ++j) { f.v[j] = 0; g.v[j] = 0; }
}

// Exactly matches secp256k1_modinv64_normalize_62
static void normalize_62(S62& r, int64_t sign, const ModInfo& mod) {
    const int64_t M62 = (int64_t)(UINT64_MAX >> 2);
    int64_t r0 = r.v[0], r1 = r.v[1], r2 = r.v[2], r3 = r.v[3], r4 = r.v[4];
    int64_t cond_add, cond_negate;

    cond_add = r4 >> 63;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    cond_add = r4 >> 63;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r.v[0] = r0; r.v[1] = r1; r.v[2] = r2; r.v[3] = r3; r.v[4] = r4;
}

static S62 limbs_to_s62(const limbs4& d) {
    constexpr uint64_t M = (1ULL << 62) - 1;
    return {{
        (int64_t)(d[0] & M),
        (int64_t)(((d[0] >> 62) | (d[1] << 2)) & M),
        (int64_t)(((d[1] >> 60) | (d[2] << 4)) & M),
        (int64_t)(((d[2] >> 58) | (d[3] << 6)) & M),
        (int64_t)(d[3] >> 56)
    }};
}

static limbs4 s62_to_limbs(const S62& s) {
    return {{
        (uint64_t)s.v[0] | ((uint64_t)s.v[1] << 62),
        ((uint64_t)s.v[1] >> 2) | ((uint64_t)s.v[2] << 60),
        ((uint64_t)s.v[2] >> 4) | ((uint64_t)s.v[3] << 58),
        ((uint64_t)s.v[3] >> 6) | ((uint64_t)s.v[4] << 56)
    }};
}

// Exactly matches secp256k1_modinv64_var
static limbs4 inverse_impl(const limbs4& x) {
    S62 d = {{0, 0, 0, 0, 0}};
    S62 e = {{1, 0, 0, 0, 0}};
    S62 f = NINFO.modulus;
    S62 g = limbs_to_s62(x);
    int len = 5;
    int64_t eta = -1;  // eta = -delta; delta starts at 1

    while (1) {
        T2x2 t;
        eta = divsteps_62_var(eta, (uint64_t)f.v[0], (uint64_t)g.v[0], t);

        update_de_62(d, e, t, NINFO);
        update_fg_62_var(len, f, g, t);

        if (g.v[0] == 0) {
            int64_t cond = 0;
            for (int j = 1; j < len; ++j) cond |= g.v[j];
            if (cond == 0) break;
        }

        int64_t fn = f.v[len - 1], gn = g.v[len - 1];
        int64_t cond = ((int64_t)len - 2) >> 63;
        cond |= fn ^ (fn >> 63);
        cond |= gn ^ (gn >> 63);
        if (cond == 0) {
            f.v[len - 2] |= (uint64_t)fn << 62;
            g.v[len - 2] |= (uint64_t)gn << 62;
            --len;
        }
    }

    normalize_62(d, f.v[len - 1], NINFO);
    return s62_to_limbs(d);
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
} // namespace scalar_safegcd

Scalar Scalar::inverse() const {
    if (is_zero()) return Scalar::zero();
    // SafeGCD divsteps -- ~10x faster than Fermat square-and-multiply
    return from_limbs(scalar_safegcd::inverse_impl(limbs_));
}

#else // !__SIZEOF_INT128__

// ============================================================================
// SafeGCD scalar modular inverse -- 30-bit divsteps (no __int128 needed)
// Port of bitcoin-core/secp256k1 modinv32 for 32-bit platforms (ESP32, etc.)
// Uses int32_t[9] signed-30 representation; intermediates fit in int64_t.
// ============================================================================
namespace scalar_safegcd30 {

struct S30  { int32_t v[9]; };
struct T2x2 { int32_t u, v, q, r; };
struct ModInfo { S30 modulus; uint32_t modulus_inv30; };

// secp256k1 order n in signed-30 form + modular inverse mod 2^30.
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// Matches bitcoin-core secp256k1_const_modinfo_scalar (8x32) exactly.
static constexpr ModInfo NINFO = {
    {{0x10364141, 0x3F497A33, 0x348A03BB, 0x2BB739AB,
      -0x146, 0, 0, 0, 65536}},
    0x2A774EC1u
};

// Count trailing zeros (32-bit, variable-time)
static inline int ctz32_var(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward(&idx, x);
    return (int)idx;
#else
    int c = 0; while (!(x & 1)) { x >>= 1; ++c; } return c;
#endif
}

// Lookup table: secp256k1_modinv32_inv256[i] = -(2*i+1)^{-1} mod 256
static const uint8_t inv256[128] = {
    0xFF,0x55,0x33,0x49,0xC7,0x5D,0x3B,0x11,0x0F,0xE5,0xC3,0x59,
    0xD7,0xED,0xCB,0x21,0x1F,0x75,0x53,0x69,0xE7,0x7D,0x5B,0x31,
    0x2F,0x05,0xE3,0x79,0xF7,0x0D,0xEB,0x41,0x3F,0x95,0x73,0x89,
    0x07,0x9D,0x7B,0x51,0x4F,0x25,0x03,0x99,0x17,0x2D,0x0B,0x61,
    0x5F,0xB5,0x93,0xA9,0x27,0xBD,0x9B,0x71,0x6F,0x45,0x23,0xB9,
    0x37,0x4D,0x2B,0x81,0x7F,0xD5,0xB3,0xC9,0x47,0xDD,0xBB,0x91,
    0x8F,0x65,0x43,0xD9,0x57,0x6D,0x4B,0xA1,0x9F,0xF5,0xD3,0xE9,
    0x67,0xFD,0xDB,0xB1,0xAF,0x85,0x63,0xF9,0x77,0x8D,0x6B,0xC1,
    0xBF,0x15,0xF3,0x09,0x87,0x1D,0xFB,0xD1,0xCF,0xA5,0x83,0x19,
    0x97,0xAD,0x8B,0xE1,0xDF,0x35,0x13,0x29,0xA7,0x3D,0x1B,0xF1,
    0xEF,0xC5,0xA3,0x39,0xB7,0xCD,0xAB,0x01
};

// Variable-time 30 divsteps (matches secp256k1_modinv32_divsteps_30_var)
static int32_t divsteps_30_var(int32_t eta, uint32_t f0, uint32_t g0, T2x2& t) {
    uint32_t u = 1, v = 0, q = 0, r = 1;
    uint32_t f = f0, g = g0, m;
    uint16_t w;
    int i = 30, limit, zeros;

    for (;;) {
        zeros = ctz32_var(g | (UINT32_MAX << i));
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= zeros;
        i -= zeros;
        if (i == 0) break;

        if (eta < 0) {
            uint32_t tmp;
            eta = -eta;
            tmp = f; f = g; g = (uint32_t)(-(int32_t)tmp);
            tmp = u; u = q; q = (uint32_t)(-(int32_t)tmp);
            tmp = v; v = r; r = (uint32_t)(-(int32_t)tmp);
        }
        limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
        m = (UINT32_MAX >> (32 - limit)) & 255U;
        w = (uint16_t)((g * inv256[(f >> 1) & 127]) & m);
        g += f * (uint32_t)w;
        q += u * (uint32_t)w;
        r += v * (uint32_t)w;
    }

    t.u = (int32_t)u; t.v = (int32_t)v;
    t.q = (int32_t)q; t.r = (int32_t)r;
    return eta;
}

// (t/2^30) * [d, e] mod modulus (matches secp256k1_modinv32_update_de_30)
static void update_de_30(S30& d, S30& e, const T2x2& t, const ModInfo& mod) {
    const int32_t M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t u = t.u, v = t.v, q = t.q, r = t.r;
    int32_t di, ei, md, me, sd, se;
    int64_t cd, ce;

    sd = d.v[8] >> 31;
    se = e.v[8] >> 31;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    di = d.v[0]; ei = e.v[0];
    cd = (int64_t)u * di + (int64_t)v * ei;
    ce = (int64_t)q * di + (int64_t)r * ei;

    md -= (int32_t)((mod.modulus_inv30 * (uint32_t)cd + (uint32_t)md) & (uint32_t)M30);
    me -= (int32_t)((mod.modulus_inv30 * (uint32_t)ce + (uint32_t)me) & (uint32_t)M30);

    cd += (int64_t)mod.modulus.v[0] * md;
    ce += (int64_t)mod.modulus.v[0] * me;
    cd >>= 30; ce >>= 30;

    for (int i = 1; i < 9; ++i) {
        di = d.v[i]; ei = e.v[i];
        cd += (int64_t)u * di + (int64_t)v * ei;
        ce += (int64_t)q * di + (int64_t)r * ei;
        cd += (int64_t)mod.modulus.v[i] * md;
        ce += (int64_t)mod.modulus.v[i] * me;
        d.v[i - 1] = (int32_t)cd & M30; cd >>= 30;
        e.v[i - 1] = (int32_t)ce & M30; ce >>= 30;
    }
    d.v[8] = (int32_t)cd;
    e.v[8] = (int32_t)ce;
}

// (t/2^30) * [f, g] variable-length (matches secp256k1_modinv32_update_fg_30_var)
static void update_fg_30_var(int len, S30& f, S30& g, const T2x2& t) {
    const int32_t M30 = (int32_t)(UINT32_MAX >> 2);
    const int32_t u = t.u, v = t.v, q = t.q, r = t.r;
    int32_t fi, gi;
    int64_t cf, cg;

    fi = f.v[0]; gi = g.v[0];
    cf = (int64_t)u * fi + (int64_t)v * gi;
    cg = (int64_t)q * fi + (int64_t)r * gi;
    cf >>= 30; cg >>= 30;

    for (int j = 1; j < len; ++j) {
        fi = f.v[j]; gi = g.v[j];
        cf += (int64_t)u * fi + (int64_t)v * gi;
        cg += (int64_t)q * fi + (int64_t)r * gi;
        f.v[j - 1] = (int32_t)((uint32_t)cf & (uint32_t)M30); cf >>= 30;
        g.v[j - 1] = (int32_t)((uint32_t)cg & (uint32_t)M30); cg >>= 30;
    }
    f.v[len - 1] = (int32_t)cf;
    g.v[len - 1] = (int32_t)cg;
    for (int j = len; j < 9; ++j) { f.v[j] = 0; g.v[j] = 0; }
}

// Normalize to [0, modulus) (matches secp256k1_modinv32_normalize_30)
static void normalize_30(S30& r, int32_t sign, const ModInfo& mod) {
    const int32_t M30 = (int32_t)(UINT32_MAX >> 2);
    int32_t r0=r.v[0], r1=r.v[1], r2=r.v[2], r3=r.v[3], r4=r.v[4],
            r5=r.v[5], r6=r.v[6], r7=r.v[7], r8=r.v[8];
    int32_t cond_add, cond_negate;

    cond_add = r8 >> 31;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    r5 += mod.modulus.v[5] & cond_add;
    r6 += mod.modulus.v[6] & cond_add;
    r7 += mod.modulus.v[7] & cond_add;
    r8 += mod.modulus.v[8] & cond_add;
    cond_negate = sign >> 31;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    r5 = (r5 ^ cond_negate) - cond_negate;
    r6 = (r6 ^ cond_negate) - cond_negate;
    r7 = (r7 ^ cond_negate) - cond_negate;
    r8 = (r8 ^ cond_negate) - cond_negate;
    r1 += r0 >> 30; r0 &= M30;
    r2 += r1 >> 30; r1 &= M30;
    r3 += r2 >> 30; r2 &= M30;
    r4 += r3 >> 30; r3 &= M30;
    r5 += r4 >> 30; r4 &= M30;
    r6 += r5 >> 30; r5 &= M30;
    r7 += r6 >> 30; r6 &= M30;
    r8 += r7 >> 30; r7 &= M30;

    cond_add = r8 >> 31;
    r0 += mod.modulus.v[0] & cond_add;
    r1 += mod.modulus.v[1] & cond_add;
    r2 += mod.modulus.v[2] & cond_add;
    r3 += mod.modulus.v[3] & cond_add;
    r4 += mod.modulus.v[4] & cond_add;
    r5 += mod.modulus.v[5] & cond_add;
    r6 += mod.modulus.v[6] & cond_add;
    r7 += mod.modulus.v[7] & cond_add;
    r8 += mod.modulus.v[8] & cond_add;
    r1 += r0 >> 30; r0 &= M30;
    r2 += r1 >> 30; r1 &= M30;
    r3 += r2 >> 30; r2 &= M30;
    r4 += r3 >> 30; r3 &= M30;
    r5 += r4 >> 30; r4 &= M30;
    r6 += r5 >> 30; r5 &= M30;
    r7 += r6 >> 30; r6 &= M30;
    r8 += r7 >> 30; r7 &= M30;

    r.v[0]=r0; r.v[1]=r1; r.v[2]=r2; r.v[3]=r3; r.v[4]=r4;
    r.v[5]=r5; r.v[6]=r6; r.v[7]=r7; r.v[8]=r8;
}

// Convert 4x64-bit limbs -> signed-30 representation
// Direct extraction per limb -- avoids accumulator overflow that drops
// high bits when shifting uint64_t left (x[1]<<4 loses bits 124-127, etc.).
static S30 limbs_to_s30(const limbs4& x) {
    S30 r{};
    const uint32_t M30 = 0x3FFFFFFFu;
    // Bit layout: x[0]=[0,64), x[1]=[64,128), x[2]=[128,192), x[3]=[192,256)
    // v[i] covers bits [i*30, (i+1)*30), v[8] covers [240,256) (16 bits).
    r.v[0] = (int32_t)( x[0]        & M30);              // bits [  0, 30)
    r.v[1] = (int32_t)((x[0] >> 30) & M30);              // bits [ 30, 60)
    r.v[2] = (int32_t)(((x[0] >> 60) | (x[1] <<  4)) & M30); // bits [ 60, 90)
    r.v[3] = (int32_t)((x[1] >> 26) & M30);              // bits [ 90,120)
    r.v[4] = (int32_t)(((x[1] >> 56) | (x[2] <<  8)) & M30); // bits [120,150)
    r.v[5] = (int32_t)((x[2] >> 22) & M30);              // bits [150,180)
    r.v[6] = (int32_t)(((x[2] >> 52) | (x[3] << 12)) & M30); // bits [180,210)
    r.v[7] = (int32_t)((x[3] >> 18) & M30);              // bits [210,240)
    r.v[8] = (int32_t)( x[3] >> 48);                     // bits [240,256)
    return r;
}

// Convert signed-30 -> 4x64-bit limbs
static limbs4 s30_to_limbs(const S30& s) {
    limbs4 r{};
    // Reassemble 9 x 30-bit limbs into 4 x 64-bit limbs
    // v[0]: bits 0-29, v[1]: bits 30-59, v[2]: bits 60-89, ...
    r[0] = ((uint64_t)(uint32_t)s.v[0])
         | ((uint64_t)(uint32_t)s.v[1] << 30)
         | ((uint64_t)(uint32_t)s.v[2] << 60);
    r[1] = ((uint64_t)(uint32_t)s.v[2] >> 4)
         | ((uint64_t)(uint32_t)s.v[3] << 26)
         | ((uint64_t)(uint32_t)s.v[4] << 56);
    r[2] = ((uint64_t)(uint32_t)s.v[4] >> 8)
         | ((uint64_t)(uint32_t)s.v[5] << 22)
         | ((uint64_t)(uint32_t)s.v[6] << 52);
    r[3] = ((uint64_t)(uint32_t)s.v[6] >> 12)
         | ((uint64_t)(uint32_t)s.v[7] << 18)
         | ((uint64_t)(uint32_t)s.v[8] << 48);
    return r;
}

// Main entry: variable-time modular inverse (matches secp256k1_modinv32_var)
static limbs4 inverse_impl(const limbs4& x) {
    S30 d{};                  // d = 0
    S30 e{}; e.v[0] = 1;     // e = 1
    S30 f = NINFO.modulus;    // f = n
    S30 g = limbs_to_s30(x); // g = x
    int len = 9;
    int32_t eta = -1;         // eta = -delta; delta starts at 1

    while (1) {
        T2x2 t;
        eta = divsteps_30_var(eta, (uint32_t)f.v[0], (uint32_t)g.v[0], t);

        update_de_30(d, e, t, NINFO);
        update_fg_30_var(len, f, g, t);

        if (g.v[0] == 0) {
            int32_t cond = 0;
            for (int j = 1; j < len; ++j) cond |= g.v[j];
            if (cond == 0) break;
        }

        int32_t fn = f.v[len - 1], gn = g.v[len - 1];
        int32_t cond = ((int32_t)len - 2) >> 31;
        cond |= fn ^ (fn >> 31);
        cond |= gn ^ (gn >> 31);
        if (cond == 0) {
            f.v[len - 2] |= (uint32_t)fn << 30;
            g.v[len - 2] |= (uint32_t)gn << 30;
            --len;
        }
    }

    normalize_30(d, f.v[len - 1], NINFO);
    return s30_to_limbs(d);
}

} // namespace scalar_safegcd30

Scalar Scalar::inverse() const {
    if (is_zero()) return Scalar::zero();
    return from_limbs(scalar_safegcd30::inverse_impl(limbs_));
}

#endif // __SIZEOF_INT128__

Scalar Scalar::negate() const {
    if (is_zero()) return Scalar::zero();
    return Scalar(sub_impl(ORDER, limbs_), true);
}

bool Scalar::is_even() const noexcept {
    return (limbs_[0] & 1) == 0;
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
// Algorithm: scan from LSB, if odd -> take +/-1, adjust remaining
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
                // k == 1 or 2 (mod 4) -> use +1
                digit = 1;
                k -= Scalar::one();
            } else {
                // k == 3 (mod 4) -> use -1 (equivalent to k-1 being even)
                digit = -1;
                k += Scalar::one();
            }
            naf.push_back(digit);
        } else {
            // k is even -> digit is 0
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
// Window width w -> digits in range {+/-1, +/-3, +/-5, ..., +/-(2^w - 1)}
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
            // k is even -> digit is 0
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
