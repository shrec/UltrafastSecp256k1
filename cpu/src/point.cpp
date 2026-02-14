#include "secp256k1/point.hpp"
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
#include "secp256k1/precompute.hpp"
#endif
#include "secp256k1/glv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace secp256k1::fast {
namespace {

// ESP32/STM32 local wNAF helpers (when precompute.hpp not included)
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
// Simple wNAF computation for ESP32 (inline, no heap allocation)
static void compute_wnaf_into(const Scalar& scalar, unsigned window_width,
                               int32_t* out, std::size_t out_capacity,
                               std::size_t& out_len) {
    // Get scalar as bytes (big-endian)
    auto bytes = scalar.to_bytes();

    // Convert to limbs (little-endian)
    uint64_t limbs[4] = {0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            limbs[i] |= static_cast<uint64_t>(bytes[31 - i*8 - j]) << (j*8);
        }
    }

    const int64_t width = 1 << window_width;     // 2^w
    const int64_t half_width = width >> 1;       // 2^(w-1)
    const uint64_t mask = static_cast<uint64_t>(width - 1);  // 2^w - 1

    out_len = 0;

    // Process until all bits consumed
    int bit_pos = 0;
    while (bit_pos < 256 || (limbs[0] | limbs[1] | limbs[2] | limbs[3]) != 0) {
        if (limbs[0] & 1) {
            // Current bit is set - compute wNAF digit
            int64_t digit = static_cast<int64_t>(limbs[0] & mask);
            if (digit >= half_width) {
                digit -= width;
            }
            out[out_len++] = static_cast<int32_t>(digit);

            // Subtract digit from scalar (handling negative digits as addition)
            if (digit > 0) {
                // Subtract positive digit
                uint64_t borrow = static_cast<uint64_t>(digit);
                for (int i = 0; i < 4 && borrow; i++) {
                    if (limbs[i] >= borrow) {
                        limbs[i] -= borrow;
                        borrow = 0;
                    } else {
                        uint64_t old = limbs[i];
                        limbs[i] -= borrow;  // wraps
                        borrow = 1;
                    }
                }
            } else if (digit < 0) {
                // Add absolute value of negative digit
                uint64_t carry = static_cast<uint64_t>(-digit);
                for (int i = 0; i < 4 && carry; i++) {
                    uint64_t sum = limbs[i] + carry;
                    carry = (sum < limbs[i]) ? 1 : 0;
                    limbs[i] = sum;
                }
            }
        } else {
            out[out_len++] = 0;
        }

        // Right-shift scalar by 1
        limbs[0] = (limbs[0] >> 1) | (limbs[1] << 63);
        limbs[1] = (limbs[1] >> 1) | (limbs[2] << 63);
        limbs[2] = (limbs[2] >> 1) | (limbs[3] << 63);
        limbs[3] >>= 1;

        bit_pos++;
        if (out_len >= out_capacity - 1) break;
    }
}

static std::vector<int32_t> compute_wnaf(const Scalar& scalar, unsigned window_bits) {
    std::array<int32_t, 260> buf{};
    std::size_t len = 0;
    compute_wnaf_into(scalar, window_bits, buf.data(), buf.size(), len);
    return std::vector<int32_t>(buf.begin(), buf.begin() + len);
}
#endif // ESP32

inline FieldElement fe_from_uint(std::uint64_t v) {
    return FieldElement::from_uint64(v);
}

// Affine coordinates (x, y) - more compact than Jacobian
struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity{true};
};

// Hot path: Point doubling - force aggressive inlining
// Optimized dbl-2007-a formula for a=0 curves (secp256k1)
// Operations: 4 squarings + 4 multiplications (vs previous 5 sqr + 3+ mul)
// Uses additions instead of multiplications for small constants (2, 3, 8)
#ifdef _MSC_VER
#pragma inline_recursion(on)
#pragma inline_depth(255)
#endif
SECP256K1_HOT_FUNCTION
JacobianPoint jacobian_double(const JacobianPoint& p) {
    if (SECP256K1_UNLIKELY(p.infinity || p.y == FieldElement::zero())) {
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    // A = X² (in-place)
    FieldElement A = p.x;           // Copy for in-place
    A.square_inplace();             // X² in-place!
    
    // B = Y² (in-place)
    FieldElement B = p.y;           // Copy for in-place
    B.square_inplace();             // Y² in-place!
    
    // C = B² (in-place)
    FieldElement C = B;             // Copy for in-place
    C.square_inplace();             // B² in-place!
    
    // D = 2*((X + B)² - A - C)
    FieldElement temp = p.x + B;
    temp.square_inplace();          // (X + B)² in-place!
    temp = temp - A;
    temp = temp - C;
    FieldElement D = temp + temp;  // *2 via addition (faster than mul)
    
    // E = 3*A
    FieldElement E = A + A;
    E = E + A;  // *3 via two additions (faster than mul)
    
    // F = E² (in-place)
    FieldElement F = E;             // Copy for in-place
    F.square_inplace();             // E² in-place!
    
    // X' = F - 2*D
    FieldElement two_D = D + D;
    FieldElement x3 = F - two_D;
    
    // Y' = E*(D - X') - 8*C
    FieldElement y3 = E * (D - x3);
    FieldElement eight_C = C + C;  // 2C
    eight_C = eight_C + eight_C;  // 4C
    eight_C = eight_C + eight_C;  // 8C (3 additions vs 1 mul)
    y3 = y3 - eight_C;
    
    // Z' = 2*Y*Z
    FieldElement z3 = p.y * p.z;
    z3 = z3 + z3;  // *2 via addition
    
    return {x3, y3, z3, false};
}

// Hot path: Mixed addition - optimize heavily
#ifdef _MSC_VER
#pragma inline_recursion(on)
#pragma inline_depth(255)
#endif
// Mixed Jacobian-Affine addition: P (Jacobian) + Q (Affine) -> Result (Jacobian)
// Optimized with dbl-2007-mix formula (7M + 4S for a=0)
[[nodiscard]] JacobianPoint jacobian_add_mixed(const JacobianPoint& p, const AffinePoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        // Convert affine to Jacobian: (x, y) -> (x, y, 1, false)
        return {q.x, q.y, FieldElement::one(), false};
    }

    // Core formula optimized for a=0 curve
    FieldElement z1z1 = p.z;                    // Copy for in-place
    z1z1.square_inplace();                      // Z1² in-place! [1S]
    FieldElement u2 = q.x * z1z1;               // U2 = X2*Z1² [1M]
    FieldElement s2 = q.y * p.z * z1z1;         // S2 = Y2*Z1³ [2M]
    
    if (SECP256K1_UNLIKELY(p.x == u2)) {
        if (p.y == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement h = u2 - p.x;                  // H = U2 - X1
    FieldElement hh = h;                        // Copy for in-place
    hh.square_inplace();                        // HH = H² in-place! [1S]
    FieldElement i = hh + hh + hh + hh;         // I = 4*HH (3 additions)
    FieldElement j = h * i;                     // J = H*I [1M]
    FieldElement r = (s2 - p.y) + (s2 - p.y);   // r = 2*(S2 - Y1)
    FieldElement v = p.x * i;                   // V = X1*I [1M]
    
    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r² in-place!
    x3 -= j + v + v;                            // X3 = r² - J - 2*V [1S]
    
    // Y3 = r*(V - X3) - 2*Y1*J — optimized: *2 via addition
    FieldElement y1j = p.y * j;                 // [1M]
    FieldElement y3 = r * (v - x3) - (y1j + y1j); // [1M]
    
    // Z3 = (Z1+H)² - Z1² - HH — in-place for performance
    FieldElement z3 = p.z + h;                  // Z1 + H
    z3.square_inplace();                        // (Z1+H)² in-place!
    z3 -= z1z1 + hh;                           // Z3 = (Z1+H)² - Z1² - HH [1S: total 7M + 4S]

    return {x3, y3, z3, false};
}

[[nodiscard]] JacobianPoint jacobian_add(const JacobianPoint& p, const JacobianPoint& q) {
    if (SECP256K1_UNLIKELY(p.infinity)) {
        return q;
    }
    if (SECP256K1_UNLIKELY(q.infinity)) {
        return p;
    }

    FieldElement z1z1 = p.z;                    // Copy for in-place
    z1z1.square_inplace();                      // z1^2 in-place!
    FieldElement z2z2 = q.z;                    // Copy for in-place
    z2z2.square_inplace();                      // z2^2 in-place!
    FieldElement u1 = p.x * z2z2;
    FieldElement u2 = q.x * z1z1;
    FieldElement s1 = p.y * q.z * z2z2;
    FieldElement s2 = q.y * p.z * z1z1;

    if (u1 == u2) {
        if (s1 == s2) {
            return jacobian_double(p);
        }
        return {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    }

    FieldElement h = u2 - u1;
    FieldElement i = h + h;                     // 2*h
    i.square_inplace();                         // (2*h)^2 in-place!
    FieldElement j = h * i;
    FieldElement r = (s2 - s1) + (s2 - s1);
    FieldElement v = u1 * i;

    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r^2 in-place!
    x3 -= j + v + v;                           // x3 = r^2 - j - 2*v
    FieldElement s1j = s1 * j;
    FieldElement y3 = r * (v - x3) - (s1j + s1j);
    FieldElement temp_z = p.z + q.z;           // z1 + z2
    temp_z.square_inplace();                   // (z1 + z2)^2 in-place!
    FieldElement z3 = (temp_z - z1z1 - z2z2) * h;

    return {x3, y3, z3, false};
}

// Batch inversion: Convert multiple Jacobian points to Affine using Montgomery's trick
// Cost: 1 inversion + (3*n - 1) multiplications for n points
// vs n inversions for individual conversion
std::vector<AffinePoint> batch_to_affine(const std::vector<JacobianPoint>& jacobian_points) {
    size_t n = jacobian_points.size();
    if (n == 0) return {};
    
    std::vector<AffinePoint> affine_points;
    affine_points.reserve(n);
    
    // Special case: single point
    if (n == 1) {
        const auto& p = jacobian_points[0];
        if (p.infinity) {
            // Infinity point - use (0, 0) as representation
            affine_points.push_back({FieldElement::zero(), FieldElement::zero()});
        } else {
            FieldElement z_inv = p.z.inverse();
            FieldElement z_inv_sq = z_inv;       // Copy for in-place
            z_inv_sq.square_inplace();           // z_inv^2 in-place!
            affine_points.push_back({
                p.x * z_inv_sq,           // x/z²
                p.y * z_inv_sq * z_inv    // y/z³
            });
        }
        return affine_points;
    }
    
    // Montgomery's trick for batch inversion
    // Step 1: Compute prefix products
    std::vector<FieldElement> prefix(n);
    prefix[0] = jacobian_points[0].z;
    for (size_t i = 1; i < n; i++) {
        prefix[i] = prefix[i-1] * jacobian_points[i].z;
    }
    
    // Step 2: Single inversion of product
    FieldElement inv_product = prefix[n-1].inverse();
    
    // Step 3: Compute individual inverses using prefix products
    std::vector<FieldElement> z_invs(n);
    z_invs[n-1] = inv_product * prefix[n-2];
    for (int i = static_cast<int>(n) - 2; i > 0; --i) {
        z_invs[i] = inv_product * prefix[i-1];
        inv_product = inv_product * jacobian_points[i+1].z;
    }
    z_invs[0] = inv_product;
    
    // Step 4: Convert to affine coordinates
    for (size_t i = 0; i < n; i++) {
        const auto& p = jacobian_points[i];
        if (p.infinity) {
            affine_points.push_back({FieldElement::zero(), FieldElement::zero()});
        } else {
            FieldElement z_inv_sq = z_invs[i];       // Copy for in-place
            z_inv_sq.square_inplace();               // z_inv^2 in-place!
            affine_points.push_back({
                p.x * z_inv_sq,              // x/z²
                p.y * z_invs[i] * z_inv_sq   // y/z³
            });
        }
    }
    
    return affine_points;
}

// No-alloc batch conversion: jacobian -> affine written into out[0..n)
static void batch_to_affine_into(const JacobianPoint* jacobian_points,
                                 std::size_t n,
                                 AffinePoint* out) {
    if (n == 0) return;

    // Conservative upper bound for this module: w in [4,7] => tables up to 2*64 = 128
    constexpr std::size_t kMaxN = 128;
    if (n > kMaxN) {
        // Fallback to allocating version for unusually large inputs
        std::vector<JacobianPoint> tmp(jacobian_points, jacobian_points + n);
        auto v = batch_to_affine(tmp);
        for (std::size_t i = 0; i < n; ++i) out[i] = v[i];
        return;
    }

    // Special case: single point
    if (n == 1) {
        const auto& p = jacobian_points[0];
        if (p.infinity) {
            out[0] = {FieldElement::zero(), FieldElement::zero()};
        } else {
            FieldElement z_inv = p.z.inverse();
            FieldElement z_inv_sq = z_inv; z_inv_sq.square_inplace();
            out[0] = { p.x * z_inv_sq, p.y * z_inv_sq * z_inv };
        }
        return;
    }

    std::array<FieldElement, kMaxN> prefix{};
    std::array<FieldElement, kMaxN> z_invs{};

    // Step 1: prefix products
    prefix[0] = jacobian_points[0].z;
    for (std::size_t i = 1; i < n; ++i) {
        prefix[i] = prefix[i - 1] * jacobian_points[i].z;
    }

    // Step 2: invert total product
    FieldElement inv_product = prefix[n - 1].inverse();

    // Step 3: individual inverses (standard backward pass)
    // inv_product initially = (z0*z1*...*z{n-1})^{-1}
    // For i from n-1..1: z_inv[i] = inv_product * prefix[i-1]; inv_product *= z_i
    // Finally z_inv[0] = inv_product
    z_invs[n - 1] = inv_product * prefix[n - 2];
    for (std::size_t i = n - 2; i > 0; --i) {
        z_invs[i] = inv_product * prefix[i - 1];
        inv_product = inv_product * jacobian_points[i].z;
    }
    z_invs[0] = inv_product;

    // Step 4: to affine
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = jacobian_points[i];
        if (p.infinity) {
            out[i] = {FieldElement::zero(), FieldElement::zero()};
        } else {
            FieldElement z_inv_sq = z_invs[i];
            z_inv_sq.square_inplace();
            out[i] = { p.x * z_inv_sq, p.y * z_invs[i] * z_inv_sq };
        }
    }
}

} // namespace

// KPlan implementation: Cache all K-dependent work for Fixed K × Variable Q
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
KPlan KPlan::from_scalar(const Scalar& k, uint8_t w) {
    // Step 1: GLV decomposition (K → k1, k2, signs)
    auto decomp = split_scalar_glv(k);
    
    // Step 2: Compute wNAF for both scalars
    auto wnaf1 = compute_wnaf(decomp.k1, w);
    auto wnaf2 = compute_wnaf(decomp.k2, w);
    
    // Return cached plan with k1, k2 scalars preserved
    KPlan plan;
    plan.window_width = w;
    plan.k1 = decomp.k1;
    plan.k2 = decomp.k2;
    plan.wnaf1 = std::move(wnaf1);
    plan.wnaf2 = std::move(wnaf2);
    plan.neg1 = decomp.neg1;
    plan.neg2 = decomp.neg2;
    return plan;
}
#else
// ESP32: simplified implementation (no GLV, just store scalar for fallback)
KPlan KPlan::from_scalar(const Scalar& k, uint8_t w) {
    KPlan plan;
    plan.window_width = w;
    plan.k1 = k;  // Store original scalar for fallback scalar_mul
    plan.neg1 = false;
    plan.neg2 = false;
    return plan;
}
#endif

Point::Point() : x_(FieldElement::zero()), y_(FieldElement::one()), z_(FieldElement::zero()), 
                 infinity_(true), is_generator_(false) {}

Point::Point(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity)
    : x_(x), y_(y), z_(z), 
      infinity_(infinity), is_generator_(false) {}

Point Point::from_jacobian_coords(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity) {
    if (infinity || z == FieldElement::zero()) {
        return Point::infinity();
    }
    return Point(x, y, z, false);
}

// Precomputed generator point G in affine coordinates (for fast mixed addition)
static const FieldElement kGeneratorX = FieldElement::from_bytes({
    0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,
    0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
    0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,
    0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98
});

static const FieldElement kGeneratorY = FieldElement::from_bytes({
    0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,
    0x5D,0xA4,0xFB,0xFC,0x0E,0x11,0x08,0xA8,
    0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,
    0x9C,0x47,0xD0,0x8F,0xFB,0x10,0xD4,0xB8
});

// Precomputed -G (negative generator) in affine coordinates
// -G = (Gx, -Gy mod p) where p = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
static const FieldElement kNegGeneratorY = FieldElement::from_bytes({
    0xB7,0xC5,0x25,0x88,0xD9,0x5C,0x3B,0x9A,
    0xA2,0x5B,0x04,0x03,0xF1,0xEE,0xF7,0x57,
    0x02,0xE8,0x4B,0xB7,0x59,0x7A,0xAB,0xE6,
    0x63,0xB8,0x2F,0x6F,0x04,0xEF,0x27,0x77
});

Point Point::generator() {
    Point g(kGeneratorX, kGeneratorY, fe_from_uint(1), false);
    g.is_generator_ = true;
    return g;
}

Point Point::infinity() {
    return Point(FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true);
}

Point Point::from_affine(const FieldElement& x, const FieldElement& y) {
    return Point(x, y, fe_from_uint(1), false);
}

Point Point::from_hex(const std::string& x_hex, const std::string& y_hex) {
    FieldElement x = FieldElement::from_hex(x_hex);
    FieldElement y = FieldElement::from_hex(y_hex);
    return from_affine(x, y);
}

FieldElement Point::x() const {
    if (infinity_) {
        return FieldElement::zero();
    }
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv;         // Copy for in-place
    z_inv2.square_inplace();             // z_inv^2 in-place!
    return x_ * z_inv2;
}

FieldElement Point::y() const {
    if (infinity_) {
        return FieldElement::zero();
    }
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv3 = z_inv;         // Copy for in-place
    z_inv3.square_inplace();             // z_inv^2 in-place!
    z_inv3 *= z_inv;                     // z_inv^3
    return y_ * z_inv3;
}

std::array<uint8_t, 16> Point::x_first_half() const {
    // Convert to affine once and reuse
    std::array<uint8_t, 32> full_x;
    x().to_bytes_into(full_x.data());
    
    std::array<uint8_t, 16> first_half;
    std::copy(full_x.begin(), full_x.begin() + 16, first_half.begin());
    return first_half;
}

std::array<uint8_t, 16> Point::x_second_half() const {
    // Convert to affine once and reuse
    std::array<uint8_t, 32> full_x;
    x().to_bytes_into(full_x.data());
    
    std::array<uint8_t, 16> second_half;
    std::copy(full_x.begin() + 16, full_x.end(), second_half.begin());
    return second_half;
}

Point Point::add(const Point& other) const {
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, other.y_, other.z_, other.infinity_};
    JacobianPoint r = jacobian_add(p, q);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
}

Point Point::dbl() const {
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint r = jacobian_double(p);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
}

Point Point::negate() const {
    if (infinity_) {
        return *this;  // Infinity is its own negation
    }
    
    // Negate in Jacobian coordinates: (x, y, z) → (x, -y, z)
    // This is much cheaper than converting to affine, negating, and converting back
    FieldElement neg_y = FieldElement::zero() - y_;
    Point result(x_, neg_y, z_, false);
    result.is_generator_ = false;
    return result;
}

Point Point::next() const {
    // Optimized: this + G using mixed Jacobian-Affine addition (8 muls vs 12)
    JacobianPoint p{x_, y_, z_, infinity_};
    AffinePoint g_affine{kGeneratorX, kGeneratorY};
    JacobianPoint r = jacobian_add_mixed(p, g_affine);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
}

Point Point::prev() const {
    // Optimized: this - G = this + (-G) using mixed addition (8 muls vs 12)
    JacobianPoint p{x_, y_, z_, infinity_};
    AffinePoint neg_g_affine{kGeneratorX, kNegGeneratorY};
    JacobianPoint r = jacobian_add_mixed(p, neg_g_affine);
    Point result = from_jacobian_coords(r.x, r.y, r.z, r.infinity);
    result.is_generator_ = false;
    return result;
}

// Mutable in-place addition: *this += G (no allocation overhead)
void Point::next_inplace() {
    if (infinity_) {
        *this = Point::generator();
        return;
    }
    
    // Mixed Jacobian-Affine addition with G
    AffinePoint g{kGeneratorX, kGeneratorY};
    
    // Use proven formula from jacobian_add_mixed
    JacobianPoint self{x_, y_, z_, infinity_};
    JacobianPoint r = jacobian_add_mixed(self, g);
    
    x_ = r.x;
    y_ = r.y;
    z_ = r.z;
    infinity_ = r.infinity;
    is_generator_ = false;
}

// Mutable in-place subtraction: *this -= G (no allocation overhead)
void Point::prev_inplace() {
    if (infinity_) {
        *this = Point::generator().negate();
        return;
    }
    
    // Mixed Jacobian-Affine addition with -G
    AffinePoint neg_g{kGeneratorX, kNegGeneratorY};
    
    // Use proven formula from jacobian_add_mixed
    JacobianPoint self{x_, y_, z_, infinity_};
    JacobianPoint r = jacobian_add_mixed(self, neg_g);
    
    x_ = r.x;
    y_ = r.y;
    z_ = r.z;
    infinity_ = r.infinity;
    is_generator_ = false;
}

// Mutable in-place addition: *this += other (no allocation overhead)
// ~12% faster than add() due to no memory allocation/copying
void Point::add_inplace(const Point& other) {
    // Fast path: if other is affine (z = 1), use mixed addition
    if (!other.infinity_ && other.z_ == FieldElement::one()) {
        JacobianPoint p{x_, y_, z_, infinity_};
        AffinePoint q{other.x_, other.y_};
        JacobianPoint r = jacobian_add_mixed(p, q);

        x_ = r.x;
        y_ = r.y;
        z_ = r.z;
        infinity_ = r.infinity;
        is_generator_ = false;
        return;
    }

    // General case: full Jacobian-Jacobian addition
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, other.y_, other.z_, other.infinity_};
    JacobianPoint r = jacobian_add(p, q);

    x_ = r.x;
    y_ = r.y;
    z_ = r.z;
    infinity_ = r.infinity;
    is_generator_ = false;
}

// Mutable in-place subtraction: *this -= other (no allocation overhead)
void Point::sub_inplace(const Point& other) {
    // In-place subtraction without allocating a temporary Point
    // this -= other  => this + (-other)
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint q{other.x_, FieldElement::zero() - other.y_, other.z_, other.infinity_};
    JacobianPoint r = jacobian_add(p, q);

    // Update in-place
    x_ = r.x;
    y_ = r.y;
    z_ = r.z;
    infinity_ = r.infinity;
    is_generator_ = false;
}

// Mutable in-place doubling: *this = 2*this (no allocation overhead)
void Point::dbl_inplace() {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(__XTENSA__) || defined(SECP256K1_PLATFORM_STM32)
    // Optimized: 5S + 2M formula (saves 2S vs generic 7S + 1M)
    // Z3 = 2·Y·Z (1M) replaces (Y+Z)²-Y²-Z² (2S), operates on fields directly
    if (infinity_) return;

    FieldElement xx = x_; xx.square_inplace();           // 1S: X²
    FieldElement yy = y_; yy.square_inplace();           // 1S: Y²
    FieldElement yyyy = yy; yyyy.square_inplace();       // 1S: Y⁴

    FieldElement s = x_ + yy;
    s.square_inplace();                                  // 1S: (X+Y²)²
    s -= xx + yyyy;
    s = s + s;                                           // S = 4·X·Y²

    FieldElement m = xx + xx + xx;                        // M = 3·X²

    // Compute Z3 BEFORE modifying x_, y_ (reads original values)
    FieldElement z3 = y_ * z_;                           // 1M: Y·Z
    z3 = z3 + z3;                                        // Z3 = 2·Y·Z

    // X3 = M² - 2·S
    x_ = m;
    x_.square_inplace();                                 // 1S: M²
    x_ -= s + s;

    // Y3 = M·(S - X3) - 8·Y⁴
    FieldElement yyyy8 = yyyy + yyyy;
    yyyy8 = yyyy8 + yyyy8;
    yyyy8 = yyyy8 + yyyy8;
    y_ = m * (s - x_) - yyyy8;                           // 1M

    z_ = z3;
    is_generator_ = false;
#else
    JacobianPoint p{x_, y_, z_, infinity_};
    JacobianPoint r = jacobian_double(p);
    x_ = r.x;
    y_ = r.y;
    z_ = r.z;
    infinity_ = r.infinity;
    is_generator_ = false;
#endif
}

// Mutable in-place negation: *this = -this (no allocation overhead)
void Point::negate_inplace() {
    // Negation in Jacobian: (X, Y, Z) → (X, -Y, Z)
    y_ = FieldElement::zero() - y_;
}

// Explicit mixed-add with affine input: this += (ax, ay)
// OPTIMIZED: Inline implementation to avoid 6 FieldElement struct copies per call
void Point::add_mixed_inplace(const FieldElement& ax, const FieldElement& ay) {
    if (SECP256K1_UNLIKELY(infinity_)) {
        // Convert affine to Jacobian: (x, y) -> (x, y, 1, false)
        x_ = ax;
        y_ = ay;
        z_ = FieldElement::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }

    // Core formula optimized for a=0 curve (inline to avoid struct copies)
    FieldElement z1z1 = z_;                     // Copy for in-place
    z1z1.square_inplace();                      // Z1² in-place! [1S]
    FieldElement u2 = ax * z1z1;                // U2 = X2*Z1² [1M]
    FieldElement s2 = ay * z_ * z1z1;           // S2 = Y2*Z1³ [2M]
    
    if (SECP256K1_UNLIKELY(x_ == u2)) {
        if (y_ == s2) {
            dbl_inplace();
            return;
        }
        // Points are inverses: result is infinity
        x_ = FieldElement::zero();
        y_ = FieldElement::one();
        z_ = FieldElement::zero();
        infinity_ = true;
        is_generator_ = false;
        return;
    }

    FieldElement h = u2 - x_;                   // H = U2 - X1
    FieldElement hh = h;                        // Copy for in-place
    hh.square_inplace();                        // HH = H² in-place! [1S]
    FieldElement i = hh + hh + hh + hh;         // I = 4*HH (3 additions)
    FieldElement j = h * i;                     // J = H*I [1M]
    FieldElement r = (s2 - y_) + (s2 - y_);     // r = 2*(S2 - Y1)
    FieldElement v = x_ * i;                    // V = X1*I [1M]
    
    FieldElement x3 = r;                        // Copy for in-place
    x3.square_inplace();                        // r² in-place!
    x3 -= j + v + v;                            // X3 = r² - J - 2*V [1S]
    
    // Y3 = r*(V - X3) - 2*Y1*J — optimized: *2 via addition
    FieldElement y1j = y_ * j;                  // [1M]
    FieldElement y3 = r * (v - x3) - (y1j + y1j); // [1M]
    
    // Z3 = (Z1+H)² - Z1² - HH — in-place for performance
    FieldElement z3 = z_ + h;                   // Z1 + H
    z3.square_inplace();                        // (Z1+H)² in-place!
    z3 -= z1z1 + hh;                            // Z3 = (Z1+H)² - Z1² - HH [1S: total 7M + 4S]

    // Update members directly (no return copy!)
    x_ = x3;
    y_ = y3;
    z_ = z3;
    infinity_ = false;
    is_generator_ = false;
}

// Explicit mixed-sub with affine input: this -= (ax, ay) == this + (ax, -ay)
// OPTIMIZED: Inline implementation to avoid 6 FieldElement struct copies per call
void Point::sub_mixed_inplace(const FieldElement& ax, const FieldElement& ay) {
    // Negate Y coordinate: subtract is add with negated Y
    FieldElement neg_ay = FieldElement::zero() - ay;
    add_mixed_inplace(ax, neg_ay);
}

// Repeated mixed addition with a fixed affine point (ax, ay, z=1)
// Changed from CRITICAL_FUNCTION to HOT_FUNCTION - flatten breaks this!
SECP256K1_HOT_FUNCTION
void Point::add_affine_constant_inplace(const FieldElement& ax, const FieldElement& ay) {
    if (infinity_) {
        x_ = ax;
        y_ = ay;
        z_ = FieldElement::one();
        infinity_ = false;
        is_generator_ = false;
        return;
    }

    FieldElement z1z1 = z_;          // z^2
    z1z1.square_inplace();
    FieldElement u2 = ax * z1z1;     // X2 * z^2
    FieldElement s2 = ay * z_ * z1z1; // Y2 * z^3

    if (x_ == u2) {
        if (y_ == s2) {
            dbl_inplace();
            return;
        } else {
            // Infinity
            *this = Point();
            return;
        }
    }

    FieldElement h = u2 - x_;
    FieldElement hh = h; hh.square_inplace(); // h^2
    FieldElement i = hh + hh + hh + hh;       // 4*HH
    FieldElement j = h * i;                   // H*I
    FieldElement r = (s2 - y_) + (s2 - y_);   // 2*(S2 - Y1)
    FieldElement v = x_ * i;                  // X1*I

    FieldElement x3 = r; x3.square_inplace(); // r^2
    x3 -= j + v + v;                          // r^2 - J - 2*V
    FieldElement y1j = y_ * j;                // Y1*J
    FieldElement y3 = r * (v - x3) - (y1j + y1j); // r*(V - X3) - 2*Y1*J
    FieldElement z3 = (z_ + h); z3.square_inplace(); z3 -= z1z1 + hh; // (Z1+H)^2 - Z1^2 - H^2

    x_ = x3; y_ = y3; z_ = z3; infinity_ = false; is_generator_ = false;
}

// ============================================================================
// Generator fixed-base multiplication (ESP32)
// Precomputes [1..15] × 2^(4i) × G in affine for i=0..31 (480 entries, 30KB)
// Turns k*G into ~60 mixed additions with NO doublings.
// One-time setup: ~50ms; steady-state: ~4ms per k*G
// ============================================================================
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
namespace {

struct GenAffine { FieldElement x, y; };

static GenAffine gen_fb_table[480];
static bool gen_fb_ready = false;

static void init_gen_fb_table() {
    if (gen_fb_ready) return;

    auto* z_orig = new FieldElement[480];
    auto* z_pfx  = new FieldElement[480];
    if (!z_orig || !z_pfx) { delete[] z_orig; delete[] z_pfx; return; }

    Point base = Point::generator();

    for (int w = 0; w < 32; w++) {
        // Compute [1..15]×base using doubling chain: 7 dbl + 7 add
        Point pts[15];
        pts[0]  = base;
        pts[1]  = base;  pts[1].dbl_inplace();                  // 2B
        pts[2]  = pts[1]; pts[2].add_inplace(base);             // 3B
        pts[3]  = pts[1]; pts[3].dbl_inplace();                 // 4B
        pts[4]  = pts[3]; pts[4].add_inplace(base);             // 5B
        pts[5]  = pts[2]; pts[5].dbl_inplace();                 // 6B
        pts[6]  = pts[5]; pts[6].add_inplace(base);             // 7B
        pts[7]  = pts[3]; pts[7].dbl_inplace();                 // 8B
        pts[8]  = pts[7]; pts[8].add_inplace(base);             // 9B
        pts[9]  = pts[4]; pts[9].dbl_inplace();                 // 10B
        pts[10] = pts[9];  pts[10].add_inplace(base);           // 11B
        pts[11] = pts[5];  pts[11].dbl_inplace();               // 12B
        pts[12] = pts[11]; pts[12].add_inplace(base);           // 13B
        pts[13] = pts[6];  pts[13].dbl_inplace();               // 14B
        pts[14] = pts[13]; pts[14].add_inplace(base);           // 15B

        for (int d = 0; d < 15; d++) {
            int idx = w * 15 + d;
            gen_fb_table[idx].x = pts[d].x_raw();
            gen_fb_table[idx].y = pts[d].y_raw();
            z_orig[idx] = pts[d].z_raw();
        }

        if (w < 31) {
            for (int d = 0; d < 4; d++) base.dbl_inplace();
        }
    }

    // Batch-to-affine: single inversion for all 480 entries
    z_pfx[0] = z_orig[0];
    for (int i = 1; i < 480; i++) z_pfx[i] = z_pfx[i - 1] * z_orig[i];

    FieldElement inv = z_pfx[479].inverse();

    for (int i = 479; i > 0; i--) {
        FieldElement z_inv = inv * z_pfx[i - 1];
        inv = inv * z_orig[i];
        FieldElement zi2 = z_inv.square();
        FieldElement zi3 = zi2 * z_inv;
        gen_fb_table[i].x = gen_fb_table[i].x * zi2;
        gen_fb_table[i].y = gen_fb_table[i].y * zi3;
    }
    {
        FieldElement zi2 = inv.square();
        FieldElement zi3 = zi2 * inv;
        gen_fb_table[0].x = gen_fb_table[0].x * zi2;
        gen_fb_table[0].y = gen_fb_table[0].y * zi3;
    }

    delete[] z_orig;
    delete[] z_pfx;
    gen_fb_ready = true;
}

inline std::uint8_t get_nybble(const Scalar& s, int pos) {
    int limb = pos / 16;
    int shift = (pos % 16) * 4;
    return static_cast<std::uint8_t>((s.limbs()[limb] >> shift) & 0xFu);
}

static Point gen_fixed_mul(const Scalar& k) {
    if (!gen_fb_ready) init_gen_fb_table();

    auto decomp = glv_decompose(k);

    static const FieldElement beta =
        FieldElement::from_bytes(glv_constants::BETA);

    Point result = Point::infinity();

    for (int w = 0; w < 32; w++) {
        std::uint8_t d1 = get_nybble(decomp.k1, w);
        if (d1 > 0) {
            const GenAffine& e = gen_fb_table[w * 15 + d1 - 1];
            FieldElement ey = decomp.k1_neg
                ? (FieldElement::zero() - e.y) : e.y;
            result.add_inplace(Point::from_affine(e.x, ey));
        }

        std::uint8_t d2 = get_nybble(decomp.k2, w);
        if (d2 > 0) {
            const GenAffine& e = gen_fb_table[w * 15 + d2 - 1];
            FieldElement px = e.x * beta;
            FieldElement py = decomp.k2_neg
                ? (FieldElement::zero() - e.y) : e.y;
            result.add_inplace(Point::from_affine(px, py));
        }
    }

    return result;
}

}  // anonymous namespace
#endif  // SECP256K1_PLATFORM_ESP32

Point Point::scalar_mul(const Scalar& scalar) const {
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
    if (is_generator_) {
        return scalar_mul_generator(scalar);
    }
#endif

#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
    // ESP32 only: generator uses precomputed fixed-base table (~4ms vs ~12ms)
    // STM32 skips this (64KB SRAM too tight for 30KB table)
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM)
    if (is_generator_) {
        return gen_fixed_mul(scalar);
    }
#endif

    // ---------------------------------------------------------------
    // Embedded: GLV decomposition + Shamir's trick
    // Splits 256-bit scalar into two ~128-bit half-scalars and processes
    // both streams simultaneously, halving the number of doublings.
    //   k*P = sign1·|k1|*P + sign2·|k2|*φ(P)
    //   where k = k1 + k2·λ (mod n), |k1|,|k2| ≈ √n
    // ---------------------------------------------------------------

    // Step 1: Decompose scalar
    GLVDecomposition decomp = glv_decompose(scalar);

    // Step 2: Handle k1 sign by negating base point before precomputation
    Point P_base = decomp.k1_neg ? this->negate() : *this;

    // Step 3: Compute wNAF for both half-scalars (stack-allocated)
    // w=4 for ~128-bit scalars: table_size=8, ~32 additions per stream
    constexpr unsigned glv_window = 4;
    constexpr int glv_table_size = (1 << (glv_window - 1));  // 8

    std::array<int32_t, 140> wnaf1_buf{}, wnaf2_buf{};
    std::size_t wnaf1_len = 0, wnaf2_len = 0;
    compute_wnaf_into(decomp.k1, glv_window,
                      wnaf1_buf.data(), wnaf1_buf.size(), wnaf1_len);
    compute_wnaf_into(decomp.k2, glv_window,
                      wnaf2_buf.data(), wnaf2_buf.size(), wnaf2_len);

    // Step 4: Precompute odd multiples [1, 3, 5, …, 15] for P
    std::array<Point, glv_table_size> tbl_P, tbl_phiP;

    tbl_P[0] = P_base;
    Point dbl_P = P_base;
    dbl_P.dbl_inplace();
    for (int i = 1; i < glv_table_size; i++) {
        tbl_P[i] = tbl_P[i - 1];
        tbl_P[i].add_inplace(dbl_P);
    }

    // Derive φ(P) table from P table using the endomorphism: φ(X:Y:Z) = (β·X:Y:Z)
    // This costs only 8 field muls vs 7 additions + 1 doubling (~10× cheaper)
    // Sign adjustment: tbl_P has k1 sign baked in; flip to k2 sign if different
    bool flip_phi = (decomp.k1_neg != decomp.k2_neg);
    for (int i = 0; i < glv_table_size; i++) {
        tbl_phiP[i] = apply_endomorphism(tbl_P[i]);
        if (flip_phi) tbl_phiP[i].negate_inplace();
    }

    // Step 5: Shamir's trick — one doubling per iteration, two lookups
    Point result = Point::infinity();
    std::size_t max_len = (wnaf1_len > wnaf2_len) ? wnaf1_len : wnaf2_len;

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        result.dbl_inplace();

        // k1 contribution
        if (static_cast<std::size_t>(i) < wnaf1_len) {
            int32_t d = wnaf1_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                result.add_inplace(tbl_P[(d - 1) / 2]);
            } else if (d < 0) {
                Point neg = tbl_P[(-d - 1) / 2];
                neg.negate_inplace();
                result.add_inplace(neg);
            }
        }

        // k2 contribution
        if (static_cast<std::size_t>(i) < wnaf2_len) {
            int32_t d = wnaf2_buf[static_cast<std::size_t>(i)];
            if (d > 0) {
                result.add_inplace(tbl_phiP[(d - 1) / 2]);
            } else if (d < 0) {
                Point neg = tbl_phiP[(-d - 1) / 2];
                neg.negate_inplace();
                result.add_inplace(neg);
            }
        }
    }

    return result;

#else
    // Desktop: wNAF with w=5 (GLV handled at higher level)
    // w=5 means 16 precomputed points, ~51 additions for 256-bit scalar
    constexpr unsigned window_width = 5;
    // No-alloc wNAF: write into stack buffer
    std::array<int32_t, 260> wnaf_buf{};
    std::size_t wnaf_len = 0;
    compute_wnaf_into(scalar, window_width, wnaf_buf.data(), wnaf_buf.size(), wnaf_len);
    
    // Precompute odd multiples: [1P, 3P, 5P, ..., 31P]
    constexpr int table_size = (1 << (window_width - 1)); // 2^4 = 16
    std::array<Point, table_size> precomp;
    
    precomp[0] = *this;  // 1P
    Point double_p = *this;        // Copy for in-place
    double_p.dbl_inplace();        // 2P in-place!
    
    for (int i = 1; i < table_size; i++) {
        precomp[i] = precomp[i-1];          // Copy previous into slot
        precomp[i].add_inplace(double_p);   // Make it next odd multiple in-place
    }
    
    Point result = Point::infinity();
    
    // Process wNAF digits from most significant to least significant
    for (int i = static_cast<int>(wnaf_len) - 1; i >= 0; --i) {
        result.dbl_inplace();  // Double in-place!
        
        int32_t digit = wnaf_buf[static_cast<std::size_t>(i)];
        if (digit > 0) {
            // Positive odd digit: 1, 3, 5, ..., 31
            int idx = (digit - 1) / 2;
            result.add_inplace(precomp[idx]);  // Add in-place!
        } else if (digit < 0) {
            // Negative odd digit: -1, -3, -5, ..., -31
            int idx = (-digit - 1) / 2;
            // Original faster path: temp copy + negate (O(1)) then mixed addition (7M)
            Point neg_point = precomp[idx];
            neg_point.negate_inplace();
            result.add_inplace(neg_point);
        }
        // digit == 0: skip (no addition needed)
    }
    
    return result;
#endif
}

// Step 1: Use existing GLV decomposition from K*G implementation
// Q * k = Q * k1 + φ(Q) * k2
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
Point Point::scalar_mul_precomputed_k(const Scalar& k) const {
    // Use the proven GLV decomposition from scalar_mul_generator
    auto decomp = split_scalar_glv(k);
    
    // Delegate to predecomposed version
    return scalar_mul_predecomposed(decomp.k1, decomp.k2, decomp.neg1, decomp.neg2);
}
#else
// ESP32: fall back to basic scalar_mul (no GLV optimization)
Point Point::scalar_mul_precomputed_k(const Scalar& k) const {
    return this->scalar_mul(k);
}
#endif

// Step 2: Runtime-only version - no decomposition overhead
// K decomposition is done once at startup, not per operation
Point Point::scalar_mul_predecomposed(const Scalar& k1, const Scalar& k2, 
                                       bool neg1, bool neg2) const {
#if !defined(SECP256K1_PLATFORM_ESP32) && !defined(ESP_PLATFORM) && !defined(SECP256K1_PLATFORM_STM32)
    // If both signs are positive, use fast Shamir's trick
    // Otherwise, fall back to separate computation (sign handling is complex)
    if (!neg1 && !neg2) {
        // Fast path: K = k1 + k2·λ (both positive)
        constexpr unsigned window_width = 4;
        auto wnaf1 = compute_wnaf(k1, window_width);
        auto wnaf2 = compute_wnaf(k2, window_width);
        return scalar_mul_precomputed_wnaf(wnaf1, wnaf2, false, false);
    }
#endif

    // Restore original variant: compute both, negate individually, then add
    Point phi_Q = apply_endomorphism(*this);
    Point term1 = this->scalar_mul(k1);
    Point term2 = phi_Q.scalar_mul(k2);
    if (neg1) term1.negate_inplace();
    if (neg2) term2.negate_inplace();
    term1.add_inplace(term2);
    return term1;
}

// Step 3: Shamir's trick with precomputed wNAF
// All K-related work (decomposition, wNAF) is done once
// Runtime only: φ(Q), tables, interleaved double-and-add
Point Point::scalar_mul_precomputed_wnaf(const std::vector<int32_t>& wnaf1,
                                          const std::vector<int32_t>& wnaf2,
                                          bool neg1, bool neg2) const {
    // Convert this point to Jacobian for internal operations
    JacobianPoint p = {x_, y_, z_, infinity_};
    
    // Compute φ(Q) - endomorphism (1 field multiplication)
    Point phi_Q = apply_endomorphism(*this);
    JacobianPoint phi_p = {phi_Q.x_, phi_Q.y_, phi_Q.z_, phi_Q.infinity_};
    
    // Determine table size from wNAF digits
    // For window w: digits are in range [-2^w+1, 2^w-1] odd
    // Table stores positive odd multiples: [1, 3, 5, ..., 2^w-1]
    // Table size = 2^(w-1)
    int max_digit = 0;
    for (auto d : wnaf1) {
        int abs_d = (d < 0) ? -d : d;
        if (abs_d > max_digit) max_digit = abs_d;
    }
    for (auto d : wnaf2) {
        int abs_d = (d < 0) ? -d : d;
        if (abs_d > max_digit) max_digit = abs_d;
    }
    
    // Table size needed to handle max_digit
    // For w=4: max_digit=15, table_size=8 (stores 1,3,5,...,15)
    // For w=5: max_digit=31, table_size=16 (stores 1,3,5,...,31)
    int table_size = (max_digit + 1) / 2;

    // Build precomputation tables for odd multiples in Jacobian (no heap alloc)
    constexpr int kMaxWindowBits = 7;
    constexpr int kMaxTableSize = (1 << (kMaxWindowBits - 1)); // 64
    constexpr int kMaxAll = kMaxTableSize * 2;                 // 128
    if (table_size > kMaxTableSize) {
        // Safety fallback: clamp to max supported table; digits beyond won't occur for w<=7
        table_size = kMaxTableSize;
    }

    std::array<JacobianPoint, kMaxTableSize> table_Q_jac{};
    std::array<JacobianPoint, kMaxTableSize> table_phi_Q_jac{};

    // Generate tables: [P, 3P, 5P, 7P, ...]
    table_Q_jac[0] = p;            // 1*Q
    table_phi_Q_jac[0] = phi_p;    // 1*φ(Q)

    JacobianPoint double_Q = jacobian_double(p);        // 2*Q
    JacobianPoint double_phi_Q = jacobian_double(phi_p); // 2*φ(Q)

    for (int i = 1; i < table_size; i++) {
        table_Q_jac[static_cast<std::size_t>(i)] = jacobian_add(table_Q_jac[static_cast<std::size_t>(i-1)], double_Q);
        table_phi_Q_jac[static_cast<std::size_t>(i)] = jacobian_add(table_phi_Q_jac[static_cast<std::size_t>(i-1)], double_phi_Q);
    }

    // Note: For maximum correctness parity with the slow path, use Jacobian-Jacobian addition
    // with the Jacobian precomputation tables directly (slightly slower than mixed-add but robust).
    
    // Shamir's trick: process both wNAF streams simultaneously
    // This is the key optimization - one doubling per iteration, not two
    JacobianPoint result = {FieldElement::zero(), FieldElement::one(), FieldElement::zero(), true};
    
    size_t max_len = std::max(wnaf1.size(), wnaf2.size());
    
    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        result = jacobian_double(result);
        
        // Add φ(Q) * k2 contribution using Jacobian-Jacobian addition
        if (i < static_cast<int>(wnaf2.size())) {
            int32_t digit2 = wnaf2[i];
            if (digit2 != 0) {
                bool is_neg = (digit2 < 0);
                // Apply global sign for k2 via XOR on per-digit sign
                if (neg2) is_neg = !is_neg;
                int idx = ((is_neg ? -digit2 : digit2) - 1) / 2;
                JacobianPoint add_p = table_phi_Q_jac[static_cast<std::size_t>(idx)];
                if (is_neg) {
                    add_p.y = FieldElement::zero() - add_p.y;
                }
                result = jacobian_add(result, add_p);
            }
        }

        // Add Q * k1 contribution using Jacobian-Jacobian addition
        if (i < static_cast<int>(wnaf1.size())) {
            int32_t digit1 = wnaf1[i];
            if (digit1 != 0) {
                bool is_neg = (digit1 < 0);
                // Apply global sign for k1 via XOR on per-digit sign
                if (neg1) is_neg = !is_neg;
                int idx = ((is_neg ? -digit1 : digit1) - 1) / 2;
                JacobianPoint add_p = table_Q_jac[static_cast<std::size_t>(idx)];
                if (is_neg) {
                    add_p.y = FieldElement::zero() - add_p.y;
                }
                result = jacobian_add(result, add_p);
            }
        }
    }

    return Point(result.x, result.y, result.z, result.infinity);
}

// Fixed K × Variable Q: Optimal performance for repeated K with different Q
// All K-dependent work is cached in KPlan (GLV decomposition + wNAF computation)
// Runtime: φ(Q), tables, Shamir's trick (if signs allow) or separate computation
Point Point::scalar_mul_with_plan(const KPlan& plan) const {
#if defined(SECP256K1_PLATFORM_ESP32) || defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_STM32)
    // Embedded: fallback to regular scalar_mul using stored k1
    return scalar_mul(plan.k1);
#else
    // Fast path: Interleaved Shamir using precomputed wNAF digits from plan
    return scalar_mul_precomputed_wnaf(plan.wnaf1, plan.wnaf2, plan.neg1, plan.neg2);
#endif
}

std::array<std::uint8_t, 33> Point::to_compressed() const {
    std::array<std::uint8_t, 33> out{};
    if (infinity_) {
        out.fill(0);
        return out;
    }
    // Compute affine coordinates with a single inversion
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv; // copy for in-place square
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv; // z_inv^3
    auto x_bytes = x_aff.to_bytes();
    auto y_bytes = y_aff.to_bytes();
    out[0] = (y_bytes[31] & 1U) ? 0x03 : 0x02;
    std::copy(x_bytes.begin(), x_bytes.end(), out.begin() + 1);
    return out;
}

std::array<std::uint8_t, 65> Point::to_uncompressed() const {
    std::array<std::uint8_t, 65> out{};
    if (infinity_) {
        out.fill(0);
        return out;
    }
    // Compute affine coordinates with a single inversion
    FieldElement z_inv = z_.inverse();
    FieldElement z_inv2 = z_inv; // copy for in-place square
    z_inv2.square_inplace();
    FieldElement x_aff = x_ * z_inv2;
    FieldElement y_aff = y_ * z_inv2 * z_inv; // z_inv^3
    auto x_bytes = x_aff.to_bytes();
    auto y_bytes = y_aff.to_bytes();
    out[0] = 0x04;
    std::copy(x_bytes.begin(), x_bytes.end(), out.begin() + 1);
    std::copy(y_bytes.begin(), y_bytes.end(), out.begin() + 33);
    return out;
}

} // namespace secp256k1::fast
