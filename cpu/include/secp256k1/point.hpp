#ifndef C870F4A3_192C_4B96_9AE6_497D1885C5D9
#define C870F4A3_192C_4B96_9AE6_497D1885C5D9

#include <string>
#include <utility>
#include "field.hpp"
#include "scalar.hpp"

// On 5x52-capable platforms, Point stores FieldElement52 internally
// for zero-conversion-overhead point arithmetic.
// FE52 native storage: only on 64-bit platforms with __int128
// Excluded on Emscripten/WASM: wasm32 emulates __int128 via compiler intrinsics,
// which is correct but gives no speed benefit over 4x64 FieldElement.
// The 52-bit dual_scalar_mul_gen_point also builds huge static tables (8192 entries)
// that are unnecessary for WASM targets.
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_PLATFORM_ESP32) && !defined(SECP256K1_PLATFORM_STM32) && !defined(__EMSCRIPTEN__)
  #ifndef SECP256K1_FAST_52BIT
    #define SECP256K1_FAST_52BIT 1
  #endif
  #include "field_52.hpp"
#endif

namespace secp256k1::fast {

// Fixed K x Variable Q optimization plan
// Caches all K-dependent work: GLV decomposition + wNAF computation
// Use this when you need to multiply many different points Q by the same scalar K
struct KPlan {
    uint8_t window_width;           // wNAF window size (typically 4 or 5)
    Scalar k1;                       // Decomposed scalar k1
    Scalar k2;                       // Decomposed scalar k2
    std::vector<int32_t> wnaf1;     // Precomputed wNAF for k1
    std::vector<int32_t> wnaf2;     // Precomputed wNAF for k2
    bool neg1;                       // Sign flag for k1
    bool neg2;                       // Sign flag for k2
    
    // Factory: Create plan from scalar K
    static KPlan from_scalar(const Scalar& k, uint8_t w = 4);
};

class Point {
public:
    Point();

    static Point generator();
    static Point infinity();
    static Point from_affine(const FieldElement& x, const FieldElement& y);
    
    // Developer-friendly: Create from hex strings (64 hex chars each)
    // Example: Point::from_hex("09af57f4...", "0947e4f9...")
    static Point from_hex(const std::string& x_hex, const std::string& y_hex);

    FieldElement x() const;
    FieldElement y() const;
    bool is_infinity() const noexcept { return infinity_; }
    bool is_gen() const noexcept { return is_generator_; }
    
    // Split x-coordinate helpers for split-keys database format
    // x_first_half(): returns first 16 bytes of x-coordinate
    // x_second_half(): returns last 16 bytes of x-coordinate
    std::array<uint8_t, 16> x_first_half() const;
    std::array<uint8_t, 16> x_second_half() const;
    
    // Direct access to Jacobian coordinates (for batch processing)
#if defined(SECP256K1_FAST_52BIT)
    FieldElement X() const noexcept;
    FieldElement Y() const noexcept;
    FieldElement z() const noexcept;
    // Direct access to 5x52 internals (for hot paths)
    const FieldElement52& X52() const noexcept { return x_; }
    const FieldElement52& Y52() const noexcept { return y_; }
    const FieldElement52& Z52() const noexcept { return z_; }
#else
    const FieldElement& X() const noexcept { return x_; }
    const FieldElement& Y() const noexcept { return y_; }
    const FieldElement& z() const noexcept { return z_; }
#endif

    Point add(const Point& other) const;
    Point dbl() const;
    Point scalar_mul(const Scalar& scalar) const;
    
    // Optimized: Q * K where K is precomputed constant
    // This will use GLV decomposition and precomputed tables
    Point scalar_mul_precomputed_k(const Scalar& k) const;
    
    // Optimized: Q * K with pre-decomposed K (k1, k2, signs)
    // Use this when K decomposition is done once at startup
    // Runtime only computes: Q*k1 + phi(Q)*k2
    Point scalar_mul_predecomposed(const Scalar& k1, const Scalar& k2, 
                                    bool neg1, bool neg2) const;
    
    // Optimized: Q * K with precomputed wNAF digits
    // This is the fastest version - all K-related work is done at compile time
    // Runtime only does: table generation + interleaved addition
    Point scalar_mul_precomputed_wnaf(const std::vector<int32_t>& wnaf1,
                                       const std::vector<int32_t>& wnaf2,
                                       bool neg1, bool neg2) const;
    
    // Fixed K x Variable Q: Use precomputed KPlan for maximum speed
    // All K-dependent work (GLV + wNAF) is cached in the plan
    // Runtime only: phi(Q), table generation, Shamir's trick
    Point scalar_mul_with_plan(const KPlan& plan) const;
    
    // Negation: returns the opposite point on the curve
    Point negate() const;  // -(x, y) = (x, -y)
    
    // Fast increment/decrement by generator (optimized with precomputed -G)
    Point next() const;  // this + G (returns new Point)
    Point prev() const;  // this - G (returns new Point)
    
    // In-place mutable versions (modify this object directly)
    // ~12% faster than immutable next() due to no memory allocation
    void next_inplace();  // this += G (modifies this)
    void prev_inplace();  // this -= G (modifies this)
    void add_inplace(const Point& other);  // this += other (modifies this, no allocation)
    void sub_inplace(const Point& other);  // this -= other (modifies this, no allocation)
    // Branchless mixed-add against affine point (z=1): avoids runtime checks
    void add_mixed_inplace(const FieldElement& ax, const FieldElement& ay);
    void sub_mixed_inplace(const FieldElement& ax, const FieldElement& ay);
    void dbl_inplace();   // this = 2*this (modifies this, no allocation)
    void negate_inplace(); // this = -this (modifies this, no allocation)
    // Optimized repeated addition by a fixed affine point (z=1)
    void add_affine_constant_inplace(const FieldElement& ax, const FieldElement& ay);

    // Y-parity check (single inversion, no full serialization)
    bool has_even_y() const;

    // Combined: returns (x_bytes, y_is_odd) with a single field inversion
    std::pair<std::array<uint8_t, 32>, bool> x_bytes_and_parity() const;

    // Dual scalar multiplication: a*G + b*P (4-stream GLV Shamir)
    // Much faster than separate generator_mul(a) + scalar_mul(b) + add
    static Point dual_scalar_mul_gen_point(const Scalar& a, const Scalar& b, const Point& P);

    std::array<std::uint8_t, 33> to_compressed() const;
    std::array<std::uint8_t, 65> to_uncompressed() const;

#if defined(SECP256K1_FAST_52BIT)
    FieldElement x_raw() const noexcept;
    FieldElement y_raw() const noexcept;
    FieldElement z_raw() const noexcept;
#else
    const FieldElement& x_raw() const noexcept { return x_; }
    const FieldElement& y_raw() const noexcept { return y_; }
    const FieldElement& z_raw() const noexcept { return z_; }
#endif

    static Point from_jacobian_coords(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity);
#if defined(SECP256K1_FAST_52BIT)
    // Zero-conversion factory: constructs Point directly from FE52 Jacobian coords
    static Point from_jacobian52(const FieldElement52& x, const FieldElement52& y, const FieldElement52& z, bool infinity);
    // Zero-conversion affine construction: (x, y, z=1) directly in FE52
    static Point from_affine52(const FieldElement52& x, const FieldElement52& y);
#endif

private:
    Point(const FieldElement& x, const FieldElement& y, const FieldElement& z, bool infinity);
#if defined(SECP256K1_FAST_52BIT)
    // Zero-conversion constructor: directly initializes FE52 members
    Point(const FieldElement52& x, const FieldElement52& y, const FieldElement52& z, bool infinity, bool is_gen);

    // Convert z_ (FE52) -> normalized FieldElement + check for zero.
    // Returns true if z is nonzero (normal case); false if z is zero.
    // On true: out_z_fe contains the normalized 4x64 FieldElement.
    // Used by x(), y(), to_compressed(), to_uncompressed(), has_even_y(),
    // x_bytes_and_parity() to avoid duplicating the defensive Z=0 guard.
    bool z_fe_nonzero(FieldElement& out_z_fe) const noexcept;
#endif

#if defined(SECP256K1_FAST_52BIT)
    FieldElement52 x_;
    FieldElement52 y_;
    FieldElement52 z_;
#else
    FieldElement x_;
    FieldElement y_;
    FieldElement z_;
#endif
    bool infinity_;
    bool is_generator_;
};

// Self-test: Verify arithmetic correctness with known test vectors
// Returns true if all tests pass, false otherwise
// Run this after any code changes to ensure math is correct!
// Set verbose=true to see detailed output for each test
bool Selftest(bool verbose = false);

} // namespace secp256k1::fast


#endif /* C870F4A3_192C_4B96_9AE6_497D1885C5D9 */
