#ifndef C870F4A3_192C_4B96_9AE6_497D1885C5D9
#define C870F4A3_192C_4B96_9AE6_497D1885C5D9

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
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

// Platform-optimal default GLV window width for k*P (scalar_mul_with_plan).
//
// Larger window = fewer point additions in the wNAF loop, but larger precompute
// table (2^(w-2) entries, each requiring a mixed add + z-ratio tracking).
//
// Tradeoff by platform (BIP-352 pipeline benchmark, 10K ops, median):
//   w=4: table=4 entries, ~33 adds per 128-bit GLV half-scalar
//   w=5: table=8 entries, ~26 adds per 128-bit GLV half-scalar
//   w=6: table=16 entries, ~21 adds (diminishing returns, precompute dominates)
//
// On in-order / narrow OoO cores (RISC-V U74, ARM Cortex-A55) where point_add
// is expensive relative to precompute, w=5 saves more than it costs.
// On wide OoO x86-64 with fast MULX, the tradeoff is roughly neutral for
// single k*P but w=5 still wins the full pipeline.
//
// Override at call site: KPlan::from_scalar(k, 6) for batch-heavy workloads
// where precompute is amortized across many points.
#if defined(SECP256K1_GLV_WINDOW_WIDTH)
  // CMake override: -DSECP256K1_GLV_WINDOW_WIDTH=6
  // Or direct compiler flag: -DSECP256K1_GLV_WINDOW_WIDTH=6
  static_assert(SECP256K1_GLV_WINDOW_WIDTH >= 4 && SECP256K1_GLV_WINDOW_WIDTH <= 7,
                "SECP256K1_GLV_WINDOW_WIDTH must be in [4,7]");
  inline constexpr uint8_t kDefaultGlvWindow = SECP256K1_GLV_WINDOW_WIDTH;
#elif defined(__riscv) || defined(__aarch64__) || defined(_M_ARM64)
  inline constexpr uint8_t kDefaultGlvWindow = 5;
#elif defined(__x86_64__) || defined(_M_X64)
  inline constexpr uint8_t kDefaultGlvWindow = 5;
#else
  inline constexpr uint8_t kDefaultGlvWindow = 4;  // ESP32, WASM, unknown
#endif

// Fixed K x Variable Q optimization plan
// Caches all K-dependent work: GLV decomposition + wNAF computation
// Use this when you need to multiply many different points Q by the same scalar K
// Maximum wNAF buffer length for a 256-bit scalar: 256 bits + 1 extra digit + 3 padding
constexpr std::size_t kWnafBufLen = 260;

struct KPlan {
    uint8_t window_width;                   // wNAF window size (see kDefaultGlvWindow)
    Scalar k1;                              // Decomposed scalar k1
    Scalar k2;                              // Decomposed scalar k2
    // wNAF digits stored in fixed-size stack buffers — no heap allocation per plan
    std::array<int32_t, kWnafBufLen> wnaf1{};
    std::size_t wnaf1_len{0};
    std::array<int32_t, kWnafBufLen> wnaf2{};
    std::size_t wnaf2_len{0};
    bool neg1;                              // Sign flag for k1
    bool neg2;                              // Sign flag for k2

    // Factory: Create plan from scalar K
    // w: wNAF window width (default: platform-optimal kDefaultGlvWindow)
    static KPlan from_scalar(const Scalar& k, uint8_t w = kDefaultGlvWindow);
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
    // Raw-pointer overload — no heap access, used by KPlan hot path
    Point scalar_mul_precomputed_wnaf(const int32_t* wnaf1, std::size_t len1,
                                       const int32_t* wnaf2, std::size_t len2,
                                       bool neg1, bool neg2) const;
    
    // Fixed K x Variable Q: Use precomputed KPlan for maximum speed
    // All K-dependent work (GLV + wNAF) is cached in the plan
    // Runtime only: phi(Q), table generation, Shamir's trick
    Point scalar_mul_with_plan(const KPlan& plan) const;

    // Jacobian output variant: identical to scalar_mul() but skips the final
    // normalize(). Result has z_one_=false (Jacobian coordinates, Z≠1).
    // Use batch_normalize / batch_to_compressed / batch_x_only_bytes to convert
    // N results to affine with ONE shared field inversion (Montgomery's trick).
    // Saves ~500 ns/call when N points are processed together.
    // Note: scalar_mul_with_plan() already returns Jacobian — no _jacobian variant needed.
    Point scalar_mul_jacobian(const Scalar& scalar) const;

    // Negation: returns the opposite point on the curve
    Point negate() const;  // -(x, y) = (x, -y)
    
    // Fast increment/decrement by generator (optimized with precomputed -G)
    Point next() const;  // this + G (returns new Point)
    Point prev() const;  // this - G (returns new Point)
    
    // In-place mutable versions (modify this object directly)
    // In-place variants: modify this directly (same perf as immutable next/prev)
    void next_inplace();  // this += G (modifies this)
    void prev_inplace();  // this -= G (modifies this)
    void add_inplace(const Point& other);  // this += other (modifies this, no allocation)
    void sub_inplace(const Point& other);  // this -= other (modifies this, no allocation)
    // Branchless mixed-add against affine point (z=1): avoids runtime checks
    void add_mixed_inplace(const FieldElement& ax, const FieldElement& ay);
    void sub_mixed_inplace(const FieldElement& ax, const FieldElement& ay);
#if defined(SECP256K1_FAST_52BIT)
    // FE52-native mixed-add: avoids FE52->FE->FE52 roundtrip in hot loops.
    // Used by effective-affine Strauss MSM where precomp is stored as FE52.
    void add_mixed52_inplace(const FieldElement52& ax, const FieldElement52& ay);
#endif
    void dbl_inplace();   // this = 2*this (modifies this, no allocation)
    void negate_inplace(); // this = -this (modifies this, no allocation)
    // Optimized repeated addition by a fixed affine point (z=1)
    void add_affine_constant_inplace(const FieldElement& ax, const FieldElement& ay);

    // Y-parity check (single inversion, no full serialization)
    bool has_even_y() const;

    // Combined: returns (x_bytes, y_is_odd) with a single field inversion
    std::pair<std::array<uint8_t, 32>, bool> x_bytes_and_parity() const;

    // Fast x-only: 32-byte big-endian x-coordinate (no Y recovery).
    // Saves one multiply vs x_bytes_and_parity() by skipping Z^(-3)*Y.
    // Use when only x is needed (e.g. BIP-352 SHA-256 input, BIP-340 x-only).
    std::array<uint8_t, 32> x_only_bytes() const;

    // Batch scalar mul: fixed K (from KPlan) × N variable points, N independent results.
    // All points share the same wNAF (from plan), so:
    //   (1) Tables for all N points built and batch-inverted together (1 field_inv per chunk).
    //   (2) Shared wNAF loop processes chunk_size accumulators in lockstep.
    //   (3) Results stored as lazy-Jacobian Points — pass to batch_to_compressed / batch_x_only_bytes.
    // Chunked internally (chunk_size ≈ 2048) to keep the working set in L2/L3 cache.
    // Fallback to per-point scalar_mul_with_plan on non-FE52 or degenerate inputs.
    // Expected speedup over N × scalar_mul_with_plan: ~15–25% on Stage 1 latency.
    static void batch_scalar_mul_fixed_k(const KPlan& plan,
                                         const Point* pts,
                                         size_t n,
                                         Point* results);

    // Batch normalize: convert N Jacobian points to affine with ONE inversion
    // via Montgomery's trick. Cost: 1 inversion + 3(N-1) multiplications.
    // For N=2048: ~9.5 ns/point vs ~1000 ns/point individually.
    // out_x, out_y: output affine coordinates (caller-owned, size >= n).
    // Skips infinity points (leaves output zero-filled).
    static void batch_normalize(const Point* points, size_t n,
                                FieldElement* out_x, FieldElement* out_y);

    // Batch to_compressed: serialize N Jacobian points using ONE inversion.
    // out: caller-owned array of 33-byte compressed pubkeys, size >= n.
    static void batch_to_compressed(const Point* points, size_t n,
                                    std::array<uint8_t, 33>* out);

    // Batch x_only_bytes: extract N x-coordinates using ONE inversion.
    // out: caller-owned array of 32-byte x coords, size >= n.
    static void batch_x_only_bytes(const Point* points, size_t n,
                                   std::array<uint8_t, 32>* out);

    // Normalize: convert Jacobian -> affine (Z=1) with ONE field inversion.
    // After this call, all serialization methods become O(1) byte copies.
    // Called automatically by scalar_mul/generator_mul/dual_scalar_mul.
    void normalize();
    bool is_normalized() const noexcept { return z_one_; }

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
    bool z_one_ = false;  // true when Z == 1 (point is affine-normalized)
};

// Self-test: Verify arithmetic correctness with known test vectors
// Returns true if all tests pass, false otherwise
// Run this after any code changes to ensure math is correct!
// Set verbose=true to see detailed output for each test
bool Selftest(bool verbose = false);

} // namespace secp256k1::fast


#endif /* C870F4A3_192C_4B96_9AE6_497D1885C5D9 */
