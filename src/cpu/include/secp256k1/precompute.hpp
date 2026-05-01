#ifndef EE840AA6_AA0C_4E9D_B58A_701AC4A267D0
#define EE840AA6_AA0C_4E9D_B58A_701AC4A267D0

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1::fast {


// No-alloc variant: write wNAF digits into caller-provided buffer.
// - Digits are written LSB-first at indices [0 .. out_len-1]
// - Valid window_bits range: [2, 16] (practically we use 4..7)
// - Throws std::runtime_error if output buffer is too small
void compute_wnaf_into(const Scalar& scalar,
                       unsigned window_bits,
                       int32_t* out,
                       std::size_t max,
                       std::size_t& out_len);
// Progress callback: (current_points, total_points, window_index, total_windows)
using ProgressCallback = void(*)(size_t, size_t, unsigned, unsigned);

struct FixedBaseConfig {
    unsigned window_bits = 18U;         // Default to 18-bit (511 MB cache, OPTIMAL: 11.755us)
    bool enable_glv = false;            // GLV disabled by default (decomposition overhead ~14us makes it 2x slower!)
    bool use_jsf = false;               // Use JSF-based Shamir for GLV (off by default)
    bool adaptive_glv = false;          // If true, automatically disable GLV for small window sizes
    unsigned glv_min_window_bits = 14U; // Minimum window_bits required to keep GLV enabled when adaptive_glv=true
    bool use_comb = false;              // Use comb method instead of windowed (smaller cache, L1-friendly)
    unsigned comb_width = 6U;           // Comb width in bits (5 or 6 recommended for L1/L2 cache)
    unsigned thread_count = 0U;         // 0 = auto-detect
    
    // Cache configuration
    bool use_cache = true;              // Enable cache system
    std::string cache_path{};           // Empty = auto-detect cache path from cache_dir
    // MSan note: std::string SSO bits are untracked with uninstrumented libc++.
    // Use this bool (plain scalar, MSan-clean) instead of cache_path.empty() checks
    // inside critical paths that run under MSan (e.g. ensure_built_locked).
    bool cache_path_set = false;        // true when cache_path was explicitly set
    std::string cache_dir = "";         // Default cache directory with all precomputed tables
    unsigned max_windows_to_load = 0U;  // Load all windows for optimal performance
    
    // Progress reporting
    ProgressCallback progress_callback = nullptr;  // Optional progress reporting

    // Auto-tune configuration (file-based, no environment variables)
    bool autotune = false;              // If true, run auto-tuning once on startup then rewrite config with best settings
    unsigned autotune_iters = 2000;     // Iterations per candidate during tuning
    unsigned autotune_min_w = 2;        // Minimum window_bits to consider
    unsigned autotune_max_w = 20;       // Maximum window_bits to consider (avoid huge caches)
    std::string autotune_log_path;      // Optional: write human-readable tuning report (e.g. "autotune.log")

    // Application-level integration (optional keys from config.ini)
    // These are not used by core precomputation logic but allow apps to pull
    // database path and external Bloom filter path from the same unified file.
    std::string db_path;                // e.g. R:\\SmallDB
    std::string bloom_filter_path;      // e.g. R:\\SmallDB\\bloom_full.blmf or "none"
    std::string fulldb_path;            // Secondary full key/value DB (used only when collision occurs)
};

struct ScalarDecomposition {
    Scalar k1;
    Scalar k2;
    bool neg1{false};
    bool neg2{false};
};

void configure_fixed_base(const FixedBaseConfig& config);
void ensure_fixed_base_ready();
bool fixed_base_ready();

// ----------------------------------------------------------------------------
// Library-level helpers to reduce per-app boilerplate
// ----------------------------------------------------------------------------
// Load FixedBase settings from a simple INI-style file and configure globally.
// Supported keys (case-insensitive):
//   cache_dir, cache_path, window_bits, enable_glv, use_jsf, use_cache,
//   max_windows, thread_count, use_comb, comb_width
// Lines starting with '#' or ';' are comments. Empty lines ignored.
// Example:
//   cache_dir=F:\\EccTables
//   window_bits=18
//   enable_glv=true
//   use_jsf=true
bool load_fixed_base_config_file(const std::string& path, FixedBaseConfig& out);

// Parse file and immediately apply via configure_fixed_base().
bool configure_fixed_base_from_file(const std::string& path);

// If env var SECP256K1_CONFIG is set, load that file and configure.
// Returns true if loaded and applied, false if env not set or load failed.
bool configure_fixed_base_from_env();

// Write a default INI with sensible values if file does not exist.
// Returns true if the file was created, false if already existed or on error.
bool write_default_fixed_base_config(const std::string& path);

// Ensure config file exists; if missing, create with defaults.
// Returns true if the file exists after this call (created or pre-existing).
bool ensure_fixed_base_config_file(const std::string& path);

// Auto: prefer env (SECP256K1_CONFIG), otherwise use "config.ini" in CWD.
// Creates a default file if missing, then loads and applies it.
bool configure_fixed_base_auto();

// ----------------------------------------------------------------------------
// Auto-tuning helpers
// ----------------------------------------------------------------------------
// Run a quick benchmark sweep across available fixed-base cache windows
// in the configured cache directory and pick the fastest configuration
// for this machine. It tries combinations of:
//   - window_bits: discovered from files like cache_w{bits}[ _glv].bin in cache_dir
//   - enable_glv:  true if GLV cache exists for that window, false if non-GLV exists
//   - use_jsf:     both true/false when GLV is enabled (JSF vs Shamir)
// It measures average time of scalar_mul_generator across 'iterations'
// and returns the best FixedBaseConfig in 'best_out'. 'report_out' (optional)
// receives a human-readable summary of tried candidates.
// Returns true on success, false if no candidates were found.
bool auto_tune_fixed_base(FixedBaseConfig& best_out,
                          std::string* report_out = nullptr,
                          unsigned iterations = 5000,
                          unsigned min_w = 2,
                          unsigned max_w = 30);

// Write a FixedBaseConfig to an INI file (overwrites existing).
// Returns true if write succeeded.
bool write_fixed_base_config(const std::string& path, const FixedBaseConfig& cfg);

// Convenience: run auto-tune and immediately write the resulting configuration
// to 'path' (e.g., "config.ini"). Returns true if tuning and write succeed.
bool auto_tune_and_write_config(const std::string& path,
                                unsigned iterations = 5000,
                                unsigned min_w = 2,
                                unsigned max_w = 30);

Point scalar_mul_generator(const Scalar& scalar);
ScalarDecomposition split_scalar_glv(const Scalar& scalar);

// GLV with pre-decomposed constants (for benchmarking when decomposition is done once)
Point scalar_mul_generator_glv_predecomposed(const Scalar& k1, const Scalar& k2, bool neg1, bool neg2);

// Batch fixed-base: compute results[i] = scalars[i] × G for all i.
// ONE mutex lock + ONE table warm-up for the entire batch — subsequent
// shamir_windowed_glv calls hit the warm L2/L3 cache.
// Thread-local digit scratch: no heap allocation in the hot path.
void batch_scalar_mul_generator(const Scalar* scalars, Point* results, std::size_t n);

// wNAF (width-w Non-Adjacent Form) computation
// Returns wNAF representation with non-adjacent digits
// window_bits: typically 4-6 for optimal performance
std::vector<int32_t> compute_wnaf(const Scalar& scalar, unsigned window_bits);

// Optimized scalar multiplication for arbitrary points (non-generator)
// Uses fixed-window method with precomputed multiples [Q, 3Q, 5Q, ..., (2^w-1)Q]
// window_bits: typically 4-6 for optimal trade-off (4=16 points, 5=32 points, 6=64 points)
// For variable-base multiplication (e.g., Q*k where Q changes frequently)
Point scalar_mul_arbitrary(const Point& base, const Scalar& scalar, unsigned window_bits = 5);

// Multi-scalar multiplication using Shamir's trick (Straus/Shamir algorithm)
// Computes k1*P + k2*Q efficiently by sharing point doublings
// Expected: 60-70% faster than two separate scalar_mul_arbitrary() calls
// Primary use case: ECDSA signature verification (s^-1*u*G + s^-1*r*PublicKey)
// window_bits: typically 4-5 for joint table (4=16 combinations, 5=64 combinations)
Point multi_scalar_mul(const Scalar& k1, const Point& P, 
                       const Scalar& k2, const Point& Q, 
                       unsigned window_bits = 4);

// ============================================================================
// Optimized API for CONSTANT SCALAR (K fixed, Q variable)
// ============================================================================
// Use case: Point navigation with fixed jump distances
//   - K is constant (reused many times)
//   - Q changes every operation
// 
// Performance: 336 us -> 190-260 us (1.3-1.8x speedup)
// 
// Usage:
//   1) ONCE (offline): PrecomputedScalar precomp = precompute_scalar_for_arbitrary(K);
//   2) MANY TIMES:     Point result = scalar_mul_arbitrary_precomputed(Q, precomp);

struct PrecomputedScalar {
    // GLV decomposition: K -> (k_1, k_2) where K == k_1 + lambda*k_2 (mod n)
    Scalar k1;
    Scalar k2;
    bool neg1;  // k_1 sign
    bool neg2;  // k_2 sign
    
    // Precomputed wNAF digits for k_1 and k_2
    std::vector<int32_t> wnaf1;  // wNAF(k_1)
    std::vector<int32_t> wnaf2;  // wNAF(k_2)
    
    unsigned window_bits;  // Window size used (typically 5)
    
    // Verify this was properly initialized
    bool is_valid() const { return window_bits >= 4 && window_bits <= 7; }
};

// Optimized precomputed scalar - ZERO runtime calculations + RLE compression!
// Uses Run-Length Encoding to skip consecutive zeros (doubles without adds)
struct PrecomputedScalarOptimized {
    // Each step: N doubles followed by 0-2 additions
    struct Step {
        uint16_t num_doubles;  // How many consecutive doubles before additions
        
        // Addition info for k_1 (Q table)
        uint8_t idx1;          // Point index [0..table_size-1], 0xFF = skip
        bool neg1;             // Negate Y coordinate?
        
        // Addition info for k_2 (psi(Q) table)
        uint8_t idx2;          // Point index [0..table_size-1], 0xFF = skip
        bool neg2;             // Negate Y coordinate?
        
        Step() : num_doubles(0), idx1(0xFF), neg1(false), idx2(0xFF), neg2(false) {}
    };
    
    Scalar k1, k2;
    bool neg1, neg2;
    std::vector<Step> steps;  // ~100-130 steps (instead of 256 iterations!)
    unsigned window_bits;
    
    bool is_valid() const { return window_bits >= 4 && window_bits <= 7 && !steps.empty(); }
};

// Precompute constant scalar K for repeated use with different points Q
// This performs GLV decomposition and wNAF generation ONCE
// Cost: ~14 us one-time overhead, amortized over many operations
// window_bits: typically 4-5 for optimal performance (4 recommended for speed)
PrecomputedScalar precompute_scalar_for_arbitrary(const Scalar& K, unsigned window_bits = 4);

// OPTIMIZED: Precompute with ALL calculations done once
// Eliminates ALL runtime conditionals and arithmetic in main loop!
// Additional cost: ~2 us one-time, but removes ~30% overhead from main loop
// Use this for MAXIMUM performance when K is truly constant
PrecomputedScalarOptimized precompute_scalar_optimized(const Scalar& K, unsigned window_bits = 4);

// Scalar multiplication with precomputed constant K
// Cost: NO decomposition overhead, only psi(Q) + 2D loop
// Expected: 190-260 us (1.3-1.8x faster than scalar_mul_arbitrary)
Point scalar_mul_arbitrary_precomputed(const Point& Q, const PrecomputedScalar& precomp);

// OPTIMIZED: Uses PrecomputedScalarOptimized for zero-overhead main loop
// Expected: 120-160 us with w=4 (2-3x faster than standard precomputed)
// Target with assembly field ops: 18-24 us (12x faster!)
Point scalar_mul_arbitrary_precomputed_optimized(const Point& Q, const PrecomputedScalarOptimized& precomp);

// NO-TABLE MODE: For single-use Q (eliminates table building overhead!)
// Uses direct +/-Q and +/-psi(Q) without precomputed tables
// Expected: ~199 us (saves 12 us from table building)
// Use when Q is used only once with constant K
Point scalar_mul_arbitrary_precomputed_notable(const Point& Q, const PrecomputedScalarOptimized& precomp);

// Cache management
bool save_precompute_cache(const std::string& path);
bool load_precompute_cache(const std::string& path, unsigned max_windows = 0);

} // namespace secp256k1::fast

#endif /* EE840AA6_AA0C_4E9D_B58A_701AC4A267D0 */
