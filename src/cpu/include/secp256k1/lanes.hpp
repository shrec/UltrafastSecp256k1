// ============================================================================
// lanes -- lane-parallel (vertical SIMD) field arithmetic: backend dispatch
// ============================================================================
//
// A lane backend holds one independent secp256k1 field element per SIMD lane,
// so one instruction stream advances several unrelated field computations at
// once (the building block for verifying several signatures together).
//
// PUBLIC DATA ONLY. Every lane backend is variable time and exists for
// verification paths, where all inputs are public. Never feed a private key,
// nonce, signing share or ECDH scalar through these types; secret-bearing code
// keeps using secp256k1::ct::*.
//
// Backends. The lane field types (namespace secp256k1::lanes::field, headers
// src/cpu/src/lanes/field_*.hpp, which document the limb-bound invariants) are
// compiled only into translation units built with that ISA's flags; nothing
// ISA-specific runs unless the dispatcher below hands its table out:
//
//   Ifma512x8     8 lanes, zmm, radix 2^52 x 5 limbs, EVEX vpmadd52luq/huq
//                 (needs AVX512F + AVX512VL + AVX512IFMA)
//   Ifma256x4     4 lanes, ymm, radix 2^52 x 5 limbs, EVEX vpmadd52luq/huq
//                 (needs AVX512VL + AVX512IFMA); only selected by UFSECP_LANES
//   AvxIfma256x4  4 lanes, ymm, radix 2^52 x 5 limbs, VEX vpmadd52luq/huq
//                 (AVX-IFMA; built only when the compiler accepts -mavxifma)
//   Avx2x4        4 lanes, ymm, radix 2^26 x 10 limbs, vpmuludq 32x32->64
//
// Default pick, from the MASKED secp256k1::cpu_features() view (so
// UFSECP_CPU_DISABLE=avx512 / ifma / avx_ifma / avx2 steer it):
//
//   Ifma512x8 > AvxIfma256x4 > Avx2x4 > None
//
// UFSECP_LANES=off|avx2|avx_ifma|ifma256|ifma512 forces one backend for
// benchmarking, but only among the available ones: a request for a backend
// this build or CPU lacks falls back to the default and the reason is
// reported (lane_selection_reason(), cpu_hwinfo, one stderr line). It can
// never select an instruction set the CPU does not have.
//
// The selection is computed once (first call) and cached, like the field mul
// dispatcher: set_cpu_disable_mask() after that first call does not change it.
// ============================================================================

#ifndef SECP256K1_LANES_HPP
#define SECP256K1_LANES_HPP

#include <cstddef>
#include <cstdint>

namespace secp256k1 {

struct CpuFeatures;

namespace lanes {

enum class LaneBackend : std::uint8_t {
    None         = 0,
    Avx2x4       = 1,
    AvxIfma256x4 = 2,
    Ifma256x4    = 3,
    Ifma512x8    = 4,
};

constexpr unsigned kLaneBackendCount = 5;

// Bit for `b` in the masks returned by available_lane_backends() and friends.
constexpr unsigned lane_backend_bit(LaneBackend b) noexcept {
    return 1u << static_cast<unsigned>(b);
}

// Short name, identical to the UFSECP_LANES token: "off", "avx2", "avx_ifma",
// "ifma256", "ifma512".
const char* lane_backend_name(LaneBackend b) noexcept;

// Number of field elements one vector of this backend holds (0 for None).
unsigned lane_backend_width(LaneBackend b) noexcept;

// Backends whose code is compiled into this library (None always is).
unsigned compiled_lane_backends() noexcept;

// Pure: does a CPU with features `f` support backend `b`? (None: always.)
bool lane_backend_supported(LaneBackend b, const CpuFeatures& f) noexcept;

// Compiled AND supported by the masked cpu_features() view (None always set).
unsigned available_lane_backends() noexcept;

// Pure: the default pick among `available_mask`.
LaneBackend default_lane_backend(unsigned available_mask) noexcept;

// Outcome of resolving UFSECP_LANES against the available backends.
enum class LaneSelectionReason : std::uint8_t {
    Default        = 0,  // UFSECP_LANES unset or empty
    Forced         = 1,  // UFSECP_LANES named an available backend
    Unavailable    = 2,  // UFSECP_LANES named a backend this build/CPU lacks
    UnknownToken   = 3,  // UFSECP_LANES value not recognised
};

struct LaneSelection {
    LaneBackend backend   = LaneBackend::None;  // what will run
    LaneBackend requested = LaneBackend::None;  // what UFSECP_LANES asked for (if recognised)
    LaneSelectionReason reason = LaneSelectionReason::Default;
};

// Pure: resolve an UFSECP_LANES value (nullptr = unset) against a mask of
// available backends. Never returns a backend outside `available_mask`.
LaneSelection resolve_lane_selection(const char* env_value, unsigned available_mask) noexcept;

// Process-wide cached selection (reads UFSECP_LANES and cpu_features() once).
const LaneSelection& lane_selection() noexcept;
LaneBackend selected_lane_backend() noexcept;

// One-line human-readable explanation of a selection reason.
const char* lane_selection_reason(LaneSelectionReason r) noexcept;

// ---- Array-wise test / benchmark kernels -----------------------------------
//
// Every element is four uint64 limbs in the fast::FieldElement layout (limb 0
// least significant). Inputs may be any 256-bit value (values >= p are fine);
// outputs are canonical (< p). `n` may be any count, including one that is not
// a multiple of the lane width. These kernels exist for the differential test
// and the --bench mode; production code templates over the lane types directly.
//
// Every kernel checks the limb-bound invariant (see src/cpu/src/lanes/) on each
// mul/sqr input and on each op result; violations are counted, not trapped,
// and read back with bound_violations().
struct LaneFieldKernels {
    LaneBackend backend;
    unsigned lanes;           // elements per vector
    unsigned limb_bits;       // internal radix (52 or 26)
    unsigned limb_count;      // internal limbs (5 or 10)
    std::uint64_t top_limb_max;  // inclusive bound of the top internal limb in weak normal form

    using Fn1  = void (*)(std::uint64_t* r, const std::uint64_t* a, std::size_t n);
    using Fn2  = void (*)(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b, std::size_t n);

    Fn1 load_store;           // r = a mod p (load, normalize, store)
    Fn1 normalize;            // r = a mod p, via a non-canonical a + 4p intermediate
    Fn2 add;
    Fn2 sub;
    Fn2 mul;
    Fn1 sqr;
    Fn1 negate;
    void (*mul_small)(std::uint64_t* r, const std::uint64_t* a, std::uint32_t k, std::size_t n);
    Fn1 inverse;              // a^(p-2); 0 -> 0
    void (*is_zero)(std::uint8_t* r, const std::uint64_t* a, std::size_t n);
    void (*equal)(std::uint8_t* r, const std::uint64_t* a, const std::uint64_t* b, std::size_t n);
    // r = mask[i] ? a : b
    void (*select)(std::uint64_t* r, const std::uint8_t* mask, const std::uint64_t* a,
                   const std::uint64_t* b, std::size_t n);
    // Mixed chain per lane: x = a, y = b, then for each step apply ops[s]:
    //   0: x = x + y   1: x = x - y   2: x = x * y   3: x = x^2
    //   4: x = -x      5: x = 3 * x   6: y = y + x   7: y = x * y
    // r = x (canonical).
    void (*chain)(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b,
                  const std::uint8_t* ops, std::size_t steps, std::size_t n);
    // Internal-form inputs: limb_count limbs per element, each below
    // 2^limb_bits except the top limb, which is <= top_limb_max (the weak
    // normal bound every op guarantees). Output canonical 4x64.
    void (*mul_limbs)(std::uint64_t* r, const std::uint64_t* a_limbs,
                      const std::uint64_t* b_limbs, std::size_t n);
    void (*sqr_limbs)(std::uint64_t* r, const std::uint64_t* a_limbs, std::size_t n);
    void (*normalize_limbs)(std::uint64_t* r, const std::uint64_t* a_limbs, std::size_t n);
    // Benchmark: load once, x = op(x, y) `iters` times (op 0 = mul, 1 = sqr,
    // 2 = add), store. Used only by test_field_lanes --bench.
    void (*bench)(int op, std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b,
                  std::size_t n, std::size_t iters);
    std::uint64_t (*bound_violations)();
};

// Kernels of backend `b` when it is available (compiled and supported by the
// masked CPU view); nullptr otherwise. Calling through a non-null result is
// always safe on this CPU.
const LaneFieldKernels* lane_field_kernels(LaneBackend b) noexcept;

} // namespace lanes
} // namespace secp256k1

#endif // SECP256K1_LANES_HPP
