// ============================================================================
// cpu_features -- one cached, thread-safe view of the x86-64 CPU
// ============================================================================
//
// Single source of truth for runtime CPU dispatch. Replaces the scattered
// per-file CPUID probes (field_asm.cpp, hash_accel.cpp, bench_unified.cpp).
//
//   * decode_cpu_features() is a pure function over an injectable CpuidSource,
//     so tests can feed synthetic CPUID / XGETBV tables.
//   * cpu_features_raw() is what the silicon (and OS) report.
//   * cpu_features() is the MASKED view every dispatcher reads: the raw view
//     with the features named by UFSECP_CPU_DISABLE / set_cpu_disable_mask()
//     turned off.
//
// The mask can only turn features OFF. It can never turn a feature on: forcing
// a feature the CPU lacks would execute an unsupported instruction (SIGILL).
//
// AVX / AVX-512 flags count only when the OS saves that register state
// (XGETBV XCR0: bits 1-2 for AVX, plus bits 5-7 for AVX-512).
//
// Non-x86 targets compile to an all-false view with vendor == Other.
// ============================================================================

#ifndef SECP256K1_CPU_FEATURES_HPP
#define SECP256K1_CPU_FEATURES_HPP

#include <cstdint>
#include <string>

namespace secp256k1 {

enum class CpuVendor : std::uint8_t {
    Other = 0,
    Intel = 1,
    AMD   = 2,
};

struct CpuFeatures {
    CpuVendor vendor = CpuVendor::Other;
    std::uint32_t family   = 0;   // display family (base + extended when base == 0xF)
    std::uint32_t model    = 0;   // display model   (extended model folded in for family 6/0xF)
    std::uint32_t stepping = 0;
    char brand[49] = {0};         // CPUID 0x80000002..4, leading/trailing blanks trimmed

    // Instruction-set flags. AVX-class flags are true only when the OS saves the state.
    bool avx2       = false;
    bool bmi1       = false;
    bool bmi2       = false;
    bool adx        = false;
    bool sha_ni     = false;
    bool avx512f    = false;
    bool avx512vl   = false;
    bool avx512ifma = false;      // CPUID.(EAX=7,ECX=0):EBX[21]
    bool avx_ifma   = false;      // CPUID.(EAX=7,ECX=1):EAX[23] (VEX-encoded, no AVX-512 needed)
    bool avx_vnni   = false;      // CPUID.(EAX=7,ECX=1):EAX[4]

    // Cache sizes in bytes (0 = unknown). Intel: CPUID leaf 4. AMD: leaf 0x8000001D.
    // On AMD, L3 is the size shared by one core complex, as reported by the CPU.
    std::uint32_t l2_cache_bytes = 0;
    std::uint32_t l3_cache_bytes = 0;
};

// ---- Injectable CPUID / XGETBV source --------------------------------------

struct CpuidRegs {
    std::uint32_t eax = 0;
    std::uint32_t ebx = 0;
    std::uint32_t ecx = 0;
    std::uint32_t edx = 0;
};

class CpuidSource {
public:
    virtual ~CpuidSource() = default;
    virtual CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) const = 0;
    virtual std::uint64_t xgetbv(std::uint32_t index) const = 0;
};

// The real CPU. On non-x86 targets every leaf reads as zero (vendor == Other).
const CpuidSource& native_cpuid_source() noexcept;

// Pure decode: no globals, no environment, no mask.
CpuFeatures decode_cpu_features(const CpuidSource& src);

// ---- Mask-down override ------------------------------------------------------

namespace cpu_mask {
constexpr std::uint32_t kAvx2      = 1u << 0;  // also clears every vector flag that depends on AVX2
constexpr std::uint32_t kBmi2      = 1u << 1;
constexpr std::uint32_t kAdx       = 1u << 2;
constexpr std::uint32_t kShaNi     = 1u << 3;
constexpr std::uint32_t kAvx512    = 1u << 4;  // avx512f, avx512vl and avx512ifma
constexpr std::uint32_t kIfma      = 1u << 5;  // avx512ifma only
constexpr std::uint32_t kAvxIfma   = 1u << 6;
constexpr std::uint32_t kAll       = (1u << 7) - 1u;
} // namespace cpu_mask

// Parse a comma list (avx2,bmi2,adx,sha_ni,avx512,ifma,avx_ifma; case-insensitive,
// blanks ignored). Unknown tokens are skipped (and appended to *unknown when
// given): an unknown name can never enable anything.
std::uint32_t parse_cpu_disable_list(const char* list, std::string* unknown = nullptr);

// Pure: returns `in` with the masked features turned off. Never turns one on.
CpuFeatures apply_cpu_disable_mask(const CpuFeatures& in, std::uint32_t mask) noexcept;

// ---- Process-wide cached views ----------------------------------------------

// What the CPU and OS report. Cached; thread-safe.
const CpuFeatures& cpu_features_raw();

// The view every dispatcher uses: raw view minus (UFSECP_CPU_DISABLE | API mask).
// Cached; thread-safe. The returned reference stays valid for the process lifetime
// (a later mask change publishes a new snapshot; it never mutates an old one).
const CpuFeatures& cpu_features();

// Set the API part of the mask (replaces the previous API mask; the environment
// part is always kept, so the API cannot undo UFSECP_CPU_DISABLE). For tests and
// benchmarks. Dispatchers that cache their choice in a function-local static
// (e.g. the field mul path) only see a mask applied before their first call.
void set_cpu_disable_mask(std::uint32_t mask);

// The effective mask (environment | API).
std::uint32_t cpu_disable_mask();

// Short name of a vendor ("Intel", "AMD", "other").
const char* cpu_vendor_name(CpuVendor v) noexcept;

} // namespace secp256k1

#endif // SECP256K1_CPU_FEATURES_HPP
