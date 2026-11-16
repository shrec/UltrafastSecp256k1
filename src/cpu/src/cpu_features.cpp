// ============================================================================
// cpu_features -- decode CPUID/XGETBV into one cached, maskable feature view
// ============================================================================
// See secp256k1/cpu_features.hpp for the contract.
// ============================================================================

#include "secp256k1/cpu_features.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define SECP256K1_CPUF_X86 1
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#endif

namespace secp256k1 {

// ============================================================================
// Native CPUID / XGETBV source
// ============================================================================

namespace {

class NativeCpuidSource final : public CpuidSource {
public:
    CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) const override {
        CpuidRegs r;
#if defined(SECP256K1_CPUF_X86)
    #if defined(_MSC_VER)
        int info[4] = {0, 0, 0, 0};
        __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
        r.eax = static_cast<std::uint32_t>(info[0]);
        r.ebx = static_cast<std::uint32_t>(info[1]);
        r.ecx = static_cast<std::uint32_t>(info[2]);
        r.edx = static_cast<std::uint32_t>(info[3]);
    #else
        unsigned int a = 0, b = 0, c = 0, d = 0;
        __cpuid_count(leaf, subleaf, a, b, c, d);
        r.eax = a; r.ebx = b; r.ecx = c; r.edx = d;
    #endif
#else
        (void)leaf; (void)subleaf;
#endif
        return r;
    }

    std::uint64_t xgetbv(std::uint32_t index) const override {
#if defined(SECP256K1_CPUF_X86)
    #if defined(_MSC_VER)
        return static_cast<std::uint64_t>(_xgetbv(index));
    #else
        // Raw instruction: avoids needing -mxsave for the intrinsic. Only called
        // after CPUID reported OSXSAVE, so XGETBV is guaranteed to be legal.
        std::uint32_t lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(index));
        return (static_cast<std::uint64_t>(hi) << 32) | lo;
    #endif
#else
        (void)index;
        return 0;
#endif
    }
};

constexpr std::uint32_t bit(std::uint32_t v, unsigned n) { return (v >> n) & 1u; }

// Walk a "deterministic cache parameters" leaf (Intel leaf 4, AMD 0x8000001D)
// and record L2 / L3 data-or-unified sizes.
void decode_caches(const CpuidSource& src, std::uint32_t leaf, CpuFeatures& f) {
    for (std::uint32_t sub = 0; sub < 32; ++sub) {
        const CpuidRegs r = src.cpuid(leaf, sub);
        const std::uint32_t type = r.eax & 0x1Fu;  // 0 = no more caches
        if (type == 0) break;
        const std::uint32_t level = (r.eax >> 5) & 0x7u;
        if (type != 1 && type != 3) continue;      // data or unified only
        const std::uint64_t ways       = static_cast<std::uint64_t>((r.ebx >> 22) & 0x3FFu) + 1;
        const std::uint64_t partitions = static_cast<std::uint64_t>((r.ebx >> 12) & 0x3FFu) + 1;
        const std::uint64_t line       = static_cast<std::uint64_t>(r.ebx & 0xFFFu) + 1;
        const std::uint64_t sets       = static_cast<std::uint64_t>(r.ecx) + 1;
        const std::uint64_t size = ways * partitions * line * sets;
        const std::uint32_t size32 = size > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<std::uint32_t>(size);
        if (level == 2) f.l2_cache_bytes = size32;
        else if (level == 3) f.l3_cache_bytes = size32;
    }
}

} // namespace

const CpuidSource& native_cpuid_source() noexcept {
    static const NativeCpuidSource src;
    return src;
}

// ============================================================================
// Decode
// ============================================================================

CpuFeatures decode_cpu_features(const CpuidSource& src) {
    CpuFeatures f;

    const CpuidRegs l0 = src.cpuid(0, 0);
    const std::uint32_t max_leaf = l0.eax;
    if (max_leaf == 0 && l0.ebx == 0 && l0.ecx == 0 && l0.edx == 0) {
        return f;  // no CPUID (non-x86): all false, vendor Other
    }

    char vendor[13];
    std::memcpy(vendor + 0, &l0.ebx, 4);
    std::memcpy(vendor + 4, &l0.edx, 4);
    std::memcpy(vendor + 8, &l0.ecx, 4);
    vendor[12] = '\0';
    if (std::strcmp(vendor, "GenuineIntel") == 0) f.vendor = CpuVendor::Intel;
    else if (std::strcmp(vendor, "AuthenticAMD") == 0) f.vendor = CpuVendor::AMD;

    bool osxsave = false;
    bool avx_bit = false;
    if (max_leaf >= 1) {
        const CpuidRegs l1 = src.cpuid(1, 0);
        f.stepping = l1.eax & 0xFu;
        const std::uint32_t base_model  = (l1.eax >> 4) & 0xFu;
        const std::uint32_t base_family = (l1.eax >> 8) & 0xFu;
        const std::uint32_t ext_model   = (l1.eax >> 16) & 0xFu;
        const std::uint32_t ext_family  = (l1.eax >> 20) & 0xFFu;
        f.family = base_family == 0xFu ? base_family + ext_family : base_family;
        f.model  = (base_family == 0x6u || base_family == 0xFu)
                       ? ((ext_model << 4) | base_model) : base_model;
        osxsave = bit(l1.ecx, 27) != 0;
        avx_bit = bit(l1.ecx, 28) != 0;
    }

    // OS-saved vector state. XGETBV is only legal when OSXSAVE is set.
    bool os_avx = false;
    bool os_avx512 = false;
    if (osxsave) {
        const std::uint64_t xcr0 = src.xgetbv(0);
        os_avx    = (xcr0 & 0x6u) == 0x6u;                      // XMM + YMM
        os_avx512 = os_avx && (xcr0 & 0xE0u) == 0xE0u;          // opmask + ZMM_Hi256 + Hi16_ZMM
    }
    const bool avx_usable    = avx_bit && os_avx;
    const bool avx512_usable = avx_bit && os_avx512;

    if (max_leaf >= 7) {
        const CpuidRegs l7 = src.cpuid(7, 0);
        f.bmi1       = bit(l7.ebx, 3) != 0;
        f.bmi2       = bit(l7.ebx, 8) != 0;
        f.adx        = bit(l7.ebx, 19) != 0;
        f.sha_ni     = bit(l7.ebx, 29) != 0;
        f.avx2       = avx_usable && bit(l7.ebx, 5) != 0;
        f.avx512f    = avx512_usable && bit(l7.ebx, 16) != 0;
        f.avx512ifma = avx512_usable && bit(l7.ebx, 21) != 0;
        f.avx512vl   = avx512_usable && bit(l7.ebx, 31) != 0;
        if (l7.eax >= 1) {  // EAX of subleaf 0 = max subleaf
            const CpuidRegs l71 = src.cpuid(7, 1);
            f.avx_vnni = avx_usable && bit(l71.eax, 4) != 0;
            f.avx_ifma = avx_usable && bit(l71.eax, 23) != 0;
        }
    }

    // Brand string
    const CpuidRegs e0 = src.cpuid(0x80000000u, 0);
    const std::uint32_t max_ext = e0.eax;
    if (max_ext >= 0x80000004u) {
        char raw[49];
        for (std::uint32_t i = 0; i < 3; ++i) {
            const CpuidRegs r = src.cpuid(0x80000002u + i, 0);
            std::memcpy(raw + i * 16 + 0, &r.eax, 4);
            std::memcpy(raw + i * 16 + 4, &r.ebx, 4);
            std::memcpy(raw + i * 16 + 8, &r.ecx, 4);
            std::memcpy(raw + i * 16 + 12, &r.edx, 4);
        }
        raw[48] = '\0';
        const char* b = raw;
        while (*b == ' ') ++b;
        std::size_t n = std::strlen(b);
        while (n > 0 && (b[n - 1] == ' ' || b[n - 1] == '\0')) --n;
        std::memcpy(f.brand, b, n);
        f.brand[n] = '\0';
    }

    // Cache sizes
    if (f.vendor == CpuVendor::Intel && max_leaf >= 4) {
        decode_caches(src, 4, f);
    } else if (f.vendor == CpuVendor::AMD && max_ext >= 0x8000001Du) {
        decode_caches(src, 0x8000001Du, f);
    }

    return f;
}

// ============================================================================
// Mask
// ============================================================================

namespace {

char lower_ascii(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

std::uint32_t token_to_mask(const std::string& t) {
    if (t == "avx2") return cpu_mask::kAvx2;
    if (t == "bmi2") return cpu_mask::kBmi2;
    if (t == "adx") return cpu_mask::kAdx;
    if (t == "sha_ni" || t == "sha-ni" || t == "shani") return cpu_mask::kShaNi;
    if (t == "avx512") return cpu_mask::kAvx512;
    if (t == "ifma") return cpu_mask::kIfma;
    if (t == "avx_ifma" || t == "avx-ifma") return cpu_mask::kAvxIfma;
    return 0;
}

} // namespace

std::uint32_t parse_cpu_disable_list(const char* list, std::string* unknown) {
    std::uint32_t mask = 0;
    if (!list) return 0;
    std::string tok;
    auto flush = [&]() {
        if (tok.empty()) return;
        const std::uint32_t m = token_to_mask(tok);
        if (m != 0) mask |= m;
        else if (unknown) {
            if (!unknown->empty()) unknown->push_back(',');
            unknown->append(tok);
        }
        tok.clear();
    };
    for (const char* p = list; *p; ++p) {
        if (*p == ',') { flush(); continue; }
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') continue;
        tok.push_back(lower_ascii(*p));
    }
    flush();
    return mask;
}

// CT boundary: the mask only moves a dispatcher onto the portable implementation
// that non-x86 builds already use. It never selects a different secret-dependent
// algorithm, and it can only clear bits -- never set one: every assignment below
// is `= false`, so a feature the CPU lacks cannot be turned on (that would be a
// SIGILL).
CpuFeatures apply_cpu_disable_mask(const CpuFeatures& in, std::uint32_t mask) noexcept {
    CpuFeatures out = in;
    if (mask & cpu_mask::kAvx2) {
        // Every vector extension used here is layered on AVX2 state.
        out.avx2 = false;
        out.avx_ifma = false;
        out.avx_vnni = false;
        out.avx512f = false;
        out.avx512vl = false;
        out.avx512ifma = false;
    }
    if (mask & cpu_mask::kBmi2) out.bmi2 = false;
    if (mask & cpu_mask::kAdx) out.adx = false;
    if (mask & cpu_mask::kShaNi) out.sha_ni = false;
    if (mask & cpu_mask::kAvx512) {
        out.avx512f = false;
        out.avx512vl = false;
        out.avx512ifma = false;
    }
    if (mask & cpu_mask::kIfma) out.avx512ifma = false;
    if (mask & cpu_mask::kAvxIfma) out.avx_ifma = false;
    return out;
}

// ============================================================================
// Process-wide cached views
// ============================================================================

namespace {

struct State {
    CpuFeatures raw;
    std::uint32_t env_mask = 0;
    std::atomic<std::uint32_t> api_mask{0};
    std::atomic<const CpuFeatures*> masked{nullptr};
    std::mutex mu;
    // Owns every published snapshot so references handed out earlier stay valid.
    std::vector<std::unique_ptr<CpuFeatures>> snapshots;

    State() : raw(decode_cpu_features(native_cpuid_source())) {
#if defined(_MSC_VER)
    #pragma warning(suppress : 4996)
#endif
        env_mask = parse_cpu_disable_list(std::getenv("UFSECP_CPU_DISABLE"));
        publish_locked(0);
    }

    void publish_locked(std::uint32_t api) {
        snapshots.push_back(std::make_unique<CpuFeatures>(apply_cpu_disable_mask(raw, env_mask | api)));
        api_mask.store(api, std::memory_order_relaxed);
        masked.store(snapshots.back().get(), std::memory_order_release);
    }
};

State& state() {
    static State s;  // C++11 magic static: thread-safe one-time init
    return s;
}

} // namespace

const CpuFeatures& cpu_features_raw() { return state().raw; }

const CpuFeatures& cpu_features() {
    return *state().masked.load(std::memory_order_acquire);
}

void set_cpu_disable_mask(std::uint32_t mask) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.publish_locked(mask & cpu_mask::kAll);
}

std::uint32_t cpu_disable_mask() {
    State& s = state();
    return s.env_mask | s.api_mask.load(std::memory_order_relaxed);
}

const char* cpu_vendor_name(CpuVendor v) noexcept {
    switch (v) {
        case CpuVendor::Intel: return "Intel";
        case CpuVendor::AMD:   return "AMD";
        default:               return "other";
    }
}

} // namespace secp256k1
