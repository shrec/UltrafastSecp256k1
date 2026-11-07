// ============================================================================
// test_cpu_features -- CPUID/XGETBV decode over synthetic CPUs, cache sizes,
// and the mask-down override
// ============================================================================
// The decode is a pure function over an injectable CpuidSource, so every CPU
// below is a hand-built table: no real hardware needed, deterministic anywhere.
// ============================================================================

#include "secp256k1/cpu_features.hpp"
#include "secp256k1/field_asm.hpp"
#include "secp256k1/hash_accel.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>

using namespace secp256k1;

namespace {

int g_fail = 0;
int g_pass = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) { ++g_pass; }                                                \
        else { ++g_fail; std::printf("  FAIL: %s  [%s:%d]\n", (msg), __FILE__, __LINE__); } \
    } while (0)

// ---- Synthetic CPUID table ---------------------------------------------------

class FakeCpu final : public CpuidSource {
public:
    std::map<std::pair<std::uint32_t, std::uint32_t>, CpuidRegs> table;
    std::uint64_t xcr0 = 0;
    mutable int xgetbv_calls = 0;

    void set(std::uint32_t leaf, std::uint32_t sub, std::uint32_t a, std::uint32_t b,
             std::uint32_t c, std::uint32_t d) {
        CpuidRegs r; r.eax = a; r.ebx = b; r.ecx = c; r.edx = d;
        table[{leaf, sub}] = r;
    }
    CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t sub) const override {
        auto it = table.find({leaf, sub});
        return it == table.end() ? CpuidRegs{} : it->second;
    }
    std::uint64_t xgetbv(std::uint32_t index) const override {
        ++xgetbv_calls;
        return index == 0 ? xcr0 : 0;
    }
};

std::uint32_t pack4(const char* s) {
    std::uint32_t v = 0;
    std::memcpy(&v, s, 4);
    return v;
}

void set_vendor(FakeCpu& c, const char* vendor12, std::uint32_t max_leaf) {
    c.set(0, 0, max_leaf, pack4(vendor12), pack4(vendor12 + 8), pack4(vendor12 + 4));
}

std::uint32_t sig(std::uint32_t family, std::uint32_t model, std::uint32_t stepping) {
    // Inverse of the display family/model rule.
    const std::uint32_t base_family = family >= 0xF ? 0xFu : family;
    const std::uint32_t ext_family  = family >= 0xF ? family - 0xF : 0;
    const std::uint32_t base_model  = model & 0xFu;
    const std::uint32_t ext_model   = (base_family == 6 || base_family == 0xF) ? (model >> 4) & 0xFu : 0;
    return (ext_family << 20) | (ext_model << 16) | (base_family << 8) | (base_model << 4) | stepping;
}

void set_brand(FakeCpu& c, const char* brand) {
    char buf[48];
    std::memset(buf, 0, sizeof buf);
    std::strncpy(buf, brand, 47);
    c.set(0x80000000u, 0, 0x80000008u, 0, 0, 0);
    for (std::uint32_t i = 0; i < 3; ++i) {
        c.set(0x80000002u + i, 0, pack4(buf + i * 16), pack4(buf + i * 16 + 4),
              pack4(buf + i * 16 + 8), pack4(buf + i * 16 + 12));
    }
}

// Cache descriptor: type 1 data / 2 instr / 3 unified.
void set_cache(FakeCpu& c, std::uint32_t leaf, std::uint32_t sub, std::uint32_t type,
               std::uint32_t level, std::uint32_t ways, std::uint32_t sets, std::uint32_t line = 64) {
    c.set(leaf, sub, type | (level << 5), ((ways - 1) << 22) | (0u << 12) | (line - 1), sets - 1, 0);
}

constexpr std::uint32_t kOsxsave = 1u << 27;
constexpr std::uint32_t kAvxBit  = 1u << 28;

constexpr std::uint32_t kBmi1 = 1u << 3, kAvx2 = 1u << 5, kBmi2 = 1u << 8, kAvx512F = 1u << 16,
                        kIfma = 1u << 21, kAdx = 1u << 19, kSha = 1u << 29, kAvx512VL = 1u << 31;
constexpr std::uint32_t kAvxVnni71 = 1u << 4, kAvxIfma71 = 1u << 23;

// ---- The six CPUs ------------------------------------------------------------

// i5-14400F-like Intel client: AVX2/BMI2/ADX/SHA-NI, no AVX-512.
FakeCpu make_client_intel() {
    FakeCpu c;
    set_vendor(c, "GenuineIntel", 0x20);
    c.set(1, 0, sig(6, 0xBF, 2), 0, kOsxsave | kAvxBit, 0);
    c.set(7, 0, 1, kBmi1 | kAvx2 | kBmi2 | kAdx | kSha, 0, 0);
    c.set(7, 1, kAvxVnni71, 0, 0, 0);
    c.xcr0 = 0x07;
    set_brand(c, "   13th Gen Intel(R) Core(TM) i5-14400F");
    set_cache(c, 4, 0, 1, 1, 12, 64);        // L1d 48K
    set_cache(c, 4, 1, 2, 1, 8, 64);         // L1i 32K
    set_cache(c, 4, 2, 3, 2, 10, 2048);      // L2 1.25 MiB
    set_cache(c, 4, 3, 3, 3, 20, 16384);     // L3 20 MiB
    return c;
}

// Sapphire Rapids-like server: AVX-512 F/VL/IFMA.
FakeCpu make_server_intel() {
    FakeCpu c;
    set_vendor(c, "GenuineIntel", 0x20);
    c.set(1, 0, sig(6, 0x8F, 8), 0, kOsxsave | kAvxBit, 0);
    c.set(7, 0, 1, kBmi1 | kAvx2 | kBmi2 | kAdx | kSha | kAvx512F | kIfma | kAvx512VL, 0, 0);
    c.set(7, 1, kAvxVnni71, 0, 0, 0);
    c.xcr0 = 0xE7;
    set_brand(c, "Intel(R) Xeon(R) Platinum 8480+");
    set_cache(c, 4, 0, 1, 1, 12, 64);
    set_cache(c, 4, 1, 2, 1, 8, 64);
    set_cache(c, 4, 2, 3, 2, 16, 2048);      // L2 2 MiB
    set_cache(c, 4, 3, 3, 3, 15, 65536);     // L3 60 MiB
    return c;
}

FakeCpu make_amd(std::uint32_t family, std::uint32_t model, bool vnni) {
    FakeCpu c;
    set_vendor(c, "AuthenticAMD", 0x10);
    c.set(1, 0, sig(family, model, 2), 0, kOsxsave | kAvxBit, 0);
    c.set(7, 0, vnni ? 1 : 0, kBmi1 | kAvx2 | kBmi2 | kAdx | kSha | kAvx512F | kIfma | kAvx512VL, 0, 0);
    if (vnni) c.set(7, 1, kAvxVnni71, 0, 0, 0);
    c.xcr0 = 0xE7;
    set_brand(c, family == 0x19 ? "AMD Ryzen 9 7950X 16-Core Processor"
                                : "AMD Ryzen 9 9950X 16-Core Processor");
    c.table[{0x80000000u, 0}].eax = 0x80000026u;
    set_cache(c, 0x8000001Du, 0, 1, 1, 8, 64);
    set_cache(c, 0x8000001Du, 1, 2, 1, 8, 64);
    set_cache(c, 0x8000001Du, 2, 3, 2, 8, 2048);      // L2 1 MiB
    set_cache(c, 0x8000001Du, 3, 3, 3, 16, 32768);    // L3 32 MiB
    // A leaf-4 table on AMD must be ignored (leaf 0x8000001D is authoritative).
    set_cache(c, 4, 2, 3, 2, 1, 1);
    return c;
}

// Intel client with AVX-IFMA (and AVX-VNNI) but no AVX-512.
FakeCpu make_client_avx_ifma() {
    FakeCpu c;
    set_vendor(c, "GenuineIntel", 0x20);
    c.set(1, 0, sig(6, 0xC6, 2), 0, kOsxsave | kAvxBit, 0);
    c.set(7, 0, 1, kBmi1 | kAvx2 | kBmi2 | kAdx | kSha, 0, 0);
    c.set(7, 1, kAvxVnni71 | kAvxIfma71, 0, 0, 0);
    c.xcr0 = 0x07;
    set_brand(c, "Intel(R) Core(TM) Ultra 9 285K");
    set_cache(c, 4, 0, 1, 1, 12, 64);
    set_cache(c, 4, 1, 2, 1, 8, 64);
    set_cache(c, 4, 2, 3, 2, 12, 4096);      // L2 3 MiB
    set_cache(c, 4, 3, 3, 3, 18, 32768);     // L3 36 MiB
    return c;
}

// CPU reports AVX2 / AVX-512 / AVX-IFMA bits but the OS does not save the state.
FakeCpu make_os_no_avx(std::uint64_t xcr0, bool osxsave) {
    FakeCpu c = make_server_intel();
    c.set(1, 0, sig(6, 0x8F, 8), 0, (osxsave ? kOsxsave : 0u) | kAvxBit, 0);
    c.set(7, 1, kAvxVnni71 | kAvxIfma71, 0, 0, 0);
    c.xcr0 = xcr0;
    return c;
}

// ---- Tests -------------------------------------------------------------------

void test_client_intel() {
    std::printf("  [i5-14400F-like client]\n");
    const FakeCpu cpu = make_client_intel();
    const CpuFeatures f = decode_cpu_features(cpu);
    CHECK(f.vendor == CpuVendor::Intel, "vendor Intel");
    CHECK(f.family == 6 && f.model == 0xBF && f.stepping == 2, "family 6 model 0xBF stepping 2");
    CHECK(std::strcmp(f.brand, "13th Gen Intel(R) Core(TM) i5-14400F") == 0, "brand trimmed");
    CHECK(f.avx2 && f.bmi1 && f.bmi2 && f.adx && f.sha_ni, "AVX2/BMI1/BMI2/ADX/SHA-NI");
    CHECK(!f.avx512f && !f.avx512vl && !f.avx512ifma, "no AVX-512");
    CHECK(f.avx_vnni && !f.avx_ifma, "AVX-VNNI without AVX-IFMA");
    CHECK(f.l2_cache_bytes == 1280u * 1024u, "L2 = 1.25 MiB");
    CHECK(f.l3_cache_bytes == 20u * 1024u * 1024u, "L3 = 20 MiB");
}

void test_server_intel() {
    std::printf("  [Sapphire Rapids-like server]\n");
    const CpuFeatures f = decode_cpu_features(make_server_intel());
    CHECK(f.vendor == CpuVendor::Intel && f.family == 6 && f.model == 0x8F, "Intel 6/0x8F");
    CHECK(f.avx2 && f.avx512f && f.avx512vl && f.avx512ifma, "AVX2 + AVX-512 F/VL/IFMA");
    CHECK(!f.avx_ifma, "no AVX-IFMA");
    CHECK(f.l2_cache_bytes == 2u * 1024u * 1024u, "L2 = 2 MiB");
    CHECK(f.l3_cache_bytes == 60u * 1024u * 1024u, "L3 = 60 MiB");
}

void test_zen4() {
    std::printf("  [AMD Zen 4]\n");
    const CpuFeatures f = decode_cpu_features(make_amd(0x19, 0x61, false));
    CHECK(f.vendor == CpuVendor::AMD, "vendor AMD");
    CHECK(f.family == 0x19 && f.model == 0x61 && f.stepping == 2, "family 0x19 model 0x61");
    CHECK(f.avx2 && f.avx512f && f.avx512vl && f.avx512ifma, "AVX2 + AVX-512 IFMA");
    CHECK(!f.avx_ifma, "no AVX-IFMA");
    CHECK(f.l2_cache_bytes == 1024u * 1024u, "L2 from leaf 0x8000001D = 1 MiB");
    CHECK(f.l3_cache_bytes == 32u * 1024u * 1024u, "L3 from leaf 0x8000001D = 32 MiB");
    CHECK(std::strcmp(f.brand, "AMD Ryzen 9 7950X 16-Core Processor") == 0, "brand");
}

void test_zen5() {
    std::printf("  [AMD Zen 5]\n");
    const CpuFeatures f = decode_cpu_features(make_amd(0x1A, 0x44, true));
    CHECK(f.vendor == CpuVendor::AMD && f.family == 0x1A && f.model == 0x44, "family 0x1A model 0x44");
    CHECK(f.avx512f && f.avx512ifma && f.avx_vnni, "AVX-512 IFMA + AVX-VNNI");
    CHECK(f.l3_cache_bytes == 32u * 1024u * 1024u, "L3 32 MiB");
}

void test_client_avx_ifma() {
    std::printf("  [Intel client with AVX-IFMA, no AVX-512]\n");
    const CpuFeatures f = decode_cpu_features(make_client_avx_ifma());
    CHECK(f.avx2 && f.avx_ifma && f.avx_vnni, "AVX2 + AVX-IFMA + AVX-VNNI");
    CHECK(!f.avx512f && !f.avx512vl && !f.avx512ifma, "no AVX-512");
    CHECK(f.l2_cache_bytes == 3u * 1024u * 1024u, "L2 = 3 MiB");
    CHECK(f.l3_cache_bytes == 36u * 1024u * 1024u, "L3 = 36 MiB");
}

void test_os_state() {
    std::printf("  [OS state (XCR0)]\n");
    {   // AVX bit + AVX2 CPUID bit, XCR0 lacks the YMM bit
        const CpuFeatures f = decode_cpu_features(make_os_no_avx(0x03, true));
        CHECK(!f.avx2, "AVX2 not usable when XCR0 lacks AVX state");
        CHECK(!f.avx512f && !f.avx512vl && !f.avx512ifma, "AVX-512 not usable either");
        CHECK(!f.avx_ifma && !f.avx_vnni, "AVX-IFMA / AVX-VNNI not usable");
        CHECK(f.bmi2 && f.adx && f.sha_ni, "GPR / SSE features unaffected");
    }
    {   // AVX state saved, AVX-512 state not
        const CpuFeatures f = decode_cpu_features(make_os_no_avx(0x07, true));
        CHECK(f.avx2 && f.avx_ifma, "AVX2 + AVX-IFMA usable with YMM state");
        CHECK(!f.avx512f && !f.avx512vl && !f.avx512ifma, "AVX-512 needs XCR0 bits 5-7");
    }
    {   // only some AVX-512 state bits: still not usable
        const CpuFeatures f = decode_cpu_features(make_os_no_avx(0x27, true));
        CHECK(!f.avx512f, "partial AVX-512 XCR0 (bits 5 only) is not usable");
    }
    {   // OSXSAVE clear: XGETBV must not be consulted, nothing AVX-class is usable
        const FakeCpu cpu = make_os_no_avx(0xE7, false);
        const CpuFeatures f = decode_cpu_features(cpu);
        CHECK(cpu.xgetbv_calls == 0, "XGETBV not executed without OSXSAVE");
        CHECK(!f.avx2 && !f.avx512f && !f.avx_ifma, "no AVX-class feature without OSXSAVE");
    }
}

void test_no_cpuid() {
    std::printf("  [no CPUID]\n");
    const FakeCpu empty;
    const CpuFeatures f = decode_cpu_features(empty);
    CHECK(f.vendor == CpuVendor::Other, "vendor other");
    CHECK(!f.avx2 && !f.bmi2 && !f.adx && !f.sha_ni && !f.avx512f && !f.avx_ifma, "all false");
    CHECK(f.l2_cache_bytes == 0 && f.l3_cache_bytes == 0 && f.brand[0] == '\0', "no caches, no brand");
}

void test_mask_parse() {
    std::printf("  [mask parsing]\n");
    CHECK(parse_cpu_disable_list(nullptr) == 0, "null list");
    CHECK(parse_cpu_disable_list("") == 0, "empty list");
    CHECK(parse_cpu_disable_list("avx2") == cpu_mask::kAvx2, "avx2");
    CHECK(parse_cpu_disable_list(" BMI2 , Adx ") == (cpu_mask::kBmi2 | cpu_mask::kAdx), "case + blanks");
    CHECK(parse_cpu_disable_list("sha_ni,avx512,ifma,avx_ifma") ==
              (cpu_mask::kShaNi | cpu_mask::kAvx512 | cpu_mask::kIfma | cpu_mask::kAvxIfma), "rest of the names");
    std::string unknown;
    CHECK(parse_cpu_disable_list("avx2,bogus,avx9999", &unknown) == cpu_mask::kAvx2, "unknown tokens add nothing");
    CHECK(unknown == "bogus,avx9999", "unknown tokens reported");
    CHECK((parse_cpu_disable_list("avx2,bmi2,adx,sha_ni,avx512,ifma,avx_ifma") & ~cpu_mask::kAll) == 0,
          "parsed mask stays inside kAll");
}

// The mask turns flags off on synthetic CPUs, and cannot turn a missing flag on.
void test_mask_synthetic() {
    std::printf("  [mask on synthetic CPUs]\n");
    const CpuFeatures server = decode_cpu_features(make_server_intel());

    CpuFeatures m = apply_cpu_disable_mask(server, cpu_mask::kBmi2);
    CHECK(!m.bmi2 && m.adx && m.avx2 && m.avx512f, "bmi2 off only");
    m = apply_cpu_disable_mask(server, cpu_mask::kAdx);
    CHECK(!m.adx && m.bmi2, "adx off only");
    m = apply_cpu_disable_mask(server, cpu_mask::kShaNi);
    CHECK(!m.sha_ni && m.avx2, "sha_ni off only");
    m = apply_cpu_disable_mask(server, cpu_mask::kAvx512);
    CHECK(!m.avx512f && !m.avx512vl && !m.avx512ifma && m.avx2, "avx512 off, avx2 kept");
    m = apply_cpu_disable_mask(server, cpu_mask::kIfma);
    CHECK(!m.avx512ifma && m.avx512f && m.avx512vl, "ifma off only");
    m = apply_cpu_disable_mask(server, cpu_mask::kAvx2);
    CHECK(!m.avx2 && !m.avx512f && !m.avx512vl && !m.avx512ifma && !m.avx_ifma && !m.avx_vnni,
          "avx2 off clears the vector flags layered on it");
    CHECK(m.bmi2 && m.adx && m.sha_ni, "avx2 mask leaves scalar features alone");

    const CpuFeatures avx_ifma = decode_cpu_features(make_client_avx_ifma());
    m = apply_cpu_disable_mask(avx_ifma, cpu_mask::kAvxIfma);
    CHECK(!m.avx_ifma && m.avx2 && m.avx_vnni, "avx_ifma off only");

    // identity / metadata untouched
    m = apply_cpu_disable_mask(server, cpu_mask::kAll);
    CHECK(m.vendor == server.vendor && m.family == server.family && m.model == server.model &&
              m.l2_cache_bytes == server.l2_cache_bytes && std::strcmp(m.brand, server.brand) == 0,
          "mask leaves vendor/model/brand/cache sizes alone");

    // A mask can never turn a missing feature on: every mask applied to a CPU
    // without a feature leaves that feature off.
    const CpuFeatures client = decode_cpu_features(make_client_intel());  // no AVX-512, no AVX-IFMA
    bool never_on = true;
    for (std::uint32_t mask = 0; mask <= cpu_mask::kAll; ++mask) {
        const CpuFeatures r = apply_cpu_disable_mask(client, mask);
        if (r.avx512f || r.avx512vl || r.avx512ifma || r.avx_ifma) never_on = false;
    }
    CHECK(never_on, "no mask turns a missing feature on (exhaustive over all masks)");

    // And a mask is monotone: result flags are a subset of the input flags.
    bool subset = true;
    for (std::uint32_t mask = 0; mask <= cpu_mask::kAll; ++mask) {
        const CpuFeatures r = apply_cpu_disable_mask(server, mask);
        if ((r.avx2 && !server.avx2) || (r.bmi2 && !server.bmi2) || (r.adx && !server.adx) ||
            (r.sha_ni && !server.sha_ni) || (r.avx512f && !server.avx512f) ||
            (r.avx512vl && !server.avx512vl) || (r.avx512ifma && !server.avx512ifma) ||
            (r.avx_ifma && !server.avx_ifma) || (r.avx_vnni && !server.avx_vnni)) subset = false;
    }
    CHECK(subset, "masked flags are always a subset of the unmasked flags");
}

// Masks reach every dispatcher through the process-wide masked view.
void test_mask_dispatchers() {
    std::printf("  [mask reaches every dispatcher]\n");
    const CpuFeatures raw = cpu_features_raw();
    const std::uint32_t env_part = cpu_disable_mask();  // UFSECP_CPU_DISABLE, if the runner set it

    set_cpu_disable_mask(0);

    // Unmasked (apart from the environment): dispatchers decide on the cpu_features() view.
    {
        const CpuFeatures& v = cpu_features();
        CHECK(secp256k1::fast::has_bmi2_support() == v.bmi2, "has_bmi2_support == cpu_features().bmi2");
        CHECK(secp256k1::fast::has_adx_support() == v.adx, "has_adx_support == cpu_features().adx");
        CHECK(secp256k1::hash::avx2_available() == v.avx2, "avx2_available == cpu_features().avx2");
        CHECK(secp256k1::hash::avx512_available() == v.avx512f, "avx512_available == cpu_features().avx512f");
        if (env_part == 0) {
            CHECK(v.bmi2 == raw.bmi2 && v.adx == raw.adx && v.sha_ni == raw.sha_ni &&
                      v.avx2 == raw.avx2 && v.avx512f == raw.avx512f,
                  "no mask: masked view == raw view");
        }
    }

    // Mask everything.
    set_cpu_disable_mask(cpu_mask::kAll);
    {
        const CpuFeatures& v = cpu_features();
        CHECK(!v.bmi2 && !v.adx && !v.sha_ni && !v.avx2 && !v.avx512f && !v.avx512vl &&
                  !v.avx512ifma && !v.avx_ifma && !v.avx_vnni, "mask-all clears the view");
        CHECK(!secp256k1::fast::has_bmi2_support(), "has_bmi2_support masked");
        CHECK(!secp256k1::fast::has_adx_support(), "has_adx_support masked");
        CHECK(!secp256k1::hash::sha_ni_available(), "sha_ni_available masked");
        CHECK(!secp256k1::hash::avx2_available(), "avx2_available masked");
        CHECK(!secp256k1::hash::avx512_available(), "avx512_available masked");
        const auto tier = secp256k1::hash::detect_hash_tier();
#ifdef SECP256K1_X86_TARGET
        const bool portable = tier == secp256k1::hash::HashTier::SCALAR;
#else
        // ARM builds report ARM_SHA2 when compiled with the SHA2 extension.
        const bool portable = tier == secp256k1::hash::HashTier::SCALAR ||
                              tier == secp256k1::hash::HashTier::ARM_SHA2;
#endif
        CHECK(portable, "detect_hash_tier falls back to the portable tier");
        CHECK(cpu_features_raw().bmi2 == raw.bmi2 && cpu_features_raw().avx2 == raw.avx2,
              "raw view is not changed by the mask");
        CHECK(cpu_disable_mask() == (env_part | cpu_mask::kAll), "effective mask reported");
    }

    // Individual flag reaches the matching dispatcher only.
    set_cpu_disable_mask(cpu_mask::kShaNi);
    CHECK(!secp256k1::hash::sha_ni_available(), "sha_ni mask reaches sha_ni_available");
    {
        const auto tier = secp256k1::hash::detect_hash_tier();
        CHECK(tier != secp256k1::hash::HashTier::SHA_NI, "detect_hash_tier no longer picks SHA-NI");
    }

    // The mask can never turn a feature the CPU lacks on: the masked view is
    // always a subset of the raw view, for every mask value.
    bool subset = true;
    for (std::uint32_t mask = 0; mask <= cpu_mask::kAll; ++mask) {
        set_cpu_disable_mask(mask);
        const CpuFeatures& v = cpu_features();
        if ((v.avx2 && !raw.avx2) || (v.bmi2 && !raw.bmi2) || (v.adx && !raw.adx) ||
            (v.sha_ni && !raw.sha_ni) || (v.avx512f && !raw.avx512f) ||
            (v.avx512vl && !raw.avx512vl) || (v.avx512ifma && !raw.avx512ifma) ||
            (v.avx_ifma && !raw.avx_ifma)) subset = false;
    }
    CHECK(subset, "cpu_features() never reports a feature that cpu_features_raw() lacks");

    // The API can lift its own mask but never the environment's.
    set_cpu_disable_mask(0);
    CHECK(cpu_disable_mask() == env_part, "clearing the API mask keeps the environment mask");

    // A reference handed out before a mask change stays valid (immutable snapshots).
    set_cpu_disable_mask(0);
    const CpuFeatures& before = cpu_features();
    const bool before_bmi2 = before.bmi2;
    set_cpu_disable_mask(cpu_mask::kAll);
    CHECK(before.bmi2 == before_bmi2, "earlier snapshot is not mutated by a later mask");

    set_cpu_disable_mask(0);
}

void test_real_cpu() {
    std::printf("  [real CPU sanity]\n");
    const CpuFeatures& r = cpu_features_raw();
    CHECK(!r.avx512ifma || r.avx512f, "AVX-512 IFMA implies AVX-512F on a real CPU");
    CHECK(!r.avx512f || r.avx2, "AVX-512F implies AVX2 on every shipping CPU");
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    CHECK(r.vendor != CpuVendor::Other || r.family == 0, "x86 vendor decoded");
#else
    CHECK(r.vendor == CpuVendor::Other && !r.avx2 && !r.bmi2 && !r.adx && !r.sha_ni,
          "non-x86: all false, vendor other");
#endif
    // Thread-safe cached instance: same object on repeat calls.
    CHECK(&cpu_features_raw() == &cpu_features_raw(), "raw view is a single cached instance");
}

} // namespace

int test_cpu_features_run() {
    g_fail = 0;
    g_pass = 0;
    std::printf("[cpu_features]\n");
    test_client_intel();
    test_server_intel();
    test_zen4();
    test_zen5();
    test_client_avx_ifma();
    test_os_state();
    test_no_cpuid();
    test_mask_parse();
    test_mask_synthetic();
    test_mask_dispatchers();
    test_real_cpu();
    std::printf("[cpu_features] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_cpu_features_run(); }
#endif
