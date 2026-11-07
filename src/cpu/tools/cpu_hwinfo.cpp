// ============================================================================
// cpu_hwinfo -- print what the library sees of this CPU and what it will use
// ============================================================================
//
// Usage: cpu_hwinfo [--json] [--help]
//
// Prints vendor, family/model/stepping, brand, every feature flag (raw, as
// the CPU and OS report it, and masked, after UFSECP_CPU_DISABLE), the L2/L3
// cache sizes and the variant each dispatcher picks, including the
// lane-parallel field backends (compiled, available on this CPU, selected
// after UFSECP_LANES, and why).
//
// Text format (stable): one "key: value" line per fact, keys are dotted paths.
//   feature.<name>: raw=<0|1> masked=<0|1>
//   dispatch.lanes.<compiled|available>: comma list of backend names
// --json prints the same facts as one JSON object.
// ============================================================================

#include "secp256k1/cpu_features.hpp"
#include "secp256k1/field_asm.hpp"
#include "secp256k1/hash_accel.hpp"
#include "secp256k1/lanes.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace secp256k1;

namespace {

struct Flag {
    const char* name;
    bool CpuFeatures::* member;
};

constexpr Flag kFlags[] = {
    {"avx2",       &CpuFeatures::avx2},
    {"bmi1",       &CpuFeatures::bmi1},
    {"bmi2",       &CpuFeatures::bmi2},
    {"adx",        &CpuFeatures::adx},
    {"sha_ni",     &CpuFeatures::sha_ni},
    {"avx512f",    &CpuFeatures::avx512f},
    {"avx512vl",   &CpuFeatures::avx512vl},
    {"avx512ifma", &CpuFeatures::avx512ifma},
    {"avx_ifma",   &CpuFeatures::avx_ifma},
    {"avx_vnni",   &CpuFeatures::avx_vnni},
};

struct MaskName {
    const char* name;
    std::uint32_t bit;
};

constexpr MaskName kMaskNames[] = {
    {"avx2",     cpu_mask::kAvx2},
    {"bmi2",     cpu_mask::kBmi2},
    {"adx",      cpu_mask::kAdx},
    {"sha_ni",   cpu_mask::kShaNi},
    {"avx512",   cpu_mask::kAvx512},
    {"ifma",     cpu_mask::kIfma},
    {"avx_ifma", cpu_mask::kAvxIfma},
};

std::string mask_to_list(std::uint32_t mask) {
    std::string s;
    for (const auto& m : kMaskNames) {
        if (mask & m.bit) {
            if (!s.empty()) s.push_back(',');
            s += m.name;
        }
    }
    return s;
}

std::string json_escape(const char* in) {
    std::string out;
    for (const char* p = in; *p; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out.push_back(static_cast<char>(c));
    }
    return out;
}

// Which field mul/square implementation the library will run on this CPU
// (mirrors the dispatch in field.cpp: the asm path needs BMI2 and ADX).
const char* field_mul_choice() {
#if defined(__x86_64__) || defined(_M_X64)
    return (fast::has_bmi2_support() && fast::has_adx_support()) ? "bmi2_adx_asm" : "portable";
#else
    return "portable";
#endif
}

// Comma list of the lane backends in `mask` (the "off" bit is not listed).
std::string lanes_to_list(unsigned mask) {
    std::string s;
    for (unsigned b = 1; b < lanes::kLaneBackendCount; ++b) {
        const auto be = static_cast<lanes::LaneBackend>(b);
        if (mask & lanes::lane_backend_bit(be)) {
            if (!s.empty()) s.push_back(',');
            s += lanes::lane_backend_name(be);
        }
    }
    return s;
}

void print_usage() {
    std::printf("Usage: cpu_hwinfo [--json] [--help]\n"
                "  Prints CPU identity, feature flags (raw and masked by UFSECP_CPU_DISABLE),\n"
                "  L2/L3 cache sizes and the variant each dispatcher picks, including the\n"
                "  lane-parallel field backend (UFSECP_LANES=off|avx2|avx_ifma|ifma256|ifma512).\n"
                "  --json   print one JSON object instead of text\n");
}

void print_text(const CpuFeatures& raw, const CpuFeatures& masked) {
    std::printf("cpu.vendor: %s\n", cpu_vendor_name(raw.vendor));
    std::printf("cpu.family: %u\n", raw.family);
    std::printf("cpu.model: %u\n", raw.model);
    std::printf("cpu.stepping: %u\n", raw.stepping);
    std::printf("cpu.brand: %s\n", raw.brand);
    std::printf("mask.disable: %s\n", mask_to_list(cpu_disable_mask()).c_str());
    for (const auto& f : kFlags) {
        std::printf("feature.%s: raw=%d masked=%d\n", f.name, raw.*(f.member) ? 1 : 0,
                    masked.*(f.member) ? 1 : 0);
    }
    std::printf("cache.l2_bytes: %u\n", raw.l2_cache_bytes);
    std::printf("cache.l3_bytes: %u\n", raw.l3_cache_bytes);
    std::printf("dispatch.field_mul: %s\n", field_mul_choice());
    std::printf("dispatch.hash_tier: %s\n", hash::hash_tier_name(hash::detect_hash_tier()));
    const lanes::LaneSelection& sel = lanes::lane_selection();
    std::printf("dispatch.lanes.compiled: %s\n", lanes_to_list(lanes::compiled_lane_backends()).c_str());
    std::printf("dispatch.lanes.available: %s\n", lanes_to_list(lanes::available_lane_backends()).c_str());
    std::printf("dispatch.lanes.selected: %s\n", lanes::lane_backend_name(sel.backend));
    std::printf("dispatch.lanes.requested: %s\n", lanes::lane_backend_name(sel.requested));
    std::printf("dispatch.lanes.reason: %s\n", lanes::lane_selection_reason(sel.reason));
}

void print_json(const CpuFeatures& raw, const CpuFeatures& masked) {
    std::printf("{\n");
    std::printf("  \"vendor\": \"%s\",\n", cpu_vendor_name(raw.vendor));
    std::printf("  \"family\": %u,\n", raw.family);
    std::printf("  \"model\": %u,\n", raw.model);
    std::printf("  \"stepping\": %u,\n", raw.stepping);
    std::printf("  \"brand\": \"%s\",\n", json_escape(raw.brand).c_str());
    std::printf("  \"mask_disable\": \"%s\",\n", mask_to_list(cpu_disable_mask()).c_str());
    std::printf("  \"features\": {\n");
    const std::size_t n = sizeof(kFlags) / sizeof(kFlags[0]);
    for (std::size_t i = 0; i < n; ++i) {
        std::printf("    \"%s\": {\"raw\": %s, \"masked\": %s}%s\n", kFlags[i].name,
                    raw.*(kFlags[i].member) ? "true" : "false",
                    masked.*(kFlags[i].member) ? "true" : "false", i + 1 < n ? "," : "");
    }
    std::printf("  },\n");
    std::printf("  \"cache\": {\"l2_bytes\": %u, \"l3_bytes\": %u},\n", raw.l2_cache_bytes, raw.l3_cache_bytes);
    const lanes::LaneSelection& sel = lanes::lane_selection();
    std::printf("  \"dispatch\": {\"field_mul\": \"%s\", \"hash_tier\": \"%s\",\n", field_mul_choice(),
                hash::hash_tier_name(hash::detect_hash_tier()));
    std::printf("    \"lanes\": {\"compiled\": \"%s\", \"available\": \"%s\", \"selected\": \"%s\", "
                "\"requested\": \"%s\", \"reason\": \"%s\"}}\n",
                lanes_to_list(lanes::compiled_lane_backends()).c_str(),
                lanes_to_list(lanes::available_lane_backends()).c_str(),
                lanes::lane_backend_name(sel.backend), lanes::lane_backend_name(sel.requested),
                json_escape(lanes::lane_selection_reason(sel.reason)).c_str());
    std::printf("}\n");
}

} // namespace

int main(int argc, char** argv) {
    bool json = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) json = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            std::fprintf(stderr, "cpu_hwinfo: unknown option '%s'\n", argv[i]);
            print_usage();
            return 2;
        }
    }
    const CpuFeatures& raw = cpu_features_raw();
    const CpuFeatures& masked = cpu_features();
    if (json) print_json(raw, masked);
    else print_text(raw, masked);
    return 0;
}
