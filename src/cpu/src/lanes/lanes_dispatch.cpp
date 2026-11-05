// ============================================================================
// lanes/lanes_dispatch.cpp -- pick a lane-parallel field backend at run time
// ============================================================================
//
// Generic code: compiled with the library's normal flags, never with an ISA
// flag. It only reads cpu_features() and hands out pointers to the per-ISA
// kernel tables; it never calls into them.
//
// SECP256K1_LANES_HAVE_{AVX2,AVX_IFMA,IFMA256,IFMA512} are set (on this file
// only) by src/cpu/CMakeLists.txt for each ISA translation unit it builds.
// See secp256k1/lanes.hpp for the selection rules.
// ============================================================================

#include "secp256k1/lanes.hpp"

#include <cstdio>
#include <cstdlib>

#include "secp256k1/cpu_features.hpp"
#include "lanes_tables.hpp"

namespace secp256k1::lanes {

namespace {

constexpr LaneBackend kAllBackends[] = {
    LaneBackend::None, LaneBackend::Avx2x4, LaneBackend::AvxIfma256x4,
    LaneBackend::Ifma256x4, LaneBackend::Ifma512x8,
};

const LaneFieldKernels* compiled_kernels(LaneBackend b) noexcept {
    switch (b) {
#if defined(SECP256K1_LANES_HAVE_AVX2)
    case LaneBackend::Avx2x4: return &detail::kLaneFieldKernelsAvx2;
#endif
#if defined(SECP256K1_LANES_HAVE_AVX_IFMA)
    case LaneBackend::AvxIfma256x4: return &detail::kLaneFieldKernelsAvxIfma;
#endif
#if defined(SECP256K1_LANES_HAVE_IFMA256)
    case LaneBackend::Ifma256x4: return &detail::kLaneFieldKernelsIfma256;
#endif
#if defined(SECP256K1_LANES_HAVE_IFMA512)
    case LaneBackend::Ifma512x8: return &detail::kLaneFieldKernelsIfma512;
#endif
    default: return nullptr;
    }
}

char lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool is_blank(char c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Case-insensitive match of [begin, end) against a lowercase literal.
bool token_is(const char* begin, const char* end, const char* lit) noexcept {
    for (; begin != end && *lit != '\0'; ++begin, ++lit)
        if (lower(*begin) != *lit) return false;
    return begin == end && *lit == '\0';
}

} // namespace

const char* lane_backend_name(LaneBackend b) noexcept {
    switch (b) {
    case LaneBackend::None: return "off";
    case LaneBackend::Avx2x4: return "avx2";
    case LaneBackend::AvxIfma256x4: return "avx_ifma";
    case LaneBackend::Ifma256x4: return "ifma256";
    case LaneBackend::Ifma512x8: return "ifma512";
    }
    return "unknown";
}

unsigned lane_backend_width(LaneBackend b) noexcept {
    switch (b) {
    case LaneBackend::None: return 0;
    case LaneBackend::Ifma512x8: return 8;
    case LaneBackend::Avx2x4:
    case LaneBackend::AvxIfma256x4:
    case LaneBackend::Ifma256x4: return 4;
    }
    return 0;
}

unsigned compiled_lane_backends() noexcept {
    unsigned mask = lane_backend_bit(LaneBackend::None);
    for (LaneBackend b : kAllBackends)
        if (compiled_kernels(b) != nullptr) mask |= lane_backend_bit(b);
    return mask;
}

bool lane_backend_supported(LaneBackend b, const CpuFeatures& f) noexcept {
    switch (b) {
    case LaneBackend::None: return true;
    case LaneBackend::Avx2x4: return f.avx2;
    // VEX AVX-IFMA operates on ymm and is decoded alongside AVX2.
    case LaneBackend::AvxIfma256x4: return f.avx2 && f.avx_ifma;
    // EVEX vpmadd52 on ymm/zmm: AVX512IFMA, with VL for the ymm form; the
    // decode only reports either when the OS saves the AVX-512 state.
    case LaneBackend::Ifma256x4:
    case LaneBackend::Ifma512x8: return f.avx512f && f.avx512vl && f.avx512ifma;
    }
    return false;
}

unsigned available_lane_backends() noexcept {
    const CpuFeatures& f = cpu_features();
    unsigned mask = 0;
    for (LaneBackend b : kAllBackends) {
        const bool compiled = (b == LaneBackend::None) || compiled_kernels(b) != nullptr;
        if (compiled && lane_backend_supported(b, f)) mask |= lane_backend_bit(b);
    }
    return mask;
}

LaneBackend default_lane_backend(unsigned available_mask) noexcept {
    // Ifma256x4 is deliberately absent: it is reachable only through
    // UFSECP_LANES=ifma256 until a measurement shows where it wins.
    constexpr LaneBackend kOrder[] = {LaneBackend::Ifma512x8, LaneBackend::AvxIfma256x4,
                                      LaneBackend::Avx2x4};
    for (LaneBackend b : kOrder)
        if (available_mask & lane_backend_bit(b)) return b;
    return LaneBackend::None;
}

LaneSelection resolve_lane_selection(const char* env_value, unsigned available_mask) noexcept {
    available_mask |= lane_backend_bit(LaneBackend::None);   // "off" is always possible
    LaneSelection sel;
    sel.backend = default_lane_backend(available_mask);
    sel.requested = sel.backend;
    sel.reason = LaneSelectionReason::Default;
    if (env_value == nullptr) return sel;

    const char* begin = env_value;
    while (*begin != '\0' && is_blank(*begin)) ++begin;
    const char* end = begin;
    while (*end != '\0') ++end;
    while (end != begin && is_blank(end[-1])) --end;
    if (begin == end) return sel;

    LaneBackend want = LaneBackend::None;
    if (token_is(begin, end, "off") || token_is(begin, end, "none")) want = LaneBackend::None;
    else if (token_is(begin, end, "avx2")) want = LaneBackend::Avx2x4;
    else if (token_is(begin, end, "avx_ifma") || token_is(begin, end, "avxifma"))
        want = LaneBackend::AvxIfma256x4;
    else if (token_is(begin, end, "ifma256")) want = LaneBackend::Ifma256x4;
    else if (token_is(begin, end, "ifma512")) want = LaneBackend::Ifma512x8;
    else {
        sel.reason = LaneSelectionReason::UnknownToken;
        return sel;
    }

    sel.requested = want;
    if (available_mask & lane_backend_bit(want)) {
        sel.backend = want;
        sel.reason = LaneSelectionReason::Forced;
    } else {
        sel.reason = LaneSelectionReason::Unavailable;   // keep the default: never SIGILL
    }
    return sel;
}

const LaneSelection& lane_selection() noexcept {
    static const LaneSelection sel = [] {
        const char* env = std::getenv("UFSECP_LANES");
        const LaneSelection s = resolve_lane_selection(env, available_lane_backends());
        // The raw environment value is never echoed (log injection).
        if (s.reason == LaneSelectionReason::Unavailable ||
            s.reason == LaneSelectionReason::UnknownToken) {
            std::fprintf(stderr, "ufsecp: UFSECP_LANES: %s (selected: %s)\n",
                         lane_selection_reason(s.reason), lane_backend_name(s.backend));
        }
        return s;
    }();
    return sel;
}

LaneBackend selected_lane_backend() noexcept { return lane_selection().backend; }

const char* lane_selection_reason(LaneSelectionReason r) noexcept {
    switch (r) {
    case LaneSelectionReason::Default: return "default pick (UFSECP_LANES unset)";
    case LaneSelectionReason::Forced: return "forced by UFSECP_LANES";
    case LaneSelectionReason::Unavailable:
        return "requested backend is not available (not built, or the CPU / UFSECP_CPU_DISABLE "
               "lacks its ISA); the default pick is used instead";
    case LaneSelectionReason::UnknownToken:
        return "value not recognised (expected off|avx2|avx_ifma|ifma256|ifma512); the default pick is used instead";
    }
    return "unknown";
}

const LaneFieldKernels* lane_field_kernels(LaneBackend b) noexcept {
    if (b == LaneBackend::None) return nullptr;
    if ((available_lane_backends() & lane_backend_bit(b)) == 0) return nullptr;
    return compiled_kernels(b);
}

} // namespace secp256k1::lanes
