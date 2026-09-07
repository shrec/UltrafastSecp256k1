#pragma once

// P6 call-boundary controls only: frozen P5 arithmetic, no new algorithm.
// All inputs must contain four canonical little-endian radix-2^64 field limbs.
// C leaf output ranges must be disjoint from every input range; a and b may
// designate the same input. Inputs are read-only. No restrict/CT certification.
#include "p5_field_product_kernels.hpp"
#include <stdexcept>

#if defined(SECP256K1_HAS_ASM) && SECP256K1_HAS_ASM && \
    !defined(SECP256K1_NO_ASM) && defined(__linux__) && defined(__x86_64__) && \
    (defined(__GNUC__) || defined(__clang__))
#define PA_P6_NATIVE_ASM_COMPILED 1
#else
#define PA_P6_NATIVE_ASM_COMPILED 0
#endif

#if PA_P6_NATIVE_ASM_COMPILED
#include <cpuid.h>
#endif

// Definitions reside exclusively in p6_field_boundary.cpp, compiled separately
// without LTO. Pointer ABI matches the native ASM leaf; output is four words.
extern "C" void pa_p6_row_mul(const std::uint64_t* a, const std::uint64_t* b,
                             std::uint64_t* output) noexcept;
extern "C" void pa_p6_row_square(const std::uint64_t* a,
                                std::uint64_t* output) noexcept;

#if PA_P6_NATIVE_ASM_COMPILED
// Existing Linux SysV x86-64 leaves. No additional outlined forwarding shim.
extern "C" void field_mul_full_asm(const std::uint64_t* a,
                                  const std::uint64_t* b, std::uint64_t* result);
extern "C" void field_sqr_full_asm(const std::uint64_t* a, std::uint64_t* result);
#endif

namespace pa_p6 {
using Limbs = pa_p5::Limbs;
inline constexpr bool native_asm_compiled = PA_P6_NATIVE_ASM_COMPILED != 0;

// Caller admission gate: run before ANY direct-ASM invocation. Unsupported hosts
// must skip/reject the direct routes. Native wrappers deliberately do not repeat
// this runtime check. Clang 18 rejects __builtin_cpu_supports("adx"), so use the
// same CPUID leaf/bits as the existing production feature-detection helpers.
inline bool native_asm_available() noexcept {
#if PA_P6_NATIVE_ASM_COMPILED
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    constexpr unsigned required = bit_BMI2 | bit_ADX;
    return (ebx & required) == required;
#else
    return false;
#endif
}

inline Limbs row_mul_inline(const Limbs& a, const Limbs& b) noexcept {
    return pa_p5::reduce_serial(pa_p5::mul_row(a, b));
}

inline Limbs row_square_inline(const Limbs& a) noexcept {
    return pa_p5::reduce_serial(pa_p5::mul_row(a, a));
}

inline Limbs row_mul_outlined(const Limbs& a, const Limbs& b) noexcept {
    Limbs result;
    pa_p6_row_mul(a.data(), b.data(), result.data());
    return result;
}

inline Limbs row_square_outlined(const Limbs& a) noexcept {
    Limbs result;
    pa_p6_row_square(a.data(), result.data());
    return result;
}

// Native-build precondition: native_asm_available() was true at caller admission.
// Calling on an unsupported native host violates this experimental contract.
// Compile-disabled builds fail closed rather than silently substituting C++.
inline Limbs asm_mul_direct(const Limbs& a, const Limbs& b) {
#if PA_P6_NATIVE_ASM_COMPILED
    Limbs result;
    field_mul_full_asm(a.data(), b.data(), result.data());
    return result;
#else
    (void)a;
    (void)b;
    throw std::runtime_error("P6 direct ASM is not compiled for this configuration");
#endif
}

inline Limbs asm_square_direct(const Limbs& a) {
#if PA_P6_NATIVE_ASM_COMPILED
    Limbs result;
    field_sqr_full_asm(a.data(), result.data());
    return result;
#else
    (void)a;
    throw std::runtime_error("P6 direct ASM is not compiled for this configuration");
#endif
}
} // namespace pa_p6
