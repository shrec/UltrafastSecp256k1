#include "p6_field_boundary.hpp"

#if defined(__GNUC__) && !defined(__clang__)
#define PA_P6_BOUNDARY __attribute__((noinline, noipa))
#elif defined(__clang__)
#define PA_P6_BOUNDARY __attribute__((noinline))
#else
#error "P6 boundary controls require GCC or Clang and separate-TU -fno-lto builds"
#endif

extern "C" PA_P6_BOUNDARY void pa_p6_row_mul(
    const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* output) noexcept {
    // Load each complete input before producing any output. Output/input overlap
    // is outside the admitted leaf contract; equal read-only inputs are allowed.
    const pa_p6::Limbs left{a[0], a[1], a[2], a[3]};
    const pa_p6::Limbs right{b[0], b[1], b[2], b[3]};
    const auto result = pa_p6::row_mul_inline(left, right);
    for (std::size_t i = 0; i < 4; ++i) output[i] = result[i];
}

extern "C" PA_P6_BOUNDARY void pa_p6_row_square(
    const std::uint64_t* a, std::uint64_t* output) noexcept {
    const pa_p6::Limbs input{a[0], a[1], a[2], a[3]};
    const auto result = pa_p6::row_square_inline(input);
    for (std::size_t i = 0; i < 4; ++i) output[i] = result[i];
}
