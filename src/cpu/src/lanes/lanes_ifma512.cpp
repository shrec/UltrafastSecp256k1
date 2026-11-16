// ============================================================================
// lanes/lanes_ifma512.cpp -- Ifma512x8 (AVX-512 IFMA, 8 lanes) lane field kernels
// ============================================================================
//
// Compiled with per-file flags -mavx512f -mavx512vl -mavx512ifma (MSVC /arch:AVX512)
// (src/cpu/CMakeLists.txt). Holds only the constant-initialised kernel table:
// no code here runs unless lanes_dispatch.cpp hands the table out, which it
// does only when the masked cpu_features() view reports the ISA.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#include "field_ifma512x8.hpp"
#include "lanes_kernels.hpp"
#include "lanes_verify_kernels.hpp"
#include "lanes_tables.hpp"

namespace secp256k1::lanes::detail {

constinit const LaneFieldKernels kLaneFieldKernelsIfma512 = field::make_lane_kernels<field::Ifma512x8>(LaneBackend::Ifma512x8);
constinit const LaneVerifyKernels kLaneVerifyKernelsIfma512 = field::make_lane_verify_kernels<field::Ifma512x8>(LaneBackend::Ifma512x8);

} // namespace secp256k1::lanes::detail
