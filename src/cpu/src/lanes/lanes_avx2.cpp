// ============================================================================
// lanes/lanes_avx2.cpp -- Avx2x4 (AVX2 radix 2^26, 4 lanes) lane field kernels
// ============================================================================
//
// Compiled with per-file flags -mavx2 (MSVC /arch:AVX2)
// (src/cpu/CMakeLists.txt). Holds only the constant-initialised kernel table:
// no code here runs unless lanes_dispatch.cpp hands the table out, which it
// does only when the masked cpu_features() view reports the ISA.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#include "field_avx2x4.hpp"
#include "lanes_kernels.hpp"
#include "lanes_tables.hpp"

namespace secp256k1::lanes::detail {

constinit const LaneFieldKernels kLaneFieldKernelsAvx2 = field::make_lane_kernels<field::Avx2x4>(LaneBackend::Avx2x4);

} // namespace secp256k1::lanes::detail
