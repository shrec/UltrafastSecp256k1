// ============================================================================
// lanes/lanes_ifma256.cpp -- Ifma256x4 (AVX-512VL IFMA on ymm, 4 lanes) lane field kernels
// ============================================================================
//
// Compiled with per-file flags -mavx512f -mavx512vl -mavx512ifma (MSVC /arch:AVX512)
// (src/cpu/CMakeLists.txt). Holds only the constant-initialised kernel table:
// no code here runs unless lanes_dispatch.cpp hands the table out, which it
// does only when the masked cpu_features() view reports the ISA.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#include "field_ifma256x4.hpp"
#include "lanes_kernels.hpp"
#include "lanes_tables.hpp"

namespace secp256k1::lanes::detail {

constinit const LaneFieldKernels kLaneFieldKernelsIfma256 = field::make_lane_kernels<field::Ifma256x4>(LaneBackend::Ifma256x4);

} // namespace secp256k1::lanes::detail
