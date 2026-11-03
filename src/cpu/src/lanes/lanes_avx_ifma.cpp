// ============================================================================
// lanes/lanes_avx_ifma.cpp -- AvxIfma256x4 (VEX AVX-IFMA, 4 lanes) lane field kernels
// ============================================================================
//
// Compiled with per-file flags -mavx2 -mavxifma
// (src/cpu/CMakeLists.txt). Holds only the constant-initialised kernel table:
// no code here runs unless lanes_dispatch.cpp hands the table out, which it
// does only when the masked cpu_features() view reports the ISA.
//
// PUBLIC DATA ONLY: variable time (verification paths).
// ============================================================================

#include "field_avx_ifma256x4.hpp"
#include "lanes_kernels.hpp"
#include "lanes_tables.hpp"

namespace secp256k1::lanes::detail {

constinit const LaneFieldKernels kLaneFieldKernelsAvxIfma = field::make_lane_kernels<field::AvxIfma256x4>(LaneBackend::AvxIfma256x4);

} // namespace secp256k1::lanes::detail
