// ============================================================================
// lanes/lanes_tables.hpp -- the per-ISA kernel tables (declarations only)
// ============================================================================
//
// Each table is defined, constant-initialised, in its own ISA translation
// unit (lanes_<isa>.cpp) and referenced only by lanes_dispatch.cpp, and only
// when the matching SECP256K1_LANES_HAVE_<ISA> macro says that TU is built.
// Constant initialisation matters: no ISA-specific code may run at static
// initialisation time on a CPU that lacks the ISA.
// ============================================================================

#ifndef SECP256K1_LANES_TABLES_HPP
#define SECP256K1_LANES_TABLES_HPP

#include "secp256k1/lanes.hpp"
#include "lanes_verify_abi.hpp"

namespace secp256k1::lanes::detail {

extern const LaneFieldKernels kLaneFieldKernelsAvx2;      // lanes_avx2.cpp
extern const LaneFieldKernels kLaneFieldKernelsAvxIfma;   // lanes_avx_ifma.cpp
extern const LaneFieldKernels kLaneFieldKernelsIfma256;   // lanes_ifma256.cpp
extern const LaneFieldKernels kLaneFieldKernelsIfma512;   // lanes_ifma512.cpp

// Lane verify kernels (lanes_verify_kernels.hpp), defined in the same ISA TUs
// and referenced only by lanes_verify.cpp under the same HAVE macros.
extern const LaneVerifyKernels kLaneVerifyKernelsAvx2;
extern const LaneVerifyKernels kLaneVerifyKernelsAvxIfma;
extern const LaneVerifyKernels kLaneVerifyKernelsIfma256;
extern const LaneVerifyKernels kLaneVerifyKernelsIfma512;

} // namespace secp256k1::lanes::detail

#endif // SECP256K1_LANES_TABLES_HPP
