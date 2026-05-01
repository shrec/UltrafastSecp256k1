/* ============================================================================
 * UltrafastSecp256k1 -- CUDA Backend Host Helpers (declarations)
 * ============================================================================
 * Pure C++ header -- no CUDA types, no nvcc-incompatible features.
 * Only POD types (uint8_t, uint64_t, size_t, bool) cross this boundary.
 * ============================================================================ */

#ifndef SECP256K1_GPU_CUDA_HOST_HELPERS_H
#define SECP256K1_GPU_CUDA_HOST_HELPERS_H

#include <cstdint>
#include <cstddef>

namespace secp256k1 {
namespace gpu {
namespace cuda_host {

/** Decompress a 33-byte compressed pubkey to Jacobian form (4×uint64 limbs).
 *  Returns false on bad prefix or no valid y. */
bool decompress_pubkey(const uint8_t pub[33],
                       uint64_t out_x[4], uint64_t out_y[4], uint64_t out_z[4]);

/** Convert Jacobian point (4×uint64 limbs) → 33-byte compressed pubkey. */
void jacobian_to_compressed(const uint64_t x[4], const uint64_t y[4],
                            const uint64_t z[4], bool infinity,
                            uint8_t out[33]);

/** Accumulate an array of Jacobian points into a single affine sum.
 *  Returns false if the result is point-at-infinity (out33 zeroed). */
bool msm_accumulate(const uint64_t* jac_x,    /* n * 4 limbs */
                    const uint64_t* jac_y,     /* n * 4 limbs */
                    const uint64_t* jac_z,     /* n * 4 limbs */
                    const bool* infinity,       /* n flags */
                    size_t n,
                    uint8_t out33[33]);

} // namespace cuda_host
} // namespace gpu
} // namespace secp256k1

#endif // SECP256K1_GPU_CUDA_HOST_HELPERS_H
