#ifndef SECP256K1_DETAIL_SEC1_CANONICAL_HPP
#define SECP256K1_DETAIL_SEC1_CANONICAL_HPP

/* ============================================================================
 * SEC1 canonical point-encoding predicates (internal, header-only)
 * ============================================================================
 * Single implementation of the range/prefix checks that the GPU C ABI host
 * gate (src/cpu/src/ufsecp_gpu_impl.cpp) applies to caller-supplied point
 * encodings before they reach a GPU kernel. The CPU parser rejects a
 * coordinate >= p (FE::parse_bytes_strict); the GPU kernels would instead
 * reduce it, accepting the x+p alias of a point the CPU rejects. This gate
 * closes that divergence.
 *
 * Scope: prefix byte and coordinate range ONLY. Curve membership and
 * decompression are not checked here. All inputs are public, so the
 * variable-time memcmp is correct.
 *
 * Dependencies: <cstdint>, <cstddef>, <cstring> only.
 * ============================================================================ */

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace secp256k1::detail {

/* secp256k1 field prime p, big-endian (see SEC 2):
 * FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F.
 * A SEC1 coordinate is canonical only when, read big-endian, it is < p. */
inline constexpr std::uint8_t kSec1FieldPrimeBE[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F,
};

/* One 32-byte big-endian coordinate: true iff value < p. */
inline bool sec1_coordinate_below_p(const std::uint8_t* coord32) noexcept {
    return std::memcmp(coord32, kSec1FieldPrimeBE, 32) < 0;
}

/* One 33-byte compressed encoding: prefix 0x02/0x03 and x < p. */
inline bool sec1_compressed_canonical(const std::uint8_t* enc33) noexcept {
    if (enc33[0] != 0x02 && enc33[0] != 0x03) {
        return false;
    }
    return sec1_coordinate_below_p(enc33 + 1);
}

/* One 65-byte uncompressed encoding: prefix 0x04, x < p and y < p. */
inline bool sec1_uncompressed_canonical(const std::uint8_t* enc65) noexcept {
    if (enc65[0] != 0x04) {
        return false;
    }
    return sec1_coordinate_below_p(enc65 + 1) &&
           sec1_coordinate_below_p(enc65 + 33);
}

/* Batch forms: `count` contiguous encodings; true iff every one is canonical.
 * count == 0 is vacuously true. */
inline bool sec1_compressed_batch_canonical(const std::uint8_t* pubkeys33,
                                            std::size_t count) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        if (!sec1_compressed_canonical(pubkeys33 + index * 33)) {
            return false;
        }
    }
    return true;
}

inline bool sec1_uncompressed_batch_canonical(const std::uint8_t* pubkeys65,
                                              std::size_t count) noexcept {
    for (std::size_t index = 0; index < count; ++index) {
        if (!sec1_uncompressed_canonical(pubkeys65 + index * 65)) {
            return false;
        }
    }
    return true;
}

}  // namespace secp256k1::detail

#endif  /* SECP256K1_DETAIL_SEC1_CANONICAL_HPP */
