/* ============================================================================
 * UltrafastSecp256k1 -- GPU Backend Host-Side Fallbacks
 * ============================================================================
 * Backend-neutral host-side fallback implementations for GPU operations that
 * do not yet have a real device-side kernel on every backend.
 *
 * These helpers let the default `GpuBackend` virtual methods do the right
 * thing (correct, deterministic, byte-identical to the CPU C ABI result)
 * even when the concrete backend has not overridden them. They run on the
 * host CPU; once a real GPU kernel lands on a backend, that backend simply
 * overrides the virtual method and bypasses the fallback.
 *
 * Adding a fallback here keeps the library true to the
 * "uniform infrastructure across all platforms" rule: every backend exposes
 * every operation, even if some operations are still computed host-side
 * pending native kernels.
 * ============================================================================ */

#include "gpu_backend.hpp"

#include <array>
#include <cstring>

#include "secp256k1/scalar.hpp"
#include "secp256k1/zk.hpp"

namespace secp256k1 {
namespace gpu {

/* ----------------------------------------------------------------------------
 * BIP-340 Schnorr SNARK foreign-field witness — host-side batch fallback.
 *
 * Iterates the input in a host loop and serialises each record into the same
 * 472-byte flat layout consumed by `ufsecp_gpu_zk_schnorr_snark_witness_batch`.
 * Layout matches `ufsecp_schnorr_snark_witness_t` (see ufsecp.h).
 *
 * PUBLIC-DATA operation: no secret values are touched.
 * --------------------------------------------------------------------------*/
GpuError schnorr_snark_witness_batch_cpu_fallback(
    const uint8_t* msgs32,
    const uint8_t* pubkeys_x32,
    const uint8_t* sigs64,
    size_t         count,
    uint8_t*       out_flat)
{
    if (count == 0) return GpuError::Ok;
    if (!msgs32 || !pubkeys_x32 || !sigs64 || !out_flat)
        return GpuError::NullArg;

    constexpr size_t REC = 472; // SCHNORR_SNARK_WITNESS_BYTES (asserted below)
    static_assert(GpuBackend::SCHNORR_SNARK_WITNESS_BYTES == REC,
                  "fallback record size out of sync with GpuBackend");

    // Field offsets within the 472-byte record (mirror ufsecp_schnorr_snark_witness_t).
    constexpr size_t OFF_MSG     = 0;
    constexpr size_t OFF_SIG_R   = 32;
    constexpr size_t OFF_SIG_S   = 64;
    constexpr size_t OFF_PUB_X   = 96;
    constexpr size_t OFF_R_Y     = 128;
    constexpr size_t OFF_PUB_Y   = 160;
    constexpr size_t OFF_E       = 192;
    constexpr size_t OFF_LMB_R   = 224; // 6 × 5 × uint64 = 240 B
    constexpr size_t LMB_BYTES   = 5 * sizeof(uint64_t); // 40
    constexpr size_t OFF_VALID   = 464;
    // 4-byte tail padding for 8-byte struct alignment (matches C struct).

    static_assert(sizeof(secp256k1::zk::ForeignFieldLimbs) == LMB_BYTES,
                  "ForeignFieldLimbs size mismatch with C ABI");

    for (size_t i = 0; i < count; ++i) {
        std::array<uint8_t, 32> msg_arr{};
        std::array<uint8_t, 32> pub_x_arr{};
        std::array<uint8_t, 32> sig_r_arr{};
        std::array<uint8_t, 32> sig_s_arr{};
        std::memcpy(msg_arr.data(),   msgs32       + i * 32, 32);
        std::memcpy(pub_x_arr.data(), pubkeys_x32  + i * 32, 32);
        std::memcpy(sig_r_arr.data(), sigs64       + i * 64,      32);
        std::memcpy(sig_s_arr.data(), sigs64       + i * 64 + 32, 32);

        // Parse s strictly; mirror ufsecp_zk_schnorr_snark_witness behaviour
        // for invalid s — emit a record with valid=0 and zeroed witness body.
        secp256k1::fast::Scalar sig_s{};
        bool s_ok = secp256k1::fast::Scalar::parse_bytes_strict(sig_s_arr, sig_s)
                    && !sig_s.is_zero();

        uint8_t* rec = out_flat + i * REC;
        std::memset(rec, 0, REC);

        // Public inputs always copied through, even on bad s, so the prover
        // can build a "rejected" witness without re-reading inputs.
        std::memcpy(rec + OFF_MSG,   msg_arr.data(),   32);
        std::memcpy(rec + OFF_SIG_R, sig_r_arr.data(), 32);
        std::memcpy(rec + OFF_SIG_S, sig_s_arr.data(), 32);
        std::memcpy(rec + OFF_PUB_X, pub_x_arr.data(), 32);

        if (!s_ok) {
            // valid = 0 already from memset
            continue;
        }

        secp256k1::zk::SchnorrSnarkWitness w =
            secp256k1::zk::schnorr_snark_witness(
                msg_arr, pub_x_arr, sig_r_arr, sig_s);

        std::memcpy(rec + OFF_R_Y,   w.bytes_r_y.data(),   32);
        std::memcpy(rec + OFF_PUB_Y, w.bytes_pub_y.data(), 32);
        std::memcpy(rec + OFF_E,     w.bytes_e.data(),     32);

        std::memcpy(rec + OFF_LMB_R + 0 * LMB_BYTES, &w.sig_r, LMB_BYTES);
        std::memcpy(rec + OFF_LMB_R + 1 * LMB_BYTES, &w.sig_s, LMB_BYTES);
        std::memcpy(rec + OFF_LMB_R + 2 * LMB_BYTES, &w.pub_x, LMB_BYTES);
        std::memcpy(rec + OFF_LMB_R + 3 * LMB_BYTES, &w.r_y,   LMB_BYTES);
        std::memcpy(rec + OFF_LMB_R + 4 * LMB_BYTES, &w.pub_y, LMB_BYTES);
        std::memcpy(rec + OFF_LMB_R + 5 * LMB_BYTES, &w.e,     LMB_BYTES);

        const int valid_int = w.valid ? 1 : 0;
        std::memcpy(rec + OFF_VALID, &valid_int, sizeof(int));
    }

    return GpuError::Ok;
}

} // namespace gpu
} // namespace secp256k1
