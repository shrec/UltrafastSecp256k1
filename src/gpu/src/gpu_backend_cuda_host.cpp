/* ============================================================================
 * UltrafastSecp256k1 -- CUDA Backend Host Helpers
 * ============================================================================
 * Compiled as C++ (not CUDA) to avoid nvcc parsing CPU headers that use
 * std::string, std::vector, and C++20 features incompatible with nvcc.
 *
 * These functions do host-side point decompression, Jacobian→affine
 * conversion, and affine point addition using the CPU FieldElement.
 * ============================================================================ */

#include <cstring>
#include <cstdint>
#include <array>
#include "secp256k1/field.hpp"

namespace secp256k1{
namespace gpu {
namespace cuda_host {

/* Limb layout is identical between CUDA FieldElement and CPU FieldElement:
   4×uint64_t LE limbs (256 bits). We pass raw limbs across the compilation
   boundary to avoid including CUDA types in this C++ TU. */

static secp256k1::fast::FieldElement fe_from_ptr(const uint64_t* p) {
    std::array<uint64_t, 4> a{p[0], p[1], p[2], p[3]};
    return secp256k1::fast::FieldElement::from_limbs(a);
}

bool decompress_pubkey(const uint8_t pub[33],
                       uint64_t out_x[4], uint64_t out_y[4], uint64_t out_z[4]) {
    uint8_t prefix = pub[0];
    if (prefix != 0x02 && prefix != 0x03) return false;

    secp256k1::fast::FieldElement cpux;
    if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub + 1, cpux))
        return false;

    auto x2 = cpux * cpux;
    auto x3 = x2 * cpux;
    auto y2 = x3 + secp256k1::fast::FieldElement::from_uint64(7);

    auto y = y2.sqrt();

    auto ybytes = y.to_bytes();
    bool y_is_odd = (ybytes[31] & 1) != 0;
    bool want_odd = (prefix == 0x03);
    if (y_is_odd != want_odd) y = y.negate();

    const auto& xarr = cpux.limbs();
    const auto& yarr = y.limbs();
    for (int i = 0; i < 4; ++i) {
        out_x[i] = xarr[i];
        out_y[i] = yarr[i];
    }
    out_z[0] = 1; out_z[1] = 0; out_z[2] = 0; out_z[3] = 0;
    return true;
}

void jacobian_to_compressed(const uint64_t x[4], const uint64_t y[4],
                            const uint64_t z[4], bool infinity,
                            uint8_t out[33]) {
    if (infinity) {
        std::memset(out, 0, 33);
        return;
    }

    auto cx = fe_from_ptr(x);
    auto cy = fe_from_ptr(y);
    auto cz = fe_from_ptr(z);

    auto zinv  = cz.inverse();
    auto zinv2 = zinv * zinv;
    auto zinv3 = zinv2 * zinv;

    auto ax = cx * zinv2;
    auto ay = cy * zinv3;

    auto ybytes = ay.to_bytes();
    out[0] = (ybytes[31] & 1) ? 0x03 : 0x02;
    auto xbytes = ax.to_bytes();
    std::memcpy(out + 1, xbytes.data(), 32);
}

/** Accumulate affine points from Jacobian-form partial results.
 *  Converts each Jacobian → affine, then sums using affine addition.
 *  Returns false if result is point at infinity. */
bool msm_accumulate(const uint64_t* jac_x,    /* n * 4 limbs */
                    const uint64_t* jac_y,     /* n * 4 limbs */
                    const uint64_t* jac_z,     /* n * 4 limbs */
                    const bool* infinity,       /* n flags */
                    size_t n,
                    uint8_t out33[33]) {
    bool have_acc = false;
    secp256k1::fast::FieldElement acc_x, acc_y;

    for (size_t i = 0; i < n; ++i) {
        if (infinity[i]) continue;

        auto cx = fe_from_ptr(jac_x + i * 4);
        auto cy = fe_from_ptr(jac_y + i * 4);
        auto cz = fe_from_ptr(jac_z + i * 4);

        auto zinv  = cz.inverse();
        auto zinv2 = zinv * zinv;
        auto zinv3 = zinv2 * zinv;
        auto px = cx * zinv2;
        auto py = cy * zinv3;

        if (!have_acc) {
            acc_x = px; acc_y = py;
            have_acc = true;
            continue;
        }

        /* Affine point addition */
        auto dx = px + acc_x.negate();
        auto dy = py + acc_y.negate();
        auto dxb = dx.to_bytes();
        bool dx_zero = true;
        for (int k = 0; k < 32 && dx_zero; ++k) dx_zero = (dxb[k] == 0);

        if (dx_zero) {
            auto dyb = dy.to_bytes();
            bool dy_zero = true;
            for (int k = 0; k < 32 && dy_zero; ++k) dy_zero = (dyb[k] == 0);
            if (dy_zero) {
                /* Doubling */
                auto x2 = acc_x * acc_x;
                auto num = x2 + x2 + x2;
                auto den = acc_y + acc_y;
                auto lam = num * den.inverse();
                auto rx = lam * lam + acc_x.negate() + acc_x.negate();
                auto ry = lam * (acc_x + rx.negate()) + acc_y.negate();
                acc_x = rx; acc_y = ry;
            } else {
                have_acc = false; /* inverse points */
            }
        } else {
            auto lam = dy * dx.inverse();
            auto rx = lam * lam + acc_x.negate() + px.negate();
            auto ry = lam * (acc_x + rx.negate()) + acc_y.negate();
            acc_x = rx; acc_y = ry;
        }
    }

    if (!have_acc) {
        std::memset(out33, 0, 33);
        return false;
    }

    auto yb = acc_y.to_bytes();
    out33[0] = (yb[31] & 1) ? 0x03 : 0x02;
    auto xb = acc_x.to_bytes();
    std::memcpy(out33 + 1, xb.data(), 32);
    return true;
}

} // namespace cuda_host
} // namespace gpu
} // namespace secp256k1
