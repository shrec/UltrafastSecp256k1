#ifndef SECP256K1_CONTEXT_HPP
#define SECP256K1_CONTEXT_HPP
#pragma once

// ============================================================================
// CurveContext -- Custom Generator Point & Curve Parameters
// ============================================================================
// Allows users to:
//   1. Use a custom generator point G (default: standard secp256k1 G)
//   2. Define custom curve configurations
//   3. Override order n, cofactor h, and curve name
//
// Design:
//   - Zero overhead for standard secp256k1 (nullptr = use defaults)
//   - constexpr-ready for compile-time known generators
//   - No heap allocation; POD-like structure
//
// Usage:
//   // Standard secp256k1 (default)
//   auto pub = secp256k1::derive_public_key(privkey);
//
//   // Custom generator
//   CurveContext ctx = CurveContext::secp256k1();
//   ctx.generator = my_custom_G;
//   auto pub = secp256k1::derive_public_key(privkey, &ctx);
// ============================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

// -- Curve Context ------------------------------------------------------------

struct CurveContext {
    // Generator point (base point G)
    fast::Point generator;
    
    // Curve order n as raw 32 bytes (big-endian).
    // Stored as raw bytes because Scalar reduces mod n, so Scalar(n) == 0.
    // Default: secp256k1 order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    std::array<std::uint8_t, 32> order{};
    
    // Cofactor h (secp256k1: h = 1)
    std::uint32_t cofactor = 1;
    
    // Curve name (for diagnostics / logging)
    // Fixed buffer to avoid heap allocation
    char name[32] = {};
    
    // -- Factory methods --------------------------------------------------
    
    // Standard secp256k1 context (default parameters)
    static CurveContext secp256k1_default() {
        CurveContext ctx;
        ctx.generator = fast::Point::generator();
        // secp256k1 order n (big-endian bytes)
        ctx.order = {
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
            0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
            0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
        };
        ctx.cofactor = 1;
        set_name(ctx, "secp256k1");
        return ctx;
    }
    
    // Create context with custom generator (same curve, different G)
    static CurveContext with_generator(const fast::Point& custom_G,
                                        const char* label = "secp256k1-custom") {
        CurveContext ctx = secp256k1_default();
        ctx.generator = custom_G;
        set_name(ctx, label);
        return ctx;
    }
    
    // Create from arbitrary parameters
    // order_n: 32-byte big-endian curve order
    static CurveContext custom(const fast::Point& G,
                               const std::array<std::uint8_t, 32>& order_n,
                               std::uint32_t h = 1,
                               const char* label = "custom") {
        CurveContext ctx;
        ctx.generator = G;
        ctx.order = order_n;
        ctx.cofactor = h;
        set_name(ctx, label);
        return ctx;
    }
    
    // -- Helpers ----------------------------------------------------------
    
    std::string_view curve_name() const noexcept {
        return std::string_view(name);
    }
    
private:
    static void set_name(CurveContext& ctx, const char* label) {
        std::memset(ctx.name, 0, sizeof(ctx.name));
        std::size_t len = 0;
        // cppcheck-suppress arrayIndexOutOfBoundsCond  ; len is bounded by sizeof(ctx.name)-1
        while (len < sizeof(ctx.name) - 1 && label[len]) {
            ctx.name[len] = label[len];
            ++len;
        }
    }
};

// -- Context-Aware Operations -------------------------------------------------

// Get the effective generator point (custom or default secp256k1 G)
inline const fast::Point& effective_generator(const CurveContext* ctx = nullptr) {
    if (ctx) return ctx->generator;
    // Return standard secp256k1 generator via static singleton
    static const fast::Point default_G = fast::Point::generator();
    return default_G;
}

// Derive public key from private key: pubkey = privkey * G
// ctx: nullptr = standard secp256k1, or custom context
inline fast::Point derive_public_key(const fast::Scalar& private_key,
                                      const CurveContext* ctx = nullptr) {
    return effective_generator(ctx).scalar_mul(private_key);
}

// Scalar multiplication with context's generator: result = scalar * G
inline fast::Point scalar_mul_G(const fast::Scalar& scalar,
                                 const CurveContext* ctx = nullptr) {
    return effective_generator(ctx).scalar_mul(scalar);
}

} // namespace secp256k1

#endif // SECP256K1_CONTEXT_HPP
