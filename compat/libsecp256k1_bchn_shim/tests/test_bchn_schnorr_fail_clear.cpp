// ============================================================================
// Regression: BCHN Schnorr sign failure clears caller signature output
// ============================================================================

#include <secp256k1.h>
#include <secp256k1_schnorr.h>

#include <cstdio>
#include <cstring>

static bool all_zero(const unsigned char* p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (p[i] != 0) return false;
    }
    return true;
}

int main() {
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ctx) {
        std::printf("[FAIL] context_create\n");
        return 1;
    }

    unsigned char sig64[64];
    unsigned char msg32[32]{};
    unsigned char bad_seckey[32]{};
    std::memset(sig64, 0xA5, sizeof(sig64));
    msg32[0] = 1;

    const int rc = secp256k1_schnorr_sign(
        ctx, sig64, msg32, bad_seckey, nullptr, nullptr);
    const bool cleared = all_zero(sig64, sizeof(sig64));

    secp256k1_context_destroy(ctx);

    if (rc != 0) {
        std::printf("[FAIL] invalid BCH Schnorr seckey returned %d\n", rc);
        return 1;
    }
    if (!cleared) {
        std::printf("[FAIL] BCH Schnorr sign failure left signature bytes visible\n");
        return 1;
    }

    std::printf("[PASS] BCH Schnorr sign failure clears signature output\n");
    return 0;
}
