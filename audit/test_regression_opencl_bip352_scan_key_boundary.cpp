// ============================================================================
// test_regression_opencl_bip352_scan_key_boundary.cpp
// Regression: OpenCL BIP-352 scan private key must use parse_bytes_strict_nonzero
// (SEC-002, Rule 11). Scalar::from_bytes silently reduces mod n; a scan key == n
// becomes 0, and n+1 becomes 1. Fix: parse_bytes_strict_nonzero, fail on invalid.
//
// SKB-1: scan_key == n  → rejected by parse_bytes_strict_nonzero
// SKB-2: scan_key == n+1 → rejected
// SKB-3: scan_key == 0  → rejected
// SKB-4: scan_key == 0xFF..FF (> n) → rejected
// SKB-5: valid scan_key (7) → accepted
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstring>
#include <cstdio>
#include <array>

#include "secp256k1/scalar.hpp"

static constexpr int ADVISORY_SKIP_CODE = 77;
static int g_fail = 0;
#define ASSERT_TRUE(cond, msg)  do { if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)
#define ASSERT_FALSE(cond, msg) do { if ( (cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)

// secp256k1 group order n (big-endian)
static const uint8_t kN[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};
static const uint8_t kNp1[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x42
};
static const uint8_t kAllFF[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF
};

static void test_scan_key_equal_n_rejected() {
    secp256k1::fast::Scalar out;
    ASSERT_FALSE(secp256k1::fast::Scalar::parse_bytes_strict_nonzero(kN, out),
                 "[SKB-1] scan_key == n must be rejected");
}
static void test_scan_key_n_plus_1_rejected() {
    secp256k1::fast::Scalar out;
    ASSERT_FALSE(secp256k1::fast::Scalar::parse_bytes_strict_nonzero(kNp1, out),
                 "[SKB-2] scan_key == n+1 must be rejected");
}
static void test_scan_key_zero_rejected() {
    uint8_t zero[32] = {};
    secp256k1::fast::Scalar out;
    ASSERT_FALSE(secp256k1::fast::Scalar::parse_bytes_strict_nonzero(zero, out),
                 "[SKB-3] scan_key == 0 must be rejected");
}
static void test_scan_key_all_ff_rejected() {
    secp256k1::fast::Scalar out;
    ASSERT_FALSE(secp256k1::fast::Scalar::parse_bytes_strict_nonzero(kAllFF, out),
                 "[SKB-4] scan_key == 0xFF..FF must be rejected");
}
static void test_valid_scan_key_accepted() {
    uint8_t sk[32] = {}; sk[31] = 7;
    secp256k1::fast::Scalar out;
    ASSERT_TRUE(secp256k1::fast::Scalar::parse_bytes_strict_nonzero(sk, out),
                "[SKB-5] valid scan_key (7) must be accepted");
    ASSERT_FALSE(out.is_zero(), "[SKB-5] parsed scalar must be non-zero");
}

int test_regression_opencl_bip352_scan_key_boundary_run() {
    g_fail = 0;
    test_scan_key_equal_n_rejected();
    test_scan_key_n_plus_1_rejected();
    test_scan_key_zero_rejected();
    test_scan_key_all_ff_rejected();
    test_valid_scan_key_accepted();
    if (g_fail == 0)
        std::printf("PASS: OpenCL BIP-352 scan key boundary (SEC-002, SKB-1..5)\n");
    else
        std::printf("FAIL: OpenCL BIP-352 scan key boundary: %d failure(s)\n", g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_opencl_bip352_scan_key_boundary_run(); }
#endif
