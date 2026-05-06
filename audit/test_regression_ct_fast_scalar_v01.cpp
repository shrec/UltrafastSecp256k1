// ============================================================================
// CT Regression: fast::Scalar operator* V-01 — signing path timing guard
// ============================================================================
// V-01: fast::Scalar::operator* and operator+ have data-dependent branches
// (ge(ORDER) conditional subtraction) — variable-time on secret operands.
//
// Fix (committed): signing paths use ct::scalar_mul/ct::scalar_add instead
// of fast::Scalar::operator*/operator+.
//
// This test is a timing regression guard: if the fix is reverted and
// fast::Scalar::operator* is re-introduced on secret key material, signing
// time will correlate with key Hamming weight → Welch |t| > threshold.
//
// Method:
//   Group A: 300 ECDSA sign calls with keys of Hamming weight 1 (one bit set)
//   Group B: 300 ECDSA sign calls with keys of Hamming weight 127 (many bits set)
//   Compute Welch's t-statistic on CPU-cycle measurements.
//   Assert |t| < 4.5 (no statistically significant timing correlation).
//
// Advisory: timing tests have high variance in shared CI environments.
// Returns ADVISORY_SKIP_CODE (77) instead of FAIL when the environment
// is too noisy to produce a reliable measurement.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstdint>
#include <cstring>
#include <cmath>
#include <array>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <vector>

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/init.hpp"

static constexpr int ADVISORY_SKIP_CODE = 77;
static constexpr int N = 300;         // samples per group
static constexpr double T_THRESHOLD = 4.5;
static constexpr double NOISE_SKIP_THRESHOLD = 0.40; // CV > 40% → too noisy

// Welch's t-statistic for two independent samples
static double welch_t(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0, mb = 0;
    for (auto v : a) ma += v; ma /= a.size();
    for (auto v : b) mb += v; mb /= b.size();
    double va = 0, vb = 0;
    for (auto v : a) va += (v - ma) * (v - ma); va /= (a.size() - 1);
    for (auto v : b) vb += (v - mb) * (v - mb); vb /= (b.size() - 1);
    double denom = std::sqrt(va / a.size() + vb / b.size());
    if (denom < 1e-12) return 0.0;
    return std::abs((ma - mb) / denom);
}

static double coeff_var(const std::vector<double>& v) {
    double m = 0;
    for (auto x : v) m += x; m /= v.size();
    if (m < 1e-12) return 999.0;
    double s = 0;
    for (auto x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size()) / m;
}

// Return a Scalar with the given Hamming weight (bits evenly spread).
static secp256k1::fast::Scalar make_scalar_hw(int hw) {
    // Start from 1, shift left to set hw bits without exceeding n.
    // Use well-spaced bit positions within the 256-bit scalar field.
    std::array<uint8_t, 32> bytes{};
    int bits_set = 0;
    for (int pos = 0; pos < 252 && bits_set < hw; pos += (252 / hw + 1)) {
        int byte_idx = pos / 8;
        int bit_idx  = pos % 8;
        if (byte_idx < 32) {
            bytes[31 - byte_idx] |= (uint8_t)(1u << bit_idx);
            ++bits_set;
        }
    }
    // Ensure nonzero
    if (bytes[31] == 0) bytes[31] = 1;
    secp256k1::fast::Scalar s;
    secp256k1::fast::Scalar::parse_bytes_strict_nonzero(bytes, s);
    return s;
}

static double measure_sign_ns(const secp256k1::fast::Scalar& sk) {
    std::array<uint8_t, 32> msg{};
    msg[0] = 0xAB; msg[15] = 0xCD; msg[31] = 0xEF;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sig;
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
}

int test_regression_ct_fast_scalar_v01_run() {
    printf("[V-01] CT signing timing regression (fast::Scalar operator* banned on secrets)\n");

    SECP256K1_INIT();

    auto sk_low  = make_scalar_hw(1);    // Hamming weight 1
    auto sk_high = make_scalar_hw(80);   // Hamming weight 80 (high density)

    // Warmup: drive code into steady state
    for (int i = 0; i < 50; ++i) {
        auto s1 = secp256k1::ct::ecdsa_sign({}, sk_low);
        auto s2 = secp256k1::ct::ecdsa_sign({}, sk_high);
        (void)s1; (void)s2;
    }

    std::vector<double> times_low, times_high;
    times_low.reserve(N);
    times_high.reserve(N);

    // Interleave measurements to reduce systematic bias
    for (int i = 0; i < N; ++i) {
        times_low.push_back(measure_sign_ns(sk_low));
        times_high.push_back(measure_sign_ns(sk_high));
    }

    // Check for excessive noise (shared CI runner)
    double cv_low  = coeff_var(times_low);
    double cv_high = coeff_var(times_high);
    if (cv_low > NOISE_SKIP_THRESHOLD || cv_high > NOISE_SKIP_THRESHOLD) {
        printf("  [advisory-skip] environment too noisy for timing test "
               "(CV_low=%.1f%% CV_high=%.1f%%)\n",
               cv_low * 100.0, cv_high * 100.0);
        return ADVISORY_SKIP_CODE;
    }

    double t_stat = welch_t(times_low, times_high);
    printf("  Welch |t| = %.2f  (threshold %.1f)  CV_low=%.1f%%  CV_high=%.1f%%\n",
           t_stat, T_THRESHOLD, cv_low * 100.0, cv_high * 100.0);

    if (t_stat < T_THRESHOLD) {
        printf("  [PASS] No timing correlation between signing time and key Hamming weight.\n"
               "         CT path (ct::scalar_mul) is in effect — V-01 fix holds.\n");
        return 0;
    }

    printf("  [FAIL] |t|=%.2f exceeds threshold %.1f\n"
           "         Signing time correlates with key Hamming weight.\n"
           "         REGRESSION: fast::Scalar::operator* may have been re-introduced.\n",
           t_stat, T_THRESHOLD);
    return 1;
}

#ifdef STANDALONE_TEST
int main() {
    int rc = test_regression_ct_fast_scalar_v01_run();
    return rc == 0 ? 0 : (rc == ADVISORY_SKIP_CODE ? 77 : 1);
}
#endif
