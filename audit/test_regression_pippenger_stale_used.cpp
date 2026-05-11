// ============================================================================
// REGRESSION: Pippenger stale used[] across windows (BUG-01)
// ============================================================================
// Root cause A (unsigned path, c<=6): a single memset(used, 0) before the outer
// window loop meant window W's dirty used[] bits persisted into window W+1.
// Fix: memset(used, 0) at the top of EVERY window iteration.
//
// Root cause B (signed-digit path, c>=7): carry propagation from the second-to-last
// window could push the last window's digit to (1<<c) = num_buckets_unsigned, producing
// abs_d == num_buckets_unsigned — an out-of-bounds access on TLS arrays sized to exactly
// num_buckets_unsigned. Fix: allocate one extra slot (tls_alloc_size = num_buckets_unsigned+1).
//
// Tests:
//   PIP-R1: n=48 (c=5, unsigned path)
//   PIP-R2: n=64 (c=5)
//   PIP-R3: n=80 (c=6)
//   PIP-R4: n=128 (c=6)
//   PIP-R5: Repeated scalar: all s_i == s forces maximum bucket reuse
//   PIP-R6: Multi-window with negated inputs
//   PIP-R7: All-zero scalars -> infinity
//   PIP-R8: n=512 (c=7, signed path)  — covers Root cause B fix
//   PIP-R9: n=1000 (c=8, signed path) — covers Root cause B fix
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>
#include <vector>
#include <cstring>

#include "secp256k1/pippenger.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("    [OK] %s\n", msg); }
    else       { ++g_fail; printf("  [FAIL] %s\n", msg); }
    fflush(stdout);
}

// Reference: naive sum(s_i * P_i)
static Point naive_msm(const std::vector<Scalar>& scalars,
                       const std::vector<Point>& points) {
    Point sum = Point::infinity();
    for (std::size_t i = 0; i < scalars.size(); ++i) {
        sum = sum.add(points[i].scalar_mul(scalars[i]));
    }
    return sum;
}

static bool points_equal(const Point& A, const Point& B) {
    if (A.is_infinity() && B.is_infinity()) return true;
    if (A.is_infinity() || B.is_infinity()) return false;
    auto ac = A.to_compressed();
    auto bc = B.to_compressed();
    return ac == bc;
}

// Deterministic scalar from seed
static Scalar scalar_from_seed(uint64_t seed) {
    uint8_t buf[32]{};
    for (int i = 0; i < 8; ++i)
        buf[i] = static_cast<uint8_t>((seed >> (i * 8)) & 0xFF);
    buf[31] = 0x01; // ensure non-zero
    return Scalar::from_bytes(buf);
}

// Build n distinct affine points: P_i = i * G
static std::vector<Point> make_points(std::size_t n) {
    std::vector<Point> pts;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Scalar s = scalar_from_seed(static_cast<uint64_t>(i + 1));
        pts.push_back(Point::generator().scalar_mul(s));
    }
    return pts;
}

static std::vector<Scalar> make_scalars(std::size_t n, uint64_t seed_offset = 0) {
    std::vector<Scalar> ss;
    ss.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        ss.push_back(scalar_from_seed(static_cast<uint64_t>(i) + 1 + seed_offset * 1000));
    }
    return ss;
}

// ─── pip_stale_used_boundary ─────────────────────────────────────────────────
static void pip_stale_used_boundary(std::size_t n, const char* label) {
    auto pts = make_points(n);
    auto scs = make_scalars(n);
    auto ref = naive_msm(scs, pts);
    auto got = secp256k1::pippenger_msm(scs.data(), pts.data(), n);
    char msg[128];
    std::snprintf(msg, sizeof(msg), "%s: pippenger_msm == naive_msm", label);
    check(points_equal(ref, got), msg);
}

// ─── pip_stale_used_repeated_scalar ──────────────────────────────────────────
// All scalars equal: maximises bucket reuse across all windows
static void pip_stale_used_repeated_scalar() {
    constexpr std::size_t N = 64;
    auto pts = make_points(N);
    Scalar s = scalar_from_seed(42);
    std::vector<Scalar> scs(N, s);
    // sum(s * P_i) = s * sum(P_i)
    Point sum_P = Point::infinity();
    for (const auto& p : pts) sum_P = sum_P.add(p);
    auto ref = sum_P.scalar_mul(s);
    auto got = secp256k1::pippenger_msm(scs.data(), pts.data(), N);
    check(points_equal(ref, got), "pip_stale_used_repeated_scalar: all s_i == s");
}

// ─── pip_stale_used_negated_inputs ───────────────────────────────────────────
// Mix of P_i and -P_i to force bucket collision patterns
static void pip_stale_used_negated_inputs() {
    constexpr std::size_t N = 80;
    auto pts = make_points(N / 2);
    // Append negated versions
    for (std::size_t i = 0; i < N / 2; ++i) {
        Point neg = pts[i]; neg.negate_inplace();
        pts.push_back(neg);
    }
    auto scs = make_scalars(N, 7);
    auto ref = naive_msm(scs, pts);
    auto got = secp256k1::pippenger_msm(scs.data(), pts.data(), N);
    check(points_equal(ref, got), "pip_stale_used_negated_inputs: mixed pos/neg points");
}

// ─── pip_stale_used_identity_check ───────────────────────────────────────────
// All-zero scalars: result must be point at infinity regardless of used[] state
static void pip_stale_used_identity_check() {
    constexpr std::size_t N = 64;
    auto pts = make_points(N);
    std::vector<Scalar> scs(N, Scalar::zero());
    auto got = secp256k1::pippenger_msm(scs.data(), pts.data(), N);
    check(got.is_infinity(), "pip_stale_used_identity_check: all-zero scalars -> infinity");
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
#else
int test_regression_pippenger_stale_used_run() {
#endif
    printf("====================================================================\n");
    printf("REGRESSION: Pippenger stale used[] / signed carry overflow (BUG-01)\n");
    printf("====================================================================\n\n");

    pip_stale_used_boundary(48,  "PIP-R1 n=48  c=5");
    pip_stale_used_boundary(64,  "PIP-R2 n=64  c=5");
    pip_stale_used_boundary(80,  "PIP-R3 n=80  c=6");
    pip_stale_used_boundary(128, "PIP-R4 n=128 c=6");
    pip_stale_used_repeated_scalar();
    pip_stale_used_negated_inputs();
    pip_stale_used_identity_check();
    pip_stale_used_boundary(512,  "PIP-R8  n=512  c=7 signed");
    pip_stale_used_boundary(1000, "PIP-R9  n=1000 c=8 signed");

    printf("\n--- Result: %d passed, %d failed ---\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
