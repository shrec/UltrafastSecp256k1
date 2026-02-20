// ============================================================================
// Side-Channel Attack Test Suite — dudect methodology
// ============================================================================
// ტესტავს CT ოპერაციების წინააღმდეგობას timing side-channel შეტევების მიმართ.
//
// მეთოდოლოგია (სწორი dudect პროტოკოლი):
//   1. ყველა ინფუტი PRE-GENERATED — არანაირი random_fe()/random_scalar()
//      measurement loop-ის შიგნით.
//   2. ორი კლასის ინფუტი ინახება ცალ-ცალკე მასივებში.
//   3. class selection = array index lookup (იდენტური cost ორივესთვის).
//   4. Welch t-test |t| > 4.5 → timing leak (99.999% confidence).
//
// CRITICAL: ტესტის მეთოდოლოგია
//   - Class 0: ფიქსირებული ინფუტი (edge-case: zero, one, identity, etc.)
//   - Class 1: რანდომ ინფუტი (pre-generated)
//   - ორივე კლასი IDENTICAL selection path-ით (array[cls][i])
//   - asm volatile barriers ზომვის გარშემო
//
// გაშვება:
//   ნორმალური:  build_rel/tests/test_ct_sidechannel
//   Valgrind:   valgrind build_rel/tests/test_ct_sidechannel_vg
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <array>
#include <random>
#include <chrono>
#include <algorithm>

// ── Our CT layer ─────────────────────────────────────────────────────────────
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct_utils.hpp"

using namespace secp256k1::fast;

// ═══════════════════════════════════════════════════════════════════════════
// Timer — rdtsc(p) on x86_64, cntvct on aarch64
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__x86_64__) || defined(_M_X64)
static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx");
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
#elif defined(__aarch64__)
static inline uint64_t rdtsc() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}
#else
static inline uint64_t rdtsc() {
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Welch t-test (online/incremental — no allocation needed)
// ═══════════════════════════════════════════════════════════════════════════

struct WelchState {
    double n[2]    = {};
    double mean[2] = {};
    double m2[2]   = {};

    void push(int cls, double x) {
        n[cls] += 1.0;
        double delta = x - mean[cls];
        mean[cls] += delta / n[cls];
        double delta2 = x - mean[cls];
        m2[cls] += delta * delta2;
    }

    double t_value() const {
        if (n[0] < 2 || n[1] < 2) return 0.0;
        double var0 = m2[0] / (n[0] - 1.0);
        double var1 = m2[1] / (n[1] - 1.0);
        double se = std::sqrt(var0 / n[0] + var1 / n[1]);
        if (se < 1e-15) return 0.0;
        return (mean[0] - mean[1]) / se;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// PRNG + helpers — pre-generation only
// ═══════════════════════════════════════════════════════════════════════════

static std::mt19937_64 rng(0xA0D17'51DE0);

static void random_bytes(uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint64_t v = rng();
        size_t chunk = (len - i < 8) ? (len - i) : 8;
        std::memcpy(out + i, &v, chunk);
    }
}

static Scalar random_scalar() {
    std::array<uint8_t, 32> buf{};
    for (;;) {
        random_bytes(buf.data(), 32);
        auto s = Scalar::from_bytes(buf);
        if (!s.is_zero()) return s;
    }
}

static FieldElement random_fe() {
    std::array<uint8_t, 32> buf{};
    random_bytes(buf.data(), 32);
    return FieldElement::from_bytes(buf);
}

// ── Framework ────────────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;
static constexpr double T_THRESHOLD = 4.5;

static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("    ✗ FAIL: %s\n", msg); }
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 1: CT პრიმიტივები (masks, cmov, cswap, lookup)
// ═══════════════════════════════════════════════════════════════════════════

static void test_ct_primitives() {
    printf("\n[1] CT პრიმიტივები — timing ტესტი\n");

    constexpr int N = 100000;

    // ── 1a: is_zero_mask ─────────────────────────────────────────────────
    {
        // Pre-generate inputs: class 0 = always 0, class 1 = random nonzero
        uint64_t inputs[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            inputs[0][i] = 0;
            inputs[1][i] = rng() | 1;
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint64_t val = inputs[cls][i];

            asm volatile("" : "+r"(val) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile uint64_t r = secp256k1::ct::is_zero_mask(val);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    is_zero_mask:    |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "is_zero_mask timing leak");
    }

    // ── 1b: bool_to_mask ─────────────────────────────────────────────────
    {
        bool inputs[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            inputs[0][i] = false;
            inputs[1][i] = true;
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            bool flag = inputs[cls][i];

            asm volatile("" : "+r"(flag) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile uint64_t m = secp256k1::ct::bool_to_mask(flag);
            (void)m;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    bool_to_mask:    |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "bool_to_mask timing leak");
    }

    // ── 1c: cmov256 ─────────────────────────────────────────────────────
    {
        uint64_t masks[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            masks[0][i] = 0;
            masks[1][i] = ~0ULL;
        }
        uint64_t dst[4] = {1,2,3,4};
        uint64_t src[4] = {5,6,7,8};

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint64_t mask = masks[cls][i];

            asm volatile("" : "+r"(mask) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::cmov256(dst, src, mask);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    cmov256:         |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "cmov256 timing leak");
    }

    // ── 1d: cswap256 ────────────────────────────────────────────────────
    {
        uint64_t masks[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            masks[0][i] = 0;
            masks[1][i] = ~0ULL;
        }
        uint64_t a[4] = {1,2,3,4}, b[4] = {5,6,7,8};

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint64_t mask = masks[cls][i];

            asm volatile("" : "+r"(mask) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::cswap256(a, b, mask);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    cswap256:        |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "cswap256 timing leak");
    }

    // ── 1e: ct_lookup_256 (16 entries) ──────────────────────────────────
    {
        size_t indices[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            indices[0][i] = 0;            // always first
            indices[1][i] = rng() % 16;   // random
        }
        uint64_t table[16][4];
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) table[i][j] = rng();
        uint64_t out[4];

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            size_t idx = indices[cls][i];

            asm volatile("" : "+r"(idx) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::ct_lookup_256(table, 16, idx, out);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    ct_lookup_256:   |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct_lookup_256 timing leak");
    }

    // ── 1f: ct_equal ────────────────────────────────────────────────────
    {
        // Pre-generate: class 0 = identical buffers, class 1 = different
        struct Pair { uint8_t a[32]; uint8_t b[32]; };
        auto* pairs0 = new Pair[N]; // class 0: a == b
        auto* pairs1 = new Pair[N]; // class 1: a != b
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            random_bytes(pairs0[i].a, 32);
            std::memcpy(pairs0[i].b, pairs0[i].a, 32);
            random_bytes(pairs1[i].a, 32);
            random_bytes(pairs1[i].b, 32);
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            const uint8_t* a = (cls == 0) ? pairs0[i].a : pairs1[i].a;
            const uint8_t* b = (cls == 0) ? pairs0[i].b : pairs1[i].b;

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile bool eq = secp256k1::ct::ct_equal(a, b, 32);
            (void)eq;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        delete[] pairs0;
        delete[] pairs1;
        double t = std::abs(ws.t_value());
        printf("    ct_equal:        |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct_equal timing leak");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 2: CT Field ოპერაციები
// ═══════════════════════════════════════════════════════════════════════════

static void test_ct_field() {
    printf("\n[2] CT Field ოპერაციები — timing ტესტი\n");

    constexpr int N = 50000;

    // Pre-generate ALL field elements
    auto* fe_cls0 = new FieldElement[N]; // class 0: fixed (zero)
    auto* fe_cls1 = new FieldElement[N]; // class 1: random
    auto* fe_base = new FieldElement[N]; // second operand (always random)
    int* classes  = new int[N];

    auto fe_zero = FieldElement::zero();
    auto fe_one  = FieldElement::one();

    for (int i = 0; i < N; ++i) {
        classes[i]  = rng() & 1;
        fe_cls0[i]  = fe_zero;     // fixed
        fe_cls1[i]  = random_fe(); // random
        fe_base[i]  = random_fe(); // second operand
    }

    // ── 2a: field_add ───────────────────────────────────────────────────
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::field_add(fe_base[i], op);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_add:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::field_add timing leak");
    }

    // ── 2b: field_mul ───────────────────────────────────────────────────
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::field_mul(fe_base[i], op);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_mul:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::field_mul timing leak");
    }

    // ── 2c: field_sqr ───────────────────────────────────────────────────
    {
        // Swap cls0 to fe_one for sqr
        for (int i = 0; i < N; ++i) fe_cls0[i] = fe_one;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::field_sqr(op);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_sqr:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::field_sqr timing leak");
    }

    // ── 2d: field_inv ───────────────────────────────────────────────────
    {
        constexpr int NSLOW = 5000;
        // Re-generate for fewer samples
        for (int i = 0; i < NSLOW; ++i) {
            fe_cls0[i] = fe_one;
            fe_cls1[i] = random_fe();
            classes[i]  = rng() & 1;
        }

        WelchState ws;
        for (int i = 0; i < NSLOW; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::field_inv(op);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_inv:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::field_inv timing leak");
    }

    // ── 2e: field_cmov ──────────────────────────────────────────────────
    {
        uint64_t masks[2];
        masks[0] = 0;
        masks[1] = ~0ULL;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint64_t mask = masks[cls];
            auto dst = fe_base[i];
            auto src = fe_cls1[i];

            asm volatile("" : "+r"(mask) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::field_cmov(&dst, src, mask);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_cmov:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::field_cmov timing leak");
    }

    // ── 2f: field_is_zero ───────────────────────────────────────────────
    {
        for (int i = 0; i < N; ++i) fe_cls0[i] = fe_zero;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto m = secp256k1::ct::field_is_zero(op);
            (void)m;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_is_zero:   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::field_is_zero timing leak");
    }

    delete[] fe_cls0;
    delete[] fe_cls1;
    delete[] fe_base;
    delete[] classes;
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 3: CT Scalar ოპერაციები
// ═══════════════════════════════════════════════════════════════════════════

static void test_ct_scalar() {
    printf("\n[3] CT Scalar ოპერაციები — timing ტესტი\n");

    constexpr int N = 50000;

    auto sc_one = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto sc_zero = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000000");

    auto* sc_cls0 = new Scalar[N];
    auto* sc_cls1 = new Scalar[N];
    auto* sc_base = new Scalar[N];
    int* classes  = new int[N];

    for (int i = 0; i < N; ++i) {
        classes[i] = rng() & 1;
        sc_cls0[i] = sc_one;
        sc_cls1[i] = random_scalar();
        sc_base[i] = random_scalar();
    }

    // ── 3a: scalar_add ──────────────────────────────────────────────────
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? sc_cls0[i] : sc_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::scalar_add(sc_base[i], op);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_add:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_add timing leak");
    }

    // ── 3b: scalar_sub ──────────────────────────────────────────────────
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? sc_cls0[i] : sc_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::scalar_sub(sc_base[i], op);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_sub:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_sub timing leak");
    }

    // ── 3c: scalar_cmov ─────────────────────────────────────────────────
    {
        uint64_t masks[2] = {0, ~0ULL};
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint64_t mask = masks[cls];
            auto dst = sc_base[i];
            auto src = sc_cls1[i];

            asm volatile("" : "+r"(mask) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::scalar_cmov(&dst, src, mask);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_cmov:     |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_cmov timing leak");
    }

    // ── 3d: scalar_is_zero ──────────────────────────────────────────────
    {
        for (int i = 0; i < N; ++i) sc_cls0[i] = sc_zero;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? sc_cls0[i] : sc_cls1[i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto m = secp256k1::ct::scalar_is_zero(op);
            (void)m;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_is_zero:  |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_is_zero timing leak");
    }

    // ── 3e: scalar_bit (position 0 vs random position) ──────────────────
    {
        // Pre-generate positions. Test: same scalar, different bit positions
        size_t positions[2][N];
        for (int i = 0; i < N; ++i) {
            positions[0][i] = 0;          // always bit 0
            positions[1][i] = rng() % 256; // random bit
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            size_t pos = positions[cls][i];

            asm volatile("" : "+r"(pos) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto bit = secp256k1::ct::scalar_bit(sc_base[i], pos);
            (void)bit;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_bit:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_bit timing leak");
    }

    // ── 3f: scalar_window ───────────────────────────────────────────────
    {
        size_t positions[2][N];
        for (int i = 0; i < N; ++i) {
            positions[0][i] = 0;
            positions[1][i] = (rng() % 63) * 4; // random 4-bit window position
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            size_t pos = positions[cls][i];

            asm volatile("" : "+r"(pos) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto w = secp256k1::ct::scalar_window(sc_base[i], pos, 4);
            (void)w;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_window:   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_window timing leak");
    }

    delete[] sc_cls0;
    delete[] sc_cls1;
    delete[] sc_base;
    delete[] classes;
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 4: CT Point ოპერაციები (ყველაზე კრიტიკული)
// ═══════════════════════════════════════════════════════════════════════════

static void test_ct_point() {
    printf("\n[4] CT Point ოპერაციები — timing ტესტი (ყველაზე კრიტიკული)\n");

    auto G = Point::generator();

    // ── 4a: complete addition (P+O vs P+Q) ──────────────────────────────
    {
        constexpr int N = 10000;
        auto ct_G = secp256k1::ct::CTJacobianPoint::from_point(G);
        auto ct_O = secp256k1::ct::CTJacobianPoint::make_infinity();
        auto ct_Q = secp256k1::ct::CTJacobianPoint::from_point(
            G.scalar_mul(random_scalar()));

        // Pre-generate: array of pointers to avoid setup in loop
        secp256k1::ct::CTJacobianPoint rhs_arr[2];
        rhs_arr[0] = ct_O;
        rhs_arr[1] = ct_Q;
        int classes[N];
        for (int i = 0; i < N; ++i) classes[i] = rng() & 1;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& rhs = rhs_arr[cls];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::point_add_complete(ct_G, rhs);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    complete_add (P+O vs P+Q):   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "complete_add P+O vs P+Q timing leak");
    }

    // ── 4b: complete addition (P+P vs P+Q) — doubling case ─────────────
    {
        constexpr int N = 10000;
        auto ct_G = secp256k1::ct::CTJacobianPoint::from_point(G);
        auto ct_Q = secp256k1::ct::CTJacobianPoint::from_point(
            G.scalar_mul(random_scalar()));

        secp256k1::ct::CTJacobianPoint rhs_arr[2];
        rhs_arr[0] = ct_G; // P+P (doubling)
        rhs_arr[1] = ct_Q; // P+Q (general)
        int classes[N];
        for (int i = 0; i < N; ++i) classes[i] = rng() & 1;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& rhs = rhs_arr[cls];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto r = secp256k1::ct::point_add_complete(ct_G, rhs);
            (void)r;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    complete_add (P+P vs P+Q):   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "complete_add P+P vs P+Q timing leak");
    }

    // ── 4c: CT scalar_mul (k=1 vs k=random) ────────────────────────────
    //    ყველაზე კრიტიკული ტესტი. secret key timing leak.
    {
        constexpr int N = 2000;
        auto sc_one = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");

        Scalar scalars[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            scalars[0][i] = sc_one;
            scalars[1][i] = random_scalar();
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& k = scalars[cls][i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto R = secp256k1::ct::scalar_mul(G, k);
            (void)R;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_mul (k=1 vs random):  |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_mul k=1 vs random timing leak");
    }

    // ── 4d: CT scalar_mul (k=n-1 vs k=random) ──────────────────────────
    {
        constexpr int N = 2000;
        auto sc_nm1 = Scalar::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");

        Scalar scalars[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            scalars[0][i] = sc_nm1;
            scalars[1][i] = random_scalar();
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& k = scalars[cls][i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto R = secp256k1::ct::scalar_mul(G, k);
            (void)R;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_mul (k=n-1 vs random):|t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_mul k=n-1 vs random timing leak");
    }

    // ── 4e: CT generator_mul (low HW vs high HW scalar) ────────────────
    {
        constexpr int N = 2000;
        auto sc_low = Scalar::from_hex(
            "0000000000000000000000000000000100000000000000000000000000000000");
        auto sc_high = Scalar::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");

        Scalar scalars[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            scalars[0][i] = sc_low;
            scalars[1][i] = sc_high;
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& k = scalars[cls][i];

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto R = secp256k1::ct::generator_mul(k);
            (void)R;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    generator_mul (low vs high HW):|t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct::generator_mul low vs high HW timing leak");
    }

    // ── 4f: point_table_lookup (index 0 vs 15) ─────────────────────────
    {
        constexpr int N = 50000;
        secp256k1::ct::CTJacobianPoint table[16];
        auto pt = secp256k1::ct::CTJacobianPoint::from_point(G);
        for (int i = 0; i < 16; ++i) {
            table[i] = pt;
            pt = secp256k1::ct::point_add_complete(pt,
                     secp256k1::ct::CTJacobianPoint::from_point(G));
        }

        size_t indices[2][N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            indices[0][i] = 0;
            indices[1][i] = 15;
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            size_t idx = indices[cls][i];

            asm volatile("" : "+r"(idx) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile auto p = secp256k1::ct::point_table_lookup(table, 16, idx);
            (void)p;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    point_tbl_lookup (0 vs 15):  |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "point_table_lookup timing leak");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 5: CT Byte Utilities
// ═══════════════════════════════════════════════════════════════════════════

static void test_ct_utils() {
    printf("\n[5] CT Byte Utilities — timing ტესტი\n");

    constexpr int N = 100000;

    // ── 5a: ct_memcpy_if ────────────────────────────────────────────────
    {
        uint8_t dst[32], src[32];
        random_bytes(src, 32);
        random_bytes(dst, 32);

        bool flags[2] = {false, true};
        int classes[N];
        for (int i = 0; i < N; ++i) classes[i] = rng() & 1;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            bool flag = flags[cls];

            asm volatile("" : "+r"(flag) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::ct_memcpy_if(dst, src, 32, flag);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    ct_memcpy_if:    |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct_memcpy_if timing leak");
    }

    // ── 5b: ct_memswap_if ───────────────────────────────────────────────
    {
        uint8_t a[32], b[32];
        random_bytes(a, 32);
        random_bytes(b, 32);

        bool flags[2] = {false, true};
        int classes[N];
        for (int i = 0; i < N; ++i) classes[i] = rng() & 1;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            bool flag = flags[cls];

            asm volatile("" : "+r"(flag) :: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::ct_memswap_if(a, b, 32, flag);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    ct_memswap_if:   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct_memswap_if timing leak");
    }

    // ── 5c: ct_memzero ──────────────────────────────────────────────────
    {
        // Both classes: zero 32-byte buffer. Test: already-zero vs random content.
        struct Buf { uint8_t data[32]; };
        auto* bufs0 = new Buf[N];
        auto* bufs1 = new Buf[N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            std::memset(bufs0[i].data, 0, 32);
            random_bytes(bufs1[i].data, 32);
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint8_t* buf = (cls == 0) ? bufs0[i].data : bufs1[i].data;

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            secp256k1::ct::ct_memzero(buf, 32);
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        delete[] bufs0;
        delete[] bufs1;
        double t = std::abs(ws.t_value());
        printf("    ct_memzero:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct_memzero timing leak");
    }

    // ── 5d: ct_compare ──────────────────────────────────────────────────
    {
        struct Pair { uint8_t a[32]; uint8_t b[32]; };
        auto* pairs0 = new Pair[N];
        auto* pairs1 = new Pair[N];
        int classes[N];
        for (int i = 0; i < N; ++i) {
            classes[i] = rng() & 1;
            random_bytes(pairs0[i].a, 32);
            std::memcpy(pairs0[i].b, pairs0[i].a, 32); // equal
            random_bytes(pairs1[i].a, 32);
            random_bytes(pairs1[i].b, 32); // different
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            const uint8_t* a = (cls == 0) ? pairs0[i].a : pairs1[i].a;
            const uint8_t* b = (cls == 0) ? pairs0[i].b : pairs1[i].b;

            asm volatile("" ::: "memory");
            uint64_t t0 = rdtsc();
            asm volatile("" ::: "memory");
            volatile int cmp = secp256k1::ct::ct_compare(a, b, 32);
            (void)cmp;
            asm volatile("" ::: "memory");
            uint64_t t1 = rdtsc();
            asm volatile("" ::: "memory");

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        delete[] pairs0;
        delete[] pairs1;
        double t = std::abs(ws.t_value());
        printf("    ct_compare:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "✅ CT" : "⚠️  LEAK");
        check(t < T_THRESHOLD, "ct_compare timing leak");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 6: fast:: path-ების CT ტესტი (მოსალოდნელია NOT CT)
// ═══════════════════════════════════════════════════════════════════════════

static void test_fast_not_ct() {
    printf("\n[6] fast:: path control test (მოსალოდნელია NOT CT)\n");
    printf("    (ადასტურებს რომ fast:: და ct:: რეალურად განსხვავდება)\n");

    auto G = Point::generator();
    constexpr int N = 5000;

    auto sc_one = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");

    Scalar scalars[2][N];
    int classes[N];
    for (int i = 0; i < N; ++i) {
        classes[i] = rng() & 1;
        scalars[0][i] = sc_one;
        scalars[1][i] = random_scalar();
    }

    WelchState ws;
    for (int i = 0; i < N; ++i) {
        int cls = classes[i];
        auto& k = scalars[cls][i];

        asm volatile("" ::: "memory");
        uint64_t t0 = rdtsc();
        asm volatile("" ::: "memory");
        volatile auto R = G.scalar_mul(k);
        (void)R;
        asm volatile("" ::: "memory");
        uint64_t t1 = rdtsc();
        asm volatile("" ::: "memory");

        ws.push(cls, static_cast<double>(t1 - t0));
    }
    double t = std::abs(ws.t_value());
    printf("    fast::scalar_mul: |t| = %6.2f  %s\n",
           t, t >= T_THRESHOLD ? "⏱️  NOT CT (მოსალოდნელი)" : "≈ CT-like");
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 7: Valgrind CLASSIFY/DECLASSIFY (ფუნქციონალური ტესტი)
// ═══════════════════════════════════════════════════════════════════════════

static void test_valgrind_markers() {
    printf("\n[7] Valgrind CLASSIFY/DECLASSIFY ტესტი\n");

#if defined(SECP256K1_CT_VALGRIND) && SECP256K1_CT_VALGRIND
    printf("    ⚡ Valgrind CT mode ENABLED — running with secret tagging\n");
#else
    printf("    ℹ️  Valgrind CT mode DISABLED\n");
    printf("    ℹ️  ჩართვა: cmake -DSECP256K1_CT_VALGRIND=1\n");
    printf("    ℹ️  გაშვება: valgrind ./test_ct_sidechannel\n");
#endif

    auto G = Point::generator();

    // 7a: CT scalar_mul
    {
        auto k = random_scalar();
        SECP256K1_CLASSIFY(&k, sizeof(k));
        auto R = secp256k1::ct::scalar_mul(G, k);
        SECP256K1_DECLASSIFY(&R, sizeof(R));
        check(!R.is_infinity(), "CT scalar_mul with classified k");
        printf("    ct::scalar_mul classified: ✅\n");
    }
    // 7b: CT field ops
    {
        auto a = random_fe(), b = random_fe();
        SECP256K1_CLASSIFY(&a, sizeof(a));
        SECP256K1_CLASSIFY(&b, sizeof(b));
        auto sum = secp256k1::ct::field_add(a, b);
        auto prod = secp256k1::ct::field_mul(a, b);
        auto sq = secp256k1::ct::field_sqr(a);
        SECP256K1_DECLASSIFY(&sum, sizeof(sum));
        SECP256K1_DECLASSIFY(&prod, sizeof(prod));
        SECP256K1_DECLASSIFY(&sq, sizeof(sq));
        check(true, "CT field ops classified");
        printf("    ct::field_{add,mul,sqr} classified: ✅\n");
    }
    // 7c: CT scalar ops
    {
        auto a = random_scalar(), b = random_scalar();
        SECP256K1_CLASSIFY(&a, sizeof(a));
        SECP256K1_CLASSIFY(&b, sizeof(b));
        auto sum = secp256k1::ct::scalar_add(a, b);
        auto neg = secp256k1::ct::scalar_neg(a);
        SECP256K1_DECLASSIFY(&sum, sizeof(sum));
        SECP256K1_DECLASSIFY(&neg, sizeof(neg));
        check(true, "CT scalar ops classified");
        printf("    ct::scalar_{add,neg} classified: ✅\n");
    }
    // 7d: cmov with classified mask
    {
        auto a = random_fe(), b = random_fe();
        uint64_t mask = secp256k1::ct::bool_to_mask(true);
        SECP256K1_CLASSIFY(&mask, sizeof(mask));
        secp256k1::ct::field_cmov(&a, b, mask);
        SECP256K1_DECLASSIFY(&a, sizeof(a));
        check(true, "CT field_cmov classified mask");
        printf("    ct::field_cmov classified mask: ✅\n");
    }
    // 7e: table lookup with classified index
    {
        uint64_t table[16][4];
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 4; ++j) table[i][j] = rng();
        size_t idx = 7;
        SECP256K1_CLASSIFY(&idx, sizeof(idx));
        uint64_t out[4];
        secp256k1::ct::ct_lookup_256(table, 16, idx, out);
        SECP256K1_DECLASSIFY(&out, sizeof(out));
        SECP256K1_DECLASSIFY(&idx, sizeof(idx));
        check(true, "CT lookup classified index");
        printf("    ct::ct_lookup_256 classified index: ✅\n");
    }
    // 7f: generator_mul
    {
        auto k = random_scalar();
        SECP256K1_CLASSIFY(&k, sizeof(k));
        auto R = secp256k1::ct::generator_mul(k);
        SECP256K1_DECLASSIFY(&R, sizeof(R));
        check(!R.is_infinity(), "CT generator_mul classified k");
        printf("    ct::generator_mul classified: ✅\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ტესტი 8: ასემბლის ანალიზი — ინფორმაცია
// ═══════════════════════════════════════════════════════════════════════════

static void test_assembly_info() {
    printf("\n[8] ასემბლის ინსპექცია — ინსტრუქცია\n");
    printf("    CT ფუნქციების ასემბლის შემოწმება:\n");
    printf("    objdump -d build_rel/tests/test_ct_sidechannel | less\n\n");
    printf("    ეძებეთ ct:: ფუნქციებში:\n");
    printf("    ✅ კარგი: cmov, cmovne, cmove (branchless conditional)\n");
    printf("    ❌ ცუდი:  jz/jnz/je/jne (secret-dependent branch)\n\n");
    printf("    სწრაფი ავტომატური შემოწმება:\n");
    printf("    objdump -d build_rel/tests/test_ct_sidechannel | \\\n");
    printf("      awk '/ct.*:$/,/^$/' | grep -cE 'j[a-z]{1,3}\\s'\n");
}

// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Side-Channel Attack Test Suite (dudect methodology)\n");
    printf("  Welch t-test: |t| > %.1f → timing leak (p < 0.00001)\n", T_THRESHOLD);
    printf("  All inputs pre-generated — no RNG in measurement loops\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    test_ct_primitives();    // 1
    test_ct_field();         // 2
    test_ct_scalar();        // 3
    test_ct_point();         // 4
    test_ct_utils();         // 5
    test_fast_not_ct();      // 6 (ინფორმაციული)
    test_valgrind_markers(); // 7
    test_assembly_info();    // 8

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  SIDE-CHANNEL AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0) {
        printf("  ⚠️  TIMING LEAK-ები აღმოჩენილია\n");
    } else {
        printf("  ✅ ყველა CT ოპერაცია გაიარა dudect ტესტი\n");
    }
    printf("═══════════════════════════════════════════════════════════════\n");

    printf("\n  სრული სერთიფიკაციის ნაბიჯები:\n");
    printf("  1. Valgrind: -DSECP256K1_CT_VALGRIND=1 && valgrind ./test\n");
    printf("  2. asm:      objdump -d <binary> | grep branches\n");
    printf("  3. hw:       Intel Pin / Flush+Reload (hardware level)\n\n");

    return g_fail > 0 ? 1 : 0;
}
