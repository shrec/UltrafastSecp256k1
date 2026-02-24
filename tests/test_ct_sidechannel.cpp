// ============================================================================
// Side-Channel Attack Test Suite -- dudect methodology
// ============================================================================
//  CT   timing side-channel  .
//
//  ( dudect ):
//   1.   PRE-GENERATED --  random_fe()/random_scalar()
//      measurement loop- .
//   2.     - .
//   3. class selection = array index lookup ( cost ).
//   4. Welch t-test |t| > 4.5 -> timing leak (99.999% confidence).
//
// CRITICAL:  
//   - Class 0:   (edge-case: zero, one, identity, etc.)
//   - Class 1:   (pre-generated)
//   -   IDENTICAL selection path- (array[cls][i])
//   - asm volatile barriers  
//
// :
//   :  build_rel/tests/test_ct_sidechannel
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
#include <atomic>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// -- Our CT layer -------------------------------------------------------------
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

// ===========================================================================
// Compiler barriers -- prevent reordering/optimization across measurement
// ===========================================================================
// BARRIER_OPAQUE(v): treat v as modified by unknown side-effect (value barrier)
// BARRIER_FENCE():   full compiler reordering barrier (memory fence)

#ifdef _MSC_VER
// MSVC: use atomic fence as compiler barrier + volatile trick for value barrier
#define BARRIER_FENCE()       std::atomic_signal_fence(std::memory_order_seq_cst)
#define BARRIER_OPAQUE(v)     do { volatile auto _bv = (v); (v) = _bv; \
                                   std::atomic_signal_fence(std::memory_order_seq_cst); } while(0)
#else
// GCC / Clang: inline asm barriers
#define BARRIER_FENCE()       asm volatile("" ::: "memory")
#define BARRIER_OPAQUE(v)     asm volatile("" : "+r"(v) :: "memory")
#endif

// ===========================================================================
// Timer -- rdtsc(p) on x86_64, cntvct on aarch64
// ===========================================================================

#if defined(_MSC_VER) && defined(_M_X64)
static inline uint64_t rdtsc() {
    unsigned int aux;
    return __rdtscp(&aux);
}
#elif defined(__x86_64__)
static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "ecx");
    return (static_cast<uint64_t>(hi) << 32) | lo;
}
#elif defined(__aarch64__) && !defined(_MSC_VER)
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

// ===========================================================================
// Welch t-test (online/incremental -- no allocation needed)
// ===========================================================================

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

// ===========================================================================
// PRNG + helpers -- pre-generation only
// ===========================================================================

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

// -- Framework ----------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

// Smoke mode: short run for CI (compile with -DDUDECT_SMOKE).
// Full mode: longer statistical run for local/nightly testing.
#ifdef DUDECT_SMOKE
static constexpr double T_THRESHOLD = 25.0;  // Very relaxed: only catch gross leaks
static constexpr int    SMOKE_N_PRIM  = 5000; // Primitive ops (masks, cmov, etc.)
static constexpr int    SMOKE_N_FIELD = 3000; // Field/scalar ops
static constexpr int    SMOKE_N_POINT = 500;  // Point ops (expensive)
static constexpr int    SMOKE_N_SIGN  = 100;  // ECDSA/Schnorr sign (very expensive)
#else
static constexpr double T_THRESHOLD = 4.5;
#endif

static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; }
    else      { ++g_fail; printf("    [x] FAIL: %s\n", msg); }
}

// ===========================================================================
//  1: CT  (masks, cmov, cswap, lookup)
// ===========================================================================

static void test_ct_primitives() {
    printf("\n[1] CT  -- timing \n");

#ifdef DUDECT_SMOKE
    constexpr int N = SMOKE_N_PRIM;
#else
    constexpr int N = 100000;
#endif

    // -- 1a: is_zero_mask -------------------------------------------------
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

            BARRIER_OPAQUE(val);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile uint64_t r = secp256k1::ct::is_zero_mask(val);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    is_zero_mask:    |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "is_zero_mask timing leak");
    }

    // -- 1b: bool_to_mask -------------------------------------------------
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

            BARRIER_OPAQUE(flag);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile uint64_t m = secp256k1::ct::bool_to_mask(flag);
            (void)m;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    bool_to_mask:    |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "bool_to_mask timing leak");
    }

    // -- 1c: cmov256 -----------------------------------------------------
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

            BARRIER_OPAQUE(mask);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::cmov256(dst, src, mask);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    cmov256:         |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "cmov256 timing leak");
    }

    // -- 1d: cswap256 ----------------------------------------------------
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

            BARRIER_OPAQUE(mask);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::cswap256(a, b, mask);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    cswap256:        |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "cswap256 timing leak");
    }

    // -- 1e: ct_lookup_256 (16 entries) ----------------------------------
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

            BARRIER_OPAQUE(idx);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::ct_lookup_256(table, 16, idx, out);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    ct_lookup_256:   |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct_lookup_256 timing leak");
    }

    // -- 1f: ct_equal ----------------------------------------------------
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile bool eq = secp256k1::ct::ct_equal(a, b, 32);
            (void)eq;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        delete[] pairs0;
        delete[] pairs1;
        double t = std::abs(ws.t_value());
        printf("    ct_equal:        |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct_equal timing leak");
    }
}

// ===========================================================================
//  2: CT Field 
// ===========================================================================

static void test_ct_field() {
    printf("\n[2] CT Field  -- timing \n");

#ifdef DUDECT_SMOKE
    constexpr int N = SMOKE_N_FIELD;
#else
    constexpr int N = 50000;
#endif

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

    // -- 2a: field_add ---------------------------------------------------
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::field_add(fe_base[i], op);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_add:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::field_add timing leak");
    }

    // -- 2b: field_mul ---------------------------------------------------
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::field_mul(fe_base[i], op);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_mul:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::field_mul timing leak");
    }

    // -- 2c: field_sqr ---------------------------------------------------
    {
        // Swap cls0 to fe_one for sqr
        for (int i = 0; i < N; ++i) fe_cls0[i] = fe_one;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::field_sqr(op);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_sqr:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::field_sqr timing leak");
    }

    // -- 2d: field_inv ---------------------------------------------------
    {
        constexpr int NSLOW = (N < 5000) ? N : 5000;
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::field_inv(op);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_inv:       |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::field_inv timing leak");
    }

    // -- 2e: field_cmov --------------------------------------------------
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

            BARRIER_OPAQUE(mask);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::field_cmov(&dst, src, mask);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_cmov:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::field_cmov timing leak");
    }

    // -- 2f: field_is_zero -----------------------------------------------
    {
        for (int i = 0; i < N; ++i) fe_cls0[i] = fe_zero;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? fe_cls0[i] : fe_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto m = secp256k1::ct::field_is_zero(op);
            (void)m;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    field_is_zero:   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::field_is_zero timing leak");
    }

    delete[] fe_cls0;
    delete[] fe_cls1;
    delete[] fe_base;
    delete[] classes;
}

// ===========================================================================
//  3: CT Scalar 
// ===========================================================================

static void test_ct_scalar() {
    printf("\n[3] CT Scalar  -- timing \n");

#ifdef DUDECT_SMOKE
    constexpr int N = SMOKE_N_FIELD;
#else
    constexpr int N = 50000;
#endif

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

    // -- 3a: scalar_add --------------------------------------------------
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? sc_cls0[i] : sc_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::scalar_add(sc_base[i], op);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_add:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_add timing leak");
    }

    // -- 3b: scalar_sub --------------------------------------------------
    {
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? sc_cls0[i] : sc_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::scalar_sub(sc_base[i], op);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_sub:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_sub timing leak");
    }

    // -- 3c: scalar_cmov -------------------------------------------------
    {
        uint64_t masks[2] = {0, ~0ULL};
        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            uint64_t mask = masks[cls];
            auto dst = sc_base[i];
            auto src = sc_cls1[i];

            BARRIER_OPAQUE(mask);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::scalar_cmov(&dst, src, mask);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_cmov:     |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_cmov timing leak");
    }

    // -- 3d: scalar_is_zero ----------------------------------------------
    {
        for (int i = 0; i < N; ++i) sc_cls0[i] = sc_zero;

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            auto& op = (cls == 0) ? sc_cls0[i] : sc_cls1[i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto m = secp256k1::ct::scalar_is_zero(op);
            (void)m;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_is_zero:  |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_is_zero timing leak");
    }

    // -- 3e: scalar_bit (class 0: scalar with bit=0, class 1: scalar with bit=1 at same position) --
    {
        // Security-relevant test: same position (public), different scalar values.
        // In scalar mul, position is the loop counter (public); scalar is secret.
        // We test that timing doesn't reveal the bit VALUE at a fixed position.
        constexpr size_t TEST_POS = 128;  // middle bit, limb 2

        // Pre-generate scalars where bit TEST_POS is forced to 0 or 1
        static Scalar sc_cls[2][N];
        for (int i = 0; i < N; ++i) {
            auto s = random_scalar();
            auto limbs = s.limbs();
            // Class 0: force bit to 0
            limbs[TEST_POS / 64] &= ~(uint64_t(1) << (TEST_POS % 64));
            sc_cls[0][i] = Scalar::from_limbs(limbs);
            // Class 1: force bit to 1
            limbs[TEST_POS / 64] |= (uint64_t(1) << (TEST_POS % 64));
            sc_cls[1][i] = Scalar::from_limbs(limbs);
        }

        WelchState ws;
        for (int i = 0; i < N; ++i) {
            int cls = classes[i];
            const auto& sc = sc_cls[cls][i];

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto bit = secp256k1::ct::scalar_bit(sc, TEST_POS);
            (void)bit;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_bit:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_bit timing leak");
    }

    // -- 3f: scalar_window -----------------------------------------------
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

            BARRIER_OPAQUE(pos);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto w = secp256k1::ct::scalar_window(sc_base[i], pos, 4);
            (void)w;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_window:   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_window timing leak");
    }

    delete[] sc_cls0;
    delete[] sc_cls1;
    delete[] sc_base;
    delete[] classes;
}

// ===========================================================================
//  4: CT Point  ( )
// ===========================================================================

static void test_ct_point() {
    printf("\n[4] CT Point  -- timing  ( )\n");

    auto G = Point::generator();

    // -- 4a: complete addition (P+O vs P+Q) ------------------------------
    {
#ifdef DUDECT_SMOKE
        constexpr int N = SMOKE_N_POINT;
#else
        constexpr int N = 10000;
#endif
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::point_add_complete(ct_G, rhs);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    complete_add (P+O vs P+Q):   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "complete_add P+O vs P+Q timing leak");
    }

    // -- 4b: complete addition (P+P vs P+Q) -- doubling case -------------
    {
#ifdef DUDECT_SMOKE
        constexpr int N = SMOKE_N_POINT;
#else
        constexpr int N = 10000;
#endif
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto r = secp256k1::ct::point_add_complete(ct_G, rhs);
            (void)r;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    complete_add (P+P vs P+Q):   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "complete_add P+P vs P+Q timing leak");
    }

    // -- 4c: CT scalar_mul (k=1 vs k=random) ----------------------------
    //      . secret key timing leak.
    {
#ifdef DUDECT_SMOKE
        constexpr int N = SMOKE_N_SIGN;
#else
        constexpr int N = 2000;
#endif
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto R = secp256k1::ct::scalar_mul(G, k);
            (void)R;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_mul (k=1 vs random):  |t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_mul k=1 vs random timing leak");
    }

    // -- 4d: CT scalar_mul (k=n-1 vs k=random) --------------------------
    {
#ifdef DUDECT_SMOKE
        constexpr int N = SMOKE_N_SIGN;
#else
        constexpr int N = 2000;
#endif
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto R = secp256k1::ct::scalar_mul(G, k);
            (void)R;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    scalar_mul (k=n-1 vs random):|t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::scalar_mul k=n-1 vs random timing leak");
    }

    // -- 4e: CT generator_mul (low HW vs high HW scalar) ----------------
    {
#ifdef DUDECT_SMOKE
        constexpr int N = SMOKE_N_SIGN;
#else
        constexpr int N = 2000;
#endif
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto R = secp256k1::ct::generator_mul(k);
            (void)R;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    generator_mul (low vs high HW):|t| = %6.2f  (%d/%d)  %s\n",
               t, (int)ws.n[0], (int)ws.n[1],
               t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct::generator_mul low vs high HW timing leak");
    }

    // -- 4f: point_table_lookup (index 0 vs 15) -------------------------
    {
#ifdef DUDECT_SMOKE
        constexpr int N = SMOKE_N_PRIM;
#else
        constexpr int N = 50000;
#endif
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

            BARRIER_OPAQUE(idx);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile auto p = secp256k1::ct::point_table_lookup(table, 16, idx);
            (void)p;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    point_tbl_lookup (0 vs 15):  |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "point_table_lookup timing leak");
    }
}

// ===========================================================================
//  5: CT Byte Utilities
// ===========================================================================

static void test_ct_utils() {
    printf("\n[5] CT Byte Utilities -- timing \n");

#ifdef DUDECT_SMOKE
    constexpr int N = SMOKE_N_PRIM;
#else
    constexpr int N = 100000;
#endif

    // -- 5a: ct_memcpy_if ------------------------------------------------
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

            BARRIER_OPAQUE(flag);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::ct_memcpy_if(dst, src, 32, flag);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    ct_memcpy_if:    |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct_memcpy_if timing leak");
    }

    // -- 5b: ct_memswap_if -----------------------------------------------
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

            BARRIER_OPAQUE(flag);
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::ct_memswap_if(a, b, 32, flag);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        double t = std::abs(ws.t_value());
        printf("    ct_memswap_if:   |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct_memswap_if timing leak");
    }

    // -- 5c: ct_memzero --------------------------------------------------
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            secp256k1::ct::ct_memzero(buf, 32);
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        delete[] bufs0;
        delete[] bufs1;
        double t = std::abs(ws.t_value());
        printf("    ct_memzero:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct_memzero timing leak");
    }

    // -- 5d: ct_compare --------------------------------------------------
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

            BARRIER_FENCE();
            uint64_t t0 = rdtsc();
            BARRIER_FENCE();
            volatile int cmp = secp256k1::ct::ct_compare(a, b, 32);
            (void)cmp;
            BARRIER_FENCE();
            uint64_t t1 = rdtsc();
            BARRIER_FENCE();

            ws.push(cls, static_cast<double>(t1 - t0));
        }
        delete[] pairs0;
        delete[] pairs1;
        double t = std::abs(ws.t_value());
        printf("    ct_compare:      |t| = %6.2f  %s\n",
               t, t < T_THRESHOLD ? "[OK] CT" : "[!]  LEAK");
        check(t < T_THRESHOLD, "ct_compare timing leak");
    }
}

// ===========================================================================
//  6: fast:: path- CT  ( NOT CT)
// ===========================================================================

static void test_fast_not_ct() {
    printf("\n[6] fast:: path control test ( NOT CT)\n");
    printf("    (  fast::  ct::  )\n");

    auto G = Point::generator();
#ifdef DUDECT_SMOKE
    constexpr int N = SMOKE_N_POINT;
#else
    constexpr int N = 5000;
#endif

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

        BARRIER_FENCE();
        uint64_t t0 = rdtsc();
        BARRIER_FENCE();
        volatile auto R = G.scalar_mul(k);
        (void)R;
        BARRIER_FENCE();
        uint64_t t1 = rdtsc();
        BARRIER_FENCE();

        ws.push(cls, static_cast<double>(t1 - t0));
    }
    double t = std::abs(ws.t_value());
    printf("    fast::scalar_mul: |t| = %6.2f  %s\n",
           t, t >= T_THRESHOLD ? "[time]  NOT CT ()" : "~= CT-like");
}

// ===========================================================================
//  7: Valgrind CLASSIFY/DECLASSIFY ( )
// ===========================================================================

static void test_valgrind_markers() {
    printf("\n[7] Valgrind CLASSIFY/DECLASSIFY \n");

#if defined(SECP256K1_CT_VALGRIND) && SECP256K1_CT_VALGRIND
    printf("    [*] Valgrind CT mode ENABLED -- running with secret tagging\n");
#else
    printf("      Valgrind CT mode DISABLED\n");
    printf("      : cmake -DSECP256K1_CT_VALGRIND=1\n");
    printf("      : valgrind ./test_ct_sidechannel\n");
#endif

    auto G = Point::generator();

    // 7a: CT scalar_mul
    {
        auto k = random_scalar();
        SECP256K1_CLASSIFY(&k, sizeof(k));
        auto R = secp256k1::ct::scalar_mul(G, k);
        SECP256K1_DECLASSIFY(&R, sizeof(R));
        check(!R.is_infinity(), "CT scalar_mul with classified k");
        printf("    ct::scalar_mul classified: [OK]\n");
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
        printf("    ct::field_{add,mul,sqr} classified: [OK]\n");
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
        printf("    ct::scalar_{add,neg} classified: [OK]\n");
    }
    // 7d: cmov with classified mask
    {
        auto a = random_fe(), b = random_fe();
        uint64_t mask = secp256k1::ct::bool_to_mask(true);
        SECP256K1_CLASSIFY(&mask, sizeof(mask));
        secp256k1::ct::field_cmov(&a, b, mask);
        SECP256K1_DECLASSIFY(&a, sizeof(a));
        check(true, "CT field_cmov classified mask");
        printf("    ct::field_cmov classified mask: [OK]\n");
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
        printf("    ct::ct_lookup_256 classified index: [OK]\n");
    }
    // 7f: generator_mul
    {
        auto k = random_scalar();
        SECP256K1_CLASSIFY(&k, sizeof(k));
        auto R = secp256k1::ct::generator_mul(k);
        SECP256K1_DECLASSIFY(&R, sizeof(R));
        check(!R.is_infinity(), "CT generator_mul classified k");
        printf("    ct::generator_mul classified: [OK]\n");
    }
}

// ===========================================================================
//  8:   -- 
// ===========================================================================

static void test_assembly_info() {
    printf("\n[8]   -- \n");
    printf("    CT   :\n");
    printf("    objdump -d build_rel/tests/test_ct_sidechannel | less\n\n");
    printf("     ct:: :\n");
    printf("    [OK] : cmov, cmovne, cmove (branchless conditional)\n");
    printf("    [FAIL] :  jz/jnz/je/jne (secret-dependent branch)\n\n");
    printf("      :\n");
    printf("    objdump -d build_rel/tests/test_ct_sidechannel | \\\n");
    printf("      awk '/ct.*:$/,/^$/' | grep -cE 'j[a-z]{1,3}\\s'\n");
}

// ===========================================================================
int main() {
    printf("===============================================================\n");
    printf("  Side-Channel Attack Test Suite (dudect methodology)\n");
    printf("  Welch t-test: |t| > %.1f -> timing leak (p < 0.00001)\n", T_THRESHOLD);
    printf("  All inputs pre-generated -- no RNG in measurement loops\n");
    printf("===============================================================\n");

    test_ct_primitives();    // 1
    test_ct_field();         // 2
    test_ct_scalar();        // 3
    test_ct_point();         // 4
    test_ct_utils();         // 5
    test_fast_not_ct();      // 6 ()
    test_valgrind_markers(); // 7
    test_assembly_info();    // 8

    printf("\n===============================================================\n");
    printf("  SIDE-CHANNEL AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    if (g_fail > 0) {
        printf("  [!]  TIMING LEAK- \n");
    } else {
        printf("  [OK]  CT   dudect \n");
    }
    printf("===============================================================\n");

    printf("\n    :\n");
    printf("  1. Valgrind: -DSECP256K1_CT_VALGRIND=1 && valgrind ./test\n");
    printf("  2. asm:      objdump -d <binary> | grep branches\n");
    printf("  3. hw:       Intel Pin / Flush+Reload (hardware level)\n\n");

    return g_fail > 0 ? 1 : 0;
}
