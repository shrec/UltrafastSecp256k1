// ============================================================================
// Constant-Time Operations Benchmark: UltrafastSecp256k1 vs libsecp256k1
// ============================================================================
// Compares constant-time operation throughput between our CT layer and
// Bitcoin Core's libsecp256k1 (which is always constant-time by design).
//
// Operations benchmarked:
//   - Key generation (pubkey from secret key)
//   - ECDSA sign / verify
//   - Schnorr BIP-340 sign / verify
//   - ECDH shared secret
//   - Scalar multiplication (CT)
//   - CT primitives (cmov, cswap, table lookup)
//
// Build:
//   Requires libsecp256k1 built from _research_repos/secp256k1/build/
//   Linked via -lsecp256k1 with appropriate include/lib paths.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <chrono>
#include <cmath>

// ── Our library ──────────────────────────────────────────────────────────────
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"

// ── libsecp256k1 (C API) ────────────────────────────────────────────────────
extern "C" {
#include <secp256k1.h>
#include <secp256k1_schnorrsig.h>
#include <secp256k1_ecdh.h>
#include <secp256k1_extrakeys.h>
#include <secp256k1_recovery.h>
}

using namespace secp256k1::fast;

// ── PRNG ─────────────────────────────────────────────────────────────────────
static std::mt19937_64 rng(0xA0D17'BE4C8);

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

static void random_seckey(uint8_t seckey[32], secp256k1_context* ctx) {
    for (;;) {
        random_bytes(seckey, 32);
        if (secp256k1_ec_seckey_verify(ctx, seckey)) return;
    }
}

// ── Benchmark harness ────────────────────────────────────────────────────────
struct BenchResult {
    const char* name;
    int         iters;
    double      total_us;
    double      per_op_ns;
    double      ops_per_sec;
};

// Warm up + measure
#define BENCH(name_str, iters_val, setup, body) \
    [&]() -> BenchResult { \
        setup; \
        int _i = 0; \
        /* warm up */ \
        for (_i = 0; _i < 10; ++_i) { body; } \
        int N = (iters_val); \
        auto t0 = std::chrono::high_resolution_clock::now(); \
        for (_i = 0; _i < N; ++_i) { body; } \
        auto t1 = std::chrono::high_resolution_clock::now(); \
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0; \
        double ns_per = (us * 1000.0) / N; \
        double ops = (us > 0) ? (N / (us / 1e6)) : 0; \
        return {name_str, N, us, ns_per, ops}; \
    }()

struct ComparisonRow {
    BenchResult ours;
    BenchResult libsecp;
};

static void print_header() {
    printf("┌────────────────────────────┬──────────────────────────────────────┬──────────────────────────────────────┬──────────┐\n");
    printf("│ %-26s │ %-36s │ %-36s │ %-8s │\n",
           "ოპერაცია", "UltrafastSecp256k1 (CT)", "libsecp256k1", "თანაფარდობა");
    printf("├────────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┼──────────┤\n");
}

static void print_row(const char* label, const BenchResult& ours, const BenchResult& lib) {
    double ratio = (lib.per_op_ns > 0) ? (ours.per_op_ns / lib.per_op_ns) : 0;
    const char* indicator;
    if (ratio < 0.85)      indicator = "✅ ჩვენი";
    else if (ratio <= 1.15) indicator = "≈  თანაბარი";
    else                    indicator = "⚠️  libsecp";

    printf("│ %-26s │ %8.1f ns/op  %10.0f op/s   │ %8.1f ns/op  %10.0f op/s   │ %5.2fx   │  %s\n",
           label, ours.per_op_ns, ours.ops_per_sec, lib.per_op_ns, lib.ops_per_sec, ratio, indicator);
}

static void print_row_single(const char* label, const BenchResult& r) {
    printf("│ %-26s │ %8.1f ns/op  %10.0f op/s   │ %-36s │ %-8s │\n",
           label, r.per_op_ns, r.ops_per_sec, "(N/A)", "—");
}

static void print_separator() {
    printf("├────────────────────────────┼──────────────────────────────────────┼──────────────────────────────────────┼──────────┤\n");
}

static void print_footer() {
    printf("└────────────────────────────┴──────────────────────────────────────┴──────────────────────────────────────┴──────────┘\n");
}

// ============================================================================
int main() {
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  CT ბენჩმარკი: UltrafastSecp256k1 vs libsecp256k1 (Bitcoin Core)\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    // ── Setup libsecp256k1 context ───────────────────────────────────────
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    {
        uint8_t seed[32];
        random_bytes(seed, 32);
        secp256k1_context_randomize(ctx, seed);
    }

    // ── Pre-generate test data ───────────────────────────────────────────
    constexpr int N_KEYGEN   = 5000;
    constexpr int N_SIGN     = 2000;
    constexpr int N_VERIFY   = 2000;
    constexpr int N_ECDH     = 2000;
    constexpr int N_SCMUL    = 1000;
    constexpr int N_PRIM     = 100000;

    // Common secret key (valid for both libraries)
    uint8_t seckey_bytes[32];
    random_seckey(seckey_bytes, ctx);
    auto our_sk = Scalar::from_bytes(std::array<uint8_t,32>{});
    std::memcpy(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&our_sk)),
                seckey_bytes, 32);
    our_sk = Scalar::from_bytes([&]{
        std::array<uint8_t,32> a{};
        std::memcpy(a.data(), seckey_bytes, 32);
        return a;
    }());

    auto G = Point::generator();

    // libsecp256k1 pubkey
    secp256k1_pubkey lib_pk;
    secp256k1_ec_pubkey_create(ctx, &lib_pk, seckey_bytes);

    // libsecp256k1 keypair for schnorr
    secp256k1_keypair lib_keypair;
    secp256k1_keypair_create(ctx, &lib_keypair, seckey_bytes);

    // libsecp256k1 xonly pubkey
    secp256k1_xonly_pubkey lib_xonly_pk;
    secp256k1_keypair_xonly_pub(ctx, &lib_xonly_pk, nullptr, &lib_keypair);

    // Our pubkey
    auto our_pk = G.scalar_mul(our_sk);

    // Message
    std::array<uint8_t, 32> msg{};
    random_bytes(msg.data(), 32);

    // Aux randomness for schnorr
    std::array<uint8_t, 32> aux{};
    random_bytes(aux.data(), 32);

    // Pre-sign for verify benchmarks
    secp256k1_ecdsa_signature lib_ecdsa_sig;
    secp256k1_ecdsa_sign(ctx, &lib_ecdsa_sig, msg.data(), seckey_bytes, nullptr, nullptr);

    auto our_ecdsa_sig = secp256k1::ecdsa_sign(msg, our_sk);

    uint8_t lib_schnorr_sig[64];
    secp256k1_schnorrsig_sign32(ctx, lib_schnorr_sig, msg.data(), &lib_keypair, aux.data());

    auto our_schnorr_pkx = secp256k1::schnorr_pubkey(our_sk);
    auto our_schnorr_sig = secp256k1::schnorr_sign(our_sk, msg, aux);

    // Second key for ECDH
    uint8_t seckey2_bytes[32];
    random_seckey(seckey2_bytes, ctx);
    secp256k1_pubkey lib_pk2;
    secp256k1_ec_pubkey_create(ctx, &lib_pk2, seckey2_bytes);

    auto our_sk2 = Scalar::from_bytes([&]{
        std::array<uint8_t,32> a{};
        std::memcpy(a.data(), seckey2_bytes, 32);
        return a;
    }());
    auto our_pk2 = G.scalar_mul(our_sk2);

    printf("  იტერაციები: keygen=%d, sign=%d, verify=%d, ecdh=%d, scalar_mul=%d, primitives=%d\n\n",
           N_KEYGEN, N_SIGN, N_VERIFY, N_ECDH, N_SCMUL, N_PRIM);

    print_header();

    // ═════════════════════════════════════════════════════════════════════
    // Section 1: Key Generation (pubkey_create)
    // ═════════════════════════════════════════════════════════════════════

    auto r_keygen_ours = BENCH("keygen_ct", N_KEYGEN, {}, {
        volatile auto p = secp256k1::ct::generator_mul(our_sk);
    });

    auto r_keygen_lib = BENCH("keygen_lib", N_KEYGEN, {}, {
        secp256k1_pubkey pk_tmp;
        volatile int ok = secp256k1_ec_pubkey_create(ctx, &pk_tmp, seckey_bytes);
    });

    print_row("Key generation (CT)", r_keygen_ours, r_keygen_lib);

    // Fast keygen for reference
    auto r_keygen_fast = BENCH("keygen_fast", N_KEYGEN, {}, {
        volatile auto p = G.scalar_mul(our_sk);
    });
    print_row_single("Key generation (fast)", r_keygen_fast);

    print_separator();

    // ═════════════════════════════════════════════════════════════════════
    // Section 2: ECDSA Sign
    // ═════════════════════════════════════════════════════════════════════

    auto r_ecdsa_sign_ours = BENCH("ecdsa_sign", N_SIGN, {}, {
        volatile auto s = secp256k1::ecdsa_sign(msg, our_sk);
    });

    auto r_ecdsa_sign_lib = BENCH("ecdsa_sign_lib", N_SIGN, {}, {
        secp256k1_ecdsa_signature sig_tmp;
        volatile int ok = secp256k1_ecdsa_sign(ctx, &sig_tmp, msg.data(), seckey_bytes, nullptr, nullptr);
    });

    print_row("ECDSA sign", r_ecdsa_sign_ours, r_ecdsa_sign_lib);

    // ═════════════════════════════════════════════════════════════════════
    // Section 3: ECDSA Verify
    // ═════════════════════════════════════════════════════════════════════

    auto r_ecdsa_verify_ours = BENCH("ecdsa_verify", N_VERIFY, {}, {
        volatile bool v = secp256k1::ecdsa_verify(msg, our_pk, our_ecdsa_sig);
    });

    auto r_ecdsa_verify_lib = BENCH("ecdsa_verify_lib", N_VERIFY, {}, {
        volatile int v = secp256k1_ecdsa_verify(ctx, &lib_ecdsa_sig, msg.data(), &lib_pk);
    });

    print_row("ECDSA verify", r_ecdsa_verify_ours, r_ecdsa_verify_lib);

    print_separator();

    // ═════════════════════════════════════════════════════════════════════
    // Section 4: Schnorr Sign
    // ═════════════════════════════════════════════════════════════════════

    auto r_schnorr_sign_ours = BENCH("schnorr_sign", N_SIGN, {}, {
        volatile auto s = secp256k1::schnorr_sign(our_sk, msg, aux);
    });

    auto r_schnorr_sign_lib = BENCH("schnorr_sign_lib", N_SIGN, {}, {
        uint8_t sig_tmp[64];
        volatile int ok = secp256k1_schnorrsig_sign32(ctx, sig_tmp, msg.data(), &lib_keypair, aux.data());
    });

    print_row("Schnorr sign", r_schnorr_sign_ours, r_schnorr_sign_lib);

    // ═════════════════════════════════════════════════════════════════════
    // Section 5: Schnorr Verify
    // ═════════════════════════════════════════════════════════════════════

    auto r_schnorr_verify_ours = BENCH("schnorr_verify", N_VERIFY, {}, {
        volatile bool v = secp256k1::schnorr_verify(our_schnorr_pkx, msg, our_schnorr_sig);
    });

    auto r_schnorr_verify_lib = BENCH("schnorr_verify_lib", N_VERIFY, {}, {
        volatile int v = secp256k1_schnorrsig_verify(ctx, lib_schnorr_sig, msg.data(), 32, &lib_xonly_pk);
    });

    print_row("Schnorr verify", r_schnorr_verify_ours, r_schnorr_verify_lib);

    print_separator();

    // ═════════════════════════════════════════════════════════════════════
    // Section 6: ECDH
    // ═════════════════════════════════════════════════════════════════════

    auto r_ecdh_ours = BENCH("ecdh", N_ECDH, {}, {
        volatile auto s = secp256k1::ecdh_compute(our_sk, our_pk2);
    });

    auto r_ecdh_lib = BENCH("ecdh_lib", N_ECDH, {}, {
        uint8_t out[32];
        volatile int ok = secp256k1_ecdh(ctx, out, &lib_pk2, seckey_bytes, nullptr, nullptr);
    });

    print_row("ECDH", r_ecdh_ours, r_ecdh_lib);

    print_separator();

    // ═════════════════════════════════════════════════════════════════════
    // Section 7: CT Scalar Multiplication
    // ═════════════════════════════════════════════════════════════════════

    auto r_ct_scmul_ours = BENCH("ct_scalar_mul", N_SCMUL, auto k = random_scalar(), {
        volatile auto p = secp256k1::ct::scalar_mul(G, k);
    });

    // libsecp256k1's ec_pubkey_tweak_mul is the closest scalar_mul equivalent
    auto r_ct_scmul_lib = BENCH("ct_scmul_lib", N_SCMUL, {
        secp256k1_pubkey pk_tmp = lib_pk;
    }, {
        secp256k1_pubkey pk_copy = lib_pk;
        volatile int ok = secp256k1_ec_pubkey_tweak_mul(ctx, &pk_copy, seckey_bytes);
    });

    print_row("CT scalar_mul", r_ct_scmul_ours, r_ct_scmul_lib);

    // CT generator_mul vs ec_pubkey_create (both are k*G)
    auto r_ct_genmul_ours = BENCH("ct_generator_mul", N_SCMUL, auto k = random_scalar(), {
        volatile auto p = secp256k1::ct::generator_mul(k);
    });

    print_row("CT generator_mul", r_ct_genmul_ours, r_keygen_lib);

    // Fast scalar_mul for comparison
    auto r_fast_scmul = BENCH("fast_scalar_mul", N_SCMUL, auto k = random_scalar(), {
        volatile auto p = G.scalar_mul(k);
    });
    print_row_single("Fast scalar_mul", r_fast_scmul);

    print_separator();

    // ═════════════════════════════════════════════════════════════════════
    // Section 8: CT Primitives (ours only — libsecp doesn't expose these)
    // ═════════════════════════════════════════════════════════════════════

    // CT cmov256
    uint64_t buf_a[4] = {1,2,3,4}, buf_b[4] = {5,6,7,8};
    auto r_cmov = BENCH("ct_cmov256", N_PRIM, {}, {
        uint64_t mask = secp256k1::ct::bool_to_mask((_i & 1) != 0);
        secp256k1::ct::cmov256(buf_a, buf_b, mask);
    });
    print_row_single("CT cmov256", r_cmov);

    // CT cswap256
    auto r_cswap = BENCH("ct_cswap256", N_PRIM, {}, {
        uint64_t mask = secp256k1::ct::bool_to_mask((_i & 1) != 0);
        secp256k1::ct::cswap256(buf_a, buf_b, mask);
    });
    print_row_single("CT cswap256", r_cswap);

    // CT table lookup (16 entries)
    secp256k1::ct::CTJacobianPoint table[16];
    {
        auto pt = secp256k1::ct::CTJacobianPoint::from_point(G);
        for (int i = 0; i < 16; ++i) {
            table[i] = pt;
            pt = secp256k1::ct::point_add_complete(pt,
                     secp256k1::ct::CTJacobianPoint::from_point(G));
        }
    }
    auto r_tbl_lookup = BENCH("ct_table_lookup_16", N_PRIM, {}, {
        volatile auto p = secp256k1::ct::point_table_lookup(table, 16, _i & 15);
    });
    print_row_single("CT table lookup (16)", r_tbl_lookup);

    // CT is_zero_mask
    auto r_is_zero = BENCH("ct_is_zero_mask", N_PRIM, {}, {
        volatile auto m = secp256k1::ct::is_zero_mask(static_cast<uint64_t>(_i));
    });
    print_row_single("CT is_zero_mask", r_is_zero);

    // CT field_add
    FieldElement fe_a = FieldElement::from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    FieldElement fe_b = FieldElement::from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

    auto r_ct_fadd = BENCH("ct_field_add", N_PRIM, {}, {
        fe_a = secp256k1::ct::field_add(fe_a, fe_b);
    });
    print_row_single("CT field_add", r_ct_fadd);

    // CT field_mul
    auto r_ct_fmul = BENCH("ct_field_mul", N_PRIM, {}, {
        fe_a = secp256k1::ct::field_mul(fe_a, fe_b);
    });
    print_row_single("CT field_mul", r_ct_fmul);

    // CT field_inv
    auto r_ct_finv = BENCH("ct_field_inv", 10000, {}, {
        fe_a = secp256k1::ct::field_inv(fe_a);
    });
    print_row_single("CT field_inv", r_ct_finv);

    // CT scalar_add
    auto sc_a = random_scalar(), sc_b = random_scalar();
    auto r_ct_sadd = BENCH("ct_scalar_add", N_PRIM, {}, {
        sc_a = secp256k1::ct::scalar_add(sc_a, sc_b);
    });
    print_row_single("CT scalar_add", r_ct_sadd);

    // CT field_cmov
    auto r_ct_fcmov = BENCH("ct_field_cmov", N_PRIM, {}, {
        uint64_t mask = secp256k1::ct::bool_to_mask((_i & 1) != 0);
        secp256k1::ct::field_cmov(&fe_a, fe_b, mask);
    });
    print_row_single("CT field_cmov", r_ct_fcmov);

    // CT complete addition
    auto ct_P = secp256k1::ct::CTJacobianPoint::from_point(G);
    auto ct_Q = secp256k1::ct::CTJacobianPoint::from_point(our_pk);
    auto r_ct_cadd = BENCH("ct_complete_add", 10000, {}, {
        ct_P = secp256k1::ct::point_add_complete(ct_P, ct_Q);
    });
    print_row_single("CT complete addition", r_ct_cadd);

    print_footer();

    // ═════════════════════════════════════════════════════════════════════
    // Summary
    // ═════════════════════════════════════════════════════════════════════
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  შეჯამება\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  ლეგენდა:\n");
    printf("    თანაფარდობა = ჩვენი_ns / libsecp_ns  (< 1.0 = ჩვენი უფრო სწრაფია)\n");
    printf("    ✅ ჩვენი   — ჩვენი ბიბლიოთეკა მნიშვნელოვნად სწრაფია (< 0.85x)\n");
    printf("    ≈  თანაბარი — შესადარებელი სიჩქარე (0.85x – 1.15x)\n");
    printf("    ⚠️  libsecp — libsecp256k1 უფრო სწრაფია (> 1.15x)\n");
    printf("\n");
    printf("  შენიშვნა:\n");
    printf("    - libsecp256k1-ის ყველა ოპერაცია CT-ია (constant-time by design)\n");
    printf("    - ჩვენი ბიბლიოთეკის 'fast' path არ არის CT, მაგრამ უფრო სწრაფია\n");
    printf("    - ჩვენი 'ct::' namespace იძლევა CT გარანტიებს fast:: ტიპებზე\n");
    printf("    - CT primitives (cmov, cswap, lookup) მხოლოდ ჩვენს ბიბლიოთეკაშია\n");
    printf("      ექსპოზირებული — libsecp256k1 არ ავლენს ამ შიდა ინტერფეისებს\n");
    printf("\n");

    secp256k1_context_destroy(ctx);
    return 0;
}
