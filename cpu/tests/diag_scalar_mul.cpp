// Detailed diagnostic: trace ct::scalar_mul GLV decomposition for failing scalars
#include "secp256k1/fast.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/sha256.hpp"
#include <iostream>
#include <iomanip>
#include <array>
#include <cstring>
#include <cstdio>

using FE = secp256k1::fast::FieldElement;
using SC = secp256k1::fast::Scalar;
using PT = secp256k1::fast::Point;
namespace ct = secp256k1::ct;

static void print_scalar(const char* label, const SC& s) {
    auto b = s.to_bytes();
    printf("%s", label);
    for (auto x : b) printf("%02x", x);
    printf("\n");
}

static void print_limbs(const char* label, const SC& s) {
    auto l = s.limbs();
    printf("%s[%016lx %016lx %016lx %016lx]\n", label,
           (unsigned long)l[3], (unsigned long)l[2],
           (unsigned long)l[1], (unsigned long)l[0]);
}

static void print_point_xy(const char* label, const PT& p) {
    if (p.is_infinity()) { printf("%sINFINITY\n", label); return; }
    auto xb = p.x().to_bytes();
    printf("%s", label);
    for (auto x : xb) printf("%02x", x);
    printf("\n");
}

struct TestRng {
    std::array<uint8_t, 32> state;
    uint64_t counter = 0;
    explicit TestRng(uint64_t seed) {
        std::memset(state.data(), 0, 32);
        state[0] = static_cast<uint8_t>(seed);
        state[1] = static_cast<uint8_t>(seed >> 8);
        state[2] = static_cast<uint8_t>(seed >> 16);
        state[3] = static_cast<uint8_t>(seed >> 24);
        state[4] = static_cast<uint8_t>(seed >> 32);
    }
    std::array<uint8_t, 32> next() {
        uint8_t buf[40];
        std::memcpy(buf, state.data(), 32);
        for (int j = 0; j < 8; j++) buf[32+j] = static_cast<uint8_t>(counter >> (j*8));
        ++counter;
        state = secp256k1::SHA256::hash(buf, 40);
        return state;
    }
    SC random_scalar() {
        for (int i = 0; i < 100; ++i) {
            auto bytes = next();
            auto s = SC::from_bytes(bytes);
            if (!s.is_zero()) return s;
        }
        return SC::from_uint64(1);
    }
};

// Replicate the K_CONST and S_OFFSET
static const SC K_CONST = SC::from_limbs({
    0xb5c2c1dcde9798d9ULL, 0x589ae84826ba29e4ULL,
    0xc2bdd6bf7c118d6bULL, 0xa4e88a7dcb13034eULL
});
static const SC S_OFFSET = SC::from_limbs({0, 0, 1, 0});

static void diagnose_scalar(const char* name, const SC& k) {
    printf("\n=== DIAGNOSE scalar: %s ===\n", name);
    print_limbs("  k       = ", k);
    
    // Step 1: s = (k + K_CONST) / 2
    SC s = ct::scalar_add(k, K_CONST);
    s = ct::scalar_half(s);
    print_limbs("  s=(k+K)/2= ", s);
    
    // Step 2: GLV decompose
    auto [k1_abs, k2_abs, k1_neg, k2_neg] = ct::ct_glv_decompose(s);
    print_limbs("  k1_abs  = ", k1_abs);
    print_limbs("  k2_abs  = ", k2_abs);
    printf("  k1_neg=%lu  k2_neg=%lu\n", (unsigned long)k1_neg, (unsigned long)k2_neg);
    
    // Check GLV bound: |ki| < 2^128 means limbs[2] and limbs[3] should be 0
    auto k1l = k1_abs.limbs();
    auto k2l = k2_abs.limbs();
    printf("  k1_abs upper: [%016lx %016lx] (expect 0)\n",
           (unsigned long)k1l[3], (unsigned long)k1l[2]);
    printf("  k2_abs upper: [%016lx %016lx] (expect 0)\n",
           (unsigned long)k2l[3], (unsigned long)k2l[2]);
    
    if (k1l[2] != 0 || k1l[3] != 0)
        printf("  *** WARNING: k1_abs > 2^128! Upper limbs non-zero!\n");
    if (k2l[2] != 0 || k2l[3] != 0)
        printf("  *** WARNING: k2_abs > 2^128! Upper limbs non-zero!\n");
    
    // Step 3: signed values and v1, v2
    SC s1 = ct::scalar_cneg(k1_abs, k1_neg);
    SC s2 = ct::scalar_cneg(k2_abs, k2_neg);
    print_limbs("  s1(signed)= ", s1);
    print_limbs("  s2(signed)= ", s2);
    
    SC v1 = ct::scalar_add(s1, S_OFFSET);
    SC v2 = ct::scalar_add(s2, S_OFFSET);
    print_limbs("  v1=s1+2^128= ", v1);
    print_limbs("  v2=s2+2^128= ", v2);
    
    // Check v1, v2 upper limbs (should be <= 2^129, so limbs[2] <= 3, limbs[3]=0)
    auto v1l = v1.limbs();
    auto v2l = v2.limbs();
    printf("  v1 upper: [%016lx %016lx]\n", (unsigned long)v1l[3], (unsigned long)v1l[2]);
    printf("  v2 upper: [%016lx %016lx]\n", (unsigned long)v2l[3], (unsigned long)v2l[2]);
    
    // Print window digits for v1 and v2
    printf("  v1 digits (group 25..0): ");
    for (int g = 25; g >= 0; --g) {
        std::uint64_t w = ct::scalar_window(v1, g*5, 5);
        printf("%02lu ", (unsigned long)w);
    }
    printf("\n");
    printf("  v2 digits (group 25..0): ");
    for (int g = 25; g >= 0; --g) {
        std::uint64_t w = ct::scalar_window(v2, g*5, 5);
        printf("%02lu ", (unsigned long)w);
    }
    printf("\n");
    
    // Verify GLV: s1 + s2*lambda should == s (mod n)
    static const SC lambda_sc = SC::from_bytes({{
        0x53,0x63,0xAD,0x4C,0xC0,0x5C,0x30,0xE0,
        0xA5,0x26,0x1C,0x02,0x88,0x12,0x64,0x5A,
        0x12,0x2E,0x22,0xEA,0x20,0x81,0x66,0x78,
        0xDF,0x02,0x96,0x7C,0x1B,0x23,0xBD,0x72
    }});
    SC s2_lambda = k2_abs * lambda_sc;
    if (k2_neg) s2_lambda = ct::scalar_cneg(s2_lambda, 1);
    SC recombined = ct::scalar_add(s1, s2_lambda); // s1 + s2*lambda should == s
    print_limbs("  s1+s2*lam= ", recombined);
    print_limbs("  s(expect)= ", s);
    
    bool glv_ok = true;
    auto sl = s.limbs();
    auto rl = recombined.limbs();
    for (int i = 0; i < 4; i++) if (sl[i] != rl[i]) glv_ok = false;
    printf("  GLV verify: %s\n", glv_ok ? "OK" : "*** MISMATCH ***");
}

int main() {
    printf("=== ct::scalar_mul detailed diagnostic ===\n\n");
    
    PT G = PT::generator();
    int fails = 0;
    
    // Run through first 64 random scalars with seed 0xCAFEBABE, report pass/fail
    TestRng rng(0xCAFEBABEu);
    printf("--- Random scalar_mul tests (seed 0xCAFEBABE) ---\n");
    for (int i = 0; i < 64; i++) {
        SC base_k = rng.random_scalar();
        PT P = G.scalar_mul(base_k);
        SC k = rng.random_scalar();
        
        PT ct_r = ct::scalar_mul(P, k);
        PT fast_r = P.scalar_mul(k);
        
        bool eq = (ct_r.is_infinity() && fast_r.is_infinity()) ||
                  (!ct_r.is_infinity() && !fast_r.is_infinity() &&
                   ct_r.x().to_bytes() == fast_r.x().to_bytes() &&
                   ct_r.y().to_bytes() == fast_r.y().to_bytes());
        
        if (!eq) {
            printf("  [%d] FAIL\n", i);
            diagnose_scalar("failing_k", k);
            fails++;
        }
    }
    printf("\nSummary: %d/64 failures\n", fails);
    
    // Also diagnose a known-good scalar
    printf("\n--- Known good scalar (k=7) ---\n");
    SC k_good = SC::from_uint64(7);
    diagnose_scalar("k=7", k_good);
    
    // And diagnose the first failing scalar directly
    printf("\n--- First failing scalar (re-check ct::scalar_mul(G,k)) ---\n");
    TestRng rng2(0xCAFEBABEu);
    rng2.random_scalar(); // skip base_k
    SC k_fail = rng2.random_scalar();
    diagnose_scalar("first_fail_k", k_fail);
    
    // Test ct::scalar_mul(G, k_fail) specifically
    PT ct_gk = ct::scalar_mul(G, k_fail);
    PT fast_gk = G.scalar_mul(k_fail);
    bool gk_eq = (!ct_gk.is_infinity() && !fast_gk.is_infinity() &&
                  ct_gk.x().to_bytes() == fast_gk.x().to_bytes());
    printf("ct::scalar_mul(G, k_fail): %s\n", gk_eq ? "OK" : "MISMATCH");
    
    return fails > 0 ? 1 : 0;
}
