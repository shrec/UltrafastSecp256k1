// ============================================================================
// test_ct_equivalence.cpp — FAST ≡ CT Property-Based Equivalence Tests
// ============================================================================
// Verifies that CT and FAST functions return bit-identical results on:
//   1. Boundary scalars (0, 1, 2, n−1, n−2, (n+1)/2)
//   2. Random 256-bit scalars (property-based)
//   3. ECDSA sign equivalence (random keys + messages)
//   4. Schnorr sign equivalence (random keys + messages)
//   5. Schnorr pubkey equivalence
//   6. Group law invariants via CT (add/double/inverse)
//
// This test is the formal proof that the dual-layer FAST/CT architecture
// maintains semantic equivalence — the cornerstone of SECURITY_CLAIMS.md.
// ============================================================================

#include "secp256k1/fast.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <array>
#include <cstdint>

using FE = secp256k1::fast::FieldElement;
using SC = secp256k1::fast::Scalar;
using PT = secp256k1::fast::Point;

namespace ct = secp256k1::ct;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) {                                             \
            ++g_pass;                                           \
        } else {                                                \
            ++g_fail;                                           \
            std::cout << "  FAIL: " << msg << "\n";             \
            std::cout.flush();                                  \
        }                                                       \
    } while (0)

// --- Deterministic PRNG (seeded SHA256 counter) ---
// Not cryptographic, but deterministic and sufficient for test generation.
struct TestRng {
    std::array<uint8_t, 32> state;
    uint64_t counter = 0;

    explicit TestRng(uint64_t seed = 0x4654455354u) {
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
        buf[32] = static_cast<uint8_t>(counter);
        buf[33] = static_cast<uint8_t>(counter >> 8);
        buf[34] = static_cast<uint8_t>(counter >> 16);
        buf[35] = static_cast<uint8_t>(counter >> 24);
        buf[36] = static_cast<uint8_t>(counter >> 32);
        buf[37] = static_cast<uint8_t>(counter >> 40);
        buf[38] = static_cast<uint8_t>(counter >> 48);
        buf[39] = static_cast<uint8_t>(counter >> 56);
        ++counter;
        state = secp256k1::SHA256::hash(buf, 40);
        return state;
    }

    SC random_scalar() {
        // Keep generating until we get a valid non-zero scalar
        for (int i = 0; i < 100; ++i) {
            auto bytes = next();
            auto s = SC::from_bytes(bytes);
            if (!s.is_zero()) return s;
        }
        return SC::from_uint64(1); // fallback
    }

    std::array<uint8_t, 32> random_bytes() { return next(); }
};

static bool pt_eq(const PT& a, const PT& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x().to_bytes() == b.x().to_bytes() &&
           a.y().to_bytes() == b.y().to_bytes();
}

// ============================================================================
// 1. Boundary scalar equivalence: ct::generator_mul vs fast::scalar_mul
// ============================================================================
static void test_boundary_generator_mul() {
    std::cout << "--- Boundary: ct::generator_mul vs fast generator mul ---\n";

    PT G = PT::generator();

    // k = 1: both should return G
    {
        SC k = SC::from_uint64(1);
        PT ct_r = ct::generator_mul(k);
        PT fast_r = G.scalar_mul(k);
        CHECK(pt_eq(ct_r, fast_r), "generator_mul(1) equivalence");
    }
    // k = 2
    {
        SC k = SC::from_uint64(2);
        PT ct_r = ct::generator_mul(k);
        PT fast_r = G.scalar_mul(k);
        CHECK(pt_eq(ct_r, fast_r), "generator_mul(2) equivalence");
    }
    // k = n-1 (secp256k1 order minus 1)
    {
        SC n_minus_1 = SC::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
        PT ct_r = ct::generator_mul(n_minus_1);
        PT fast_r = G.scalar_mul(n_minus_1);
        CHECK(pt_eq(ct_r, fast_r), "generator_mul(n-1) equivalence");
    }
    // k = n-2
    {
        SC n_minus_2 = SC::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD036413F");
        PT ct_r = ct::generator_mul(n_minus_2);
        PT fast_r = G.scalar_mul(n_minus_2);
        CHECK(pt_eq(ct_r, fast_r), "generator_mul(n-2) equivalence");
    }
    // k = (n+1)/2  (midpoint of the group)
    {
        SC half_n = SC::from_hex(
            "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A1");
        PT ct_r = ct::generator_mul(half_n);
        PT fast_r = G.scalar_mul(half_n);
        CHECK(pt_eq(ct_r, fast_r), "generator_mul((n+1)/2) equivalence");
    }
}

// ============================================================================
// 2. Property-based: random scalars × G
// ============================================================================
static void test_random_generator_mul() {
    std::cout << "--- Property: 64 random ct::generator_mul vs fast ---\n";

    TestRng rng(0xDEADBEEFu);
    PT G = PT::generator();

    for (int i = 0; i < 64; ++i) {
        SC k = rng.random_scalar();
        PT ct_r = ct::generator_mul(k);
        PT fast_r = G.scalar_mul(k);
        CHECK(pt_eq(ct_r, fast_r), "random generator_mul equivalence #" + std::to_string(i));
    }
}

// ============================================================================
// 3. Property-based: random scalars × arbitrary P (ct::scalar_mul)
// ============================================================================
static void test_random_scalar_mul() {
    std::cout << "--- Property: 64 random ct::scalar_mul(P, k) vs fast ---\n";

    TestRng rng(0xCAFEBABEu);
    PT G = PT::generator();

    for (int i = 0; i < 64; ++i) {
        // Random base point
        SC base_k = rng.random_scalar();
        PT P = G.scalar_mul(base_k);

        // Random scalar
        SC k = rng.random_scalar();

        PT ct_r = ct::scalar_mul(P, k);
        PT fast_r = P.scalar_mul(k);
        bool eq = pt_eq(ct_r, fast_r);
        if (!eq) {
            // Gold-standard: (base_k * k) * G should equal k * P
            SC combined = base_k * k;
            PT gold = G.scalar_mul(combined);
            auto ct_xb = ct_r.x().to_bytes();
            auto fast_xb = fast_r.x().to_bytes();
            auto gold_xb = gold.x().to_bytes();
            std::cout << "  [DIAG #" << i << "] ct_x   =";
            for (auto b : ct_xb) std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)b;
            std::cout << "\n  [DIAG #" << i << "] fast_x =";
            for (auto b : fast_xb) std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)b;
            std::cout << "\n  [DIAG #" << i << "] gold_x =";
            for (auto b : gold_xb) std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)b;
            std::cout << std::dec << "\n";
            std::cout << "  [DIAG #" << i << "] ct==gold? " << pt_eq(ct_r, gold)
                      << " fast==gold? " << pt_eq(fast_r, gold) << "\n";
            std::cout.flush();
        }
        CHECK(eq, "random scalar_mul equivalence #" + std::to_string(i));
    }
}

// ============================================================================
// 4. Boundary scalar × arbitrary P
// ============================================================================
static void test_boundary_scalar_mul() {
    std::cout << "--- Boundary: ct::scalar_mul edge scalars ---\n";

    PT G = PT::generator();
    SC k7 = SC::from_uint64(7);
    PT P = G.scalar_mul(k7); // use 7*G as a non-trivial base point

    // k = 0: result should be infinity
    {
        SC k = SC::from_uint64(0);
        PT ct_r = ct::scalar_mul(P, k);
        PT fast_r = P.scalar_mul(k);
        CHECK(ct_r.is_infinity() && fast_r.is_infinity(),
              "scalar_mul(P, 0) == O");
    }
    // k = 1
    {
        SC k = SC::from_uint64(1);
        PT ct_r = ct::scalar_mul(P, k);
        CHECK(pt_eq(ct_r, P), "scalar_mul(P, 1) == P");
    }
    // k = 2
    {
        SC k = SC::from_uint64(2);
        PT ct_r = ct::scalar_mul(P, k);
        PT fast_r = P.scalar_mul(k);
        CHECK(pt_eq(ct_r, fast_r), "scalar_mul(P, 2) equivalence");
    }
    // k = n-1: P*(n-1) = -P
    {
        SC n_minus_1 = SC::from_hex(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
        PT ct_r = ct::scalar_mul(P, n_minus_1);
        PT fast_r = P.scalar_mul(n_minus_1);
        CHECK(pt_eq(ct_r, fast_r), "scalar_mul(P, n-1) equivalence (= -P)");
    }
}

// ============================================================================
// 5. ECDSA sign equivalence: 32 random key+msg pairs
// ============================================================================
static void test_ecdsa_sign_equivalence() {
    std::cout << "--- Property: 32 random ECDSA sign CT≡FAST ---\n";

    TestRng rng(0xEC05Au);
    PT G = PT::generator();

    for (int i = 0; i < 32; ++i) {
        SC privkey = rng.random_scalar();
        auto msg_hash = rng.random_bytes();

        auto ct_sig = secp256k1::ct::ecdsa_sign(msg_hash, privkey);
        auto fast_sig = secp256k1::ecdsa_sign(msg_hash, privkey);

        bool r_eq = (ct_sig.r.to_bytes() == fast_sig.r.to_bytes());
        bool s_eq = (ct_sig.s.to_bytes() == fast_sig.s.to_bytes());
        CHECK(r_eq && s_eq,
              "ECDSA sign equivalence #" + std::to_string(i));

        // Also verify the CT signature
        PT pubkey = G.scalar_mul(privkey);
        CHECK(secp256k1::ecdsa_verify(msg_hash, pubkey, ct_sig),
              "ECDSA CT sig verifies #" + std::to_string(i));
    }
}

// ============================================================================
// 6. Schnorr sign equivalence: 32 random key+msg pairs
// ============================================================================
static void test_schnorr_sign_equivalence() {
    std::cout << "--- Property: 32 random Schnorr sign CT≡FAST ---\n";

    TestRng rng(0x5CA00Bu);

    for (int i = 0; i < 32; ++i) {
        SC privkey = rng.random_scalar();
        auto msg = rng.random_bytes();
        auto aux = rng.random_bytes();

        auto ct_kp  = secp256k1::ct::schnorr_keypair_create(privkey);
        auto fast_kp = secp256k1::schnorr_keypair_create(privkey);

        // Keypair must match
        CHECK(ct_kp.px == fast_kp.px,
              "Schnorr keypair.px equivalence #" + std::to_string(i));

        auto ct_sig  = secp256k1::ct::schnorr_sign(ct_kp, msg, aux);
        auto fast_sig = secp256k1::schnorr_sign(fast_kp, msg, aux);

        bool r_eq = (ct_sig.r == fast_sig.r);
        bool s_eq = (ct_sig.s.to_bytes() == fast_sig.s.to_bytes());
        CHECK(r_eq && s_eq,
              "Schnorr sign equivalence #" + std::to_string(i));

        // Verify CT signature
        CHECK(secp256k1::schnorr_verify(ct_kp.px, msg, ct_sig),
              "Schnorr CT sig verifies #" + std::to_string(i));
    }
}

// ============================================================================
// 7. Schnorr pubkey equivalence: boundary + random
// ============================================================================
static void test_schnorr_pubkey_equivalence() {
    std::cout << "--- Schnorr pubkey CT≡FAST (boundary + random) ---\n";

    // k=1
    {
        SC k = SC::from_uint64(1);
        auto ct_px = secp256k1::ct::schnorr_pubkey(k);
        auto fast_px = secp256k1::schnorr_pubkey(k);
        CHECK(ct_px == fast_px, "schnorr_pubkey(1) equivalence");
    }
    // k=2
    {
        SC k = SC::from_uint64(2);
        auto ct_px = secp256k1::ct::schnorr_pubkey(k);
        auto fast_px = secp256k1::schnorr_pubkey(k);
        CHECK(ct_px == fast_px, "schnorr_pubkey(2) equivalence");
    }
    // Random
    TestRng rng(0xA1B2C3u);
    for (int i = 0; i < 16; ++i) {
        SC k = rng.random_scalar();
        auto ct_px = secp256k1::ct::schnorr_pubkey(k);
        auto fast_px = secp256k1::schnorr_pubkey(k);
        CHECK(ct_px == fast_px,
              "schnorr_pubkey random equivalence #" + std::to_string(i));
    }
}

// ============================================================================
// 8. CT group law: associativity, commutativity, identity, inverse
// ============================================================================
static void test_ct_group_law() {
    std::cout << "--- CT group law invariants ---\n";

    PT G = PT::generator();
    SC k3 = SC::from_uint64(3);
    SC k5 = SC::from_uint64(5);
    SC k7 = SC::from_uint64(7);

    PT P = G.scalar_mul(k3);
    PT Q = G.scalar_mul(k5);
    PT R = G.scalar_mul(k7);

    auto J_P = ct::CTJacobianPoint::from_point(P);
    auto J_Q = ct::CTJacobianPoint::from_point(Q);
    auto J_R = ct::CTJacobianPoint::from_point(R);
    auto J_O = ct::CTJacobianPoint::make_infinity();

    // Commutativity: P + Q == Q + P
    {
        auto pq = ct::point_add_complete(J_P, J_Q).to_point();
        auto qp = ct::point_add_complete(J_Q, J_P).to_point();
        CHECK(pt_eq(pq, qp), "CT commutativity: P+Q == Q+P");
    }
    // Associativity: (P+Q)+R == P+(Q+R)
    {
        auto pq = ct::point_add_complete(J_P, J_Q);
        auto pq_r = ct::point_add_complete(pq, J_R).to_point();
        auto qr = ct::point_add_complete(J_Q, J_R);
        auto p_qr = ct::point_add_complete(J_P, qr).to_point();
        CHECK(pt_eq(pq_r, p_qr), "CT associativity: (P+Q)+R == P+(Q+R)");
    }
    // Identity: P + O == P
    {
        auto po = ct::point_add_complete(J_P, J_O).to_point();
        CHECK(pt_eq(po, P), "CT identity: P+O == P");
    }
    // Inverse: P + (-P) == O
    {
        auto neg_P = ct::point_neg(J_P);
        auto sum = ct::point_add_complete(J_P, neg_P).to_point();
        CHECK(sum.is_infinity(), "CT inverse: P+(-P) == O");
    }
    // Doubling consistency: P+P == ct::point_dbl(P)
    {
        auto pp = ct::point_add_complete(J_P, J_P).to_point();
        auto dbl = ct::point_dbl(J_P).to_point();
        CHECK(pt_eq(pp, dbl), "CT doubling: P+P == dbl(P)");
    }
}

// ============================================================================
// Main
// ============================================================================

int test_ct_equivalence_run() {
    std::cout << "=== FAST ≡ CT Equivalence Tests ===\n\n";

    test_boundary_generator_mul();
    test_random_generator_mul();
    test_random_scalar_mul();
    test_boundary_scalar_mul();
    test_ecdsa_sign_equivalence();
    test_schnorr_sign_equivalence();
    test_schnorr_pubkey_equivalence();
    test_ct_group_law();

    std::cout << "\n=== CT Equivalence: " << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail > 0 ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_ct_equivalence_run(); }
#endif
