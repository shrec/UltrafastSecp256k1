// ============================================================================
// test_soundness_adaptor_dleq_forgery.cpp
// ============================================================================
// NEGATIVE-SOUNDNESS PROBE (the "test the WRONG case" methodology).
//
// Most tests assert that *correct* inputs are ACCEPTED (honest sign -> verify
// roundtrips). That is exactly why GHSA-c7q2-gv3g-rgxm slipped every review: the
// old ECDSA-adaptor "binding scalar" cancelled in verify and bound nothing, yet
// every honest roundtrip still passed. The only thing that would have caught it
// is a test built on the INVERTED invariant — construct an input that VIOLATES
// the security property and assert the verifier BLOCKS it. If the system cannot
// block the wrong case, that is the hole.
//
// INVARIANT (ECDSA adaptor pre-signature adaptability, GHSA-c7q2):
//     ecdsa_adaptor_verify(pre, P, m, T) == true
//         =>  log_G(pre.R_hat) == log_T(pre.R)
//     i.e. R_hat = k*G and R = k*T share the SAME secret k, so r = R.x is bound
//     to the adaptor base T. This is enforced by the Chaum-Pedersen DLEQ proof
//     (dleq_e, dleq_s). Without it, a verifying pre-signature need NOT be
//     adaptable — the whole point of the adaptor is void.
//
// PROBE: forge a pre-signature with log_G(R_hat) != log_T(R) that STILL satisfies
//     the other two verify checks (r == R.x, and the ECDSA relation
//     s_hat*R_hat == z*G + r*P, achieved by choosing s_hat = (z + r*x)*k1^-1).
//     Only the DLEQ binding can distinguish it. verify MUST reject. If it
//     accepts, the binding is vacuous (the GHSA-c7q2 hole) -> FAIL.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>

#include "secp256k1/adaptor.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// Fixed, reproducible secret material.
static const Scalar SK = Scalar::from_bytes({{
    0x3B,0x91,0x8F,0xC2,0x3A,0xA4,0x77,0xD8, 0xDE,0x9B,0x28,0x12,0xF3,0xBE,0x60,0xCE,
    0x8B,0x7E,0x45,0x26,0xA3,0x81,0x25,0x60, 0xB9,0x92,0x21,0x3F,0x19,0x33,0xAE,0x71}});
static const Scalar T_SECRET = Scalar::from_bytes({{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x2B}});
static const std::array<std::uint8_t, 32> MSG = {{
    0xDE,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xAD}};
// Two DISTINCT nonce scalars: k1 = dlog_G(R_hat), k2 = dlog_T(R). k1 != k2 is the
// whole point — the forged pre-signature has mismatched discrete logs.
static const Scalar K1 = Scalar::from_bytes({{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x17}});
static const Scalar K2 = Scalar::from_bytes({{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x53}});

int test_soundness_adaptor_dleq_forgery_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Negative-soundness probe: ECDSA adaptor DLEQ binding (GHSA-c7q2)\n");
    printf("  Invariant: verify==OK  =>  log_G(R_hat) == log_T(R)\n");
    printf("======================================================================\n\n");

    Point const T  = Point::generator().scalar_mul(T_SECRET);  // adaptor base T = t*G
    Point const PK = Point::generator().scalar_mul(SK);        // public key P = x*G
    Scalar const z = Scalar::from_bytes(MSG);

    // ── Control: an honest pre-signature MUST verify (the binding is live) ─────
    auto honest = ecdsa_adaptor_sign(SK, MSG, T);
    CHECK(ecdsa_adaptor_verify(honest, PK, MSG, T),
          "control: honest pre-signature verifies");

    // ── FORGE 1: mismatched discrete logs, ECDSA relation + r-check satisfied ──
    // R_hat = k1*G, R = k2*T with k1 != k2. Choose s_hat = (z + r*x) * k1^-1 so the
    // ECDSA relation s_hat*R_hat == z*G + r*P holds by construction, and r = R.x so
    // the r-check holds. The DLEQ proof is the ONLY thing that can reject it. We
    // borrow a well-formed (dleq_e, dleq_s) from the honest pre-sig; it cannot
    // validate for a mismatched (R_hat, R), so verify must reject.
    {
        Point  const R_hat = Point::generator().scalar_mul(K1);  // k1 * G
        Point  const R     = T.scalar_mul(K2);                   // k2 * T   (k2 != k1)
        Scalar const r     = Scalar::from_bytes(R.x().to_bytes());
        Scalar const s_hat = (z + r * SK) * K1.inverse();        // passes s_hat*R_hat == z*G + r*P
        ECDSAAdaptorSig forged{R_hat, R, s_hat, r, honest.dleq_e, honest.dleq_s};

        bool const accepted = ecdsa_adaptor_verify(forged, PK, MSG, T);
        CHECK(!accepted,
              "FORGE: log_G(R_hat) != log_T(R) (ECDSA relation OK) MUST be rejected by DLEQ (GHSA-c7q2)");
    }

    // ── FORGE 2: take an honest pre-sig and shift R off the proven point ──────
    // R' = R + T  => log_T(R') = k+1 != k = log_G(R_hat). r is recomputed from R'.
    {
        ECDSAAdaptorSig tampered = honest;
        tampered.R = honest.R.add(T);                          // (k+1)*T
        tampered.r = Scalar::from_bytes(tampered.R.x().to_bytes());
        CHECK(!ecdsa_adaptor_verify(tampered, PK, MSG, T),
              "FORGE: R shifted by +T (dlog mismatch) MUST be rejected");
    }

    // ── FORGE 3: tamper the DLEQ response s on an otherwise-honest pre-sig ────
    {
        ECDSAAdaptorSig tampered = honest;
        tampered.dleq_s = honest.dleq_s + Scalar::from_bytes({{
            0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1}});
        CHECK(!ecdsa_adaptor_verify(tampered, PK, MSG, T),
              "FORGE: tampered DLEQ response dleq_s MUST be rejected");
    }

    printf("\n[soundness_adaptor_dleq_forgery] %d/%d probes passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_soundness_adaptor_dleq_forgery_run(); }
#endif
