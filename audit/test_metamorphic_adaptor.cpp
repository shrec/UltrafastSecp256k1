// ============================================================================
// test_metamorphic_adaptor.cpp
// ============================================================================
// POSITIVE-METAMORPHIC PROBE — the complement to the negative-soundness probe.
//
// A negative-soundness probe forges a VIOLATING input and asserts rejection
// (test_soundness_adaptor_dleq_forgery.cpp, GHSA-c7q2). A metamorphic probe
// asserts an ALGEBRAIC RELATION that must be preserved across a protocol
// transformation — not "this one honest roundtrip works", but "for the family
// of inputs, applying transform F then G yields a value related to the inputs
// by an identity that cannot hold by accident".
//
// Why this matters as its own class: a roundtrip test (sign then verify==OK)
// proves a single happy path. A metamorphic relation pins the *structure* of
// the transform — e.g. adapt(pre, t) must produce a signature whose witness,
// when extracted, is exactly +/-t, and whose r is the pre-signature's r
// unchanged. A bug that breaks the adapt/extract correspondence (the atomicity
// guarantee that makes DLCs / atomic swaps safe) escapes a plain roundtrip but
// breaks the relation. These are the ECDSA-adaptor C++-API relations; the C-ABI
// Schnorr roundtrip is covered separately (test_exploit_adaptor_extraction_soundness).
//
// RELATIONS (all over T = t*G, signer key x, message m):
//   MR1  adapt-validity:        ecdsa_verify(m, P, adapt(pre, t)) == true
//   MR2  extract inverts adapt: extract(pre, adapt(pre, t)) == (+/-t, ok)
//   MR3  r invariant:           adapt(pre, t).r == pre.r   (adapt acts only on s)
//   MR4  pre-sig boundary:      {pre.r, pre.s_hat} as a plain sig MUST NOT verify,
//                               yet adapt(pre, t) DOES — the transform crosses the
//                               validity boundary (a pre-signature is not a sig).
//   MR5  adapt determinism:     adapt(pre, t) is a pure function (equal bytes)
//   MR6  witness correspondence: distinct T1=t1*G, T2=t2*G -> each pre-sig's
//                               adapt/extract recovers its OWN +/-t (no crossover)
//
// If any relation breaks, the adapt/extract machinery is structurally wrong even
// when honest roundtrips pass. That is exactly the gap a metamorphic gate closes.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>

#include "secp256k1/adaptor.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// Fixed, reproducible material.
static const Scalar SK = Scalar::from_bytes({{
    0x2C,0x41,0x9A,0x77,0x0E,0xB2,0x5D,0x18, 0xA9,0x63,0x44,0xF1,0x07,0xCB,0x90,0x3E,
    0x55,0x8B,0xE7,0x10,0xD3,0x2A,0x6C,0x84, 0x19,0xFA,0x2D,0xB6,0x71,0x0C,0x35,0x9B}});
static const std::array<std::uint8_t, 32> MSG = {{
    0xA5,0x00,0x11,0x22,0x33,0x44,0x55,0x66, 0x77,0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,
    0xFF,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x5A}};
// Two distinct adaptor witnesses t1, t2 for the cross-correspondence relation.
static const Scalar T1 = Scalar::from_bytes({{
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0x3D}});
static const Scalar T2 = Scalar::from_bytes({{
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0x77}});

static bool recovers(const Scalar& got, const Scalar& want) {
    return got == want || got == want.negate();
}

int test_metamorphic_adaptor_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Metamorphic probe: ECDSA adaptor adapt/extract relations\n");
    printf("  Positive complement to the GHSA-c7q2 negative-soundness probe\n");
    printf("======================================================================\n\n");

    Point const PK = Point::generator().scalar_mul(SK);
    Point const Tp1 = Point::generator().scalar_mul(T1);
    Point const Tp2 = Point::generator().scalar_mul(T2);

    auto pre1 = ecdsa_adaptor_sign(SK, MSG, Tp1);
    CHECK(ecdsa_adaptor_verify(pre1, PK, MSG, Tp1),
          "control: honest pre-signature over T1 verifies");

    // MR1 — adapt-validity: the adapted pre-signature is a real ECDSA signature.
    ECDSASignature const full1 = ecdsa_adaptor_adapt(pre1, T1);
    CHECK(ecdsa_verify(MSG.data(), PK, full1),
          "MR1: ecdsa_verify(m, P, adapt(pre, t)) == true");

    // MR2 — extract inverts adapt: the recovered witness is +/- the original t.
    {
        auto [ext, ok] = ecdsa_adaptor_extract(pre1, full1);
        CHECK(ok && recovers(ext, T1),
              "MR2: extract(pre, adapt(pre, t)) recovers +/-t");
    }

    // MR3 — r invariant: adapt acts only on the s component, never on r.
    CHECK(full1.r == pre1.r,
          "MR3: adapt(pre, t).r == pre.r (r is invariant under adapt)");

    // MR4 — pre-sig boundary: the pre-signature's encrypted scalar is NOT a valid
    // plain signature, but adapt crosses the boundary into validity.
    {
        ECDSASignature const raw{pre1.r, pre1.s_hat};
        bool const raw_ok = ecdsa_verify(MSG.data(), PK, raw);
        CHECK(!raw_ok,
              "MR4a: {pre.r, pre.s_hat} as a plain sig MUST NOT verify (pre-sig != sig)");
        CHECK(ecdsa_verify(MSG.data(), PK, full1),
              "MR4b: ...yet adapt(pre, t) DOES verify (transform crosses the boundary)");
    }

    // MR5 — adapt determinism: adapt is a pure function of (pre, t).
    {
        ECDSASignature const again = ecdsa_adaptor_adapt(pre1, T1);
        CHECK(again.to_compact() == full1.to_compact(),
              "MR5: adapt(pre, t) is deterministic (identical bytes)");
    }

    // MR6 — witness correspondence across distinct adaptors: each pre-sig's
    // adapt/extract recovers its OWN witness; no crossover between T1 and T2.
    {
        auto pre2 = ecdsa_adaptor_sign(SK, MSG, Tp2);
        CHECK(ecdsa_adaptor_verify(pre2, PK, MSG, Tp2),
              "control: honest pre-signature over T2 verifies");

        ECDSASignature const full2 = ecdsa_adaptor_adapt(pre2, T2);
        auto [e1, ok1] = ecdsa_adaptor_extract(pre1, full1);
        auto [e2, ok2] = ecdsa_adaptor_extract(pre2, full2);
        CHECK(ok1 && recovers(e1, T1) && !recovers(e1, T2),
              "MR6a: T1 pre-sig recovers +/-t1, never +/-t2");
        CHECK(ok2 && recovers(e2, T2) && !recovers(e2, T1),
              "MR6b: T2 pre-sig recovers +/-t2, never +/-t1");
    }

    printf("\n[metamorphic_adaptor] %d/%d relations held\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_metamorphic_adaptor_run(); }
#endif
