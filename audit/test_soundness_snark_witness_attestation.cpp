// ============================================================================
// test_soundness_snark_witness_attestation.cpp
// ============================================================================
// NEGATIVE-SOUNDNESS PROBE for the SNARK foreign-field WITNESS attestations.
//
// zk::ecdsa_snark_witness / zk::schnorr_snark_witness produce a witness whose
// `valid` flag is consumed by a SNARK circuit (eprint 2025/695) as GROUND TRUTH:
// the circuit proves "I know a valid signature" by trusting witness.valid. The
// witnesses REIMPLEMENT the verification equation rather than calling the audited
// verifier — exactly the GHSA-c7q2 shape (a verifying-but-unsound attestation that
// the bool-`*verify*` soundness scan never saw, because they return a struct).
//
// SOUNDNESS INVARIANT:
//     witness.valid == true   MUST IMPLY   canonical_verify == OK
//     for the SAME (msg, pubkey, signature). If the attestation can say "valid"
//     for an input the network's verifier rejects (non-canonical r>=p, tampered
//     scalar, malleable s, ...), a prover can attest an invalid signature.
//
// We assert the stronger EQUALITY witness.valid == verify across honest + forged
// inputs. A mismatch is an unsound attestation -> FAIL. Honest controls also
// assert both sides accept (completeness sanity).
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>

#include "secp256k1/zk.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

static const Scalar SK = Scalar::from_bytes({{
    0x6B,0x12,0x77,0xA9,0x3E,0x55,0x8B,0xE7, 0x10,0xD3,0x2A,0x6C,0x84,0x19,0xFA,0x2D,
    0xB6,0x71,0x0C,0x35,0x9B,0x2C,0x41,0x9A, 0x77,0x0E,0xB2,0x5D,0x18,0xA9,0x63,0x44}});
static const std::array<std::uint8_t, 32> MSG = {{
    0x9E,0x11,0x22,0x33,0x44,0x55,0x66,0x77, 0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,
    0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08, 0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0xA9}};
static const Scalar ONE = Scalar::from_bytes({{
    0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1}});

static ECDSASignature honest_ecdsa() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return ecdsa_sign(MSG, SK);   // VT signer: test setup only (no secret-timing claim here)
#pragma GCC diagnostic pop
}

int test_soundness_snark_witness_attestation_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Negative-soundness probe: SNARK-witness attestation (eprint 2025/695)\n");
    printf("  Invariant: witness.valid == canonical_verify  (valid==1 => verify==OK)\n");
    printf("======================================================================\n\n");

    Point const P = Point::generator().scalar_mul(SK);

    // ── ECDSA ────────────────────────────────────────────────────────────────
    ECDSASignature const sig = honest_ecdsa();

    {   // control: honest signature — both accept
        auto w = zk::ecdsa_snark_witness(MSG, P, sig.r, sig.s);
        bool v = ecdsa_verify(MSG.data(), P, sig);
        CHECK(v && w.valid == v, "ECDSA control: witness.valid == verify == true");
    }
    {   // forge: tampered s — verify rejects; attestation MUST agree
        ECDSASignature bad = sig; bad.s = sig.s + ONE;
        bool wv = zk::ecdsa_snark_witness(MSG, P, bad.r, bad.s).valid;
        bool vv = ecdsa_verify(MSG.data(), P, bad);
        CHECK(wv == vv, "ECDSA tampered-s: attestation valid MUST match verify");
    }
    {   // forge: tampered r
        ECDSASignature bad = sig; bad.r = sig.r + ONE;
        bool wv = zk::ecdsa_snark_witness(MSG, P, bad.r, bad.s).valid;
        bool vv = ecdsa_verify(MSG.data(), P, bad);
        CHECK(wv == vv, "ECDSA tampered-r: attestation valid MUST match verify");
    }
    {   // forge: malleable high-s (s -> n-s) — attestation MUST track verify's policy
        ECDSASignature mall = sig; mall.s = Scalar::zero() - sig.s;
        bool wv = zk::ecdsa_snark_witness(MSG, P, mall.r, mall.s).valid;
        bool vv = ecdsa_verify(MSG.data(), P, mall);
        CHECK(wv == vv, "ECDSA high-s malleable: attestation valid MUST match verify");
    }

    // ── Schnorr (BIP-340) ─────────────────────────────────────────────────────
    std::array<std::uint8_t, 32> const PX = schnorr_pubkey(SK);
    std::array<std::uint8_t, 32> const AUX{};
    SchnorrSignature const ssig = schnorr_sign(SK, MSG, AUX);

    {   // control
        auto w = zk::schnorr_snark_witness(MSG, PX, ssig.r, ssig.s);
        bool v = schnorr_verify(PX, MSG, ssig);
        CHECK(v && w.valid == v, "Schnorr control: witness.valid == verify == true");
    }
    {   // forge: non-canonical r >= p (all-0xFF). schnorr_verify strict-rejects (lift_x
        // fails for r>=p); the witness parses sig_r via FieldElement::from_bytes (silent
        // mod-p reduce). valid MUST NOT be true while verify rejects.
        std::array<std::uint8_t, 32> rbig; rbig.fill(0xFF);
        SchnorrSignature sb = ssig; sb.r = rbig;
        bool wv = zk::schnorr_snark_witness(MSG, PX, rbig, ssig.s).valid;
        bool vv = schnorr_verify(PX, MSG, sb);
        CHECK(wv == vv, "Schnorr r>=p non-canonical: attestation valid MUST match verify");
    }
    {   // forge: s == 0
        SchnorrSignature s0 = ssig; s0.s = Scalar::zero();
        bool wv = zk::schnorr_snark_witness(MSG, PX, ssig.r, Scalar::zero()).valid;
        bool vv = schnorr_verify(PX, MSG, s0);
        CHECK(wv == vv, "Schnorr s==0: attestation valid MUST match verify");
    }
    {   // forge: tampered r (flip a byte, stays < p)
        std::array<std::uint8_t, 32> rt = ssig.r; rt[20] ^= 0x01;
        SchnorrSignature st = ssig; st.r = rt;
        bool wv = zk::schnorr_snark_witness(MSG, PX, rt, ssig.s).valid;
        bool vv = schnorr_verify(PX, MSG, st);
        CHECK(wv == vv, "Schnorr tampered-r: attestation valid MUST match verify");
    }
    {   // forge: wrong message
        std::array<std::uint8_t, 32> m2 = MSG; m2[0] ^= 0xFF;
        bool wv = zk::schnorr_snark_witness(m2, PX, ssig.r, ssig.s).valid;
        bool vv = schnorr_verify(PX, m2, ssig);
        CHECK(wv == vv, "Schnorr wrong-msg: attestation valid MUST match verify");
    }

    printf("\n[soundness_snark_witness_attestation] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_soundness_snark_witness_attestation_run(); }
#endif
