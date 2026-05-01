// ============================================================================
// GPU CT Layer Compilation + Smoke Test
// ============================================================================
// Exercises all CT headers to verify compilation and basic correctness.
// Kernels: ct_sign ECDSA, ct_sign Schnorr, CT field/scalar/point ops.
// ============================================================================

#include "ct/ct_sign.cuh"
#include "ct/ct_zk.cuh"
#include <cstdio>
#include <cstring>

using namespace secp256k1::cuda;

// -- Known test vector (from BIP-340 / Wycheproof) ----------------------------
// Private key: 1 (simplest non-trivial key)

__device__ static const Scalar TEST_PRIVKEY = {
    {1ULL, 0ULL, 0ULL, 0ULL}
};

// SHA-256("test") truncated to 32 bytes
__device__ static const uint8_t TEST_MSG[32] = {
    0x9f, 0x86, 0xd0, 0x81, 0x88, 0x4c, 0x7d, 0x65,
    0x9a, 0x2f, 0xea, 0xa0, 0xc5, 0x5a, 0xd0, 0x15,
    0xa3, 0xbf, 0x4f, 0x1b, 0x2b, 0x0b, 0x82, 0x2c,
    0xd1, 0x5d, 0x6c, 0x15, 0xb0, 0xf0, 0x0a, 0x08
};

__device__ static const uint8_t ZERO_AUX[32] = {0};

// ---------- CT ECDSA Sign + Verify -------------------------------------------

__global__ void test_ct_ecdsa_kernel(int* result) {
    ECDSASignatureGPU sig;
    bool ok = ct::ct_ecdsa_sign(TEST_MSG, &TEST_PRIVKEY, &sig);

    if (!ok) { *result = 1; return; }

    // Verify: compute pubkey via CT, then verify with fast path
    JacobianPoint pubkey;
    ct::ct_generator_mul(&TEST_PRIVKEY, &pubkey);

    bool verified = ecdsa_verify(TEST_MSG, &pubkey, &sig);
    *result = verified ? 0 : 2;
}

// ---------- CT Schnorr Sign + Verify -----------------------------------------

__global__ void test_ct_schnorr_kernel(int* result) {
    SchnorrSignatureGPU sig;
    bool ok = ct::ct_schnorr_sign(&TEST_PRIVKEY, TEST_MSG, ZERO_AUX, &sig);

    if (!ok) { *result = 1; return; }

    // Get pubkey for verification
    uint8_t pubkey_x[32];
    ct::ct_schnorr_pubkey(&TEST_PRIVKEY, pubkey_x);

    bool verified = schnorr_verify(pubkey_x, TEST_MSG, &sig);
    *result = verified ? 0 : 2;
}

// ---------- CT Schnorr Keypair Sign + Verify ---------------------------------

__global__ void test_ct_schnorr_keypair_kernel(int* result) {
    ct::CTSchnorrKeypairGPU kp;
    bool ok = ct::ct_schnorr_keypair_create(&TEST_PRIVKEY, &kp);
    if (!ok) { *result = 1; return; }

    SchnorrSignatureGPU sig;
    ok = ct::ct_schnorr_sign_with_keypair(&kp, TEST_MSG, ZERO_AUX, &sig);
    if (!ok) { *result = 2; return; }

    bool verified = schnorr_verify(kp.px, TEST_MSG, &sig);
    *result = verified ? 0 : 3;
}

// ---------- CT vs Fast ECDSA Cross-Check ------------------------------------
// Both should produce the same r value (same nonce from RFC6979, same k*G)

__global__ void test_ct_fast_ecdsa_parity_kernel(int* result) {
    ECDSASignatureGPU ct_sig, fast_sig;

    ct::ct_ecdsa_sign(TEST_MSG, &TEST_PRIVKEY, &ct_sig);
    ecdsa_sign(TEST_MSG, &TEST_PRIVKEY, &fast_sig);

    // r must match (same nonce → same R.x → same r)
    bool r_match = true;
    for (int i = 0; i < 4; i++) {
        if (ct_sig.r.limbs[i] != fast_sig.r.limbs[i]) r_match = false;
    }
    // s must match (same k_inv, same z+rd)
    bool s_match = true;
    for (int i = 0; i < 4; i++) {
        if (ct_sig.s.limbs[i] != fast_sig.s.limbs[i]) s_match = false;
    }

    if (!r_match) { *result = 1; return; }
    if (!s_match) { *result = 2; return; }
    *result = 0;
}

// ---------- CT Field/Scalar/Point Ops ----------------------------------------

__global__ void test_ct_ops_kernel(int* result) {
    // Test CT field add/sub round-trip
    FieldElement a, b, sum, diff;
    field_set_zero(&a); a.limbs[0] = 0x123456789ABCDEF0ULL;
    field_set_zero(&b); b.limbs[0] = 0xFEDCBA9876543210ULL;

    ct::field_add(&a, &b, &sum);
    ct::field_sub(&sum, &b, &diff);

    // diff should equal a
    bool field_ok = true;
    for (int i = 0; i < 4; i++) {
        if (diff.limbs[i] != a.limbs[i]) field_ok = false;
    }

    if (!field_ok)  { *result = 1; return; }
    *result = 0;
}

__global__ void test_ct_scalar_basic_kernel(int* result) {
    // Test CT scalar add/sub round-trip
    Scalar sa, sb, ssum, sdiff;
    sa.limbs[0] = 42; sa.limbs[1] = 0; sa.limbs[2] = 0; sa.limbs[3] = 0;
    sb.limbs[0] = 99; sb.limbs[1] = 0; sb.limbs[2] = 0; sb.limbs[3] = 0;

    ct::scalar_add(&sa, &sb, &ssum);
    ct::scalar_sub(&ssum, &sb, &sdiff);

    bool scalar_ok = true;
    for (int i = 0; i < 4; i++) {
        if (sdiff.limbs[i] != sa.limbs[i]) scalar_ok = false;
    }

    // Test CT scalar_is_high: value 1 should NOT be high
    uint64_t high = ct::scalar_is_high(&sa);
    bool high_ok = (high == 0);

    // Test CT scalar_cneg
    Scalar neg_sa;
    ct::scalar_cneg(&neg_sa, &sa, ~(uint64_t)0);  // negate
    Scalar re_neg;
    ct::scalar_cneg(&re_neg, &neg_sa, ~(uint64_t)0);  // negate again

    bool cneg_ok = true;
    for (int i = 0; i < 4; i++) {
        if (re_neg.limbs[i] != sa.limbs[i]) cneg_ok = false;
    }

    if (!scalar_ok) { *result = 2; return; }
    if (!high_ok)   { *result = 3; return; }
    if (!cneg_ok)   { *result = 4; return; }
    *result = 0;
}

__global__ void test_ct_scalar_inv_kernel(int* result) {
    // Test CT scalar_inverse: a * a^{-1} = 1
    Scalar sa;
    sa.limbs[0] = 42; sa.limbs[1] = 0; sa.limbs[2] = 0; sa.limbs[3] = 0;
    Scalar sa_inv, product;
    ct::scalar_inverse(&sa, &sa_inv);
    ct::scalar_mul(&sa, &sa_inv, &product);

    bool inv_ok = (product.limbs[0] == 1 && product.limbs[1] == 0 &&
                   product.limbs[2] == 0 && product.limbs[3] == 0);

    if (!inv_ok)    { *result = 5; return; }
    *result = 0;
}

// ---------- CT ZK Knowledge Proof: Prove + Verify ----------------------------

__global__ void test_ct_zk_knowledge_kernel(int* result) {
    // Compute pubkey P = secret * G (CT)
    JacobianPoint pubkey;
    ct::ct_generator_mul(&TEST_PRIVKEY, &pubkey);

    // Prove knowledge of secret
    KnowledgeProofGPU proof;
    bool ok = ct::ct_knowledge_prove_generator_device(
        &TEST_PRIVKEY, &pubkey, TEST_MSG, ZERO_AUX, &proof);
    if (!ok) { *result = 1; return; }

    // Verify with fast path: convert pubkey to affine
    FieldElement ax, ay;
    jacobian_to_affine(&pubkey, &ax, &ay);
    AffinePoint pub_aff;
    pub_aff.x = ax;
    pub_aff.y = ay;

    bool verified = knowledge_verify_device(&proof, &pub_aff, TEST_MSG);
    *result = verified ? 0 : 2;
}

// ---------- CT ZK DLEQ Proof: Prove + Verify --------------------------------

__global__ void test_ct_zk_dleq_kernel(int* result) {
    // Set up generator G
    JacobianPoint G;
    for (int i = 0; i < 4; ++i) {
        G.x.limbs[i] = GENERATOR_X[i];
        G.y.limbs[i] = GENERATOR_Y[i];
    }
    field_set_one(&G.z);
    G.infinity = false;

    // Use H = 2*G as second base (deterministic, always valid)
    JacobianPoint H;
    jacobian_double(&G, &H);

    // P = secret * G, Q = secret * H  (CT)
    JacobianPoint P, Q;
    ct::ct_scalar_mul(&G, &TEST_PRIVKEY, &P);
    ct::ct_scalar_mul(&H, &TEST_PRIVKEY, &Q);

    // Prove DLEQ
    DLEQProofGPU proof;
    bool ok = ct::ct_dleq_prove_device(
        &TEST_PRIVKEY, &G, &H, &P, &Q, ZERO_AUX, &proof);
    if (!ok) { *result = 1; return; }

    // Verify: convert all to affine
    AffinePoint Ga, Ha, Pa, Qa;
    FieldElement tmp_x, tmp_y;
    jacobian_to_affine(&G, &tmp_x, &tmp_y);
    Ga.x = tmp_x; Ga.y = tmp_y;
    jacobian_to_affine(&H, &tmp_x, &tmp_y);
    Ha.x = tmp_x; Ha.y = tmp_y;
    jacobian_to_affine(&P, &tmp_x, &tmp_y);
    Pa.x = tmp_x; Pa.y = tmp_y;
    jacobian_to_affine(&Q, &tmp_x, &tmp_y);
    Qa.x = tmp_x; Qa.y = tmp_y;

    bool verified = dleq_verify_device(&proof, &Ga, &Ha, &Pa, &Qa);
    *result = verified ? 0 : 2;
}

// ---------- Main -------------------------------------------------------------

int main() {
    int* d_result;
    int h_result;
    cudaMalloc(&d_result, sizeof(int));

    auto run = [&](const char* name, auto kernel_fn) -> bool {
        h_result = -1;
        cudaMemset(d_result, 0xFF, sizeof(int));
        kernel_fn<<<1, 1>>>(d_result);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  [FAIL] %s  (launch error: %s)\n", name, cudaGetErrorString(err));
            return false;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  [FAIL] %s  (sync error: %s)\n", name, cudaGetErrorString(err));
            return false;
        }
        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_result == 0) {
            printf("  [OK]   %s\n", name);
            return true;
        } else {
            printf("  [FAIL] %s  (code=%d)\n", name, h_result);
            return false;
        }
    };

    printf("=== GPU CT Layer Smoke Test ===\n\n");
    fflush(stdout);

    int pass = 0, fail = 0;

    if (run("CT field/scalar ops", test_ct_ops_kernel)) pass++; else fail++;
    if (run("CT scalar basic ops", test_ct_scalar_basic_kernel)) pass++; else fail++;
    if (run("CT scalar inverse", test_ct_scalar_inv_kernel)) pass++; else fail++;
    if (run("CT ECDSA sign + verify", test_ct_ecdsa_kernel)) pass++; else fail++;
    if (run("CT Schnorr sign + verify", test_ct_schnorr_kernel)) pass++; else fail++;
    if (run("CT Schnorr keypair sign", test_ct_schnorr_keypair_kernel)) pass++; else fail++;
    if (run("CT vs Fast ECDSA parity", test_ct_fast_ecdsa_parity_kernel)) pass++; else fail++;
    if (run("CT ZK knowledge prove+verify", test_ct_zk_knowledge_kernel)) pass++; else fail++;
    if (run("CT ZK DLEQ prove+verify", test_ct_zk_dleq_kernel)) pass++; else fail++;

    printf("\n%d/%d passed\n", pass, pass + fail);

    cudaFree(d_result);
    return fail > 0 ? 1 : 0;
}
