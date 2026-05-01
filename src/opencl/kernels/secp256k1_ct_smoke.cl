// =============================================================================
// secp256k1_ct_smoke.cl -- Smoke-test kernels for GPU CT layer (OpenCL)
// =============================================================================
// Exports __kernel entry points that exercise the branchless CT primitives,
// CT field/scalar/point ops, and CT ECDSA + Schnorr signing.
// Each kernel writes 0 on success or a non-zero error bitmap on failure.
// Used exclusively by opencl_audit_runner.cpp (gpu_ct_smoke section).
//
// Dependency chain:
//   secp256k1_extended.cl   -> field, point, scalar, ECDSA/Schnorr fast-path
//   secp256k1_ct_ops.cl     -> value_barrier, masks, cmov, cswap
//   secp256k1_ct_field.cl   -> branchless field add/sub/reduce
//   secp256k1_ct_scalar.cl  -> branchless scalar add/sub/neg/normalize_low_s
//   secp256k1_ct_point.cl   -> CT point add, CT generator mul
//   secp256k1_ct_sign.cl    -> CT ECDSA sign, CT Schnorr sign
// =============================================================================

#include "secp256k1_extended.cl"
#include "secp256k1_ct_ops.cl"
#include "secp256k1_ct_field.cl"
#include "secp256k1_ct_scalar.cl"
#include "secp256k1_ct_point.cl"
#include "secp256k1_ct_sign.cl"

// ---------------------------------------------------------------------------
// Shared test message: SHA-256("test")
// ---------------------------------------------------------------------------
__constant uchar CT_SMOKE_MSG[32] = {
    0x9f, 0x86, 0xd0, 0x81, 0x88, 0x4c, 0x7d, 0x65,
    0x9a, 0x2f, 0xea, 0xa0, 0xc5, 0x5a, 0xd0, 0x15,
    0xa3, 0xbf, 0x4f, 0x1b, 0x2b, 0x0b, 0x82, 0x2c,
    0xd1, 0x5d, 0x6c, 0x15, 0xb0, 0xf0, 0x0a, 0x08
};

__constant uchar CT_SMOKE_AUX[32] = {0};

// ---------------------------------------------------------------------------
// Kernel 1: CT mask generators
//   Tests: ct_is_zero_mask, ct_is_nonzero_mask, ct_eq_mask, ct_bool_to_mask
//   Error bits: 1=zero_mask fail, 2=nonzero_mask fail, 4=eq_mask fail,
//               8=bool_true_mask fail, 16=bool_false_mask fail
// ---------------------------------------------------------------------------
__kernel void ct_smoke_masks(__global int* result) {
    if (get_global_id(0) != 0) return;
    int fail = 0;

    // ct_is_zero_mask(0) must be all-ones; ct_is_zero_mask(1) must be zero
    ulong zm0 = ct_is_zero_mask(0UL);
    ulong zm1 = ct_is_zero_mask(1UL);
    if (zm0 != 0xFFFFFFFFFFFFFFFFUL) fail |= 1;
    if (zm1 != 0UL)                  fail |= 1;

    // ct_is_nonzero_mask
    ulong nz0 = ct_is_nonzero_mask(0UL);
    ulong nz1 = ct_is_nonzero_mask(42UL);
    if (nz0 != 0UL)                   fail |= 2;
    if (nz1 != 0xFFFFFFFFFFFFFFFFUL)  fail |= 2;

    // ct_eq_mask
    ulong eq0 = ct_eq_mask(7UL, 7UL);
    ulong eq1 = ct_eq_mask(7UL, 8UL);
    if (eq0 != 0xFFFFFFFFFFFFFFFFUL)  fail |= 4;
    if (eq1 != 0UL)                   fail |= 4;

    // ct_bool_to_mask
    ulong bt = ct_bool_to_mask(1);
    ulong bf = ct_bool_to_mask(0);
    if (bt != 0xFFFFFFFFFFFFFFFFUL) fail |= 8;
    if (bf != 0UL)                  fail |= 16;

    *result = fail;
}

// ---------------------------------------------------------------------------
// Kernel 2: CT cmov256 / cswap256 on 256-bit values
//   Error bits: 1=cmov_select fail, 2=cmov_keep fail,
//               4=cswap_swap fail, 8=cswap_noswap fail
// ---------------------------------------------------------------------------
__kernel void ct_smoke_cmov(__global int* result) {
    if (get_global_id(0) != 0) return;
    int fail = 0;

    ulong all1 = 0xFFFFFFFFFFFFFFFFUL;
    ulong all0 = 0UL;

    ulong a[4] = {1UL, 2UL, 3UL, 4UL};
    ulong b[4] = {5UL, 6UL, 7UL, 8UL};

    // cmov: mask=all-ones -> select b into r
    ulong r[4] = {1UL, 2UL, 3UL, 4UL};
    ct_cmov256(r, b, all1);
    for (int i = 0; i < 4; ++i) if (r[i] != b[i]) { fail |= 1; break; }

    // cmov: mask=0 -> keep original a
    ulong r2[4] = {1UL, 2UL, 3UL, 4UL};
    ct_cmov256(r2, b, all0);
    for (int i = 0; i < 4; ++i) if (r2[i] != a[i]) { fail |= 2; break; }

    // cswap: mask=all-ones -> a and b swap
    ulong sa[4] = {1UL, 2UL, 3UL, 4UL};
    ulong sb[4] = {5UL, 6UL, 7UL, 8UL};
    ct_cswap256(sa, sb, all1);
    for (int i = 0; i < 4; ++i) {
        if (sa[i] != b[i] || sb[i] != a[i]) { fail |= 4; break; }
    }

    // cswap: mask=0 -> no swap
    ulong ca[4] = {1UL, 2UL, 3UL, 4UL};
    ulong cb[4] = {5UL, 6UL, 7UL, 8UL};
    ct_cswap256(ca, cb, all0);
    for (int i = 0; i < 4; ++i) {
        if (ca[i] != a[i] || cb[i] != b[i]) { fail |= 8; break; }
    }

    *result = fail;
}

// ---------------------------------------------------------------------------
// Kernel 3: CT ECDSA sign (privkey=1) + fast-path verify
//   privkey=1 -> pubkey=G (well-known generator point)
//   Error: 1=sign failed, 2=verify failed
// ---------------------------------------------------------------------------
__kernel void ct_smoke_ecdsa(__global int* result) {
    if (get_global_id(0) != 0) return;

    Scalar privkey;
    privkey.limbs[0] = 1UL;
    privkey.limbs[1] = 0UL;
    privkey.limbs[2] = 0UL;
    privkey.limbs[3] = 0UL;

    ECDSASignature sig;
    int ok = ct_ecdsa_sign_impl(CT_SMOKE_MSG, &privkey, &sig);
    if (!ok) { *result = 1; return; }

    // Derive pubkey = 1*G via CT generator mul
    CTJacobianPoint pub_jac;
    ct_generator_mul_impl(&privkey, &pub_jac);
    JacobianPoint pub = ct_point_to_jacobian(&pub_jac);

    int verified = ecdsa_verify_impl(CT_SMOKE_MSG, &pub, &sig);
    *result = verified ? 0 : 2;
}

// ---------------------------------------------------------------------------
// Kernel 4: CT Schnorr sign (privkey=1) + fast-path verify
//   Output: sig[64] (R.x || s bytes), verified via schnorr_verify_impl
//   Error: 1=sign failed, 2=verify failed
// ---------------------------------------------------------------------------
__kernel void ct_smoke_schnorr(__global int* result) {
    if (get_global_id(0) != 0) return;

    Scalar privkey;
    privkey.limbs[0] = 1UL;
    privkey.limbs[1] = 0UL;
    privkey.limbs[2] = 0UL;
    privkey.limbs[3] = 0UL;

    uchar sig64[64];
    int ok = ct_schnorr_sign_impl(&privkey, CT_SMOKE_MSG, CT_SMOKE_AUX, sig64);
    if (!ok) { *result = 1; return; }

    // Derive pubkey x-coordinate via CT (BIP-340 xonly pubkey)
    FieldElement pub_x;
    ct_schnorr_pubkey_impl(&privkey, &pub_x);

    // Serialize pub_x to bytes (big-endian)
    uchar pub_x_bytes[32];
    for (int i = 0; i < 4; ++i) {
        ulong v = pub_x.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            pub_x_bytes[i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }

    // Reconstruct SchnorrSignature from raw bytes
    SchnorrSignature ssig;
    for (int i = 0; i < 32; ++i) ssig.r[i] = sig64[i];
    scalar_from_bytes_impl(sig64 + 32, &ssig.s);

    int verified = schnorr_verify_impl(pub_x_bytes, CT_SMOKE_MSG, &ssig);
    *result = verified ? 0 : 2;
}
