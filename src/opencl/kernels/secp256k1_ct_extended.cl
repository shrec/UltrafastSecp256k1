// =============================================================================
// secp256k1_ct_extended.cl -- CT-backed drop-in for secp256k1_extended.cl
// =============================================================================
// Provides the same kernel entry points as secp256k1_extended.cl but routes
// secret-bearing operations (ecdsa_sign, schnorr_sign) through constant-time
// primitives from secp256k1_ct_sign.cl.
//
// Dependency chain (same as secp256k1_ct_smoke.cl):
//   secp256k1_extended.cl   -> field, point, scalar, verify kernels (non-CT)
//   secp256k1_ct_ops.cl     -> value_barrier, masks, cmov, cswap
//   secp256k1_ct_field.cl   -> branchless field ops
//   secp256k1_ct_scalar.cl  -> branchless scalar ops
//   secp256k1_ct_point.cl   -> CT generator mul
//   secp256k1_ct_sign.cl    -> ct_ecdsa_sign_impl, ct_schnorr_sign_impl
//
// The preprocessor guard SECP256K1_CT_SIGN_KERNELS suppresses the variable-time
// ecdsa_sign / schnorr_sign kernel bodies inside secp256k1_extended.cl, so
// this file can redefine them as CT-backed entry points without conflict.
// =============================================================================

#define SECP256K1_CT_SIGN_KERNELS

#include "secp256k1_extended.cl"
#include "secp256k1_ct_ops.cl"
#include "secp256k1_ct_field.cl"
#include "secp256k1_ct_scalar.cl"
#include "secp256k1_ct_point.cl"
#include "secp256k1_ct_sign.cl"

// ---------------------------------------------------------------------------
// CT-backed ecdsa_sign kernel (replaces variable-time version from extended.cl)
// Same interface — callers require no changes.
// ---------------------------------------------------------------------------
__kernel void ecdsa_sign(
    __global const uchar* msg_hashes,
    __global const Scalar* private_keys,
    __global ECDSASignature* signatures,
    __global int* success_flags,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_hashes[gid * 32 + i];

    Scalar priv = private_keys[gid];
    ECDSASignature sig;
    // CT path: branchless k*G and k^-1 — no secret-dependent branches
    int ok = ct_ecdsa_sign_impl(msg, &priv, &sig);
    if (ok) {
        signatures[gid] = sig;
    } else {
        // Fail-closed: zero output on failure
        ECDSASignature zero_sig;
        for (int i = 0; i < 4; i++) { zero_sig.r.limbs[i] = 0; zero_sig.s.limbs[i] = 0; }
        signatures[gid] = zero_sig;
    }
    success_flags[gid] = ok;
}

// ---------------------------------------------------------------------------
// CT-backed schnorr_sign kernel (replaces variable-time version from extended.cl)
// Same interface — callers require no changes.
// ---------------------------------------------------------------------------
__kernel void schnorr_sign(
    __global const uchar* messages,
    __global const Scalar* private_keys,
    __global const uchar* aux_rands,
    __global SchnorrSignature* signatures,
    __global int* success_flags,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar msg[32], aux[32];
    for (int i = 0; i < 32; i++) { msg[i] = messages[gid*32+i]; aux[i] = aux_rands[gid*32+i]; }

    Scalar priv = private_keys[gid];

    // ct_schnorr_sign_impl writes R.x (32 bytes) || s (32 bytes) into sig64
    uchar sig64[64];
    int ok = ct_schnorr_sign_impl(&priv, msg, aux, sig64);
    if (ok) {
        for (int i = 0; i < 32; i++) signatures[gid].r[i] = sig64[i];
        scalar_from_bytes_impl(sig64 + 32, &signatures[gid].s);
    } else {
        // Fail-closed: zero output on failure
        for (int i = 0; i < 32; i++) signatures[gid].r[i] = 0;
        for (int i = 0; i < 4; i++) signatures[gid].s.limbs[i] = 0;
    }
    success_flags[gid] = ok;
}
