// =============================================================================
// secp256k1_ecdh.cl -- ECDH Key Agreement for OpenCL
// =============================================================================
// Computes shared secret from private key + peer public key.
// Three variants:
//   - ecdh_compute_raw_impl:   raw 32-byte x-coordinate
//   - ecdh_compute_xonly_impl: SHA-256(x) x-only hash
//   - ecdh_compute_impl:       SHA-256(0x02|x) standard compressed hash
// =============================================================================

#ifndef SECP256K1_ECDH_CL
#define SECP256K1_ECDH_CL

// ---------------------------------------------------------------------------
// CT scalar multiplication for ECDH: fixed 256-iteration double-and-add.
// No secret-dependent branches; scalar bits accessed via branchless mask.
// ---------------------------------------------------------------------------
inline void ct_ecdh_scalar_mul(JacobianPoint* r, const JacobianPoint* pk,
                                const Scalar* sk)
{
    JacobianPoint R, T;
    point_set_infinity(&R);

    for (int i = 255; i >= 0; --i) {
        point_double_impl(&R, &R);
        point_add_impl(&T, &R, pk);

        // CT select: if bit i of sk is 1, R = T; else R = R.
        // mask = all-ones if bit=1, all-zeros if bit=0.
        int word = i >> 6;
        int bit  = (int)((sk->limbs[word] >> (i & 63)) & 1UL);
        ulong mask = -(ulong)bit;
        for (int j = 0; j < 4; ++j) {
            R.x.limbs[j] ^= mask & (R.x.limbs[j] ^ T.x.limbs[j]);
            R.y.limbs[j] ^= mask & (R.y.limbs[j] ^ T.y.limbs[j]);
            R.z.limbs[j] ^= mask & (R.z.limbs[j] ^ T.z.limbs[j]);
        }
        R.infinity = (int)(((uint)R.infinity & (uint)~(uint)mask) |
                           ((uint)T.infinity & (uint)mask));
    }
    *r = R;
}

// ECDH: raw x-coordinate of shared secret (CT path — secret scalar)
// shared_secret = x-coordinate of sk * PK (32 bytes, big-endian)
inline int ecdh_compute_raw_impl(const Scalar* private_key,
                                  const JacobianPoint* peer_pubkey,
                                  uchar out[32])
{
    JacobianPoint shared;
    ct_ecdh_scalar_mul(&shared, peer_pubkey, private_key);
    if (shared.infinity) return 0;

    FieldElement z_inv, z_inv2, x_aff;
    field_inv_impl(&z_inv, &shared.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&x_aff, &shared.x, &z_inv2);

    // Serialize x in big-endian
    for (int i = 0; i < 4; ++i) {
        ulong v = x_aff.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            out[i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }
    return 1;
}

// ECDH: x-only hash: SHA-256(x)
inline int ecdh_compute_xonly_impl(const Scalar* private_key,
                                    const JacobianPoint* peer_pubkey,
                                    uchar out[32])
{
    uchar x_bytes[32];
    if (!ecdh_compute_raw_impl(private_key, peer_pubkey, x_bytes))
        return 0;

    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, x_bytes, 32);
    sha256_final(&ctx, out);
    return 1;
}

// ECDH: standard compressed hash: SHA-256(0x02 || x) (CT path — secret scalar)
inline int ecdh_compute_impl(const Scalar* private_key,
                              const JacobianPoint* peer_pubkey,
                              uchar out[32])
{
    JacobianPoint shared;
    ct_ecdh_scalar_mul(&shared, peer_pubkey, private_key);
    if (shared.infinity) return 0;

    // BUG-3 FIX: compute y_aff to derive the correct compressed-point prefix.
    // Previously hardcoded 0x02, producing wrong output for ~50% of key pairs
    // (those where the shared point has odd Y). Must match CPU/CUDA behaviour.
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv_impl(&z_inv, &shared.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv, &z_inv2);
    field_mul_impl(&x_aff, &shared.x, &z_inv2);
    field_mul_impl(&y_aff, &shared.y, &z_inv3);

    uchar x_bytes[32];
    for (int i = 0; i < 4; ++i) {
        ulong v = x_aff.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            x_bytes[i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }

    // limbs[0] holds the least-significant 64-bit word; bit 0 is the Y parity.
    uchar prefix = (y_aff.limbs[0] & 1UL) ? 0x03 : 0x02;
    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, &prefix, 1);
    sha256_update(&ctx, x_bytes, 32);
    sha256_final(&ctx, out);
    return 1;
}

// Batch ECDH kernel
__kernel void ecdh_batch_kernel(
    __global const Scalar* private_keys,
    __global const JacobianPoint* peer_pubkeys,
    __global uchar* shared_secrets,
    __global uchar* results,
    uint count)
{
    uint idx = get_global_id(0);
    if (idx >= count) return;

    Scalar sk = private_keys[idx];
    JacobianPoint pk = peer_pubkeys[idx];
    uchar secret[32];
    results[idx] = (uchar)ecdh_compute_impl(&sk, &pk, secret);
    for (int i = 0; i < 32; ++i)
        shared_secrets[idx * 32 + i] = secret[i];
}

#endif // SECP256K1_ECDH_CL
