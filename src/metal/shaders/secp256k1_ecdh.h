// =============================================================================
// secp256k1_ecdh.h -- ECDH Key Agreement for Metal
// =============================================================================
// Three variants:
//   - ecdh_compute_raw:   raw 32-byte x-coordinate
//   - ecdh_compute_xonly: SHA-256(x)
//   - ecdh_compute:       SHA-256(0x02|x) standard compressed hash
// =============================================================================

#ifndef SECP256K1_ECDH_H
#define SECP256K1_ECDH_H

// ECDH: raw x-coordinate (8x32 limbs -> big-endian bytes)
inline bool ecdh_compute_raw_metal(thread const Scalar256& private_key,
                                   thread const JacobianPoint& peer_pubkey,
                                   thread uchar* out) {
    JacobianPoint shared = scalar_mul(peer_pubkey, private_key);
    if (shared.infinity) return false;

    FieldElement z_inv = field_inv(shared.z);
    FieldElement z_inv2 = field_sqr(z_inv);
    FieldElement x_aff = field_mul(shared.x, z_inv2);

    for (int i = 0; i < 8; ++i) {
        uint v = x_aff.limbs[7 - i];
        out[i * 4 + 0] = (uchar)(v >> 24);
        out[i * 4 + 1] = (uchar)(v >> 16);
        out[i * 4 + 2] = (uchar)(v >> 8);
        out[i * 4 + 3] = (uchar)(v);
    }
    return true;
}

// ECDH: x-only hash SHA-256(x)
inline bool ecdh_compute_xonly_metal(thread const Scalar256& private_key,
                                     thread const JacobianPoint& peer_pubkey,
                                     thread uchar* out) {
    uchar x_bytes[32];
    if (!ecdh_compute_raw_metal(private_key, peer_pubkey, x_bytes))
        return false;

    SHA256Ctx ctx = sha256_init();
    ctx = sha256_update(ctx, x_bytes, 32);
    sha256_final(ctx, out);
    return true;
}

// ECDH: standard compressed hash SHA-256(prefix || x)
// prefix = 0x02 if Y is even, 0x03 if Y is odd — must match CPU/CUDA behaviour.
inline bool ecdh_compute_metal(thread const Scalar256& private_key,
                                thread const JacobianPoint& peer_pubkey,
                                thread uchar* out) {
    JacobianPoint shared = scalar_mul(peer_pubkey, private_key);
    if (shared.infinity) return false;

    // BUG-3 FIX: compute y_aff to derive the correct compressed-point prefix.
    // Previously hardcoded 0x02, producing wrong output for ~50% of key pairs
    // (those where the shared point has odd Y). limbs[0] is the least-significant
    // 32-bit word (8×32 LE layout); bit 0 is the Y parity.
    FieldElement z_inv = field_inv(shared.z);
    FieldElement z_inv2 = field_sqr(z_inv);
    FieldElement z_inv3 = field_mul(z_inv, z_inv2);
    FieldElement x_aff = field_mul(shared.x, z_inv2);
    FieldElement y_aff = field_mul(shared.y, z_inv3);

    uchar x_bytes[32];
    for (int i = 0; i < 8; ++i) {
        uint v = x_aff.limbs[7 - i];
        x_bytes[i * 4 + 0] = (uchar)(v >> 24);
        x_bytes[i * 4 + 1] = (uchar)(v >> 16);
        x_bytes[i * 4 + 2] = (uchar)(v >> 8);
        x_bytes[i * 4 + 3] = (uchar)(v);
    }

    uchar prefix = (y_aff.limbs[0] & 1u) ? 0x03u : 0x02u;
    SHA256Ctx ctx = sha256_init();
    ctx = sha256_update(ctx, &prefix, 1);
    ctx = sha256_update(ctx, x_bytes, 32);
    sha256_final(ctx, out);
    return true;
}

#endif // SECP256K1_ECDH_H
