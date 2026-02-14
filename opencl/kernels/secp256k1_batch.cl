// =============================================================================
// UltrafastSecp256k1 OpenCL Kernels - Batch Operations
// =============================================================================
// Optimized batch operations for high throughput
// =============================================================================

// Include point operations (which includes field)
// Note: In actual build, these are combined or included properly
// For standalone kernel build, include the definitions

#ifndef SECP256K1_BATCH_CL
#define SECP256K1_BATCH_CL

// These should be included from secp256k1_point.cl
// Assuming combined kernel build

// =============================================================================
// Batch Montgomery Inversion
// =============================================================================
// Computes inverses of n elements using only 1 inversion + 3(n-1) multiplications
// Algorithm: Batch inversion using Montgomery's trick
//
// For elements a[0], a[1], ..., a[n-1]:
// 1. Compute prefix products: p[i] = a[0] * a[1] * ... * a[i]
// 2. Compute q = p[n-1]^(-1) (single inversion)
// 3. Compute inverses backwards:
//    a[n-1]^(-1) = q * p[n-2]
//    a[n-2]^(-1) = q * a[n-1] * p[n-3]
//    etc.

__kernel void batch_inversion(
    __global const FieldElement* inputs,
    __global FieldElement* outputs,
    __global FieldElement* scratch,  // Size: 2 * count
    const uint count
) {
    // This kernel runs with a single work group for the inversion step
    // Each work item handles part of the prefix computation

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_size = get_local_size(0);

    // Phase 1: Compute prefix products (parallel-ish)
    // Each thread handles a chunk
    uint chunk_size = (count + group_size - 1) / group_size;
    uint start = lid * chunk_size;
    uint end = min(start + chunk_size, count);

    __global FieldElement* prefix = scratch;

    if (start < count) {
        // First element in chunk
        if (start == 0) {
            prefix[0] = inputs[0];
        } else {
            prefix[start] = inputs[start];
        }

        // Build prefix within chunk
        for (uint i = start + 1; i < end; i++) {
            field_mul_impl((FieldElement*)&prefix[i],
                          (const FieldElement*)&prefix[i-1],
                          (const FieldElement*)&inputs[i]);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Phase 2: Combine chunks (sequential part - only thread 0)
    if (lid == 0) {
        for (uint i = 1; i < group_size; i++) {
            uint chunk_start = i * chunk_size;
            if (chunk_start < count) {
                uint prev_end = chunk_start - 1;
                uint chunk_end = min(chunk_start + chunk_size, count);

                // Multiply all elements in this chunk by prefix of previous chunk
                for (uint j = chunk_start; j < chunk_end; j++) {
                    FieldElement temp;
                    field_mul_impl(&temp,
                                  (const FieldElement*)&prefix[j],
                                  (const FieldElement*)&prefix[prev_end]);
                    prefix[j] = temp;
                }
            }
        }

        // Phase 3: Compute single inversion of total product
        FieldElement total_inv;
        field_inv_impl(&total_inv, (const FieldElement*)&prefix[count - 1]);

        // Store in scratch space
        scratch[count] = total_inv;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Phase 4: Compute individual inverses (backwards)
    // Each thread handles its chunk backwards
    if (start < count) {
        FieldElement running_inv = scratch[count];

        // Multiply by prefixes of later chunks
        for (uint i = group_size - 1; i > lid; i--) {
            uint chunk_end = min((i + 1) * chunk_size, count);
            if (chunk_end > 0) {
                uint idx = chunk_end - 1;
                FieldElement temp;
                field_mul_impl(&temp, &running_inv, (const FieldElement*)&inputs[idx]);
                running_inv = temp;
            }
        }

        // Process our chunk backwards
        for (uint i = end; i > start; i--) {
            uint idx = i - 1;
            if (idx == 0) {
                outputs[0] = running_inv;
            } else {
                FieldElement temp;
                field_mul_impl(&temp, &running_inv, (const FieldElement*)&prefix[idx - 1]);
                outputs[idx] = temp;

                field_mul_impl(&temp, &running_inv, (const FieldElement*)&inputs[idx]);
                running_inv = temp;
            }
        }
    }
}

// =============================================================================
// Batch Jacobian to Affine Conversion
// =============================================================================
// Uses batch inversion for efficiency

__kernel void batch_jacobian_to_affine(
    __global const JacobianPoint* jacobians,
    __global AffinePoint* affines,
    __global FieldElement* z_scratch,   // Size: count (for Z values)
    __global FieldElement* inv_scratch, // Size: count (for Z^(-1) values)
    const uint count
) {
    uint gid = get_global_id(0);

    // Phase 1: Extract Z values and handle infinity
    if (gid < count) {
        if (point_is_infinity(&jacobians[gid])) {
            // For infinity, use Z = 1 (will be handled specially)
            z_scratch[gid].limbs[0] = 1;
            z_scratch[gid].limbs[1] = 0;
            z_scratch[gid].limbs[2] = 0;
            z_scratch[gid].limbs[3] = 0;
        } else {
            z_scratch[gid] = jacobians[gid].z;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // Phase 2: Batch inversion is done separately
    // (Assuming inv_scratch already contains Z^(-1) values)

    // Phase 3: Compute affine coordinates
    if (gid < count) {
        if (point_is_infinity(&jacobians[gid])) {
            // Point at infinity - set to (0, 0) or handle specially
            affines[gid].x.limbs[0] = 0; affines[gid].x.limbs[1] = 0;
            affines[gid].x.limbs[2] = 0; affines[gid].x.limbs[3] = 0;
            affines[gid].y.limbs[0] = 0; affines[gid].y.limbs[1] = 0;
            affines[gid].y.limbs[2] = 0; affines[gid].y.limbs[3] = 0;
        } else {
            FieldElement z_inv = inv_scratch[gid];
            FieldElement z_inv2, z_inv3;

            // z_inv2 = z_inv^2
            field_sqr_impl(&z_inv2, &z_inv);

            // z_inv3 = z_inv^3
            field_mul_impl(&z_inv3, &z_inv2, &z_inv);

            // x = X * z_inv2
            field_mul_impl((FieldElement*)&affines[gid].x,
                          (const FieldElement*)&jacobians[gid].x,
                          &z_inv2);

            // y = Y * z_inv3
            field_mul_impl((FieldElement*)&affines[gid].y,
                          (const FieldElement*)&jacobians[gid].y,
                          &z_inv3);
        }
    }
}

// =============================================================================
// Simple Batch Field Inversion (single work-item version for small batches)
// =============================================================================

__kernel void batch_field_inv_simple(
    __global const FieldElement* inputs,
    __global FieldElement* outputs,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    FieldElement r;
    field_inv_impl(&r, &inputs[gid]);
    outputs[gid] = r;
}

// =============================================================================
// Batch Scalar Multiplication (with precomputation hints)
// =============================================================================

// Window size for wNAF representation
#define WNAF_WINDOW 4
#define PRECOMP_SIZE (1 << (WNAF_WINDOW - 1))  // 8 points

__kernel void batch_scalar_mul_precomp(
    __global const Scalar* scalars,
    __global const AffinePoint* base_point,  // Single base point
    __global const AffinePoint* precomp,     // Precomputed multiples: G, 3G, 5G, ..., 15G
    __global JacobianPoint* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    Scalar k = scalars[gid];

    // Check for zero scalar
    if ((k.limbs[0] | k.limbs[1] | k.limbs[2] | k.limbs[3]) == 0) {
        point_set_infinity(&results[gid]);
        return;
    }

    JacobianPoint R;
    point_set_infinity(&R);

    // Simple double-and-add with precomputed table
    // Process 4 bits at a time
    for (int i = 63; i >= 0; i--) {
        // Double 4 times
        point_double_impl(&R, &R);
        point_double_impl(&R, &R);
        point_double_impl(&R, &R);
        point_double_impl(&R, &R);

        // Extract 4-bit window from each limb
        for (int limb = 3; limb >= 0; limb--) {
            uint window = (k.limbs[limb] >> (i * 4)) & 0xF;

            if (window != 0) {
                // Add precomputed point
                // For odd windows: use precomp[(window-1)/2]
                // For even windows: use doubled versions

                // Simplified: just use the base point lookup
                if (window < PRECOMP_SIZE * 2) {
                    uint idx = (window & 1) ? ((window - 1) >> 1) : (window >> 1);
                    if (idx < PRECOMP_SIZE) {
                        point_add_mixed_impl(&R, &R, &precomp[idx]);
                    }
                }
            }
        }
    }

    results[gid] = R;
}

#endif // SECP256K1_BATCH_CL

