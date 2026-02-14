// =============================================================================
// UltrafastSecp256k1 OpenCL Kernels - Field Arithmetic
// =============================================================================
// secp256k1 field: F_p where p = 2^256 - 2^32 - 977
// Little-endian 256-bit integers using 4x64-bit limbs
// =============================================================================

// Field prime p = 2^256 - 0x1000003D1
// In 64-bit limbs (little-endian):
// p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}

// Constants
#define SECP256K1_P0 0xFFFFFFFEFFFFFC2FUL
#define SECP256K1_P1 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P2 0xFFFFFFFFFFFFFFFFUL
#define SECP256K1_P3 0xFFFFFFFFFFFFFFFFUL

// K = 2^32 + 977 = 0x1000003D1 (for fast reduction)
#define SECP256K1_K 0x1000003D1UL

// =============================================================================
// 64-bit Multiplication Helpers
// =============================================================================

// Multiply two 64-bit numbers, get 128-bit result as (hi, lo)
inline ulong2 mul64_full(ulong a, ulong b) {
    // Use OpenCL's mul_hi for high part
    ulong lo = a * b;
    ulong hi = mul_hi(a, b);
    return (ulong2)(lo, hi);
}

// Add with carry: result = a + b + carry_in, returns new carry
inline ulong add_with_carry(ulong a, ulong b, ulong carry_in, ulong* carry_out) {
    ulong sum = a + b;
    ulong c1 = (sum < a) ? 1UL : 0UL;
    sum += carry_in;
    ulong c2 = (sum < carry_in) ? 1UL : 0UL;
    *carry_out = c1 + c2;
    return sum;
}

// Subtract with borrow: result = a - b - borrow_in, returns new borrow
inline ulong sub_with_borrow(ulong a, ulong b, ulong borrow_in, ulong* borrow_out) {
    ulong diff = a - b;
    ulong b1 = (a < b) ? 1UL : 0UL;
    ulong temp = diff;
    diff -= borrow_in;
    ulong b2 = (temp < borrow_in) ? 1UL : 0UL;
    *borrow_out = b1 + b2;
    return diff;
}

// =============================================================================
// Field Element Type (256-bit)
// =============================================================================

typedef struct {
    ulong limbs[4];  // Little-endian: limbs[0] is LSB
} FieldElement;

// =============================================================================
// Field Reduction: r = a mod p
// Uses the fact that p = 2^256 - K where K = 0x1000003D1
// So 2^256 ≡ K (mod p), meaning we can reduce by replacing high bits with K*high
// =============================================================================

inline void field_reduce(FieldElement* r, const ulong* a8) {
    // a8 is 512-bit number (8 limbs), reduce to 256-bit mod p
    // Since p = 2^256 - K, we have: a mod p = a_low + K * a_high (mod p)

    ulong carry = 0;
    ulong temp[5];

    // First reduction: fold a[4..7] into a[0..3] using K
    // temp = a[0..3] + K * a[4..7]

    // Process each high limb
    ulong2 prod;

    // limb 0: a[0] + K * a[4]
    prod = mul64_full(SECP256K1_K, a8[4]);
    temp[0] = a8[0] + prod.x;
    carry = (temp[0] < a8[0]) ? 1UL : 0UL;
    carry += prod.y;

    // limb 1: a[1] + K * a[5] + carry
    prod = mul64_full(SECP256K1_K, a8[5]);
    temp[1] = a8[1] + carry;
    ulong c1 = (temp[1] < carry) ? 1UL : 0UL;
    temp[1] += prod.x;
    c1 += (temp[1] < prod.x) ? 1UL : 0UL;
    carry = c1 + prod.y;

    // limb 2: a[2] + K * a[6] + carry
    prod = mul64_full(SECP256K1_K, a8[6]);
    temp[2] = a8[2] + carry;
    c1 = (temp[2] < carry) ? 1UL : 0UL;
    temp[2] += prod.x;
    c1 += (temp[2] < prod.x) ? 1UL : 0UL;
    carry = c1 + prod.y;

    // limb 3: a[3] + K * a[7] + carry
    prod = mul64_full(SECP256K1_K, a8[7]);
    temp[3] = a8[3] + carry;
    c1 = (temp[3] < carry) ? 1UL : 0UL;
    temp[3] += prod.x;
    c1 += (temp[3] < prod.x) ? 1UL : 0UL;
    temp[4] = c1 + prod.y;

    // Second reduction: if temp[4] > 0, fold it in
    if (temp[4] != 0) {
        prod = mul64_full(SECP256K1_K, temp[4]);
        temp[0] += prod.x;
        carry = (temp[0] < prod.x) ? 1UL : 0UL;
        carry += prod.y;

        temp[1] += carry;
        carry = (temp[1] < carry) ? 1UL : 0UL;

        temp[2] += carry;
        carry = (temp[2] < carry) ? 1UL : 0UL;

        temp[3] += carry;
        // At this point result fits in 256 bits (plus possible 1-bit overflow)
    }

    // Final reduction: if result >= p, subtract p
    // Check if result >= p by comparing limbs
    ulong borrow = 0;
    ulong diff[4];

    diff[0] = sub_with_borrow(temp[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(temp[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(temp[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(temp[3], SECP256K1_P3, borrow, &borrow);

    // If no borrow, result >= p, use subtracted value
    // Otherwise, use original value
    // Branchless selection
    ulong mask = (borrow == 0) ? ~0UL : 0UL;

    r->limbs[0] = (diff[0] & mask) | (temp[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (temp[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (temp[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (temp[3] & ~mask);
}

// =============================================================================
// Field Addition: r = (a + b) mod p
// =============================================================================

inline void field_add_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
    ulong carry = 0;
    ulong sum[4];

    // Add with carry chain
    sum[0] = add_with_carry(a->limbs[0], b->limbs[0], 0, &carry);
    sum[1] = add_with_carry(a->limbs[1], b->limbs[1], carry, &carry);
    sum[2] = add_with_carry(a->limbs[2], b->limbs[2], carry, &carry);
    sum[3] = add_with_carry(a->limbs[3], b->limbs[3], carry, &carry);

    // Reduce: if carry or sum >= p, subtract p
    ulong borrow = 0;
    ulong diff[4];

    diff[0] = sub_with_borrow(sum[0], SECP256K1_P0, 0, &borrow);
    diff[1] = sub_with_borrow(sum[1], SECP256K1_P1, borrow, &borrow);
    diff[2] = sub_with_borrow(sum[2], SECP256K1_P2, borrow, &borrow);
    diff[3] = sub_with_borrow(sum[3], SECP256K1_P3, borrow, &borrow);

    // If carry from addition or no borrow from subtraction, use diff
    ulong use_diff = (carry != 0) | (borrow == 0);
    ulong mask = use_diff ? ~0UL : 0UL;

    r->limbs[0] = (diff[0] & mask) | (sum[0] & ~mask);
    r->limbs[1] = (diff[1] & mask) | (sum[1] & ~mask);
    r->limbs[2] = (diff[2] & mask) | (sum[2] & ~mask);
    r->limbs[3] = (diff[3] & mask) | (sum[3] & ~mask);
}

// =============================================================================
// Field Subtraction: r = (a - b) mod p
// =============================================================================

inline void field_sub_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
    ulong borrow = 0;
    ulong diff[4];

    // Subtract with borrow chain
    diff[0] = sub_with_borrow(a->limbs[0], b->limbs[0], 0, &borrow);
    diff[1] = sub_with_borrow(a->limbs[1], b->limbs[1], borrow, &borrow);
    diff[2] = sub_with_borrow(a->limbs[2], b->limbs[2], borrow, &borrow);
    diff[3] = sub_with_borrow(a->limbs[3], b->limbs[3], borrow, &borrow);

    // If borrow, add p (result was negative)
    ulong mask = borrow ? ~0UL : 0UL;

    ulong carry = 0;
    ulong adj[4];
    adj[0] = add_with_carry(diff[0], SECP256K1_P0 & mask, 0, &carry);
    adj[1] = add_with_carry(diff[1], SECP256K1_P1 & mask, carry, &carry);
    adj[2] = add_with_carry(diff[2], SECP256K1_P2 & mask, carry, &carry);
    adj[3] = add_with_carry(diff[3], SECP256K1_P3 & mask, carry, &carry);

    r->limbs[0] = adj[0];
    r->limbs[1] = adj[1];
    r->limbs[2] = adj[2];
    r->limbs[3] = adj[3];
}

// =============================================================================
// Field Multiplication: r = (a * b) mod p
// =============================================================================

inline void field_mul_impl(FieldElement* r, const FieldElement* a, const FieldElement* b) {
    ulong product[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Schoolbook multiplication with 64-bit limbs
    // product = a * b (512-bit result)
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong2 mul = mul64_full(a->limbs[i], b->limbs[j]);

            // Add to product[i+j] with carry
            ulong sum = product[i + j] + mul.x;
            ulong c1 = (sum < product[i + j]) ? 1UL : 0UL;
            sum += carry;
            ulong c2 = (sum < carry) ? 1UL : 0UL;
            product[i + j] = sum;
            carry = mul.y + c1 + c2;
        }
        product[i + 4] += carry;
    }

    // Reduce 512-bit product to 256-bit result mod p
    field_reduce(r, product);
}

// =============================================================================
// Field Squaring: r = a² mod p
// Optimized: only need upper triangle of multiplication
// =============================================================================

inline void field_sqr_impl(FieldElement* r, const FieldElement* a) {
    ulong product[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // Compute off-diagonal terms (doubled)
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < 4; j++) {
            ulong2 mul = mul64_full(a->limbs[i], a->limbs[j]);

            // Double the product (will be done after accumulation)
            ulong sum = product[i + j] + mul.x;
            ulong c1 = (sum < product[i + j]) ? 1UL : 0UL;
            sum += carry;
            ulong c2 = (sum < carry) ? 1UL : 0UL;
            product[i + j] = sum;
            carry = mul.y + c1 + c2;
        }
        product[i + 4] += carry;
    }

    // Double the off-diagonal terms
    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        ulong doubled = (product[i] << 1) | carry;
        carry = product[i] >> 63;
        product[i] = doubled;
    }

    // Add diagonal terms (a[i]²)
    carry = 0;
    for (int i = 0; i < 4; i++) {
        ulong2 sq = mul64_full(a->limbs[i], a->limbs[i]);

        ulong sum = product[2*i] + sq.x;
        ulong c1 = (sum < product[2*i]) ? 1UL : 0UL;
        sum += carry;
        ulong c2 = (sum < carry) ? 1UL : 0UL;
        product[2*i] = sum;

        sum = product[2*i + 1] + sq.y + c1 + c2;
        carry = (sum < product[2*i + 1]) ? 1UL : 0UL;
        product[2*i + 1] = sum;
    }

    // Reduce
    field_reduce(r, product);
}

// =============================================================================
// Field Negation: r = -a mod p = p - a
// =============================================================================

inline void field_neg_impl(FieldElement* r, const FieldElement* a) {
    // Check if a is zero
    ulong is_zero = ((a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0) ? 1UL : 0UL;

    ulong borrow = 0;
    r->limbs[0] = sub_with_borrow(SECP256K1_P0, a->limbs[0], 0, &borrow);
    r->limbs[1] = sub_with_borrow(SECP256K1_P1, a->limbs[1], borrow, &borrow);
    r->limbs[2] = sub_with_borrow(SECP256K1_P2, a->limbs[2], borrow, &borrow);
    r->limbs[3] = sub_with_borrow(SECP256K1_P3, a->limbs[3], borrow, &borrow);

    // If a was zero, result should be zero
    ulong mask = is_zero ? 0UL : ~0UL;
    r->limbs[0] &= mask;
    r->limbs[1] &= mask;
    r->limbs[2] &= mask;
    r->limbs[3] &= mask;
}

// =============================================================================
// Field Inversion: r = a^(-1) mod p
// Using Fermat's little theorem: a^(-1) = a^(p-2) mod p
// =============================================================================

inline void field_inv_impl(FieldElement* r, const FieldElement* a) {
    // Compute a^(p-2) using square-and-multiply
    // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D

    FieldElement base = *a;
    FieldElement result;
    result.limbs[0] = 1; result.limbs[1] = 0;
    result.limbs[2] = 0; result.limbs[3] = 0;

    // Binary exponentiation
    // p-2 in binary has a specific pattern we can optimize

    // First, handle the lower bits (least significant 32 bits of p-2)
    // 0xFFFFFC2D = 0b11111111111111111111110000101101

    // Process bit by bit for the full 256-bit exponent
    // This is a reference implementation; production would use addition chains

    const ulong exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,  // p-2 limb 0
        0xFFFFFFFFFFFFFFFFULL,  // p-2 limb 1
        0xFFFFFFFFFFFFFFFFULL,  // p-2 limb 2
        0xFFFFFFFFFFFFFFFFULL   // p-2 limb 3
    };

    for (int limb = 0; limb < 4; limb++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[limb] >> bit) & 1) {
                field_mul_impl(&result, &result, &base);
            }
            field_sqr_impl(&base, &base);
        }
    }

    *r = result;
}

// =============================================================================
// OpenCL Kernels
// =============================================================================

__kernel void field_add(
    __global const FieldElement* a,
    __global const FieldElement* b,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_add_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

__kernel void field_sub(
    __global const FieldElement* a,
    __global const FieldElement* b,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_sub_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

__kernel void field_mul(
    __global const FieldElement* a,
    __global const FieldElement* b,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement b_local = b[gid];
    FieldElement r;
    field_mul_impl(&r, &a_local, &b_local);
    result[gid] = r;
}

__kernel void field_sqr(
    __global const FieldElement* a,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement r;
    field_sqr_impl(&r, &a_local);
    result[gid] = r;
}

__kernel void field_inv(
    __global const FieldElement* a,
    __global FieldElement* result,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    FieldElement a_local = a[gid];
    FieldElement r;
    field_inv_impl(&r, &a_local);
    result[gid] = r;
}

