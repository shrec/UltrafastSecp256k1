// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
// Rust port Copyright (c) 2026 CyberAshven
// Port of add_cc/sub_cc/mul64 and conditional reductions in secp256k1.cuh.

#[inline(always)]
pub(crate) fn add(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    #[cfg(target_arch = "nvptx64")]
    return crate::ptx::add(a, b);
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let mut carry = 0u128;
        let out = core::array::from_fn(|i| {
            carry += a[i] as u128 + b[i] as u128;
            let limb = carry as u64;
            carry >>= 64;
            limb
        });
        (out, carry as u64)
    }
}

#[inline(always)]
pub(crate) fn sub(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    #[cfg(target_arch = "nvptx64")]
    return crate::ptx::sub(a, b);
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let mut borrow = false;
        let out = core::array::from_fn(|i| {
            let (x, c1) = a[i].overflowing_sub(b[i]);
            let (y, c2) = x.overflowing_sub(u64::from(borrow));
            borrow = c1 | c2;
            y
        });
        (out, u64::from(borrow))
    }
}

#[inline(always)]
pub(crate) fn select(a: [u64; 4], b: [u64; 4], use_b: bool) -> [u64; 4] {
    let mask = 0u64.wrapping_sub(u64::from(use_b));
    core::array::from_fn(|i| (a[i] & !mask) | (b[i] & mask))
}

#[inline(always)]
pub(crate) fn reduce(a: [u64; 4], modulus: [u64; 4]) -> [u64; 4] {
    let (b, borrow) = sub(a, modulus);
    select(a, b, borrow == 0)
}

#[inline(always)]
pub(crate) fn add_mod(a: [u64; 4], b: [u64; 4], modulus: [u64; 4]) -> [u64; 4] {
    let (sum, carry) = add(a, b);
    let (reduced, borrow) = sub(sum, modulus);
    select(sum, reduced, (carry != 0) | (borrow == 0))
}

#[inline(always)]
pub(crate) fn sub_mod(a: [u64; 4], b: [u64; 4], modulus: [u64; 4]) -> [u64; 4] {
    let (difference, borrow) = sub(a, b);
    select(difference, add(difference, modulus).0, borrow != 0)
}

#[inline(always)]
pub(crate) fn mul64(a: u64, b: u64) -> (u64, u64) {
    #[cfg(target_arch = "nvptx64")]
    {
        let (lo, hi);
        // SAFETY: register-only multiplication, both outputs fully initialized.
        unsafe {
            core::arch::asm!(
                "mul.lo.u64 {lo}, {a}, {b};", "mul.hi.u64 {hi}, {a}, {b};",
                lo = out(reg64) lo, hi = out(reg64) hi,
                a = in(reg64) a, b = in(reg64) b, options(pure, nomem, nostack)
            )
        };
        (lo, hi)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let product = a as u128 * b as u128;
        (product as u64, (product >> 64) as u64)
    }
}

// Schoolbook rows from scalar_mul_mod_n. Each product plus existing limb and
// carry fits in 128 bits; unlike a whole Comba column it cannot overflow u128.
pub(crate) fn multiply<const A: usize, const B: usize, const N: usize>(
    a: [u64; A],
    b: [u64; B],
) -> [u64; N] {
    let mut out = [0u64; N];
    for i in 0..A {
        let mut carry = 0u64;
        for j in 0..B.min(N - i) {
            let (lo, hi) = mul64(a[i], b[j]);
            let (x, c1) = lo.overflowing_add(carry);
            let (y, c2) = x.overflowing_add(out[i + j]);
            out[i + j] = y;
            carry = hi.wrapping_add(u64::from(c1)).wrapping_add(u64::from(c2));
        }
        if i + B < N {
            out[i + B] = carry;
        }
    }
    out
}

pub(crate) fn from_bytes(bytes: [u8; 32]) -> [u64; 4] {
    core::array::from_fn(|i| u64::from_be_bytes(core::array::from_fn(|j| bytes[24 - i * 8 + j])))
}

pub(crate) fn to_bytes(limbs: [u64; 4]) -> [u8; 32] {
    core::array::from_fn(|i| limbs[3 - i / 8].to_be_bytes()[i % 8])
}
