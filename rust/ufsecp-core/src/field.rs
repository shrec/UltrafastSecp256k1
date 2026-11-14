// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
// Rust port Copyright (c) 2026 CyberAshven
// Source: src/cuda/include/secp256k1.cuh and src/cpu/src/field.cpp.
use crate::limbs;

/// Canonical element modulo p = 2^256 - 2^32 - 977, little-endian 4x64 layout.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldElement([u64; 4]);

impl FieldElement {
    /// The secp256k1 base field modulus p.
    pub const MODULUS: [u64; 4] = [0xfffffffefffffc2f, u64::MAX, u64::MAX, u64::MAX];
    /// Additive identity.
    pub const ZERO: Self = Self([0; 4]);
    /// Multiplicative identity.
    pub const ONE: Self = Self([1, 0, 0, 0]);
    pub(crate) const fn generator_x() -> Self {
        Self([
            0x59f2815b16f81798,
            0x029bfcdb2dce28d9,
            0x55a06295ce870b07,
            0x79be667ef9dcbbac,
        ])
    }
    pub(crate) const fn generator_y() -> Self {
        Self([
            0x9c47d08ffb10d4b8,
            0xfd17b448a6855419,
            0x5da4fbfc0e1108a8,
            0x483ada7726a3c465,
        ])
    }

    /// Rejects noncanonical input; useful at an external encoding boundary.
    pub fn from_limbs(limbs: [u64; 4]) -> Option<Self> {
        (crate::limbs::sub(limbs, Self::MODULUS).1 != 0).then_some(Self(limbs))
    }
    /// Reduces any 256-bit integer modulo p (one subtraction is sufficient).
    #[inline(always)]
    pub fn from_limbs_reduced(limbs: [u64; 4]) -> Self {
        Self(crate::limbs::reduce(limbs, Self::MODULUS))
    }
    /// Returns the canonical little-endian limb encoding.
    pub const fn to_limbs(self) -> [u64; 4] {
        self.0
    }
    /// Rejects a noncanonical 32-byte big-endian field encoding.
    pub fn from_be_bytes(bytes: [u8; 32]) -> Option<Self> {
        Self::from_limbs(limbs::from_bytes(bytes))
    }
    /// Returns the canonical 32-byte big-endian encoding.
    pub fn to_be_bytes(self) -> [u8; 32] {
        limbs::to_bytes(self.0)
    }
    #[inline(always)]
    /// Tests the additive identity.
    pub fn is_zero(self) -> bool {
        self.0[0] | self.0[1] | self.0[2] | self.0[3] == 0
    }
    #[inline(always)]
    /// Modular addition of canonical operands.
    pub fn add_mod(self, other: Self) -> Self {
        Self(limbs::add_mod(self.0, other.0, Self::MODULUS))
    }
    #[inline(always)]
    /// Modular subtraction of canonical operands.
    pub fn sub_mod(self, other: Self) -> Self {
        Self(limbs::sub_mod(self.0, other.0, Self::MODULUS))
    }
    /// Canonical negation, including -0 = 0.
    pub fn negate(self) -> Self {
        Self::ZERO.sub_mod(self)
    }

    #[inline(always)]
    /// Modular multiplication; portable CPU or hybrid Comba GPU backend.
    pub fn mul_mod(self, other: Self) -> Self {
        #[cfg(target_arch = "nvptx64")]
        return Self(crate::ptx::field_mul(self.0, other.0));
        #[cfg(not(target_arch = "nvptx64"))]
        Self(reduce_wide(limbs::multiply(self.0, other.0)))
    }
    #[inline(always)]
    /// Modular square; GPU uses the symmetric Comba product.
    pub fn square(self) -> Self {
        #[cfg(target_arch = "nvptx64")]
        return Self(crate::ptx::field_square(self.0));
        #[cfg(not(target_arch = "nvptx64"))]
        self.mul_mod(self)
    }
    #[inline(always)]
    fn square_n(mut self, count: usize) -> Self {
        for _ in 0..count {
            self = self.square();
        }
        self
    }

    /// Fermat chain from field_inv_fermat_chain_impl. inverse(0) is defined as 0.
    #[inline(never)]
    pub fn inverse(self) -> Self {
        let x2 = self.square().mul_mod(self);
        let x3 = x2.square().mul_mod(self);
        let x6 = x3.square_n(3).mul_mod(x3);
        let x9 = x6.square_n(3).mul_mod(x3);
        let x11 = x9.square_n(2).mul_mod(x2);
        let x22 = x11.square_n(11).mul_mod(x11);
        let x44 = x22.square_n(22).mul_mod(x22);
        let x88 = x44.square_n(44).mul_mod(x44);
        let x176 = x88.square_n(88).mul_mod(x88);
        let x223 = x176.square_n(44).mul_mod(x44).square_n(3).mul_mod(x3);
        let t = x223.square_n(23).mul_mod(x22).square_n(4);
        let t = t.square().mul_mod(self).square_n(2).mul_mod(self);
        t.square().mul_mod(self).square_n(2).mul_mod(self)
    }

    /// Upstream field_sqrt exponent (p+1)/4; checked before returning a root.
    pub fn sqrt(self) -> Option<Self> {
        let root = self.sqrt_candidate();
        (root.square() == self).then_some(root)
    }
    #[inline(never)]
    pub fn is_square(self) -> bool {
        self.sqrt().is_some()
    }
    fn sqrt_candidate(self) -> Self {
        let x2 = self.square().mul_mod(self);
        let x3 = x2.square().mul_mod(self);
        let x6 = x3.square_n(3).mul_mod(x3);
        let x11 = x6.square_n(3).mul_mod(x3).square_n(2).mul_mod(x2);
        let x22 = x11.square_n(11).mul_mod(x11);
        let x44 = x22.square_n(22).mul_mod(x22);
        let x88 = x44.square_n(44).mul_mod(x44);
        let t = x88.square_n(88).mul_mod(x88).square_n(44).mul_mod(x44);
        let t = t.square_n(2).mul_mod(x2).square().mul_mod(self);
        t.square_n(23)
            .mul_mod(x22)
            .square_n(8)
            .mul_mod(x2.square_n(2))
    }
}

// Portable CPU fold from field.cpp::reduce. Retain the full five-limb carry
// cascade: the high-magnitude boundary tests exercise every overflow position.
#[cfg(not(target_arch = "nvptx64"))]
fn reduce_wide(t: [u64; 8]) -> [u64; 4] {
    let mut r = [t[0], t[1], t[2], t[3], 0];
    for i in 0..4 {
        let high = t[i + 4];
        let product = high as u128 * 977;
        let acc = r[i] as u128 + product as u64 as u128 + (high << 32) as u128;
        r[i] = acc as u64;
        let acc = r[i + 1] as u128 + (product >> 64) + (high >> 32) as u128 + (acc >> 64);
        r[i + 1] = acc as u64;
        let mut carry = acc >> 64;
        for limb in &mut r[i + 2..] {
            let acc = *limb as u128 + carry;
            *limb = acc as u64;
            carry = acc >> 64;
        }
    }
    for _ in 0..2 {
        let term = r[4] as u128 * 0x1000003d1;
        r[4] = 0;
        let acc = r[0] as u128 + term as u64 as u128;
        r[0] = acc as u64;
        let acc = r[1] as u128 + (term >> 64) + (acc >> 64);
        r[1] = acc as u64;
        let mut carry = acc >> 64;
        for limb in &mut r[2..] {
            let acc = *limb as u128 + carry;
            *limb = acc as u64;
            carry = acc >> 64;
        }
    }
    limbs::reduce([r[0], r[1], r[2], r[3]], FieldElement::MODULUS)
}
