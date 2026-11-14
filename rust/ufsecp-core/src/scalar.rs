// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
// Rust port Copyright (c) 2026 CyberAshven
// Port of scalar_add/sub/negate/mul_mod_n/inverse in secp256k1.cuh.
use crate::limbs;

/// Canonical scalar modulo the group order, little-endian 4x64 layout.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar([u64; 4]);

impl Scalar {
    /// The secp256k1 group order n.
    pub const ORDER: [u64; 4] = [
        0xbfd25e8cd0364141,
        0xbaaedce6af48a03b,
        0xfffffffffffffffe,
        u64::MAX,
    ];
    /// Additive identity.
    pub const ZERO: Self = Self([0; 4]);
    /// Multiplicative identity.
    pub const ONE: Self = Self([1, 0, 0, 0]);
    /// Rejects noncanonical input at an external encoding boundary.
    pub fn from_limbs(limbs: [u64; 4]) -> Option<Self> {
        (crate::limbs::sub(limbs, Self::ORDER).1 != 0).then_some(Self(limbs))
    }
    #[inline(always)]
    /// Reduces any 256-bit integer modulo n.
    pub fn from_limbs_reduced(limbs: [u64; 4]) -> Self {
        Self(crate::limbs::reduce(limbs, Self::ORDER))
    }
    /// Returns the canonical little-endian limb encoding.
    pub const fn to_limbs(self) -> [u64; 4] {
        self.0
    }
    /// Reduces a 32-byte big-endian integer modulo n.
    pub fn from_be_bytes(bytes: [u8; 32]) -> Self {
        Self::from_limbs_reduced(limbs::from_bytes(bytes))
    }
    /// Returns the canonical 32-byte big-endian encoding.
    pub fn to_be_bytes(self) -> [u8; 32] {
        limbs::to_bytes(self.0)
    }
    #[inline(always)]
    /// Modular addition of canonical operands.
    pub fn add_mod(self, other: Self) -> Self {
        Self(limbs::add_mod(self.0, other.0, Self::ORDER))
    }
    /// Modular subtraction of canonical operands.
    pub fn sub_mod(self, other: Self) -> Self {
        Self(limbs::sub_mod(self.0, other.0, Self::ORDER))
    }
    /// Canonical additive inverse, including -0 = 0.
    pub fn negate(self) -> Self {
        Self::ZERO.sub_mod(self)
    }
    /// Upstream 4x4 schoolbook product and Barrett reduction, including both
    /// final subtraction passes. Not an audited constant-time secret API.
    pub fn mul_mod(self, other: Self) -> Self {
        const MU: [u64; 5] = [0x402da1732fc9bec0, 0x4551231950b75fc4, 1, 0, 1];
        let prod: [u64; 8] = limbs::multiply(self.0, other.0);
        let qmu: [u64; 9] = limbs::multiply([prod[4], prod[5], prod[6], prod[7]], MU);
        let qn: [u64; 5] = limbs::multiply([qmu[4], qmu[5], qmu[6], qmu[7]], Self::ORDER);
        let (mut r, borrow) = limbs::sub(
            [prod[0], prod[1], prod[2], prod[3]],
            [qn[0], qn[1], qn[2], qn[3]],
        );
        let mut r4 = prod[4].wrapping_sub(qn[4]).wrapping_sub(borrow);
        for _ in 0..2 {
            let (candidate, borrow) = limbs::sub(r, Self::ORDER);
            let need = (r4 != 0) | (borrow == 0);
            r = limbs::select(r, candidate, need);
            r4 = r4.wrapping_sub(borrow & 0u64.wrapping_sub(u64::from(need)));
        }
        Self(r)
    }
    /// Fermat inverse from scalar_inverse. inverse(0) is defined as 0.
    pub fn inverse(self) -> Self {
        let exponent = [
            0xbfd25e8cd036413f,
            Self::ORDER[1],
            Self::ORDER[2],
            Self::ORDER[3],
        ];
        let mut result = Self::ONE;
        for i in (0..256).rev() {
            result = result.mul_mod(result);
            if (exponent[i / 64] >> (i % 64)) & 1 != 0 {
                result = result.mul_mod(self);
            }
        }
        result
    }
}
