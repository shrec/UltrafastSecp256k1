// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
// Rust port Copyright (c) 2026 CyberAshven
// Direct port of jacobian_double / jacobian_add_mixed in secp256k1.cuh.
use crate::{FieldElement as F, Scalar};

/// Finite affine point. Construct with new() to validate untrusted coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffinePoint {
    pub x: F,
    pub y: F,
}

impl AffinePoint {
    /// Standard secp256k1 base point G.
    pub const GENERATOR: Self = Self {
        x: F::generator_x(),
        y: F::generator_y(),
    };
    /// Checks y^2 = x^3 + 7. Infinity has no finite affine encoding.
    pub fn new(x: F, y: F) -> Option<Self> {
        let seven = F::from_limbs_reduced([7, 0, 0, 0]);
        (y.square() == x.square().mul_mod(x).add_mod(seven)).then_some(Self { x, y })
    }
}

/// Internal Jacobian representation, with upstream's explicit infinity flag.
/// Coordinates must describe a validated point; all operations are variable-time.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct JacobianPoint {
    pub x: F,
    pub y: F,
    pub z: F,
    pub infinity: bool,
}

impl JacobianPoint {
    /// Canonical infinity with zero coordinates.
    pub const INFINITY: Self = Self {
        x: F::ZERO,
        y: F::ZERO,
        z: F::ZERO,
        infinity: true,
    };
    /// Lifts a validated finite point with z = 1.
    pub fn from_affine(point: AffinePoint) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: F::ONE,
            infinity: false,
        }
    }
    /// Jacobian doubling; infinity and y = 0 produce canonical infinity.
    pub fn double(self) -> Self {
        if self.infinity || self.y.is_zero() {
            return Self::INFINITY;
        }
        let yy = self.y.square();
        let s = self.x.mul_mod(yy);
        let s = s.add_mod(s);
        let s = s.add_mod(s);
        let xx = self.x.square();
        let m = xx.add_mod(xx).add_mod(xx);
        let x = m.square().sub_mod(s.add_mod(s));
        let yyyy = yy.square();
        let yyyy = yyyy.add_mod(yyyy);
        let yyyy = yyyy.add_mod(yyyy);
        let yyyy = yyyy.add_mod(yyyy);
        let yz = self.y.mul_mod(self.z);
        Self {
            x,
            y: m.mul_mod(s.sub_mod(x)).sub_mod(yyyy),
            z: yz.add_mod(yz),
            infinity: false,
        }
    }
    /// Upstream add-2007-bl mixed addition, including doubling and inverse pairs.
    #[inline(never)]
    pub fn add_mixed(self, q: AffinePoint) -> Self {
        if self.infinity {
            return Self::from_affine(q);
        }
        let z1z1 = self.z.square();
        let u2 = q.x.mul_mod(z1z1);
        let s2 = q.y.mul_mod(self.z).mul_mod(z1z1);
        let h = u2.sub_mod(self.x);
        let delta = s2.sub_mod(self.y);
        if h.is_zero() {
            return if delta.is_zero() {
                self.double()
            } else {
                Self::INFINITY
            };
        }
        let hh = h.square();
        let i = hh.add_mod(hh);
        let i = i.add_mod(i);
        let j = h.mul_mod(i);
        let rr = delta.add_mod(delta);
        let v = self.x.mul_mod(i);
        let x = rr.square().sub_mod(j).sub_mod(v.add_mod(v));
        let yj = self.y.mul_mod(j);
        Self {
            x,
            y: rr.mul_mod(v.sub_mod(x)).sub_mod(yj.add_mod(yj)),
            z: self.z.add_mod(h).square().sub_mod(z1z1).sub_mod(hh),
            infinity: false,
        }
    }
    /// Normalizes a finite point, returning None for infinity or z = 0.
    pub fn to_affine(self) -> Option<AffinePoint> {
        if self.infinity || self.z.is_zero() {
            return None;
        }
        let zi = self.z.inverse();
        let zi2 = zi.square();
        Some(AffinePoint {
            x: self.x.mul_mod(zi2),
            y: self.y.mul_mod(zi2).mul_mod(zi),
        })
    }
    /// Basic double-and-add for public scalars. Not the upstream GLV/CT engines.
    pub fn generator_mul(scalar: Scalar) -> Self {
        let mut result = Self::INFINITY;
        for byte in scalar.to_be_bytes() {
            for bit in (0..8).rev() {
                result = result.double();
                if byte & (1 << bit) != 0 {
                    result = result.add_mixed(AffinePoint::GENERATOR);
                }
            }
        }
        result
    }
}
