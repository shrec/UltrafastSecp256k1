// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
// Rust port Copyright (c) 2026 CyberAshven
//! Experimental Rust port of UltrafastSecp256k1's public arithmetic.
//! No FFI, allocation, runtime dependency, or signing API. Point operations are
//! variable-time: never use this backend for secret keys or signing nonces.
//! See README.md for the exact source revision and port boundaries.
#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(asm_experimental_arch))]
#![deny(unsafe_op_in_unsafe_fn)]

mod field;
mod limbs;
mod point;
#[cfg(target_arch = "nvptx64")]
mod ptx;
mod scalar;

pub use field::FieldElement;
pub use point::{AffinePoint, JacobianPoint};
pub use scalar::Scalar;
