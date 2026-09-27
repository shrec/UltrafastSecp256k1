// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
//! Test-only CUDA entry point, built directly by rustc's NVPTX backend.
#![cfg_attr(target_os = "cuda", no_std)]
#![cfg_attr(target_os = "cuda", feature(abi_ptx, stdarch_nvptx))]

#[cfg(target_os = "cuda")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { core::arch::nvptx::trap() }
}

/// Each input contains count*8 u64 words (a,b). Each output contains count*44
/// words: field add/sub/mul/square/inverse, scalar add/sub/mul/inverse, affine G*a.
/// Only synthetic/public values are permitted. Buffers must not overlap.
///
/// # Safety
/// The caller allocates the full input/output buffers and synchronizes before
/// reading output; the kernel must be launched with a one-dimensional grid.
#[cfg(target_os = "cuda")]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn ufsecp_arithmetic_probe(
    input: *const u64,
    output: *mut u64,
    count: u32,
) {
    use core::arch::nvptx;
    use ufsecp_core::{FieldElement as F, JacobianPoint as J, Scalar as S};
    let index = unsafe { nvptx::_block_idx_x() * nvptx::_block_dim_x() + nvptx::_thread_idx_x() };
    if index >= count {
        return;
    }
    let offset = index as usize * 8;
    let a = core::array::from_fn(|i| unsafe { *input.add(offset + i) });
    let b = core::array::from_fn(|i| unsafe { *input.add(offset + 4 + i) });
    let x = F::from_limbs_reduced(a);
    let y = F::from_limbs_reduced(b);
    let s = S::from_limbs_reduced(a);
    let t = S::from_limbs_reduced(b);
    let point = J::generator_mul(s).to_affine();
    let values = [
        x.add_mod(y).to_limbs(),
        x.sub_mod(y).to_limbs(),
        x.mul_mod(y).to_limbs(),
        x.square().to_limbs(),
        x.inverse().to_limbs(),
        s.add_mod(t).to_limbs(),
        s.sub_mod(t).to_limbs(),
        s.mul_mod(t).to_limbs(),
        s.inverse().to_limbs(),
        point.map_or([0; 4], |p| p.x.to_limbs()),
        point.map_or([0; 4], |p| p.y.to_limbs()),
    ];
    for (operation, limbs) in values.into_iter().enumerate() {
        for (limb, value) in limbs.into_iter().enumerate() {
            unsafe {
                *output.add(index as usize * 44 + operation * 4 + limb) = value;
            }
        }
    }
}
