// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Vano Chkheidze
// Rust port Copyright (c) 2026 CyberAshven
// Rust transcription of secp256k1_32_hybrid_final.cuh's Comba products and
// four reduction phases. PTX carry chains remain inside single asm blocks.
// All unsafe blocks below only touch declared registers, initialize outputs,
// access no memory, and neither adjust the stack nor depend on external flags.
use core::arch::asm;

#[inline(always)]
pub(super) fn add(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    let (r0, r1, r2, r3, flag): (u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            "add.cc.u64 {r0}, {a0}, {b0};",
            "addc.cc.u64 {r1}, {a1}, {b1};",
            "addc.cc.u64 {r2}, {a2}, {b2};",
            "addc.cc.u64 {r3}, {a3}, {b3};",
            "addc.u64 {flag}, 0, 0;",
            r0 = out(reg64) r0,
            r1 = out(reg64) r1,
            r2 = out(reg64) r2,
            r3 = out(reg64) r3,
            flag = out(reg64) flag,
            a0 = in(reg64) a[0],
            a1 = in(reg64) a[1],
            a2 = in(reg64) a[2],
            a3 = in(reg64) a[3],
            b0 = in(reg64) b[0],
            b1 = in(reg64) b[1],
            b2 = in(reg64) b[2],
            b3 = in(reg64) b[3],
            options(pure, nomem, nostack)
        )
    };
    ([r0, r1, r2, r3], flag)
}

#[inline(always)]
pub(super) fn sub(a: [u64; 4], b: [u64; 4]) -> ([u64; 4], u64) {
    let (r0, r1, r2, r3, flag): (u64, u64, u64, u64, u64);
    unsafe {
        asm!(
            "sub.cc.u64 {r0}, {a0}, {b0};",
            "subc.cc.u64 {r1}, {a1}, {b1};",
            "subc.cc.u64 {r2}, {a2}, {b2};",
            "subc.cc.u64 {r3}, {a3}, {b3};",
            "subc.u64 {flag}, 0, 0;",
            r0 = out(reg64) r0,
            r1 = out(reg64) r1,
            r2 = out(reg64) r2,
            r3 = out(reg64) r3,
            flag = out(reg64) flag,
            a0 = in(reg64) a[0],
            a1 = in(reg64) a[1],
            a2 = in(reg64) a[2],
            a3 = in(reg64) a[3],
            b0 = in(reg64) b[0],
            b1 = in(reg64) b[1],
            b2 = in(reg64) b[2],
            b3 = in(reg64) b[3],
            options(pure, nomem, nostack)
        )
    };
    ([r0, r1, r2, r3], flag & 1)
}

#[inline(always)]
fn accumulate(mut r: [u32; 3], a: u32, b: u32) -> [u32; 3] {
    unsafe {
        asm!(
            "mad.lo.cc.u32 {r0}, {a}, {b}, {r0};",
            "madc.hi.cc.u32 {r1}, {a}, {b}, {r1};",
            "addc.u32 {r2}, {r2}, 0;",
            r0 = inout(reg32) r[0], r1 = inout(reg32) r[1], r2 = inout(reg32) r[2],
            a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)
        )
    };
    r
}

#[inline(always)]
fn accumulate_twice(mut r: [u32; 3], a: u32, b: u32) -> [u32; 3] {
    unsafe {
        asm!(
            "{{ .reg .u32 lo, hi;",
            "mul.lo.u32 lo, {a}, {b};", "mul.hi.u32 hi, {a}, {b};",
            "add.cc.u32 {r0}, {r0}, lo;", "addc.cc.u32 {r1}, {r1}, hi;", "addc.u32 {r2}, {r2}, 0;",
            "add.cc.u32 {r0}, {r0}, lo;", "addc.cc.u32 {r1}, {r1}, hi;", "addc.u32 {r2}, {r2}, 0;",
            "}}",
            r0 = inout(reg32) r[0], r1 = inout(reg32) r[1], r2 = inout(reg32) r[2],
            a = in(reg32) a, b = in(reg32) b, options(pure, nomem, nostack)
        )
    };
    r
}

#[inline(always)]
fn words(a: [u64; 4]) -> [u32; 8] {
    core::array::from_fn(|i| (a[i / 2] >> ((i % 2) * 32)) as u32)
}

#[inline(always)]
pub(super) fn field_mul(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let a = words(a);
    let b = words(b);
    let mut r = [0u32; 3];
    let mut t = [0u32; 16];
    r = accumulate(r, a[0], b[0]);
    t[0] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[1]);
    r = accumulate(r, a[1], b[0]);
    t[1] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[2]);
    r = accumulate(r, a[1], b[1]);
    r = accumulate(r, a[2], b[0]);
    t[2] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[3]);
    r = accumulate(r, a[1], b[2]);
    r = accumulate(r, a[2], b[1]);
    r = accumulate(r, a[3], b[0]);
    t[3] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[4]);
    r = accumulate(r, a[1], b[3]);
    r = accumulate(r, a[2], b[2]);
    r = accumulate(r, a[3], b[1]);
    r = accumulate(r, a[4], b[0]);
    t[4] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[5]);
    r = accumulate(r, a[1], b[4]);
    r = accumulate(r, a[2], b[3]);
    r = accumulate(r, a[3], b[2]);
    r = accumulate(r, a[4], b[1]);
    r = accumulate(r, a[5], b[0]);
    t[5] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[6]);
    r = accumulate(r, a[1], b[5]);
    r = accumulate(r, a[2], b[4]);
    r = accumulate(r, a[3], b[3]);
    r = accumulate(r, a[4], b[2]);
    r = accumulate(r, a[5], b[1]);
    r = accumulate(r, a[6], b[0]);
    t[6] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[0], b[7]);
    r = accumulate(r, a[1], b[6]);
    r = accumulate(r, a[2], b[5]);
    r = accumulate(r, a[3], b[4]);
    r = accumulate(r, a[4], b[3]);
    r = accumulate(r, a[5], b[2]);
    r = accumulate(r, a[6], b[1]);
    r = accumulate(r, a[7], b[0]);
    t[7] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[1], b[7]);
    r = accumulate(r, a[2], b[6]);
    r = accumulate(r, a[3], b[5]);
    r = accumulate(r, a[4], b[4]);
    r = accumulate(r, a[5], b[3]);
    r = accumulate(r, a[6], b[2]);
    r = accumulate(r, a[7], b[1]);
    t[8] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[2], b[7]);
    r = accumulate(r, a[3], b[6]);
    r = accumulate(r, a[4], b[5]);
    r = accumulate(r, a[5], b[4]);
    r = accumulate(r, a[6], b[3]);
    r = accumulate(r, a[7], b[2]);
    t[9] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[3], b[7]);
    r = accumulate(r, a[4], b[6]);
    r = accumulate(r, a[5], b[5]);
    r = accumulate(r, a[6], b[4]);
    r = accumulate(r, a[7], b[3]);
    t[10] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[4], b[7]);
    r = accumulate(r, a[5], b[6]);
    r = accumulate(r, a[6], b[5]);
    r = accumulate(r, a[7], b[4]);
    t[11] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[5], b[7]);
    r = accumulate(r, a[6], b[6]);
    r = accumulate(r, a[7], b[5]);
    t[12] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[6], b[7]);
    r = accumulate(r, a[7], b[6]);
    t[13] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[7], b[7]);
    t[14] = r[0];
    t[15] = r[1];
    reduce(t)
}

#[inline(always)]
pub(super) fn field_square(a: [u64; 4]) -> [u64; 4] {
    let a = words(a);
    let mut r = [0u32; 3];
    let mut t = [0u32; 16];
    r = accumulate(r, a[0], a[0]);
    t[0] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[1]);
    t[1] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[2]);
    r = accumulate(r, a[1], a[1]);
    t[2] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[3]);
    r = accumulate_twice(r, a[1], a[2]);
    t[3] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[4]);
    r = accumulate_twice(r, a[1], a[3]);
    r = accumulate(r, a[2], a[2]);
    t[4] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[5]);
    r = accumulate_twice(r, a[1], a[4]);
    r = accumulate_twice(r, a[2], a[3]);
    t[5] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[6]);
    r = accumulate_twice(r, a[1], a[5]);
    r = accumulate_twice(r, a[2], a[4]);
    r = accumulate(r, a[3], a[3]);
    t[6] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[0], a[7]);
    r = accumulate_twice(r, a[1], a[6]);
    r = accumulate_twice(r, a[2], a[5]);
    r = accumulate_twice(r, a[3], a[4]);
    t[7] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[1], a[7]);
    r = accumulate_twice(r, a[2], a[6]);
    r = accumulate_twice(r, a[3], a[5]);
    r = accumulate(r, a[4], a[4]);
    t[8] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[2], a[7]);
    r = accumulate_twice(r, a[3], a[6]);
    r = accumulate_twice(r, a[4], a[5]);
    t[9] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[3], a[7]);
    r = accumulate_twice(r, a[4], a[6]);
    r = accumulate(r, a[5], a[5]);
    t[10] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[4], a[7]);
    r = accumulate_twice(r, a[5], a[6]);
    t[11] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[5], a[7]);
    r = accumulate(r, a[6], a[6]);
    t[12] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate_twice(r, a[6], a[7]);
    t[13] = r[0];
    r = [r[1], r[2], 0];
    r = accumulate(r, a[7], a[7]);
    t[14] = r[0];
    t[15] = r[1];
    reduce(t)
}

#[inline(always)]
fn reduce(t: [u32; 16]) -> [u64; 4] {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    unsafe {
        asm!(
            "{{ .reg .u32 a<10>, low<8>, carry, elo, ehi, ec, plo, phi, m1, ek1, ekc, ek2;",
            ".reg .u64 r<4>, eklo, ekhi, c, cmask, cfold, s<4>, borrow, mask, tmp;",
            "mul.lo.u32 a0, {t8}, 977;",
            "mul.hi.u32 a1, {t8}, 977;",
            "mad.lo.cc.u32 a1, {t9}, 977, a1;",
            "madc.hi.u32 a2, {t9}, 977, 0;",
            "mad.lo.cc.u32 a2, {t10}, 977, a2;",
            "madc.hi.u32 a3, {t10}, 977, 0;",
            "mad.lo.cc.u32 a3, {t11}, 977, a3;",
            "madc.hi.u32 a4, {t11}, 977, 0;",
            "mad.lo.cc.u32 a4, {t12}, 977, a4;",
            "madc.hi.u32 a5, {t12}, 977, 0;",
            "mad.lo.cc.u32 a5, {t13}, 977, a5;",
            "madc.hi.u32 a6, {t13}, 977, 0;",
            "mad.lo.cc.u32 a6, {t14}, 977, a6;",
            "madc.hi.u32 a7, {t14}, 977, 0;",
            "mad.lo.cc.u32 a7, {t15}, 977, a7;",
            "madc.hi.u32 a8, {t15}, 977, 0;",
            "add.cc.u32 a1, a1, {t8};",
            "addc.cc.u32 a2, a2, {t9};",
            "addc.cc.u32 a3, a3, {t10};",
            "addc.cc.u32 a4, a4, {t11};",
            "addc.cc.u32 a5, a5, {t12};",
            "addc.cc.u32 a6, a6, {t13};",
            "addc.cc.u32 a7, a7, {t14};",
            "addc.cc.u32 a8, a8, {t15};",
            "addc.u32 a9, 0, 0;",
            "add.cc.u32 low0, {t0}, a0;",
            "addc.cc.u32 low1, {t1}, a1;",
            "addc.cc.u32 low2, {t2}, a2;",
            "addc.cc.u32 low3, {t3}, a3;",
            "addc.cc.u32 low4, {t4}, a4;",
            "addc.cc.u32 low5, {t5}, a5;",
            "addc.cc.u32 low6, {t6}, a6;",
            "addc.cc.u32 low7, {t7}, a7;",
            "addc.u32 carry, 0, 0;",
            "add.cc.u32 elo, a8, carry;",
            "addc.u32 ec, 0, 0;",
            "add.u32 ehi, a9, ec;",
            "mul.lo.u32 plo, elo, 977;",
            "mul.hi.u32 phi, elo, 977;",
            "mad.lo.u32 m1, ehi, 977, phi;",
            "add.cc.u32 ek1, m1, elo;",
            "addc.u32 ekc, 0, 0;",
            "add.u32 ek2, ehi, ekc;",
            "mov.b64 r0, {{low0, low1}};",
            "mov.b64 r1, {{low2, low3}};",
            "mov.b64 r2, {{low4, low5}};",
            "mov.b64 r3, {{low6, low7}};",
            "mov.b64 eklo, {{plo, ek1}};",
            "cvt.u64.u32 ekhi, ek2;",
            "add.cc.u64 r0, r0, eklo;",
            "addc.cc.u64 r1, r1, ekhi;",
            "addc.cc.u64 r2, r2, 0;",
            "addc.cc.u64 r3, r3, 0;",
            "addc.u64 c, 0, 0;",
            "neg.s64 cmask, c;",
            "and.b64 cfold, cmask, 4294968273;",
            "add.cc.u64 r0, r0, cfold;",
            "addc.cc.u64 r1, r1, 0;",
            "addc.cc.u64 r2, r2, 0;",
            "addc.u64 r3, r3, 0;",
            "sub.cc.u64 s0, r0, 18446744069414583343;",
            "subc.cc.u64 s1, r1, 18446744073709551615;",
            "subc.cc.u64 s2, r2, 18446744073709551615;",
            "subc.cc.u64 s3, r3, 18446744073709551615;",
            "subc.u64 borrow, 0, 0;",
            "not.b64 mask, borrow;",
            "xor.b64 tmp, s0, r0;",
            "and.b64 tmp, tmp, mask;",
            "xor.b64 {o0}, r0, tmp;",
            "xor.b64 tmp, s1, r1;",
            "and.b64 tmp, tmp, mask;",
            "xor.b64 {o1}, r1, tmp;",
            "xor.b64 tmp, s2, r2;",
            "and.b64 tmp, tmp, mask;",
            "xor.b64 {o2}, r2, tmp;",
            "xor.b64 tmp, s3, r3;",
            "and.b64 tmp, tmp, mask;",
            "xor.b64 {o3}, r3, tmp;",
            "}}",
            o0 = out(reg64) o0,
            o1 = out(reg64) o1,
            o2 = out(reg64) o2,
            o3 = out(reg64) o3,
            t0 = in(reg32) t[0],
            t1 = in(reg32) t[1],
            t2 = in(reg32) t[2],
            t3 = in(reg32) t[3],
            t4 = in(reg32) t[4],
            t5 = in(reg32) t[5],
            t6 = in(reg32) t[6],
            t7 = in(reg32) t[7],
            t8 = in(reg32) t[8],
            t9 = in(reg32) t[9],
            t10 = in(reg32) t[10],
            t11 = in(reg32) t[11],
            t12 = in(reg32) t[12],
            t13 = in(reg32) t[13],
            t14 = in(reg32) t[14],
            t15 = in(reg32) t[15],
            options(pure, nomem, nostack)
        )
    };
    [o0, o1, o2, o3]
}
