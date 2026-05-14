// ============================================================================
// Constant-Time Field Arithmetic -- Implementation
// ============================================================================
// All operations have data-independent execution traces.
// p = 2^256 - 2^32 - 977 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//
// PLATFORM DISPATCH (field_mul / field_sqr):
//   x86-64:  Inline FE52 5x52 multiply -- zero call overhead, best codegen.
//   ARM64:   fast::operator* -> native ASM (already fastest for ARM64).
//   Generic: fast::operator* + field_normalize (MSVC, 32-bit -- not perf-critical).
//
// PLATFORM DISPATCH (field_inv):
//   x86-64 / ARM64 (__int128): SafeGCD 10x59=590 divsteps (CT, matches libsecp).
//   Generic (MSVC/32-bit):     SafeGCD30 25x30=750 divsteps (CT, no __int128).
// ============================================================================

#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/config.hpp"
#include <cstring>
#include <cstdint>

// Include 5x52 inline implementations for __int128 platforms.
// The inline 5x52 C path produces identical asm to hand-written mul/sqr
// but with zero function-call overhead and superior register allocation.
#if defined(__SIZEOF_INT128__)
#include "secp256k1/field_52.hpp"
#include "secp256k1/field_52_impl.hpp"
#endif

// Unified sanitizer detection. The __SANITIZE_THREAD__ / __SANITIZE_ADDRESS__
// / __SANITIZE_MEMORY__ predefined macros are GCC-only — Clang does NOT
// define them and instead exposes sanitizer state via __has_feature(). Code
// that relied on the GCC-only spellings silently allowed the asm-clobber /
// __builtin_addcll path to run under Clang sanitizers, producing TSan/ASan/
// MSan false positives (e.g. "FAIL: ct field_add" in test_comprehensive).
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer) || __has_feature(address_sanitizer) || __has_feature(memory_sanitizer)
#    define SECP256K1_HAS_SANITIZER 1
#  endif
#endif
#if !defined(SECP256K1_HAS_SANITIZER) && \
    (defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_MEMORY__))
#  define SECP256K1_HAS_SANITIZER 1
#endif

// Disable LTO-defeating asm barriers when ALL of the following are true:
//   1. We're on Clang (Clang's -O0 register allocator handles
//      `asm volatile("" : "+r"(ptr) : : "memory")` clobbers on input
//      pointers by re-fetching the address but writing through stale
//      stack slots — observed in test_comprehensive::test_ct_field
//      "FAIL: ct field_add" on CI / linux (clang-17, Debug)).
//   2. Optimizations are OFF (__OPTIMIZE__ undefined → -O0 / Debug).
// In Debug-O0 builds the compiler does NOT propagate constants across
// function boundaries anyway, so the barrier is a no-op for the property
// it's meant to guarantee (defeat LTO constant propagation). Dropping it
// under Debug-O0 only is safe; production Release / -O2 builds keep the
// barrier, preserving the CT timing invariant.
#if defined(__clang__) && !defined(__OPTIMIZE__)
#  define SECP256K1_NO_LTO_BARRIER 1
#endif

// --- Platform dispatch -------------------------------------------------------
// x86-64 (Clang/GCC): inline FE52 5x52 multiply -- zero call overhead, best codegen.
// ARM64 (Clang/GCC): operator* delegates to native ASM via fast:: -- already optimal.
// Generic (MSVC/32-bit): operator* + field_normalize (functional, not perf-critical).

namespace secp256k1::ct {

// secp256k1 prime p in 4 x 64-bit limbs (little-endian)
static constexpr std::uint64_t P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// 2^256 mod p = 2^32 + 977 = 0x1000003D1
[[maybe_unused]] static constexpr std::uint64_t MOD_K = 0x1000003D1ULL;

// --- Internal helpers --------------------------------------------------------

static inline std::uint64_t add_carry_u64(std::uint64_t a,
                                          std::uint64_t b,
                                          std::uint64_t sum) noexcept {
    return ((a & b) | ((a | b) & ~sum)) >> 63;
}

static inline std::uint64_t sub_borrow_u64(std::uint64_t a,
                                           std::uint64_t b,
                                           std::uint64_t diff) noexcept {
    return (a ^ ((a ^ b) | (diff ^ a))) >> 63;
}

// CT 256-bit addition with carry out. Returns carry (0 or 1).
//
// Implementation note: previously used Clang's __builtin_addcll to emit an
// ADCX/ADOX carry chain on x86-64. Under Clang -O3 + ThinLTO (the actual
// flags used by CI / linux (clang-17, Debug+Release) — see CMakeLists.txt
// which forces -O3 -flto=thin onto every target), Clang miscompiled the
// builtin: r[0..3] received zeros while the sum-of-low-limbs ended up in
// the return register (the carry). This made test_ct_field fail with
// "FAIL: ct field_add" across all Clang variants on x86-64.
//
// The __int128 path below produces correct code under every optimization
// level we exercise (-O0 / -O2 / -O3 / -O3+ThinLTO) and Clang still emits
// a tight ADC chain. We keep the portable manual-carry chain as the
// no-int128 fallback for 32-bit / MSVC targets.
__attribute__((noinline))
static std::uint64_t add256(std::uint64_t r[4],
                                    const std::uint64_t a[4],
                                    const std::uint64_t b[4]) noexcept {
#if 0  // __builtin_addcll path disabled — see note above.
    // Portable add-with-carry chain. Originally written with two add_carry_u64
    // calls + a `r[i] = sum` per iteration, but at Clang -O0 that pattern
    // miscompiled (the loop's `carry` accumulator got fused with the per-iter
    // `sum` slot, returning the sum-of-low-limb as the carry-out and leaving
    // r[0..3] untouched — observed in test_comprehensive::test_ct_field on
    // CI / linux (clang-17, Debug)). The __int128 version below compiles
    // correctly at every optimization level and produces identical results.
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_NO_INT128)
    using u128 = unsigned __int128;
    u128 acc = static_cast<u128>(a[0]) + b[0];
    r[0] = static_cast<std::uint64_t>(acc);
    acc = static_cast<u128>(a[1]) + b[1] + static_cast<std::uint64_t>(acc >> 64);
    r[1] = static_cast<std::uint64_t>(acc);
    acc = static_cast<u128>(a[2]) + b[2] + static_cast<std::uint64_t>(acc >> 64);
    r[2] = static_cast<std::uint64_t>(acc);
    acc = static_cast<u128>(a[3]) + b[3] + static_cast<std::uint64_t>(acc >> 64);
    r[3] = static_cast<std::uint64_t>(acc);
    return static_cast<std::uint64_t>(acc >> 64);
#else
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const sum_lo = a[i] + b[i];
        std::uint64_t const c1 = add_carry_u64(a[i], b[i], sum_lo);
        std::uint64_t const sum = sum_lo + carry;
        std::uint64_t const c2 = add_carry_u64(sum_lo, carry, sum);
        r[i] = sum;
        carry = c1 | c2;
    }
    return carry;
#endif
#endif
}

// CT 256-bit subtraction with borrow out. Returns borrow (0 or 1).
// Uses __builtin_subcll on Clang x86-64 only to emit SBB instructions (B-10).
// Same platform and sanitizer restrictions as add256 / __builtin_addcll above:
//   - x86-64 only: on ARM64 the borrow-flag chain has the same carry-chain
//     reliability issue as __builtin_addcll (observed macOS M-series failures).
//   - Excluded under sanitizers: TSan/ASan instrumentation inserted between
//     calls clobbers the hardware borrow flag, producing wrong results.
// GCC-13 does NOT have __builtin_subcll; always uses the portable path.
static inline std::uint64_t sub256(std::uint64_t r[4],
                                    const std::uint64_t a[4],
                                    const std::uint64_t b[4]) noexcept {
#if defined(__clang__) && defined(__x86_64__) && \
    !defined(SECP256K1_HAS_SANITIZER) && !defined(LLVM_PROFILE_ENABLED) && \
    !defined(SECP256K1_NO_LTO_BARRIER)
    unsigned long long borrow = 0;
    borrow = __builtin_subcll(a[0], b[0], borrow, (unsigned long long*)&r[0]);
    borrow = __builtin_subcll(a[1], b[1], borrow, (unsigned long long*)&r[1]);
    borrow = __builtin_subcll(a[2], b[2], borrow, (unsigned long long*)&r[2]);
    borrow = __builtin_subcll(a[3], b[3], borrow, (unsigned long long*)&r[3]);
    return (std::uint64_t)borrow;
#else
    // Same Clang-O0 miscompile as add256; use __int128 carry chain.
#if defined(__SIZEOF_INT128__) && !defined(SECP256K1_NO_INT128)
    using u128 = unsigned __int128;
    // diff = a - b - borrow_in; borrow_out = (a < b + borrow_in) as 0/1.
    // Implement via 1-128-bit math: extend a to u128, subtract b + borrow_in,
    // top bit of the truncated u128 indicates borrow.
    u128 acc = static_cast<u128>(a[0]) - b[0];
    r[0] = static_cast<std::uint64_t>(acc);
    std::uint64_t borrow = static_cast<std::uint64_t>(acc >> 64) & 1ULL;
    acc = static_cast<u128>(a[1]) - b[1] - borrow;
    r[1] = static_cast<std::uint64_t>(acc);
    borrow = static_cast<std::uint64_t>(acc >> 64) & 1ULL;
    acc = static_cast<u128>(a[2]) - b[2] - borrow;
    r[2] = static_cast<std::uint64_t>(acc);
    borrow = static_cast<std::uint64_t>(acc >> 64) & 1ULL;
    acc = static_cast<u128>(a[3]) - b[3] - borrow;
    r[3] = static_cast<std::uint64_t>(acc);
    return static_cast<std::uint64_t>(acc >> 64) & 1ULL;
#else
    std::uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const diff = a[i] - b[i];
        std::uint64_t const b1 = sub_borrow_u64(a[i], b[i], diff);
        std::uint64_t const result = diff - borrow;
        std::uint64_t const b2 = sub_borrow_u64(diff, borrow, result);
        r[i] = result;
        borrow = b1 | b2;
    }
    return borrow;
#endif
#endif
}

// CT normalize: reduce to [0, p) without branches
// If value >= p, subtract p. Uses cmov.
static inline void ct_reduce_once(std::uint64_t r[4]) noexcept {
    std::uint64_t tmp[4];
    std::uint64_t const borrow = sub256(tmp, r, P);
    // If borrow == 0, r >= p -> use tmp (reduced). Else keep r.
    // mask = 0xFFF...F if no borrow (r >= p), else 0
    std::uint64_t const mask = is_zero_mask(borrow);
    cmov256(r, tmp, mask);
}

// --- Public API --------------------------------------------------------------

FieldElement field_normalize(const FieldElement& a) noexcept {
    std::uint64_t r[4];
    std::memcpy(r, &a.limbs()[0], 32);
    ct_reduce_once(r);
    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

// field_add / field_sub / field_neg delegate to the fast layer.
//
// The fast::FieldElement operator+ / operator- / negate paths in
// src/cpu/src/field.cpp are already branchless (see add_impl: 4 add64 +
// conditional reduce via xor-mask, no data-dependent branches). Maintaining
// a parallel CT-only implementation here was the source of TWO independent
// miscompiles:
//   1. Clang ThinLTO at -O3 reordered the add256 + sub256 + cmov256 chain
//      so r[] received zeros while the sum-of-limb-0 ended up in the
//      carry return slot — observed on every CI / linux (clang-17, *) job.
//   2. The __builtin_addcll / asm-volatile barriers fought with Clang -O0
//      register allocation under sanitizer instrumentation.
// Delegating preserves the CT guarantee (the fast path IS the CT path) and
// removes both miscompile classes at once.
FieldElement field_add(const FieldElement& a, const FieldElement& b) noexcept {
    return a + b;
}

FieldElement field_sub(const FieldElement& a, const FieldElement& b) noexcept {
    return a - b;
}

FieldElement field_mul(const FieldElement& a, const FieldElement& b) noexcept {
#if defined(__SIZEOF_INT128__)
    // FE52 5x52 multiply with integrated reduction -- best codegen on
    // x86-64 (BMI2), ARM64, and RISC-V (dedicated asm: fe52_mul_inner_riscv64).
    using FE52 = secp256k1::fast::FieldElement52;
    // Barrier both FE52 limb arrays to prevent compiler constant-propagation
    // of known inputs (e.g. fe_one * x) into the multiply inner kernel.
    FE52 fa = FE52::from_fe(a); // NOLINT(misc-const-correctness)
    FE52 fb = FE52::from_fe(b); // NOLINT(misc-const-correctness)
#if (defined(__GNUC__) || defined(__clang__)) && !defined(SECP256K1_HAS_SANITIZER) && !defined(SECP256K1_NO_LTO_BARRIER)
    asm volatile("" : "+r"(fa.n[0]), "+r"(fa.n[1]), "+r"(fa.n[2]),
                      "+r"(fa.n[3]), "+r"(fa.n[4]));
    asm volatile("" : "+r"(fb.n[0]), "+r"(fb.n[1]), "+r"(fb.n[2]),
                      "+r"(fb.n[3]), "+r"(fb.n[4]));
#endif
    return (fa * fb).to_fe();
#else
    // MSVC / ESP32: 4x64 Comba with separate reduction.
    return a * b;
#endif
}

FieldElement field_sqr(const FieldElement& a) noexcept {
#if defined(__SIZEOF_INT128__)
    // FE52 5x52 square with integrated reduction -- best codegen on
    // x86-64 (BMI2), ARM64, and RISC-V (dedicated asm: fe52_sqr_inner_riscv64).
    using FE52 = secp256k1::fast::FieldElement52;
    // Barrier all FE52 limbs to prevent the compiler from propagating
    // known-limb patterns (e.g. fe_one, fe_zero) into the square kernel,
    // which would create measurable timing differences vs random inputs.
    // Applied on all platforms (RISC-V, x86-64, ARM64) for uniform CT behavior.
    FE52 tmp = FE52::from_fe(a);  // NOLINT(misc-const-correctness) -- +r clobber
#if (defined(__GNUC__) || defined(__clang__)) && !defined(SECP256K1_HAS_SANITIZER) && !defined(SECP256K1_NO_LTO_BARRIER)
    asm volatile("" : "+r"(tmp.n[0]), "+r"(tmp.n[1]), "+r"(tmp.n[2]),
                      "+r"(tmp.n[3]), "+r"(tmp.n[4]));
#endif
    return tmp.square().to_fe();
#else
    // MSVC / ESP32: 4x64 square with separate reduction.
    return a.square();
#endif
}

FieldElement field_neg(const FieldElement& a) noexcept {
    // -a mod p = 0 - a (both fast::operator- and fast::operator unary - are
    // branchless, see field.cpp::sub_impl). Delegate to avoid the same Clang
    // ThinLTO miscompile that hit field_add / field_sub.
    return FieldElement::zero() - a;
}

FieldElement field_half(const FieldElement& a) noexcept {
    // r = a/2 mod p. Branchless.
    // If a is odd: r = (a + p) / 2; if even: r = a / 2.
    const auto& al = a.limbs();
    std::uint64_t const odd = 0ULL - (al[0] & 1);  // all-ones if odd, 0 if even

    // Conditionally add p (only if odd)
    std::uint64_t t[4];
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t const addend = P[i] & odd;
        std::uint64_t const sum_lo = al[i] + addend;
        auto const c1 = static_cast<std::uint64_t>(sum_lo < al[i]);
        std::uint64_t const sum = sum_lo + carry;
        auto const c2 = static_cast<std::uint64_t>(sum < sum_lo);
        t[i] = sum;
        carry = c1 + c2;
    }

    // Right shift 257-bit value (t[0..3] + carry*2^256) by 1
    std::uint64_t const r0 = (t[0] >> 1) | (t[1] << 63);
    std::uint64_t const r1 = (t[1] >> 1) | (t[2] << 63);
    std::uint64_t const r2 = (t[2] >> 1) | (t[3] << 63);
    std::uint64_t const r3 = (t[3] >> 1) | (carry << 63);

    return FieldElement::from_limbs_raw({r0, r1, r2, r3});
}

// -- CT SafeGCD building blocks (extracted for register allocation) ----------
#if defined(__SIZEOF_INT128__)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
namespace {

struct SG62   { int64_t v[5]; };
struct SGTrans { int64_t u, v, q, r; };

// secp256k1 prime in signed-62:  p = 2^256 - 2^32 - 977
static constexpr SG62 SGP = {{
    -(int64_t)0x1000003D1LL, 0, 0, 0, 256
}};
// (-p)^{-1} mod 2^62
static constexpr uint64_t P_INV62 = 0x27C7F6E22DDACACFULL;

// CT batch of 59 divsteps using conditional-add pattern (not conditional-swap).
// Result matrix is scaled by 2^62 (initial 8xidentity + 59 halvings).
// zeta = -(delta + 1/2); swap condition is zeta < 0 (i.e., delta > 0).
// Matches libsecp: 10 x 59 = 590 divsteps (proven sufficient for 256-bit).
static int64_t ct_divsteps_59(int64_t zeta, uint64_t f0, uint64_t g0,
                               SGTrans& t) noexcept {
    // Start with 8xidentity (2^3) so output is scaled by 2^62 (3 + 59 = 62).
    uint64_t u = 8, v = 0, q = 0, r = 8;
    uint64_t f = f0, g = g0;

    for (int i = 3; i < 62; ++i) {
        auto c1 = (uint64_t)(zeta >> 63);    // all-ones if zeta < 0
        uint64_t const c2 = -(g & 1);                   // all-ones if g odd

        // Conditionally negate (f,u,v) based on zeta sign
        uint64_t const x = (f ^ c1) - c1;
        uint64_t const y = (u ^ c1) - c1;
        uint64_t const z = (v ^ c1) - c1;

        // Conditionally add to (g,q,r) if g is odd
        g += x & c2;
        q += y & c2;
        r += z & c2;

        // Combined mask: zeta < 0 AND g was odd
        c1 &= c2;

        // zeta update: swap -> -zeta-2, no-swap -> zeta-1
        zeta = (zeta ^ (int64_t)c1) - 1;

        // Conditionally add (g,q,r) into (f,u,v) when both conditions met
        f += g & c1;
        u += q & c1;
        v += r & c1;

        g >>= 1;
        u <<= 1;
        v <<= 1;
    }

    t.u = (int64_t)u; t.v = (int64_t)v;
    t.q = (int64_t)q; t.r = (int64_t)r;
    return zeta;
}

// Apply transition matrix to (d,e) mod p.
// Exploits secp256k1 prime structure: p.v[1..3]=0, p.v[4]=256=2^8.
static void ct_update_de(SG62& d, SG62& e, const SGTrans& t) noexcept {
    const uint64_t M62 = UINT64_MAX >> 2;
    const int64_t d0=d.v[0], d1=d.v[1], d2=d.v[2], d3=d.v[3], d4=d.v[4];
    const int64_t e0=e.v[0], e1=e.v[1], e2=e.v[2], e3=e.v[3], e4=e.v[4];
    int64_t md = 0, me = 0;
    __int128 cd = 0, ce = 0;

    md = (t.u & (d4 >> 63)) + (t.v & (e4 >> 63));
    me = (t.q & (d4 >> 63)) + (t.r & (e4 >> 63));

    cd = (__int128)t.u * d0 + (__int128)t.v * e0;
    ce = (__int128)t.q * d0 + (__int128)t.r * e0;

    md -= (int64_t)((P_INV62 * (uint64_t)cd + (uint64_t)md) & M62);
    me -= (int64_t)((P_INV62 * (uint64_t)ce + (uint64_t)me) & M62);

    cd += (__int128)SGP.v[0] * md;
    ce += (__int128)SGP.v[0] * me;
    cd >>= 62;
    ce >>= 62;

    cd += (__int128)t.u * d1 + (__int128)t.v * e1;
    ce += (__int128)t.q * d1 + (__int128)t.r * e1;
    d.v[0] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[0] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (__int128)t.u * d2 + (__int128)t.v * e2;
    ce += (__int128)t.q * d2 + (__int128)t.r * e2;
    d.v[1] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[1] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (__int128)t.u * d3 + (__int128)t.v * e3;
    ce += (__int128)t.q * d3 + (__int128)t.r * e3;
    d.v[2] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[2] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    cd += (__int128)t.u * d4 + (__int128)t.v * e4 + ((__int128)md << 8);
    ce += (__int128)t.q * d4 + (__int128)t.r * e4 + ((__int128)me << 8);
    d.v[3] = (int64_t)((uint64_t)cd & M62); cd >>= 62;
    e.v[3] = (int64_t)((uint64_t)ce & M62); ce >>= 62;

    d.v[4] = (int64_t)cd;
    e.v[4] = (int64_t)ce;
}

// Apply transition matrix to full-precision (f,g).  Always 5 limbs.
static void ct_update_fg(SG62& f, SG62& g, const SGTrans& t) noexcept {
    const auto M62 = (int64_t)((uint64_t)(-1) >> 2);
    __int128 cf = 0, cg = 0;

    cf = (__int128)t.u * f.v[0] + (__int128)t.v * g.v[0];
    cg = (__int128)t.q * f.v[0] + (__int128)t.r * g.v[0];
    cf >>= 62;
    cg >>= 62;

    cf += (__int128)t.u * f.v[1] + (__int128)t.v * g.v[1];
    cg += (__int128)t.q * f.v[1] + (__int128)t.r * g.v[1];
    f.v[0] = (int64_t)cf & M62; cf >>= 62;
    g.v[0] = (int64_t)cg & M62; cg >>= 62;

    cf += (__int128)t.u * f.v[2] + (__int128)t.v * g.v[2];
    cg += (__int128)t.q * f.v[2] + (__int128)t.r * g.v[2];
    f.v[1] = (int64_t)cf & M62; cf >>= 62;
    g.v[1] = (int64_t)cg & M62; cg >>= 62;

    cf += (__int128)t.u * f.v[3] + (__int128)t.v * g.v[3];
    cg += (__int128)t.q * f.v[3] + (__int128)t.r * g.v[3];
    f.v[2] = (int64_t)cf & M62; cf >>= 62;
    g.v[2] = (int64_t)cg & M62; cg >>= 62;

    cf += (__int128)t.u * f.v[4] + (__int128)t.v * g.v[4];
    cg += (__int128)t.q * f.v[4] + (__int128)t.r * g.v[4];
    f.v[3] = (int64_t)cf & M62; cf >>= 62;
    g.v[3] = (int64_t)cg & M62; cg >>= 62;

    f.v[4] = (int64_t)cf;
    g.v[4] = (int64_t)cg;
}

// Normalize result to [0, p): conditional add/negate/carry propagation.
static void ct_sg_normalize(SG62& r, int64_t f_sign) noexcept {
    const auto M62 = (int64_t)(UINT64_MAX >> 2);
    int64_t r0=r.v[0], r1=r.v[1], r2=r.v[2], r3=r.v[3], r4=r.v[4];

    int64_t ca = r4 >> 63;
    r0 += SGP.v[0] & ca;
    r4 += SGP.v[4] & ca;

    int64_t const cn = f_sign >> 63;
    r0 = (r0 ^ cn) - cn;
    r1 = (r1 ^ cn) - cn;
    r2 = (r2 ^ cn) - cn;
    r3 = (r3 ^ cn) - cn;
    r4 = (r4 ^ cn) - cn;

    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    ca = r4 >> 63;
    r0 += SGP.v[0] & ca;
    r4 += SGP.v[4] & ca;

    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r.v[0]=r0; r.v[1]=r1; r.v[2]=r2; r.v[3]=r3; r.v[4]=r4;
}

} // anonymous namespace
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif // __SIZEOF_INT128__

// -- CT SafeGCD30 building blocks (MSVC / no __int128) -----------------------
// Bernstein-Yang divstep with 30-bit batches, using only int32_t/int64_t.
// Constant-time: fixed 30 iterations per batch, branchless swap/negate.
// Matches bitcoin-core/secp256k1 secp256k1_modinv32 (MIT license).
#if !defined(__SIZEOF_INT128__)
namespace {

struct S30CT  { int32_t v[9]; };
struct T2x2CT { int32_t u, v, q, r; };
struct ModInfoCT { S30CT modulus; std::uint32_t modulus_inv30; };

static inline int32_t arshift32(int32_t x, unsigned s) noexcept {
    std::uint32_t const ux = static_cast<std::uint32_t>(x);
    std::uint32_t const sign = 0U - (ux >> 31);
    if (s == 0) {
        return x;
    }
    return static_cast<int32_t>((ux >> s) | (sign << (32U - s)));
}

// secp256k1 prime p in signed-30 representation:
// p = 2^256 - 2^32 - 977
static constexpr ModInfoCT PINFO_CT = {
    {{-0x3D1, -4, 0, 0, 0, 0, 0, 0, 65536}},
    0x2DDACACFU
};

// Constant-time 30 divsteps -- branchless, no ctz, fixed iteration count.
// zeta tracks -(delta + 1/2); delta starts at 1/2, so zeta starts at -1.
static int32_t ct_divsteps_30(int32_t zeta, std::uint32_t f0, std::uint32_t g0,
                               T2x2CT& t) noexcept {
    std::uint32_t u = 1, v = 0, q = 0, r = 1;
    std::uint32_t f = f0, g = g0;

    for (int i = 0; i < 30; ++i) {
        // c1 = all-ones if zeta < 0 (i.e. delta > 0), else 0
        auto c1 = 0U - (static_cast<std::uint32_t>(zeta) >> 31);
        // c2 = all-ones if g is odd, else 0
        std::uint32_t const c2 = 0U - (g & 1U);

        // Conditionally negate f,u,v (if zeta < 0)
        std::uint32_t const x = (f ^ c1) - c1;
        std::uint32_t const y = (u ^ c1) - c1;
        std::uint32_t const z = (v ^ c1) - c1;

        // Conditionally add negated f to g (if g odd)
        g += x & c2;
        q += y & c2;
        r += z & c2;

        // Combined mask: swap iff zeta < 0 AND g was odd
        c1 &= c2;

        // zeta update: swap -> -(old_zeta)-2; no-swap -> zeta-1
        zeta = static_cast<int32_t>(static_cast<std::uint32_t>(zeta) ^ c1) - 1;

        // Conditionally swap: add new (g,q,r) into (f,u,v)
        f += g & c1;
        u += q & c1;
        v += r & c1;

        // Halve g, double (u,v) columns
        g >>= 1;
        u <<= 1;
        v <<= 1;
    }

    t.u = static_cast<int32_t>(u); t.v = static_cast<int32_t>(v);
    t.q = static_cast<int32_t>(q); t.r = static_cast<int32_t>(r);
    return zeta;
}

// Apply transition matrix to (d,e) mod p.  Constant-time.
// Computes (d', e') = (t / 2^30) * (d, e)  mod p.
static void ct_update_de_30(S30CT& d, S30CT& e, const T2x2CT& t,
                             const ModInfoCT& mod) noexcept {
    const auto M30 = static_cast<int32_t>(UINT32_MAX >> 2);
    const int32_t u = t.u, v = t.v, q = t.q, r = t.r;
    int32_t di, ei, md, me, sd, se;
    int64_t cd, ce;

    sd = arshift32(d.v[8], 31);
    se = arshift32(e.v[8], 31);
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    di = d.v[0]; ei = e.v[0];
    cd = static_cast<int64_t>(u) * di + static_cast<int64_t>(v) * ei;
    ce = static_cast<int64_t>(q) * di + static_cast<int64_t>(r) * ei;

    md -= static_cast<int32_t>((mod.modulus_inv30 * static_cast<std::uint32_t>(cd)
          + static_cast<std::uint32_t>(md)) & static_cast<std::uint32_t>(M30));
    me -= static_cast<int32_t>((mod.modulus_inv30 * static_cast<std::uint32_t>(ce)
          + static_cast<std::uint32_t>(me)) & static_cast<std::uint32_t>(M30));

    cd += static_cast<int64_t>(mod.modulus.v[0]) * md;
    ce += static_cast<int64_t>(mod.modulus.v[0]) * me;
    cd >>= 30; ce >>= 30;

    for (int i = 1; i < 9; ++i) {
        di = d.v[i]; ei = e.v[i];
        cd += static_cast<int64_t>(u) * di + static_cast<int64_t>(v) * ei;
        ce += static_cast<int64_t>(q) * di + static_cast<int64_t>(r) * ei;
        cd += static_cast<int64_t>(mod.modulus.v[i]) * md;
        ce += static_cast<int64_t>(mod.modulus.v[i]) * me;
        d.v[i - 1] = static_cast<int32_t>(cd) & M30; cd >>= 30;
        e.v[i - 1] = static_cast<int32_t>(ce) & M30; ce >>= 30;
    }
    d.v[8] = static_cast<int32_t>(cd);
    e.v[8] = static_cast<int32_t>(ce);
}

// Apply transition matrix to (f,g).  Constant-time, always full 9 limbs.
static void ct_update_fg_30(S30CT& f, S30CT& g, const T2x2CT& t) noexcept {
    const auto M30 = static_cast<int32_t>(UINT32_MAX >> 2);
    int32_t fi, gi;
    int64_t cf, cg;

    fi = f.v[0]; gi = g.v[0];
    cf = static_cast<int64_t>(t.u) * fi + static_cast<int64_t>(t.v) * gi;
    cg = static_cast<int64_t>(t.q) * fi + static_cast<int64_t>(t.r) * gi;
    cf >>= 30; cg >>= 30;

    for (int j = 1; j < 9; ++j) {
        fi = f.v[j]; gi = g.v[j];
        cf += static_cast<int64_t>(t.u) * fi + static_cast<int64_t>(t.v) * gi;
        cg += static_cast<int64_t>(t.q) * fi + static_cast<int64_t>(t.r) * gi;
        f.v[j - 1] = static_cast<int32_t>(static_cast<std::uint32_t>(cf) & static_cast<std::uint32_t>(M30));
        cf >>= 30;
        g.v[j - 1] = static_cast<int32_t>(static_cast<std::uint32_t>(cg) & static_cast<std::uint32_t>(M30));
        cg >>= 30;
    }
    f.v[8] = static_cast<int32_t>(cf);
    g.v[8] = static_cast<int32_t>(cg);
}

// Normalize signed-30 result to [0, p).  Constant-time (branchless).
static void ct_normalize_30(S30CT& r, int32_t sign, const ModInfoCT& mod) noexcept {
    const auto M30 = static_cast<int32_t>(UINT32_MAX >> 2);
    int32_t r0=r.v[0], r1=r.v[1], r2=r.v[2], r3=r.v[3], r4=r.v[4],
            r5=r.v[5], r6=r.v[6], r7=r.v[7], r8=r.v[8];
    int32_t cond;

    // If r < 0, add modulus
    cond = arshift32(r8, 31);
    r0 += mod.modulus.v[0] & cond;
    r1 += mod.modulus.v[1] & cond;
    r2 += mod.modulus.v[2] & cond;
    r3 += mod.modulus.v[3] & cond;
    r4 += mod.modulus.v[4] & cond;
    r5 += mod.modulus.v[5] & cond;
    r6 += mod.modulus.v[6] & cond;
    r7 += mod.modulus.v[7] & cond;
    r8 += mod.modulus.v[8] & cond;

    // Conditionally negate based on sign of f
    cond = arshift32(sign, 31);
    r0 = (r0 ^ cond) - cond;
    r1 = (r1 ^ cond) - cond;
    r2 = (r2 ^ cond) - cond;
    r3 = (r3 ^ cond) - cond;
    r4 = (r4 ^ cond) - cond;
    r5 = (r5 ^ cond) - cond;
    r6 = (r6 ^ cond) - cond;
    r7 = (r7 ^ cond) - cond;
    r8 = (r8 ^ cond) - cond;

    // Carry propagation
    r1 += arshift32(r0, 30); r0 &= M30;
    r2 += arshift32(r1, 30); r1 &= M30;
    r3 += arshift32(r2, 30); r2 &= M30;
    r4 += arshift32(r3, 30); r3 &= M30;
    r5 += arshift32(r4, 30); r4 &= M30;
    r6 += arshift32(r5, 30); r5 &= M30;
    r7 += arshift32(r6, 30); r6 &= M30;
    r8 += arshift32(r7, 30); r7 &= M30;

    // Second conditional add (may still be negative after negate)
    cond = arshift32(r8, 31);
    r0 += mod.modulus.v[0] & cond;
    r1 += mod.modulus.v[1] & cond;
    r2 += mod.modulus.v[2] & cond;
    r3 += mod.modulus.v[3] & cond;
    r4 += mod.modulus.v[4] & cond;
    r5 += mod.modulus.v[5] & cond;
    r6 += mod.modulus.v[6] & cond;
    r7 += mod.modulus.v[7] & cond;
    r8 += mod.modulus.v[8] & cond;

    // Final carry propagation
    r1 += arshift32(r0, 30); r0 &= M30;
    r2 += arshift32(r1, 30); r1 &= M30;
    r3 += arshift32(r2, 30); r2 &= M30;
    r4 += arshift32(r3, 30); r3 &= M30;
    r5 += arshift32(r4, 30); r4 &= M30;
    r6 += arshift32(r5, 30); r5 &= M30;
    r7 += arshift32(r6, 30); r6 &= M30;
    r8 += arshift32(r7, 30); r7 &= M30;

    r.v[0]=r0; r.v[1]=r1; r.v[2]=r2; r.v[3]=r3; r.v[4]=r4;
    r.v[5]=r5; r.v[6]=r6; r.v[7]=r7; r.v[8]=r8;
}

// Convert 4x64-bit limbs -> signed-30 representation
static S30CT ct_limbs_to_s30(const std::uint64_t* x) noexcept {
    S30CT r{};
    const std::uint32_t M30 = 0x3FFFFFFFu;
    r.v[0] = static_cast<int32_t>( x[0]        & M30);
    r.v[1] = static_cast<int32_t>((x[0] >> 30) & M30);
    r.v[2] = static_cast<int32_t>(((x[0] >> 60) | (x[1] <<  4)) & M30);
    r.v[3] = static_cast<int32_t>((x[1] >> 26) & M30);
    r.v[4] = static_cast<int32_t>(((x[1] >> 56) | (x[2] <<  8)) & M30);
    r.v[5] = static_cast<int32_t>((x[2] >> 22) & M30);
    r.v[6] = static_cast<int32_t>(((x[2] >> 52) | (x[3] << 12)) & M30);
    r.v[7] = static_cast<int32_t>((x[3] >> 18) & M30);
    r.v[8] = static_cast<int32_t>( x[3] >> 48);
    return r;
}

// Convert signed-30 -> 4x64-bit limbs
static void ct_s30_to_u64(const S30CT& s, std::uint64_t* r) noexcept {
    r[0] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[0])))
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[1])) << 30)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[2])) << 60);
    r[1] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[2])) >> 4)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[3])) << 26)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[4])) << 56);
    r[2] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[4])) >> 8)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[5])) << 22)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[6])) << 52);
    r[3] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[6])) >> 12)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[7])) << 18)
         | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(s.v[8])) << 48);
}

} // anonymous namespace
#endif // !__SIZEOF_INT128__

FieldElement field_inv(const FieldElement& a) noexcept {
#if defined(__SIZEOF_INT128__)
    // -- CT SafeGCD inverse -----------------------------------------------
    // Bernstein-Yang divstep: 10 x 59 = 590 branchless divsteps.
    // Matches bitcoin-core/secp256k1 proven bound for 256-bit modular inverse.

    // fe -> s62
    const auto& al = a.limbs();
    constexpr uint64_t M = (1ULL << 62) - 1;

    SG62 d = {{0,0,0,0,0}};
    SG62 e = {{1,0,0,0,0}};
    SG62 f = SGP;
    SG62 g = {{
        (int64_t)(al[0] & M),
        (int64_t)(((al[0] >> 62) | (al[1] << 2)) & M),
        (int64_t)(((al[1] >> 60) | (al[2] << 4)) & M),
        (int64_t)(((al[2] >> 58) | (al[3] << 6)) & M),
        (int64_t)(al[3] >> 56)
    }};
    int64_t zeta = -1;  // zeta = -(delta + 1/2); delta starts at 1/2

    // 10 x 59 = 590 divsteps (proven sufficient for 256-bit; same as libsecp)
    for (int iter = 0; iter < 10; ++iter) {
        SGTrans t;
        zeta = ct_divsteps_59(zeta, (uint64_t)f.v[0], (uint64_t)g.v[0], t);
        ct_update_de(d, e, t);
        ct_update_fg(f, g, t);
    }

    ct_sg_normalize(d, f.v[4]);

    // s62 -> fe
    return FieldElement::from_limbs({
        (uint64_t)d.v[0] | ((uint64_t)d.v[1] << 62),
        ((uint64_t)d.v[1] >> 2)  | ((uint64_t)d.v[2] << 60),
        ((uint64_t)d.v[2] >> 4)  | ((uint64_t)d.v[3] << 58),
        ((uint64_t)d.v[3] >> 6)  | ((uint64_t)d.v[4] << 56)
    });

#else
    // -- CT SafeGCD30 inverse (MSVC / no __int128) -------------------------
    // Bernstein-Yang divstep: 25 x 30 = 750 branchless divsteps.
    // Matches bitcoin-core/secp256k1 secp256k1_modinv32 proven bound.
    // No __int128 required -- uses int32_t/int64_t only.
    // Replaces the Fermat chain (a^(p-2)) which leaked via non-CT multiply.

    const auto& al = a.limbs();

    S30CT d{};
    S30CT e{}; e.v[0] = 1;
    S30CT f = PINFO_CT.modulus;
    S30CT g = ct_limbs_to_s30(al.data());
    int32_t zeta = -1;  // zeta = -(delta + 1/2); delta starts at 1/2

    // 25 x 30 = 750 divsteps (proven sufficient for 256-bit; same as libsecp)
    for (int iter = 0; iter < 25; ++iter) {
        T2x2CT t;
        zeta = ct_divsteps_30(zeta, static_cast<std::uint32_t>(f.v[0]),
                              static_cast<std::uint32_t>(g.v[0]), t);
        ct_update_de_30(d, e, t, PINFO_CT);
        ct_update_fg_30(f, g, t);
    }

    ct_normalize_30(d, f.v[8], PINFO_CT);

    // s30 -> fe
    std::uint64_t out[4];
    ct_s30_to_u64(d, out);
    return FieldElement::from_limbs({out[0], out[1], out[2], out[3]});
#endif  // __SIZEOF_INT128__
}

// --- Conditional Operations --------------------------------------------------

void field_cmov(FieldElement* r, const FieldElement& a,
                std::uint64_t mask) noexcept {
    // In-place XOR-mask conditional move -- no temporary FieldElement.
    // mask is all-ones (select a) or all-zeros (keep r).
    auto* rd = r->limbs_mut().data();
    const auto* ad = a.limbs().data();
    rd[0] ^= (rd[0] ^ ad[0]) & mask;
    rd[1] ^= (rd[1] ^ ad[1]) & mask;
    rd[2] ^= (rd[2] ^ ad[2]) & mask;
    rd[3] ^= (rd[3] ^ ad[3]) & mask;
}

void field_cswap(FieldElement* a, FieldElement* b,
                 std::uint64_t mask) noexcept {
    // Direct in-place XOR swap -- no temporaries.
    auto* ad = a->limbs_mut().data();
    auto* bd = b->limbs_mut().data();
    for (int i = 0; i < 4; ++i) {
        std::uint64_t const diff = (ad[i] ^ bd[i]) & mask;
        ad[i] ^= diff;
        bd[i] ^= diff;
    }
}

FieldElement field_select(const FieldElement& a, const FieldElement& b,
                          std::uint64_t mask) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    return FieldElement::from_limbs_raw({
        ct_select(al[0], bl[0], mask),
        ct_select(al[1], bl[1], mask),
        ct_select(al[2], bl[2], mask),
        ct_select(al[3], bl[3], mask)
    });
}

FieldElement field_cneg(const FieldElement& a, std::uint64_t mask) noexcept {
    FieldElement const neg = field_neg(a);
    return field_select(neg, a, mask);
}

// --- Comparison --------------------------------------------------------------

std::uint64_t field_is_zero(const FieldElement& a) noexcept {
    const auto& l = a.limbs();
    // value_barrier each limb before OR to prevent the compiler from
    // constant-propagating zero limbs -> zero OR -> optimized is_zero_mask.
    std::uint64_t a0 = l[0], a1 = l[1], a2 = l[2], a3 = l[3];
    value_barrier(a0);
    value_barrier(a1);
    value_barrier(a2);
    value_barrier(a3);
    std::uint64_t const z = a0 | a1 | a2 | a3;
    return is_zero_mask(z);
}

std::uint64_t field_eq(const FieldElement& a, const FieldElement& b) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    // value_barrier each XOR result to prevent compiler from
    // short-circuiting when a == b (XOR of same provenance -> 0).
    std::uint64_t d0 = al[0] ^ bl[0];
    std::uint64_t d1 = al[1] ^ bl[1];
    std::uint64_t d2 = al[2] ^ bl[2];
    std::uint64_t d3 = al[3] ^ bl[3];
    value_barrier(d0);
    value_barrier(d1);
    value_barrier(d2);
    value_barrier(d3);
    std::uint64_t const diff = d0 | d1 | d2 | d3;
    return is_zero_mask(diff);
}

} // namespace secp256k1::ct
