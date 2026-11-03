// ============================================================================
// test_field_lanes -- lane-parallel field backends vs fast::FieldElement
// ============================================================================
//
// For every lane backend available on the running CPU (secp256k1/lanes.hpp),
// every op is compared element by element with the scalar fast::FieldElement:
//
//   * 2^16 random inputs plus edge values (0, 1, p-1, p-2, p, 2^255,
//     2^256-1 raw and mod p, values straddling every radix-2^52 and
//     radix-2^26 limb boundary), edge x edge pairs for the binary ops;
//   * internal-form inputs exactly at the documented weak-normal bound
//     (mul_limbs / sqr_limbs / normalize_limbs);
//   * a 10^4-step mixed op chain per lane with a different value in every lane;
//   * the kernels' limb-bound assert counter must stay 0.
//
// The pure dispatch helpers (resolve_lane_selection, default_lane_backend,
// lane_backend_supported) are checked against synthetic inputs first.
//
// Usage: test_field_lanes [--require=<off|avx2|avx_ifma|ifma256|ifma512>]...
//                         [--bench] [--seed=<n>]
//   --require  fail (exit 1), not skip, when that backend is unavailable
//   --bench    after the checks, print ns per element for mul/sqr/add per
//              backend and for scalar FieldElement (stdout only)
// Exit: 0 pass, 1 failure, 77 no lane backend available (and none required).
// ============================================================================

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "secp256k1/cpu_features.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/lanes.hpp"

using secp256k1::fast::FieldElement;
namespace lanes = secp256k1::lanes;
using lanes::LaneBackend;
using lanes::LaneFieldKernels;

namespace {

using W4 = std::array<std::uint64_t, 4>;

int g_failures = 0;

constexpr W4 kP = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                   0xFFFFFFFFFFFFFFFFULL};

void check(bool ok, const char* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

FieldElement fe_of(const std::uint64_t* w) { return FieldElement::from_limbs({w[0], w[1], w[2], w[3]}); }

std::string hex(const std::uint64_t* w) {
    char buf[80];
    std::snprintf(buf, sizeof buf, "%016llx%016llx%016llx%016llx", static_cast<unsigned long long>(w[3]),
                  static_cast<unsigned long long>(w[2]), static_cast<unsigned long long>(w[1]),
                  static_cast<unsigned long long>(w[0]));
    return buf;
}

// 256-bit helpers on raw 4x64 values.
W4 add_raw(const W4& a, const W4& b, bool* carry_out = nullptr) {
    W4 r{};
    std::uint64_t c = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const std::uint64_t s = a[i] + c;
        const std::uint64_t c1 = s < c ? 1u : 0u;
        r[i] = s + b[i];
        c = c1 + (r[i] < s ? 1u : 0u);
    }
    if (carry_out) *carry_out = c != 0;
    return r;
}

W4 sub_small(const W4& a, std::uint64_t s) {
    W4 r = a;
    for (std::size_t i = 0; i < 4; ++i) {
        const std::uint64_t before = r[i];
        r[i] = before - s;
        if (before >= s) break;
        s = 1;
    }
    return r;
}

W4 pow2(unsigned bit) {
    W4 r{};
    if (bit < 256) r[bit / 64] = std::uint64_t{1} << (bit % 64);
    return r;
}

std::vector<W4> edge_values() {
    std::vector<W4> v;
    const W4 zero{};
    const W4 ones = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    v.push_back(zero);
    v.push_back({1, 0, 0, 0});
    v.push_back({2, 0, 0, 0});
    v.push_back({3, 0, 0, 0});
    v.push_back(sub_small(kP, 1));            // p - 1
    v.push_back(sub_small(kP, 2));            // p - 2
    v.push_back(kP);                          // p (raw, == 0)
    v.push_back(add_raw(kP, {1, 0, 0, 0}));   // p + 1 (raw)
    v.push_back(pow2(255));
    v.push_back(ones);                        // 2^256 - 1 (raw)
    v.push_back({0x1000003D0ULL, 0, 0, 0});   // 2^256 - 1 mod p
    v.push_back({0x1000003D1ULL, 0, 0, 0});   // 2^256 mod p
    v.push_back({0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL});
    v.push_back({0x5555555555555555ULL, 0x5555555555555555ULL, 0x5555555555555555ULL, 0x5555555555555555ULL});
    v.push_back({~0ULL, 0, 0, 0});
    v.push_back({0, ~0ULL, 0, 0});
    v.push_back({0, 0, ~0ULL, 0});
    v.push_back({0, 0, 0, ~0ULL});
    // Radix 2^52 and radix 2^26 limb boundaries: 2^k - 1, 2^k, 2^k + 1.
    for (unsigned k = 26; k < 256; k += 26) {
        const W4 b = pow2(k);
        v.push_back(sub_small(b, 1));
        v.push_back(b);
        v.push_back(add_raw(b, {1, 0, 0, 0}));
    }
    // (Every radix-2^52 boundary 52k is also a multiple of 26, so the sweep
    // above covers both radixes.)
    v.push_back(pow2(248));
    v.push_back(sub_small(pow2(208), 1));
    // Top limb of each radix at its maximum: 2^256 - 2^208 and 2^256 - 2^234.
    v.push_back({0, 0, 0, 0xFFFF000000000000ULL});
    v.push_back({0, 0, 0, 0xFFFFFC0000000000ULL});
    // Just below / at / above p in the low limb.
    v.push_back(sub_small(kP, 0x1000003D1ULL));
    v.push_back(add_raw(kP, {0x1000003D0ULL, 0, 0, 0}));   // 2^256 - 1 again via p + K - 1
    return v;
}

W4 random_w4(std::mt19937_64& rng) { return {rng(), rng(), rng(), rng()}; }

void put(std::uint64_t* dst, const W4& w) { std::memcpy(dst, w.data(), sizeof(W4)); }

void put_fe(std::uint64_t* dst, const FieldElement& f) {
    const auto& l = f.limbs();
    for (int i = 0; i < 4; ++i) dst[i] = l[static_cast<std::size_t>(i)];
}

// Compare n kernel outputs (r) with expected (e); report a few mismatches.
void compare(const char* backend, const char* op, const std::vector<std::uint64_t>& r,
             const std::vector<std::uint64_t>& e, const std::vector<std::uint64_t>* a = nullptr,
             const std::vector<std::uint64_t>* b = nullptr) {
    const std::size_t n = e.size() / 4;
    int shown = 0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&r[4 * i], &e[4 * i], 32) == 0) continue;
        ++bad;
        if (shown++ < 5) {
            std::printf("  [%s] %s mismatch at %zu:\n", backend, op, i);
            if (a) std::printf("    a   = %s\n", hex(&(*a)[4 * i]).c_str());
            if (b) std::printf("    b   = %s\n", hex(&(*b)[4 * i]).c_str());
            std::printf("    got = %s\n    exp = %s\n", hex(&r[4 * i]).c_str(), hex(&e[4 * i]).c_str());
        }
    }
    char what[160];
    std::snprintf(what, sizeof what, "[%s] %s: %zu of %zu elements differ", backend, op, bad, n);
    check(bad == 0, what);
}

void compare_bytes(const char* backend, const char* op, const std::vector<std::uint8_t>& r,
                   const std::vector<std::uint8_t>& e) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < e.size(); ++i)
        if (r[i] != e[i]) {
            if (bad < 5) std::printf("  [%s] %s mismatch at %zu: got %u exp %u\n", backend, op, i, r[i], e[i]);
            ++bad;
        }
    char what[160];
    std::snprintf(what, sizeof what, "[%s] %s: %zu of %zu elements differ", backend, op, bad, e.size());
    check(bad == 0, what);
}

// ---- Dispatch helpers -----------------------------------------------------------

void test_dispatch_pure() {
    using lanes::lane_backend_bit;
    using lanes::LaneSelectionReason;
    const unsigned none = lane_backend_bit(LaneBackend::None);
    const unsigned all = none | lane_backend_bit(LaneBackend::Avx2x4) |
                         lane_backend_bit(LaneBackend::AvxIfma256x4) |
                         lane_backend_bit(LaneBackend::Ifma256x4) | lane_backend_bit(LaneBackend::Ifma512x8);
    check(lanes::default_lane_backend(all) == LaneBackend::Ifma512x8, "default: ifma512 first");
    check(lanes::default_lane_backend(none | lane_backend_bit(LaneBackend::AvxIfma256x4) |
                                      lane_backend_bit(LaneBackend::Avx2x4)) == LaneBackend::AvxIfma256x4,
          "default: avx_ifma before avx2");
    check(lanes::default_lane_backend(none | lane_backend_bit(LaneBackend::Ifma256x4) |
                                      lane_backend_bit(LaneBackend::Avx2x4)) == LaneBackend::Avx2x4,
          "default: ifma256 is never the default");
    check(lanes::default_lane_backend(none) == LaneBackend::None, "default: none");

    const unsigned avx2_only = none | lane_backend_bit(LaneBackend::Avx2x4);
    auto s = lanes::resolve_lane_selection(nullptr, avx2_only);
    check(s.backend == LaneBackend::Avx2x4 && s.reason == LaneSelectionReason::Default, "resolve: unset");
    s = lanes::resolve_lane_selection("  ", avx2_only);
    check(s.backend == LaneBackend::Avx2x4 && s.reason == LaneSelectionReason::Default, "resolve: blank");
    s = lanes::resolve_lane_selection("off", avx2_only);
    check(s.backend == LaneBackend::None && s.reason == LaneSelectionReason::Forced, "resolve: off");
    s = lanes::resolve_lane_selection(" AVX2 ", avx2_only);
    check(s.backend == LaneBackend::Avx2x4 && s.reason == LaneSelectionReason::Forced, "resolve: case/blanks");
    s = lanes::resolve_lane_selection("ifma512", avx2_only);
    check(s.backend == LaneBackend::Avx2x4 && s.requested == LaneBackend::Ifma512x8 &&
              s.reason == LaneSelectionReason::Unavailable,
          "resolve: unavailable request falls back to the default");
    s = lanes::resolve_lane_selection("avx_ifma", none);
    check(s.backend == LaneBackend::None && s.reason == LaneSelectionReason::Unavailable,
          "resolve: unavailable with nothing available");
    s = lanes::resolve_lane_selection("ifma256", all);
    check(s.backend == LaneBackend::Ifma256x4 && s.reason == LaneSelectionReason::Forced, "resolve: ifma256");
    s = lanes::resolve_lane_selection("avx512", all);
    check(s.backend == LaneBackend::Ifma512x8 && s.reason == LaneSelectionReason::UnknownToken,
          "resolve: unknown token keeps the default");

    secp256k1::CpuFeatures f{};
    check(!lanes::lane_backend_supported(LaneBackend::Avx2x4, f), "supported: avx2 needs avx2");
    check(lanes::lane_backend_supported(LaneBackend::None, f), "supported: none always");
    f.avx2 = true;
    check(lanes::lane_backend_supported(LaneBackend::Avx2x4, f), "supported: avx2");
    check(!lanes::lane_backend_supported(LaneBackend::AvxIfma256x4, f), "supported: avx_ifma needs avx_ifma");
    f.avx_ifma = true;
    check(lanes::lane_backend_supported(LaneBackend::AvxIfma256x4, f), "supported: avx_ifma");
    f.avx512f = true;
    f.avx512vl = true;
    check(!lanes::lane_backend_supported(LaneBackend::Ifma512x8, f), "supported: ifma512 needs avx512ifma");
    f.avx512ifma = true;
    check(lanes::lane_backend_supported(LaneBackend::Ifma512x8, f), "supported: ifma512");
    check(lanes::lane_backend_supported(LaneBackend::Ifma256x4, f), "supported: ifma256");
    f.avx512vl = false;
    check(!lanes::lane_backend_supported(LaneBackend::Ifma256x4, f), "supported: ifma256 needs avx512vl");

    // The live view never hands out kernels for an unavailable backend.
    const unsigned avail = lanes::available_lane_backends();
    for (unsigned b = 1; b < lanes::kLaneBackendCount; ++b) {
        const auto be = static_cast<LaneBackend>(b);
        const bool listed = (avail & lane_backend_bit(be)) != 0;
        check(listed == (lanes::lane_field_kernels(be) != nullptr), "kernels iff available");
        if (listed) check((lanes::compiled_lane_backends() & lane_backend_bit(be)) != 0, "available implies compiled");
    }
    const LaneBackend sel = lanes::selected_lane_backend();
    check(sel == LaneBackend::None || (avail & lane_backend_bit(sel)) != 0, "selection is available");
}

// ---- Per-backend differential test -----------------------------------------------

struct Inputs {
    std::vector<std::uint64_t> a, b;   // 4 limbs per element
    std::size_t n = 0;
};

Inputs make_inputs(std::mt19937_64& rng) {
    const std::vector<W4> edges = edge_values();
    Inputs in;
    auto push = [&](const W4& x, const W4& y) {
        in.a.insert(in.a.end(), x.begin(), x.end());
        in.b.insert(in.b.end(), y.begin(), y.end());
    };
    for (const W4& x : edges)
        for (const W4& y : edges) push(x, y);
    for (int i = 0; i < (1 << 16); ++i) push(random_w4(rng), random_w4(rng));
    // Random values in [p, 2^256): the non-canonical inputs load must accept.
    for (int i = 0; i < 1024; ++i) {
        bool carry = false;
        const W4 x = add_raw(kP, {rng() % 0x1000003D1ULL, 0, 0, 0}, &carry);
        push(x, random_w4(rng));
    }
    // Edges against random partners.
    for (const W4& x : edges) push(x, random_w4(rng));
    // Odd count, so every backend also runs a partial tail block.
    push({5, 0, 0, 0}, {7, 0, 0, 0});
    in.n = in.a.size() / 4;
    if (in.n % 8 == 0) push({11, 0, 0, 0}, {13, 0, 0, 0});
    in.n = in.a.size() / 4;
    return in;
}

// Internal-form limbs of value sum l_i 2^(bits i) mod p, as a FieldElement.
FieldElement value_of_limbs(const std::uint64_t* l, unsigned bits, unsigned count) {
    FieldElement acc = FieldElement::zero();
    for (unsigned i = 0; i < count; ++i) {
        const W4 p2 = pow2(bits * i);
        acc = acc + FieldElement::from_uint64(l[i]) * fe_of(p2.data());
    }
    return acc;
}

void test_limbs_at_bound(const LaneFieldKernels& k, const char* name, std::mt19937_64& rng) {
    const unsigned L = k.limb_count;
    const std::uint64_t lmax = (std::uint64_t{1} << k.limb_bits) - 1u;
    const std::uint64_t tmax = k.top_limb_max;
    std::vector<std::vector<std::uint64_t>> elems;
    auto make = [&](std::uint64_t low, std::uint64_t top) {
        std::vector<std::uint64_t> e(L, low);
        e[L - 1] = top;
        elems.push_back(e);
    };
    make(lmax, tmax);   // every limb at its bound: v = 2^256 + 2^(256-top_bits) - 1
    make(0, tmax);      // exactly 2^256
    make(lmax, 0);
    make(lmax, tmax - 1u);
    make(0, 0);
    make(1, 0);
    for (unsigned j = 0; j < L; ++j) {   // a single limb at its bound
        std::vector<std::uint64_t> e(L, 0);
        e[j] = (j + 1 == L) ? tmax : lmax;
        elems.push_back(e);
    }
    for (int i = 0; i < 4096; ++i) {
        std::vector<std::uint64_t> e(L);
        for (unsigned j = 0; j + 1 < L; ++j) {
            const std::uint64_t r = rng();
            e[j] = (r & 3u) == 0 ? lmax : (r >> 2) & lmax;   // a quarter of limbs at the bound
        }
        const std::uint64_t r = rng();
        e[L - 1] = (r & 3u) == 0 ? tmax : (r >> 2) % (tmax + 1u);
        elems.push_back(e);
    }
    if (elems.size() % 8 == 0) elems.push_back(std::vector<std::uint64_t>(L, 0));
    const std::size_t n = elems.size();
    std::vector<std::uint64_t> la(L * n), lb(L * n), r(4 * n), e(4 * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (unsigned j = 0; j < L; ++j) {
            la[L * i + j] = elems[i][j];
            lb[L * i + j] = elems[(i * 7 + 3) % n][j];
        }
    }
    std::vector<FieldElement> va(n), vb(n);
    for (std::size_t i = 0; i < n; ++i) {
        va[i] = value_of_limbs(&la[L * i], k.limb_bits, L);
        vb[i] = value_of_limbs(&lb[L * i], k.limb_bits, L);
    }
    k.normalize_limbs(r.data(), la.data(), n);
    for (std::size_t i = 0; i < n; ++i) put_fe(&e[4 * i], va[i]);
    compare(name, "normalize_limbs (at bound)", r, e);
    k.mul_limbs(r.data(), la.data(), lb.data(), n);
    for (std::size_t i = 0; i < n; ++i) put_fe(&e[4 * i], va[i] * vb[i]);
    compare(name, "mul_limbs (at bound)", r, e);
    k.sqr_limbs(r.data(), la.data(), n);
    for (std::size_t i = 0; i < n; ++i) put_fe(&e[4 * i], va[i].square());
    compare(name, "sqr_limbs (at bound)", r, e);
}

FieldElement chain_ref(FieldElement x, FieldElement y, const std::vector<std::uint8_t>& ops) {
    const FieldElement three = FieldElement::from_uint64(3);
    for (std::uint8_t op : ops) {
        switch (op & 7u) {
        case 0: x = x + y; break;
        case 1: x = x - y; break;
        case 2: x = x * y; break;
        case 3: x = x.square(); break;
        case 4: x = FieldElement::zero() - x; break;
        case 5: x = x * three; break;
        case 6: y = y + x; break;
        default: y = x * y; break;
        }
    }
    return x;
}

void test_chain(const LaneFieldKernels& k, const char* name, std::mt19937_64& rng) {
    constexpr std::size_t kSteps = 10000;
    // 37 elements: every lane of every block holds a different value, and
    // the last block is partial.
    constexpr std::size_t kN = 37;
    const std::vector<W4> edges = edge_values();
    std::vector<std::uint64_t> a(4 * kN), b(4 * kN), r(4 * kN), e(4 * kN);
    for (std::size_t i = 0; i < kN; ++i) {
        put(&a[4 * i], i < edges.size() / 2 ? edges[i] : random_w4(rng));
        put(&b[4 * i], random_w4(rng));
    }
    for (int variant = 0; variant < 3; ++variant) {
        std::vector<std::uint8_t> ops(kSteps);
        for (auto& op : ops) {
            const std::uint64_t x = rng();
            if (variant == 0) op = static_cast<std::uint8_t>(x & 7u);
            // Add/sub/negate heavy: the paths that stress carry-back to weak form.
            else if (variant == 1) {
                static constexpr std::uint8_t kAddy[] = {0, 1, 4, 5, 6, 0, 1, 2};
                op = kAddy[x & 7u];
            } else {
                static constexpr std::uint8_t kMuly[] = {2, 3, 7, 2, 3, 5, 0, 1};
                op = kMuly[x & 7u];
            }
        }
        k.chain(r.data(), a.data(), b.data(), ops.data(), ops.size(), kN);
        for (std::size_t i = 0; i < kN; ++i) put_fe(&e[4 * i], chain_ref(fe_of(&a[4 * i]), fe_of(&b[4 * i]), ops));
        char what[64];
        std::snprintf(what, sizeof what, "chain variant %d (10^4 steps)", variant);
        compare(name, what, r, e, &a, &b);
    }
}

void test_backend(const LaneFieldKernels& k, const Inputs& in, std::mt19937_64& rng) {
    const char* name = lanes::lane_backend_name(k.backend);
    std::printf("-- backend %s: %u lanes, radix 2^%u x %u limbs\n", name, k.lanes, k.limb_bits, k.limb_count);
    const std::size_t n = in.n;
    std::vector<FieldElement> fa(n), fb(n);
    for (std::size_t i = 0; i < n; ++i) {
        fa[i] = fe_of(&in.a[4 * i]);
        fb[i] = fe_of(&in.b[4 * i]);
    }
    std::vector<std::uint64_t> r(4 * n), e(4 * n);
    auto expect = [&](auto&& f) {
        for (std::size_t i = 0; i < n; ++i) put_fe(&e[4 * i], f(i));
    };

    k.load_store(r.data(), in.a.data(), n);
    expect([&](std::size_t i) { return fa[i]; });
    compare(name, "load/store", r, e, &in.a);

    k.normalize(r.data(), in.a.data(), n);
    compare(name, "normalize (a + 4p)", r, e, &in.a);

    k.add(r.data(), in.a.data(), in.b.data(), n);
    expect([&](std::size_t i) { return fa[i] + fb[i]; });
    compare(name, "add", r, e, &in.a, &in.b);

    k.sub(r.data(), in.a.data(), in.b.data(), n);
    expect([&](std::size_t i) { return fa[i] - fb[i]; });
    compare(name, "sub", r, e, &in.a, &in.b);

    k.mul(r.data(), in.a.data(), in.b.data(), n);
    expect([&](std::size_t i) { return fa[i] * fb[i]; });
    compare(name, "mul", r, e, &in.a, &in.b);

    k.sqr(r.data(), in.a.data(), n);
    expect([&](std::size_t i) { return fa[i].square(); });
    compare(name, "sqr", r, e, &in.a);

    k.negate(r.data(), in.a.data(), n);
    expect([&](std::size_t i) { return FieldElement::zero() - fa[i]; });
    compare(name, "negate", r, e, &in.a);

    for (std::uint32_t c : {0u, 1u, 2u, 3u, 4u, 7u, 8u, 977u, 0x10000u, 0xFFFFFFFFu}) {
        k.mul_small(r.data(), in.a.data(), c, n);
        const FieldElement fc = FieldElement::from_uint64(c);
        expect([&](std::size_t i) { return fa[i] * fc; });
        char what[48];
        std::snprintf(what, sizeof what, "mul_small(%u)", c);
        compare(name, what, r, e, &in.a);
    }

    k.inverse(r.data(), in.a.data(), n);
    const FieldElement zero = FieldElement::zero();
    expect([&](std::size_t i) { return fa[i] == zero ? zero : fa[i].inverse(); });
    compare(name, "inverse", r, e, &in.a);

    std::vector<std::uint8_t> rb(n), eb(n);
    k.is_zero(rb.data(), in.a.data(), n);
    std::size_t zeros = 0;
    for (std::size_t i = 0; i < n; ++i) {
        eb[i] = fa[i] == zero ? 1 : 0;
        zeros += eb[i];
    }
    compare_bytes(name, "is_zero", rb, eb);
    check(zeros >= 2, "is_zero corpus contains zero values (0 and p)");

    // equal: a against itself, against a + p when that fits in 256 bits, and
    // against an unrelated b.
    std::vector<std::uint64_t> eqb = in.b;
    for (std::size_t i = 0; i < n; ++i) {
        W4 ai;
        std::memcpy(ai.data(), &in.a[4 * i], 32);
        if (i % 3 == 0) {
            put(&eqb[4 * i], ai);
        } else if (i % 3 == 1) {
            bool carry = false;
            const W4 alias = add_raw(ai, kP, &carry);
            if (!carry) put(&eqb[4 * i], alias);
        }
    }
    k.equal(rb.data(), in.a.data(), eqb.data(), n);
    std::size_t equal_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        eb[i] = fa[i] == fe_of(&eqb[4 * i]) ? 1 : 0;
        equal_count += eb[i];
    }
    compare_bytes(name, "equal", rb, eb);
    check(equal_count > n / 3, "equal corpus contains equal pairs");

    std::vector<std::uint8_t> mask(n);
    for (auto& m : mask) m = static_cast<std::uint8_t>(rng() & 1u);
    k.select(r.data(), mask.data(), in.a.data(), in.b.data(), n);
    expect([&](std::size_t i) { return mask[i] ? fa[i] : fb[i]; });
    compare(name, "select", r, e, &in.a, &in.b);

    test_limbs_at_bound(k, name, rng);
    test_chain(k, name, rng);

    char what[96];
    std::snprintf(what, sizeof what, "[%s] limb-bound assert: %llu violations", name,
                  static_cast<unsigned long long>(k.bound_violations()));
    check(k.bound_violations() == 0, what);
}

// ---- --bench -------------------------------------------------------------------------

double ns_per(std::chrono::steady_clock::duration d, std::size_t ops) {
    return std::chrono::duration<double, std::nano>(d).count() / static_cast<double>(ops);
}

void bench(const std::vector<const LaneFieldKernels*>& ks, std::mt19937_64& rng) {
    constexpr std::size_t kN = 1024;
    constexpr std::size_t kIters = 2000;
    std::vector<std::uint64_t> a(4 * kN), b(4 * kN), r(4 * kN);
    for (std::size_t i = 0; i < kN; ++i) {
        put(&a[4 * i], random_w4(rng));
        put(&b[4 * i], random_w4(rng));
    }
    static const char* kOps[] = {"mul", "sqr", "add"};
    std::printf("\nbench: ns per element-op (n=%zu, %zu dependent ops per element; stdout only)\n", kN, kIters);
    for (const LaneFieldKernels* k : ks) {
        for (int op = 0; op < 3; ++op) {
            k->bench(op, r.data(), a.data(), b.data(), kN, kIters / 10);   // warm-up
            const auto t0 = std::chrono::steady_clock::now();
            k->bench(op, r.data(), a.data(), b.data(), kN, kIters);
            const auto t1 = std::chrono::steady_clock::now();
            std::printf("  %-9s %-4s %8.3f\n", lanes::lane_backend_name(k->backend), kOps[op],
                        ns_per(t1 - t0, kN * kIters));
        }
    }
    std::vector<FieldElement> fa(kN), fb(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        fa[i] = fe_of(&a[4 * i]);
        fb[i] = fe_of(&b[4 * i]);
    }
    std::uint64_t sink = 0;
    for (int op = 0; op < 3; ++op) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kN; ++i) {
            FieldElement x = fa[i];
            const FieldElement y = fb[i];
            if (op == 0) for (std::size_t j = 0; j < kIters; ++j) x = x * y;
            else if (op == 1) for (std::size_t j = 0; j < kIters; ++j) x = x.square();
            else for (std::size_t j = 0; j < kIters; ++j) x = x + y;
            sink ^= x.limbs()[0];
        }
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("  %-9s %-4s %8.3f\n", "scalar", kOps[op], ns_per(t1 - t0, kN * kIters));
    }
    std::printf("  (sink %llx)\n", static_cast<unsigned long long>(sink));
}

bool parse_backend(const char* s, LaneBackend* out) {
    for (unsigned b = 0; b < lanes::kLaneBackendCount; ++b) {
        const auto be = static_cast<LaneBackend>(b);
        if (std::strcmp(s, lanes::lane_backend_name(be)) == 0) {
            *out = be;
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<LaneBackend> required;
    bool want_bench = false;
    std::uint64_t seed = 0x6c616e6573ULL;   // "lanes"
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strncmp(arg, "--require=", 10) == 0) {
            LaneBackend b;
            if (!parse_backend(arg + 10, &b)) {
                std::fprintf(stderr, "test_field_lanes: unknown backend '%s' (off|avx2|avx_ifma|ifma256|ifma512)\n",
                             arg + 10);
                return 2;
            }
            required.push_back(b);
        } else if (std::strcmp(arg, "--bench") == 0) {
            want_bench = true;
        } else if (std::strncmp(arg, "--seed=", 7) == 0) {
            seed = std::strtoull(arg + 7, nullptr, 0);
        } else {
            std::fprintf(stderr, "usage: test_field_lanes [--require=<backend>]... [--bench] [--seed=<n>]\n");
            return 2;
        }
    }

    const auto& sel = lanes::lane_selection();
    std::printf("lanes: compiled=0x%x available=0x%x selected=%s (%s)\n", lanes::compiled_lane_backends(),
                lanes::available_lane_backends(), lanes::lane_backend_name(sel.backend),
                lanes::lane_selection_reason(sel.reason));

    test_dispatch_pure();

    for (LaneBackend b : required) {
        if (b == LaneBackend::None) continue;
        if (lanes::lane_field_kernels(b) == nullptr) {
            const bool compiled = (lanes::compiled_lane_backends() & lanes::lane_backend_bit(b)) != 0;
            std::printf("FAIL: --require=%s but the backend is %s\n", lanes::lane_backend_name(b),
                        compiled ? "not supported by this CPU (or masked by UFSECP_CPU_DISABLE)"
                                 : "not built into this library");
            ++g_failures;
        }
    }

    std::vector<const LaneFieldKernels*> ks;
    for (unsigned b = 1; b < lanes::kLaneBackendCount; ++b)
        if (const LaneFieldKernels* k = lanes::lane_field_kernels(static_cast<LaneBackend>(b))) ks.push_back(k);

    if (ks.empty()) {
        if (g_failures != 0) {
            std::printf("test_field_lanes: %d failure(s)\n", g_failures);
            return 1;
        }
        std::printf("test_field_lanes: no lane backend available on this CPU/build -- skipped (77)\n");
        return 77;
    }

    std::mt19937_64 rng(seed);
    const Inputs in = make_inputs(rng);
    std::printf("corpus: %zu elements (seed 0x%llx)\n", in.n, static_cast<unsigned long long>(seed));
    for (const LaneFieldKernels* k : ks) test_backend(*k, in, rng);

    if (g_failures != 0) {
        std::printf("test_field_lanes: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_field_lanes: all checks passed for %zu backend(s)\n", ks.size());
    if (want_bench) bench(ks, rng);
    return 0;
}
