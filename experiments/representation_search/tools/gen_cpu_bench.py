#!/usr/bin/env python3
"""Generate a single-binary side-by-side harness for every point-formula variant.

Why one binary rather than two library builds: two builds differ in code layout,
inlining decisions and I-cache placement, none of which is the formula under
test. Putting every variant in one translation unit, driven by one loop over one
input set, removes those confounders. What remains is the formula.

Two reference points are emitted, deliberately:

  *_handwritten   the production body transcribed verbatim from point.cpp,
                  in-place ops and all
  *_generated     the SAME formula emitted by this tool's code generator, in the
                  same style as every candidate

The gap between those two is the CODE-GENERATION penalty, not a formula
difference. Every candidate is compared against *_generated, so the comparison
is like-for-like; *_handwritten tells us how much the generator itself costs.

    python3 tools/gen_cpu_bench.py > cpu_point_formula_bench.cpp
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repsearch import pointforms
from repsearch.rewrite import search

INPUT_MAGS = {"X": 7, "Y": 4, "Z": 1, "X1": 7, "Y1": 4, "Z1": 1, "X2": 1, "Y2": 1}
DECLARED = {"X": 8, "Y": 4}

HANDWRITTEN = r"""
// ---------------------------------------------------------------------------
// Reference: the production bodies, transcribed VERBATIM from
// src/cpu/src/point.cpp. In-place operations, same temporaries, same order.
// These exist to measure the code-generation penalty of the emitted variants,
// NOT to be compared against them as if they were a formula difference.
// ---------------------------------------------------------------------------

static inline void dbl_handwritten(const FE& in_x, const FE& in_y, const FE& in_z,
                                   FE& out_x, FE& out_y, FE& out_z) noexcept {
    FE x = in_x, y = in_y, z = in_z;
    FE s = y.square();
    FE l = x.square();
    l.mul_int_assign(3);
    l.half_assign();
    z.mul_assign(y);
    y = s.negate(1);
    y.mul_assign(x);
    x = l.square();
    x.add_assign(y);
    x.add_assign(y);
    s.square_inplace();
    y.add_assign(x);
    y.mul_assign(l);
    y.add_assign(s);
    y.negate_assign(2);
    out_x = x; out_y = y; out_z = z;
}

static inline void madd_handwritten(const FE& X1, const FE& Y1, const FE& Z1,
                                    const FE& X2, const FE& Y2,
                                    FE& out_x, FE& out_y, FE& out_z) noexcept {
    FE px = X1, py = Y1, pz = Z1;
    FE const zz = pz.square();
    FE const u2 = X2 * zz;
    FE s2 = Y2 * zz;
    s2.mul_assign(pz);
    FE const negX1 = px.negate(8);
    FE const h = u2 + negX1;
    pz.mul_assign(h);
    FE const negS2 = s2.negate(1);
    FE const i_val = py + negS2;
    FE h2 = h.square();
    FE const i2 = i_val.square();
    h2.negate_assign(1);
    FE h3 = h2 * h;
    FE t = px * h2;
    px = i2 + h3;
    px.add_assign(t);
    px.add_assign(t);
    t.add_assign(px);
    h3.mul_assign(py);
    py = t * i_val;
    py.add_assign(h3);
    out_x = px; out_y = py; out_z = pz;
}
"""

HARNESS = r"""
// ===========================================================================
// Correctness: every variant must agree with the handwritten reference on the
// POINT, i.e. projectively.  Different formulas legitimately land on different
// representatives of the same class, so raw coordinate identity is NOT
// required -- (X:Y:Z) ~ (L^2 X : L^3 Y : L Z) is.
// ===========================================================================

// Small constants: FE52 has no from_uint, so build from big-endian bytes.
static FE fe_small(uint64_t v) {
    uint8_t b[32] = {0};
    for (int i = 0; i < 8; ++i) b[31 - i] = uint8_t(v >> (8 * i));
    return FE::from_bytes(b);
}

static bool fe_eq(const FE& a, const FE& b) {
    FE d = a + b.negate(32);
    return d.normalizes_to_zero();
}

static bool projectively_equal(const FE& x1, const FE& y1, const FE& z1,
                               const FE& x2, const FE& y2, const FE& z2) {
    FE z1s = z1.square(), z2s = z2.square();
    FE z1c = z1s * z1, z2c = z2s * z2;
    return fe_eq(x2 * z1s, x1 * z2s) && fe_eq(y2 * z1c, y1 * z2c);
}

// SplitMix64 -- deterministic, seeded, identical across machines and runs.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

static FE random_fe(Rng& rng) {
    uint8_t b[32];
    for (int i = 0; i < 32; i += 8) {
        uint64_t v = rng.next();
        for (int j = 0; j < 8; ++j) b[i + j] = uint8_t(v >> (8 * j));
    }
    b[0] &= 0x7f;  // keep it comfortably below p
    return FE::from_bytes(b);
}

struct JPoint { FE x, y, z; };

// Build valid curve points: pick x until y^2 = x^3 + 7 has a root, then apply a
// random nonzero Z so no formula can pass by ignoring Z.
static std::vector<JPoint> make_points(size_t n, uint64_t seed) {
    Rng rng(seed);
    std::vector<JPoint> out;
    FE seven = fe_small(7);
    while (out.size() < n) {
        FE x = random_fe(rng);
        FE rhs = x.square() * x + seven;
        FE y = rhs.sqrt();
        if (!fe_eq(y.square(), rhs)) continue;   // not a residue, try again
        FE z = random_fe(rng);
        if (z.normalizes_to_zero()) continue;
        FE z2 = z.square(), z3 = z2 * z;
        out.push_back(JPoint{x * z2, y * z3, z});
    }
    return out;
}

// A checksum the optimiser cannot discard: it depends on every produced value.
static uint64_t fold(const FE& a) {
    FE t = a;
    t.normalize();
    uint8_t b[32];
    t.to_bytes_into(b);
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < 32; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

typedef void (*DblFn)(const FE&, const FE&, const FE&, FE&, FE&, FE&);
typedef void (*MaddFn)(const FE&, const FE&, const FE&, const FE&, const FE&, FE&, FE&, FE&);

struct Variant { const char* name; const char* note; DblFn dbl; MaddFn madd; };

static double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
}

// Chain of dependent doublings: this is how the primitive is actually used, and
// it is the only way the magnitude fixed point and the true critical path show
// up.  An independent-iteration loop would measure the loop, not the formula.
static double time_dbl_chain(DblFn fn, const std::vector<JPoint>& pts,
                             int chain, uint64_t& checksum) {
    double best = 1e30;
    for (const auto& p : pts) {
        FE x = p.x, y = p.y, z = p.z;
        double t0 = now_ns();
        for (int i = 0; i < chain; ++i) {
            FE nx, ny, nz;
            fn(x, y, z, nx, ny, nz);
            x = nx; y = ny; z = nz;
        }
        double dt = (now_ns() - t0) / double(chain);
        checksum = checksum * 1000003ULL + fold(x) + 31ULL * fold(y) + 961ULL * fold(z);
        if (dt < best) best = dt;
    }
    return best;
}

static double time_madd_chain(MaddFn fn, const std::vector<JPoint>& pts,
                              const FE& ax, const FE& ay, int chain,
                              uint64_t& checksum) {
    double best = 1e30;
    for (const auto& p : pts) {
        FE x = p.x, y = p.y, z = p.z;
        double t0 = now_ns();
        for (int i = 0; i < chain; ++i) {
            FE nx, ny, nz;
            fn(x, y, z, ax, ay, nx, ny, nz);
            x = nx; y = ny; z = nz;
        }
        double dt = (now_ns() - t0) / double(chain);
        checksum = checksum * 1000003ULL + fold(x) + 31ULL * fold(y) + 961ULL * fold(z);
        if (dt < best) best = dt;
    }
    return best;
}

struct Stat { double med, lo, hi; int groups; };

// Noise-robust estimator for a shared machine.
//
// Contention and frequency scaling can only ADD time to a deterministic
// CPU-bound kernel, never remove it, so the minimum of a group of passes is the
// least-contaminated estimate of the true cost.  Taking one minimum would give
// no way to tell a stable estimate from a lucky one, so the passes are split
// into groups, the minimum is taken within each group, and the SPREAD ACROSS
// GROUP MINIMA is what the verdict uses.  A tight spread means the estimate
// itself is reproducible; a wide one means the machine is too loaded to decide.
static Stat robust(const std::vector<double>& v, int groups) {
    if (groups < 1) groups = 1;
    size_t per = v.size() / size_t(groups);
    if (per == 0) { per = 1; groups = int(v.size()); }
    std::vector<double> mins;
    for (int g = 0; g < groups; ++g) {
        double m = 1e30;
        for (size_t i = size_t(g) * per; i < size_t(g + 1) * per && i < v.size(); ++i)
            m = std::min(m, v[i]);
        if (m < 1e29) mins.push_back(m);
    }
    std::sort(mins.begin(), mins.end());
    return Stat{mins[mins.size() / 2], mins.front(), mins.back(), int(mins.size())};
}
"""


def emit_variants():
    """Emit every formula through the SAME code generator, plus the best
    programs the automatic search found for each."""
    chunks = []
    registry_dbl = []
    registry_madd = []

    def add(name, slp, note):
        chunks.append(slp.to_cpp(fn_name=name, fe="FE", input_mags=INPUT_MAGS))
        return name

    for name, fn in pointforms.DOUBLING.items():
        add(name + "_generated", fn(), fn().note)
        registry_dbl.append((name + "_generated", fn().note))

    for name, fn in pointforms.MIXED_ADD.items():
        add(name + "_generated", fn(), fn().note)
        registry_madd.append((name + "_generated", fn().note))

    # Best programs the automatic search found, for the two production formulas.
    for base_name, fn, bucket in (("dbl_production", pointforms.dbl_production, registry_dbl),
                                  ("madd_production", pointforms.madd_production, registry_madd)):
        ref = fn()
        res = search(ref, INPUT_MAGS, DECLARED, depth=3, beam=40, corpus_size=12)
        emitted = 0
        for c in res.candidates:
            if not c.path or emitted >= 3:
                continue
            nm = "%s_search%d" % (base_name, emitted)
            chunks.append(c.slp.to_cpp(fn_name=nm, fe="FE", input_mags=INPUT_MAGS))
            bucket.append((nm, "auto-search: " + " > ".join(c.path)))
            emitted += 1

    return "\n\n".join(chunks), registry_dbl, registry_madd


def main():
    variants_src, reg_dbl, reg_madd = emit_variants()

    dbl_rows = ['    {"dbl_handwritten", "production, verbatim from point.cpp", dbl_handwritten, nullptr},']
    for nm, note in reg_dbl:
        dbl_rows.append('    {"%s", "%s", %s, nullptr},' % (nm, note.replace('"', "'")[:70], nm))

    madd_rows = ['    {"madd_handwritten", "production, verbatim from point.cpp", nullptr, madd_handwritten},']
    for nm, note in reg_madd:
        madd_rows.append('    {"%s", "%s", nullptr, %s},' % (nm, note.replace('"', "'")[:70], nm))

    print(r'''// ===========================================================================
// CPU point-formula representation comparison -- EXPERIMENT ONLY
//
// GENERATED by experiments/representation_search/tools/gen_cpu_bench.py.
// Do not edit by hand; regenerate.
//
// Every variant in this file is proven exactly equivalent to the reference
// group law by the Python oracle before it is emitted, and is re-checked
// projectively at runtime here against the handwritten production body.
//
// Nothing in this file is built by default and nothing here is production code.
// ===========================================================================

#include "secp256k1/field_52.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

using FE = secp256k1::fast::FieldElement52;
''')
    print(HANDWRITTEN)
    print("// --- generated variants "
          "----------------------------------------------------\n")
    print(variants_src)
    print(HARNESS)
    print("static const Variant DBL_VARIANTS[] = {")
    print("\n".join(dbl_rows))
    print("};")
    print("static const Variant MADD_VARIANTS[] = {")
    print("\n".join(madd_rows))
    print("};")
    print(r'''
int main(int argc, char** argv) {
    int points = 24, chain = 4000, passes = 9;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--points") && i + 1 < argc) points = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) chain = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--passes") && i + 1 < argc) passes = std::atoi(argv[++i]);
    }

    std::vector<JPoint> pts = make_points(size_t(points), 0x5EC0256B1ULL);
    std::printf("points=%zu chain=%d passes=%d\n\n", pts.size(), chain, passes);

    // ---- correctness before timing -------------------------------------
    int bad = 0;
    for (const auto& p : pts) {
        FE rx, ry, rz;
        dbl_handwritten(p.x, p.y, p.z, rx, ry, rz);
        for (const auto& v : DBL_VARIANTS) {
            if (!v.dbl) continue;
            FE cx, cy, cz;
            v.dbl(p.x, p.y, p.z, cx, cy, cz);
            if (!projectively_equal(rx, ry, rz, cx, cy, cz)) {
                std::printf("  MISMATCH dbl %s\n", v.name);
                ++bad;
                break;
            }
        }
    }
    FE ax = pts[0].x, ay = pts[0].y;   // affine-ish operand; Z2 == 1 by contract
    {
        FE zi = pts[0].z.inverse();
        FE zi2 = zi.square();
        ax = pts[0].x * zi2;
        ay = pts[0].y * zi2 * zi;
    }
    for (size_t k = 1; k < pts.size(); ++k) {
        const auto& p = pts[k];
        FE rx, ry, rz;
        madd_handwritten(p.x, p.y, p.z, ax, ay, rx, ry, rz);
        for (const auto& v : MADD_VARIANTS) {
            if (!v.madd) continue;
            FE cx, cy, cz;
            v.madd(p.x, p.y, p.z, ax, ay, cx, cy, cz);
            if (!projectively_equal(rx, ry, rz, cx, cy, cz)) {
                std::printf("  MISMATCH madd %s\n", v.name);
                ++bad;
                break;
            }
        }
    }
    if (bad) { std::printf("\n%d variant(s) disagree with the reference. Refusing to time.\n", bad); return 1; }
    std::printf("all variants projectively equal to the production body\n\n");

    uint64_t checksum = 0;

    // ---- warm-up -------------------------------------------------------
    // The first touch of each variant pays I-cache misses and, on a scaling
    // governor, a frequency ramp.  Those land entirely in the first pass and
    // would otherwise inflate every arm's max, making every range overlap and
    // every verdict "inconclusive" for a reason that has nothing to do with the
    // formulas.  Warm up untimed instead of discarding samples after the fact.
    for (int w = 0; w < 2; ++w) {
        for (const auto& v : DBL_VARIANTS)
            if (v.dbl) (void)time_dbl_chain(v.dbl, pts, chain / 4 + 1, checksum);
        for (const auto& v : MADD_VARIANTS)
            if (v.madd) (void)time_madd_chain(v.madd, pts, ax, ay, chain / 4 + 1, checksum);
    }

    // ---- interleaved timing --------------------------------------------
    // Every pass runs every variant, in a rotating order, so no variant sits
    // permanently in a cold or a warm slot.
    size_t nd = sizeof(DBL_VARIANTS) / sizeof(DBL_VARIANTS[0]);
    size_t nm = sizeof(MADD_VARIANTS) / sizeof(MADD_VARIANTS[0]);
    std::vector<std::vector<double>> dsamp(nd), msamp(nm);

    for (int pass = 0; pass < passes; ++pass) {
        for (size_t i = 0; i < nd; ++i) {
            size_t k = (i + size_t(pass)) % nd;
            dsamp[k].push_back(time_dbl_chain(DBL_VARIANTS[k].dbl, pts, chain, checksum));
        }
        for (size_t i = 0; i < nm; ++i) {
            size_t k = (i + size_t(pass)) % nm;
            msamp[k].push_back(time_madd_chain(MADD_VARIANTS[k].madd, pts, ax, ay, chain, checksum));
        }
    }

    auto report = [&](const char* title, const Variant* vs, size_t n,
                      std::vector<std::vector<double>>& samp) {
        std::printf("== %s ==\n", title);
        Stat base = robust(samp[0], 4);
        std::printf("  %-34s %9s %9s %9s %9s\n", "variant", "est ns", "lo", "hi", "vs ref");
        for (size_t i = 0; i < n; ++i) {
            Stat s = robust(samp[i], 4);
            double delta = (s.med - base.med) / base.med * 100.0;
            const char* verdict = "";
            if (i) {
                // A single sample has min == max, which would make every
                // comparison look decisive.  Require at least 3 samples per arm
                // before any verdict is allowed to be stated.
                if (s.groups < 3 || base.groups < 3) verdict = "  (too few samples for a verdict)";
                else if (s.hi < base.lo) verdict = "  FASTER (ranges separate)";
                else if (s.lo > base.hi) verdict = "  SLOWER (ranges separate)";
                else verdict = "  inconclusive (ranges overlap)";
            }
            std::printf("  %-34s %9.3f %9.3f %9.3f %+8.2f%%%s\n",
                        vs[i].name, s.med, s.lo, s.hi, i ? delta : 0.0, verdict);
        }
        std::printf("\n");
    };

    report("point doubling (dependent chain)", DBL_VARIANTS, nd, dsamp);
    report("mixed addition (dependent chain)", MADD_VARIANTS, nm, msamp);

    std::printf("checksum %016llx  (printed so no variant can be optimised away)\n",
                (unsigned long long)checksum);
    return 0;
}
''')


if __name__ == "__main__":
    main()
