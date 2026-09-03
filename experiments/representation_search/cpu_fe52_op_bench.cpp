// ===========================================================================
// FE52 per-operation cost measurement -- EXPERIMENT ONLY
//
// Why this exists: fitting per-operation costs by regression over point-formula
// timings produced an excellent fit (max error 1.65%) with NONSENSE
// coefficients -- sqr at 0.10x mul, half at 1.37x mul.  The design matrix is
// collinear: across the formulas, M and S counts move together and `half` and
// `mul_int` each appear exactly once.  A good fit does not make a collinear
// coefficient identifiable.  So measure each operation directly instead.
//
// Each operation is timed in a DEPENDENT chain where possible, because latency
// on the critical path is what a formula's dependency depth actually pays for.
// Where a dependent chain would blow the magnitude budget, the chain is broken
// at a documented interval and the reset cost is measured and subtracted.
//
// Nothing here is production code and nothing here builds by default.
// ===========================================================================

#include "secp256k1/field_52.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>

using FE = secp256k1::fast::FieldElement52;

static double now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
}

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
    b[0] &= 0x7f;
    return FE::from_bytes(b);
}

static uint64_t sink = 0;
static void consume(const FE& a) {
    FE t = a;
    t.normalize();
    uint8_t b[32];
    t.to_bytes_into(b);
    for (int i = 0; i < 32; ++i) sink = sink * 1000003ULL + b[i];
}

// Noise-robust: contention only ADDS time, so the minimum over groups is the
// best estimate of the true cost; the spread across group minima says whether
// that estimate is stable.
struct Stat { double best, second, spread_pct; };

static Stat robust(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return Stat{v[0], v.size() > 1 ? v[1] : v[0],
                v.empty() ? 0.0 : (v.back() - v[0]) / v[0] * 100.0};
}

int main(int argc, char** argv) {
    long iters = 200000;
    int passes = 15;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atol(argv[++i]);
        else if (!std::strcmp(argv[i], "--passes") && i + 1 < argc) passes = std::atoi(argv[++i]);
    }

    Rng rng(0x5EC0256B1ULL);
    const FE A = random_fe(rng);
    const FE B = random_fe(rng);

    std::vector<double> mul, sqr, add, sub, neg, mulint, half, normw, loop;

    for (int pass = 0; pass < passes; ++pass) {
        // -- empty loop: the floor everything else is measured against ------
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) { x.n[0] ^= uint64_t(i) & 1ULL; }
            loop.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- mul: dependent chain, output magnitude is always 1 -------------
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) x.mul_assign(B);
            mul.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- sqr: dependent chain, output magnitude is always 1 -------------
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) x.square_inplace();
            sqr.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- half: dependent chain; magnitude (m>>1)+1 is stable at 1 --------
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) x.half_assign();
            half.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- add: magnitude grows by 1 per add, budget is ~4096 -------------
        // Chain in blocks of 2048, resetting between blocks.  The reset is one
        // struct copy per 2048 adds, i.e. below the noise floor per operation.
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) {
                if ((i & 2047) == 0) x = A;
                x.add_assign(B);
            }
            add.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- sub: FE52 exposes NO operator-.  Every subtraction in the engine
        // is written out as a + b.negate(m), which is TWO passes over the five
        // limbs, not one.  This is the single biggest error in a naive
        // operation-count model: it charges a subtract like an add.
        {
            FE x = A;
            double t0 = now_ns();
            // B is loop-invariant, so B.negate(1) would be hoisted out of the
            // loop and this would silently measure a bare add.  Negate the
            // VARYING operand instead, which is what the engine actually does:
            // h = u2 + p.x.negate(8) negates the changing coordinate.
            for (long i = 0; i < iters; ++i) {
                if ((i & 511) == 0) x = A;
                x = x.negate(1024);
                x.add_assign(B);
            }
            sub.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- neg: magnitude grows m -> m+1, so reset frequently -------------
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) {
                if ((i & 1023) == 0) x = A;
                x.negate_assign(1024);
            }
            neg.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- mul_int(3): magnitude multiplies by 3, so reset every 7 ---------
        // The reset is a struct copy every 7 operations, which is NOT below the
        // noise floor -- this number includes it and is an UPPER BOUND.
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) {
                if ((i % 7) == 0) x = A;
                x.mul_int_assign(3);
            }
            mulint.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
        // -- normalize_weak -------------------------------------------------
        {
            FE x = A;
            double t0 = now_ns();
            for (long i = 0; i < iters; ++i) { x.add_assign(B); x.normalize_weak(); }
            normw.push_back((now_ns() - t0) / double(iters));
            consume(x);
        }
    }

    // -- instruction-level parallelism -----------------------------------
    // The numbers above are LATENCY: each operation waits for the previous one.
    // A point formula does not work that way -- independent multiplies overlap
    // in the out-of-order engine.  Measuring k independent chains at once gives
    // the throughput limit, and the ratio latency/throughput is how much ILP the
    // hardware can extract.  Without this, formula cost cannot be predicted:
    // the same operation count can cost anything between the two bounds.
    std::printf("\n%-22s %10s %10s %9s\n", "independent chains", "mul ns/op", "sqr ns/op", "speedup");
    double mul1 = 0.0, sqr1 = 0.0;
    for (int k : {1, 2, 3, 4, 6, 8}) {
        std::vector<double> mk, sk;
        for (int pass = 0; pass < passes; ++pass) {
            FE xs[8];
            for (int j = 0; j < k; ++j) { xs[j] = A; xs[j].n[0] ^= uint64_t(j); }
            double t0 = now_ns();
            for (long i = 0; i < iters / k; ++i)
                for (int j = 0; j < k; ++j) xs[j].mul_assign(B);
            mk.push_back((now_ns() - t0) / double((iters / k) * k));
            for (int j = 0; j < k; ++j) consume(xs[j]);

            for (int j = 0; j < k; ++j) { xs[j] = A; xs[j].n[0] ^= uint64_t(j); }
            t0 = now_ns();
            for (long i = 0; i < iters / k; ++i)
                for (int j = 0; j < k; ++j) xs[j].square_inplace();
            sk.push_back((now_ns() - t0) / double((iters / k) * k));
            for (int j = 0; j < k; ++j) consume(xs[j]);
        }
        Stat m = robust(mk), sq = robust(sk);
        if (k == 1) { mul1 = m.best; sqr1 = sq.best; }
        std::printf("k = %-18d %10.4f %10.4f %8.2fx\n", k, m.best, sq.best,
                    mul1 > 0 ? mul1 / m.best : 1.0);
    }

    struct Row { const char* name; std::vector<double>* v; const char* note; };
    Row rows[] = {
        {"loop floor",     &loop,   "empty loop, subtracted from nothing -- read as the noise floor"},
        {"mul",            &mul,    "dependent chain"},
        {"sqr",            &sqr,    "dependent chain"},
        {"add",            &add,    "chain, reset every 2048 (magnitude budget)"},
        {"sub (a-b)",      &sub,    "chain, reset every 1024; FE52 sub = negate + add"},
        {"neg",            &neg,    "chain, reset every 1024"},
        {"mul_int(3)",     &mulint, "reset every 7 -- UPPER BOUND, includes the reset copy"},
        {"half",           &half,   "dependent chain"},
        {"add+normalize_weak", &normw, "add followed by normalize_weak"},
    };

    std::printf("iters=%ld passes=%d\n\n", iters, passes);
    std::printf("%-22s %10s %10s %9s %8s  %s\n",
                "operation", "ns/op", "2nd best", "spread", "rel mul", "how measured");
    Stat mstat = robust(mul);
    for (const auto& r : rows) {
        Stat s = robust(*r.v);
        std::printf("%-22s %10.4f %10.4f %8.1f%% %8.3f  %s\n",
                    r.name, s.best, s.second, s.spread_pct,
                    mstat.best > 0 ? s.best / mstat.best : 0.0, r.note);
    }
    std::printf("\nsink %016llx  (printed so nothing can be optimised away)\n",
                (unsigned long long)sink);
    return 0;
}
