// ============================================================================
// bench_bch_schnorr -- Bitcoin Cash 2019 Schnorr verification, single and
// batch: this engine against the libsecp256k1 fork that Knuth vendors
// (GitHub issue #466; Knuth PRs #825 and #826)
// ============================================================================
// Spec: https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/ (NOT
// BIP-340). Every input is public, so every timed path is variable-time by
// design (CLAUDE.md CT-VERIFY).
//
// Rows, for every batch size N (default 1,2,8,32,128,512,2048; --sizes):
//   ufsecp single       N x (bch_schnorr_pubkey_parse + bch_schnorr_verify)
//   ufsecp batch        N x bch_schnorr_pubkey_parse, then one
//                       bch_schnorr_batch_verify over the parsed entries
//   ufsecp C ABI batch  one ufsecp_bch_schnorr_batch_verify on the 129-byte
//                       rows (the call parses the keys itself)
//   ref single          N x (secp256k1_ec_pubkey_parse + secp256k1_schnorr_verify)
//   ref batch           N x secp256k1_ec_pubkey_parse, then one
//                       secp256k1_schnorr_verify_batch
// The ref rows exist only when CMake was given
// UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR and UFSECP_BCH_REF_SECP256K1_LIBRARY
// for a libsecp256k1 built with -DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON
// (pinned: Knuth's vendored v0.8.0, k-nuth/kth
// d1b9a8ac50cce15843c6925374709e2391e7aba4, src/secp256k1). The BCHN shim
// (compat/libsecp256k1_bchn_shim) is never linked: it exports the same
// secp256k1_schnorr_* symbols as the reference.
//
// Both batch rows time the same work: N key parses plus one batch call. The
// message and signature bytes are laid out once, before timing, for both.
//
// Inputs: N distinct keys and messages, signed by fixture_sign() below.
// Before anything is timed, every implementation must accept every set and
// reject the same set with one corrupted signature; any disagreement exits 1.
// Every calibration, warm-up and timed call is checked as well: a wrong
// verdict exits 1 and no number is reported for that run.
//
// Timing: bench_unified's method, through its harness (bench::Harness in
// src/cpu/include/secp256k1/benchmark_harness.hpp): warm-up calls, then
// --passes passes (default 11), each the mean of a calibrated number of
// calls, IQR outlier removal, median. A CPU frequency warm-up runs before the
// first row, as bench_unified.cpp cpu_frequency_warmup() does. One thread.
// ns/sig is the call time divided by N.
//
// Output: a summary table on stdout; --json PATH writes the full record
// (machine, build, method, cross-check, per-row median/min/max). This
// program never edits docs or benchmark artifacts.
//
// Usage:
//   bench_bch_schnorr [--sizes 1,2,8,32] [--passes 11] [--pass-ms 50]
//                     [--warmup CALLS] [--freq-warmup-ms 3000] [--pin]
//                     [--note TEXT] [--json PATH]
// Exit: 0 ok, 1 cross-check / verdict / I/O failure, 2 bad arguments.
// Reproduction guide: docs/BCH_SCHNORR_BATCH_BENCH.md
// ============================================================================

#if defined(UFSECP_BCH_BENCH_HAVE_REF)
// The reference headers come first. Older forks' secp256k1.h define
// SECP256K1_INLINE in the public header and this engine's config.hpp defines
// its own; the reference copy is dropped here so neither redefines the other
// (nothing below uses it).
#  include "secp256k1.h"
#  include "ufsecp_bch_ref_schnorr.h"  // generated: the header declaring secp256k1_schnorr_verify_batch
#  if defined(SECP256K1_INLINE)
#    undef SECP256K1_INLINE
#  endif
#endif

#include "secp256k1/bch_schnorr.hpp"
#include "secp256k1/cpu_features.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/lanes.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/sha256.hpp"

#if defined(UFSECP_BCH_BENCH_HAVE_CABI)
#  include "ufsecp/ufsecp.h"
#endif

#include "bench_bch_schnorr_build_info.h"  // generated: configure-time build strings

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#if defined(__linux__)
#  include <sched.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#  include <sys/utsname.h>
#endif
#if defined(__APPLE__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif

// Last: on Windows this pulls in <windows.h> (with NOMINMAX).
#include "secp256k1/benchmark_harness.hpp"

#define UFSECP_BCH_STR2(x) #x
#define UFSECP_BCH_STR(x) UFSECP_BCH_STR2(x)

namespace {

using secp256k1::BchSchnorrBatchPointEntry;
using secp256k1::fast::FieldElement;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

constexpr std::size_t kPubLen = 33;
constexpr std::size_t kMsgLen = 32;
constexpr std::size_t kSigLen = 64;
constexpr std::size_t kRowLen = kPubLen + kMsgLen + kSigLen;  // ufsecp C ABI batch entry
constexpr std::size_t kMaxSize = std::size_t{1} << 20;        // the C ABI batch limit

constexpr const char* kPinnedRefCommit = "d1b9a8ac50cce15843c6925374709e2391e7aba4";

#if defined(UFSECP_BCH_BENCH_CONFIG)
constexpr const char* kBuildConfig = UFSECP_BCH_STR(UFSECP_BCH_BENCH_CONFIG);
#else
constexpr const char* kBuildConfig = "";
#endif

// ---- options ---------------------------------------------------------------

struct Options {
    std::vector<std::size_t> sizes = {1, 2, 8, 32, 128, 512, 2048};
    int passes = 11;
    double pass_ms = 50.0;
    long warmup = -1;            // warm-up calls per row; -1 = one pass worth
    long freq_warmup_ms = 3000;  // bench_unified warms up for 3 s as well
    bool pin = false;
    std::string note;
    std::string json_path;
};

std::string trim(const std::string& s) {
    const char* const ws = " \t\r\n";
    const std::size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return std::string();
    const std::size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Same splitter as benchmarks/gpu/bench_gpu_abi_batch.cpp.
std::vector<std::string> split_csv(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur.push_back(*p);
        }
    }
    return out;
}

bool parse_uint(const char* s, unsigned long long lo, unsigned long long hi,
                unsigned long long& out) {
    if (s == nullptr || *s < '0' || *s > '9') return false;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || v < lo || v > hi) return false;
    out = v;
    return true;
}

bool parse_real(const char* s, double lo, double hi, double& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v >= lo) || !(v <= hi)) return false;
    out = v;
    return true;
}

void print_usage(const char* argv0) {
    std::printf(
        "usage: %s [--sizes N,N,..] [--passes N] [--pass-ms MS] [--warmup CALLS]\n"
        "          [--freq-warmup-ms MS] [--pin] [--note TEXT] [--json PATH]\n"
        "  --sizes           batch sizes (default 1,2,8,32,128,512,2048; each 1..1048576)\n"
        "  --passes          timed passes per row (default 11; IQR trim, then median)\n"
        "  --pass-ms         target wall time of one pass; sets calls per pass (default 50)\n"
        "  --warmup          warm-up calls per row before the passes (default: one pass worth)\n"
        "  --freq-warmup-ms  CPU frequency warm-up before the first row (default 3000, 0 = off)\n"
        "  --pin             pin to logical CPU 0 and raise priority (for Windows; on Linux\n"
        "                    use taskset instead, --pin would move the thread to CPU 0)\n"
        "  --note TEXT       free text stored in the JSON (turbo, pinning, cooling, ...)\n"
        "  --json PATH       write the full record as JSON\n"
        "Guide: docs/BCH_SCHNORR_BATCH_BENCH.md\n",
        argv0);
}

// 0 = run, 1 = help printed, 2 = bad arguments.
int parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = nullptr;
        auto value = [&](const char* flag) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return false;
            }
            v = argv[++i];
            return true;
        };
        unsigned long long u = 0;
        if (a == "--sizes") {
            if (!value("--sizes")) return 2;
            o.sizes.clear();
            for (const std::string& t : split_csv(v)) {
                if (!parse_uint(t.c_str(), 1, kMaxSize, u)) {
                    std::fprintf(stderr, "bad batch size '%s' (1..%zu)\n", t.c_str(), kMaxSize);
                    return 2;
                }
                o.sizes.push_back(static_cast<std::size_t>(u));
            }
            if (o.sizes.empty()) {
                std::fprintf(stderr, "--sizes needs at least one size\n");
                return 2;
            }
        } else if (a == "--passes") {
            if (!value("--passes") || !parse_uint(v, 1, 1001, u)) {
                std::fprintf(stderr, "--passes needs an integer in 1..1001\n");
                return 2;
            }
            o.passes = static_cast<int>(u);
        } else if (a == "--pass-ms") {
            if (!value("--pass-ms") || !parse_real(v, 0.1, 60000.0, o.pass_ms)) {
                std::fprintf(stderr, "--pass-ms needs a number in 0.1..60000\n");
                return 2;
            }
        } else if (a == "--warmup") {
            if (!value("--warmup") || !parse_uint(v, 0, 100000000, u)) {
                std::fprintf(stderr, "--warmup needs an integer in 0..100000000\n");
                return 2;
            }
            o.warmup = static_cast<long>(u);
        } else if (a == "--freq-warmup-ms") {
            if (!value("--freq-warmup-ms") || !parse_uint(v, 0, 600000, u)) {
                std::fprintf(stderr, "--freq-warmup-ms needs an integer in 0..600000\n");
                return 2;
            }
            o.freq_warmup_ms = static_cast<long>(u);
        } else if (a == "--pin") {
            o.pin = true;
        } else if (a == "--note") {
            if (!value("--note")) return 2;
            o.note = v;
        } else if (a == "--json") {
            if (!value("--json")) return 2;
            o.json_path = v;
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 1;
        } else {
            std::fprintf(stderr, "unknown argument '%s' (try --help)\n", a.c_str());
            return 2;
        }
    }
    std::sort(o.sizes.begin(), o.sizes.end());
    o.sizes.erase(std::unique(o.sizes.begin(), o.sizes.end()), o.sizes.end());
    return 0;
}

// ---- fixture: N distinct keys and messages, BCH-signed ---------------------

// SHA256(label || index_le32 || attempt_le32): the fixture's only entropy, so
// every run signs exactly the same sets. (Pattern of stream32() in
// audit/test_exploit_bch_schnorr_batch_forgery.cpp.)
std::array<std::uint8_t, 32> stream32(const char* label, std::uint32_t index,
                                      std::uint32_t attempt) {
    secp256k1::SHA256 h;
    h.update(label, std::strlen(label));
    std::uint8_t ctr[8];
    for (int b = 0; b < 4; ++b) {
        ctr[b] = static_cast<std::uint8_t>(index >> (8 * b));
        ctr[4 + b] = static_cast<std::uint8_t>(attempt >> (8 * b));
    }
    h.update(ctr, sizeof(ctr));
    return h.finalize();
}

Scalar nonzero_scalar(const char* label, std::uint32_t index) {
    for (std::uint32_t attempt = 0;; ++attempt) {
        const Scalar s = Scalar::from_bytes(stream32(label, index, attempt));
        if (!s.is_zero()) return s;
    }
}

// p == 3 (mod 4), so sqrt(y)^2 == y iff y is a quadratic residue.
bool is_qr(const FieldElement& y) {
    if (y == FieldElement::zero()) return false;
    const FieldElement t = y.sqrt();
    return t.square() == y;
}

Scalar challenge(const std::uint8_t* r32, const std::uint8_t* pub33, const std::uint8_t* msg32) {
    secp256k1::SHA256 h;
    h.update(r32, 32);
    h.update(pub33, 33);
    h.update(msg32, 32);
    return Scalar::from_bytes(h.finalize());
}

// BCH 2019 signer for the fixture only, the construction of the spec signer in
// audit/test_exploit_bch_schnorr_batch_forgery.cpp: R = k*G, k negated when
// R.y is not a quadratic residue (R and -R share x), e = SHA256(x(R) || P33 ||
// m) mod n, s = k + e*d. Keys and nonces come from public labels, so nothing
// here is secret; ct::generator_mul is used anyway, as in the audit signer.
// Not a signing API: the production BCH signer is the BCHN shim's CT one,
// which this program must not link. Every signature it makes is checked by
// every verifier below before anything is timed.
bool fixture_sign(const Scalar& d, const std::uint8_t* pub33, const std::uint8_t* msg32,
                  std::uint32_t index, std::uint8_t* sig64) {
    for (std::uint32_t attempt = 0; attempt < 64; ++attempt) {
        Scalar k = Scalar::from_bytes(stream32("bench_bch_schnorr/nonce", index, attempt));
        if (k.is_zero()) continue;
        const Point R = secp256k1::ct::generator_mul(k);
        if (!is_qr(R.y())) k = k.negate();
        const auto rx = R.x().to_bytes();
        const Scalar e = challenge(rx.data(), pub33, msg32);
        const Scalar s = k + e * d;
        if (s.is_zero()) continue;
        const auto sb = s.to_bytes();
        std::memcpy(sig64, rx.data(), 32);
        std::memcpy(sig64 + 32, sb.data(), 32);
        return true;
    }
    return false;
}

struct Fixture {
    std::size_t n = 0;
    std::vector<std::uint8_t> pub;   // n x 33, SEC1 compressed
    std::vector<std::uint8_t> msg;   // n x 32
    std::vector<std::uint8_t> sig;   // n x 64, r || s
    std::vector<std::uint8_t> rows;  // n x 129, pub | msg | sig (ufsecp C ABI layout)
};

// The first n entries of a fixture.
struct SetView {
    std::size_t n = 0;
    const std::uint8_t* pub = nullptr;
    const std::uint8_t* msg = nullptr;
    const std::uint8_t* sig = nullptr;
    const std::uint8_t* rows = nullptr;
};

SetView view_of(const Fixture& f, std::size_t n) {
    SetView v;
    v.n = n;
    v.pub = f.pub.data();
    v.msg = f.msg.data();
    v.sig = f.sig.data();
    v.rows = f.rows.data();
    return v;
}

bool build_fixture(std::size_t n, Fixture& f) {
    f.n = n;
    f.pub.assign(n * kPubLen, 0);
    f.msg.assign(n * kMsgLen, 0);
    f.sig.assign(n * kSigLen, 0);
    f.rows.assign(n * kRowLen, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto idx = static_cast<std::uint32_t>(i);
        const Scalar d = nonzero_scalar("bench_bch_schnorr/key", idx);
        const Point P = secp256k1::ct::generator_mul(d);
        const auto pub33 = P.to_compressed();
        const auto msg32 = stream32("bench_bch_schnorr/msg", idx, 0);
        std::array<std::uint8_t, 64> sig64{};
        if (!fixture_sign(d, pub33.data(), msg32.data(), idx, sig64.data())) return false;
        std::memcpy(f.pub.data() + i * kPubLen, pub33.data(), kPubLen);
        std::memcpy(f.msg.data() + i * kMsgLen, msg32.data(), kMsgLen);
        std::memcpy(f.sig.data() + i * kSigLen, sig64.data(), kSigLen);
        std::uint8_t* row = f.rows.data() + i * kRowLen;
        std::memcpy(row, pub33.data(), kPubLen);
        std::memcpy(row + kPubLen, msg32.data(), kMsgLen);
        std::memcpy(row + kPubLen + kMsgLen, sig64.data(), kSigLen);
    }
    return true;
}

// The first n entries with signature `bad` made invalid but still well formed:
// s := s + 1 (mod n), never 0, so every verifier has to reach the equation and
// reject there rather than at a structural check.
Fixture corrupted_copy(const Fixture& f, std::size_t n, std::size_t bad) {
    Fixture c;
    c.n = n;
    c.pub.assign(f.pub.begin(), f.pub.begin() + static_cast<std::ptrdiff_t>(n * kPubLen));
    c.msg.assign(f.msg.begin(), f.msg.begin() + static_cast<std::ptrdiff_t>(n * kMsgLen));
    c.sig.assign(f.sig.begin(), f.sig.begin() + static_cast<std::ptrdiff_t>(n * kSigLen));
    c.rows.assign(f.rows.begin(), f.rows.begin() + static_cast<std::ptrdiff_t>(n * kRowLen));
    std::uint8_t* s_bytes = c.sig.data() + bad * kSigLen + 32;
    Scalar s = Scalar::from_bytes(s_bytes);
    s = s + Scalar::one();
    if (s.is_zero()) s = s + Scalar::one();
    const auto sb = s.to_bytes();
    std::memcpy(s_bytes, sb.data(), 32);
    std::memcpy(c.rows.data() + bad * kRowLen + kPubLen + kMsgLen + 32, sb.data(), 32);
    return c;
}

// ---- implementations under test --------------------------------------------

class Impl {
public:
    virtual ~Impl() = default;
    virtual const char* name() const = 0;  // row label
    virtual const char* lib() const = 0;   // "ufsecp" or "ref"
    virtual const char* api() const = 0;   // "single", "batch", "c_abi_batch"
    // Lays out what one call needs (not timed).
    virtual void prepare(const SetView& set) = 0;
    // One verification of the whole prepared set; true iff every signature is valid.
    virtual bool run() = 0;
    // Extra context for a failure message.
    virtual std::string detail() const { return std::string(); }
};

class UfsecpSingle final : public Impl {
public:
    const char* name() const override { return "ufsecp single"; }
    const char* lib() const override { return "ufsecp"; }
    const char* api() const override { return "single"; }
    void prepare(const SetView& set) override { set_ = set; }
    bool run() override {
        std::size_t good = 0;
        for (std::size_t i = 0; i < set_.n; ++i) {
            const std::uint8_t* pub = set_.pub + i * kPubLen;
            Point P;
            if (secp256k1::bch_schnorr_pubkey_parse(pub, P) &&
                secp256k1::bch_schnorr_verify(set_.sig + i * kSigLen, set_.msg + i * kMsgLen, P, pub)) {
                ++good;
            }
        }
        return good == set_.n;
    }

private:
    SetView set_;
};

class UfsecpBatch final : public Impl {
public:
    const char* name() const override { return "ufsecp batch"; }
    const char* lib() const override { return "ufsecp"; }
    const char* api() const override { return "batch"; }
    void prepare(const SetView& set) override {
        set_ = set;
        entries_.assign(set.n, BchSchnorrBatchPointEntry{});
        for (std::size_t i = 0; i < set.n; ++i) {
            std::memcpy(entries_[i].pubkey33.data(), set.pub + i * kPubLen, kPubLen);
            std::memcpy(entries_[i].message.data(), set.msg + i * kMsgLen, kMsgLen);
            std::memcpy(entries_[i].signature.data(), set.sig + i * kSigLen, kSigLen);
        }
    }
    bool run() override {
        bool parsed = true;
        for (std::size_t i = 0; i < set_.n; ++i) {
            parsed = secp256k1::bch_schnorr_pubkey_parse(set_.pub + i * kPubLen, entries_[i].point) && parsed;
        }
        return parsed && secp256k1::bch_schnorr_batch_verify(entries_.data(), entries_.size(), nullptr);
    }

private:
    SetView set_;
    std::vector<BchSchnorrBatchPointEntry> entries_;
};

#if defined(UFSECP_BCH_BENCH_HAVE_CABI)
class UfsecpAbiBatch final : public Impl {
public:
    explicit UfsecpAbiBatch(ufsecp_ctx* ctx) : ctx_(ctx) {}
    const char* name() const override { return "ufsecp C ABI batch"; }
    const char* lib() const override { return "ufsecp"; }
    const char* api() const override { return "c_abi_batch"; }
    void prepare(const SetView& set) override { set_ = set; }
    bool run() override {
        last_ = ufsecp_bch_schnorr_batch_verify(ctx_, set_.rows, set_.n);
        return last_ == UFSECP_OK;
    }
    std::string detail() const override {
        return " (ufsecp_error_t " + std::to_string(static_cast<long long>(last_)) + ")";
    }

private:
    ufsecp_ctx* ctx_;
    SetView set_;
    ufsecp_error_t last_ = UFSECP_OK;
};

// RAII owner of the C ABI context.
struct AbiContext {
    ufsecp_ctx* ctx = nullptr;
    AbiContext() {
        if (ufsecp_ctx_create(&ctx) != UFSECP_OK) ctx = nullptr;
    }
    ~AbiContext() {
        if (ctx != nullptr) ufsecp_ctx_destroy(ctx);
    }
    AbiContext(const AbiContext&) = delete;
    AbiContext& operator=(const AbiContext&) = delete;
};
#endif

#if defined(UFSECP_BCH_BENCH_HAVE_REF)
#  if defined(SECP256K1_CONTEXT_NONE)
#    define UFSECP_BCH_REF_CTX_FLAGS SECP256K1_CONTEXT_NONE
#  else
#    define UFSECP_BCH_REF_CTX_FLAGS SECP256K1_CONTEXT_VERIFY
#  endif

class RefSingle final : public Impl {
public:
    explicit RefSingle(const secp256k1_context* ctx) : ctx_(ctx) {}
    const char* name() const override { return "ref single"; }
    const char* lib() const override { return "ref"; }
    const char* api() const override { return "single"; }
    void prepare(const SetView& set) override { set_ = set; }
    bool run() override {
        std::size_t good = 0;
        for (std::size_t i = 0; i < set_.n; ++i) {
            secp256k1_pubkey pk;
            if (secp256k1_ec_pubkey_parse(ctx_, &pk, set_.pub + i * kPubLen, kPubLen) == 1 &&
                secp256k1_schnorr_verify(ctx_, set_.sig + i * kSigLen, set_.msg + i * kMsgLen, &pk) == 1) {
                ++good;
            }
        }
        return good == set_.n;
    }

private:
    const secp256k1_context* ctx_;
    SetView set_;
};

class RefBatch final : public Impl {
public:
    explicit RefBatch(const secp256k1_context* ctx) : ctx_(ctx) {}
    const char* name() const override { return "ref batch"; }
    const char* lib() const override { return "ref"; }
    const char* api() const override { return "batch"; }
    void prepare(const SetView& set) override {
        set_ = set;
        keys_.assign(set.n, secp256k1_pubkey{});
        sig_ptrs_.assign(set.n, nullptr);
        msg_ptrs_.assign(set.n, nullptr);
        key_ptrs_.assign(set.n, nullptr);
        for (std::size_t i = 0; i < set.n; ++i) {
            sig_ptrs_[i] = set.sig + i * kSigLen;
            msg_ptrs_[i] = set.msg + i * kMsgLen;
            key_ptrs_[i] = &keys_[i];
        }
    }
    bool run() override {
        bool parsed = true;
        for (std::size_t i = 0; i < set_.n; ++i) {
            parsed = (secp256k1_ec_pubkey_parse(ctx_, &keys_[i], set_.pub + i * kPubLen, kPubLen) == 1) && parsed;
        }
        return parsed && secp256k1_schnorr_verify_batch(ctx_, sig_ptrs_.data(), msg_ptrs_.data(),
                                                        key_ptrs_.data(), set_.n, nullptr) == 1;
    }

private:
    const secp256k1_context* ctx_;
    SetView set_;
    std::vector<secp256k1_pubkey> keys_;
    std::vector<const unsigned char*> sig_ptrs_;
    std::vector<const unsigned char*> msg_ptrs_;
    std::vector<const secp256k1_pubkey*> key_ptrs_;
};

// RAII owner of the reference context.
struct RefContext {
    secp256k1_context* ctx = nullptr;
    RefContext() : ctx(secp256k1_context_create(UFSECP_BCH_REF_CTX_FLAGS)) {}
    ~RefContext() {
        if (ctx != nullptr) secp256k1_context_destroy(ctx);
    }
    RefContext(const RefContext&) = delete;
    RefContext& operator=(const RefContext&) = delete;
};
#endif

// ---- cross-check and timing --------------------------------------------------

// Every implementation must accept `valid` and reject `corrupted` (the same
// set with signature `bad` broken). Prints every disagreement.
bool cross_check(const std::vector<Impl*>& impls, const SetView& valid, const SetView& corrupted,
                 std::size_t bad) {
    bool agree = true;
    for (Impl* im : impls) {
        im->prepare(valid);
        const bool accepts_valid = im->run();
        const std::string why_valid = im->detail();
        im->prepare(corrupted);
        const bool accepts_corrupted = im->run();
        if (!accepts_valid) {
            std::fprintf(stderr,
                         "[bench_bch_schnorr] CROSS-CHECK FAILED: %s rejected the valid set, N=%zu%s\n",
                         im->name(), valid.n, why_valid.c_str());
            agree = false;
        }
        if (accepts_corrupted) {
            std::fprintf(stderr,
                         "[bench_bch_schnorr] CROSS-CHECK FAILED: %s accepted the N=%zu set whose "
                         "signature %zu is corrupted\n",
                         im->name(), corrupted.n, bad);
            agree = false;
        }
    }
    return agree;
}

struct Row {
    const char* label = "";
    const char* lib = "";
    const char* api = "";
    std::size_t n = 0;
    int calls_per_pass = 0;
    int warmup_calls = 0;
    int passes = 0;
    bench::Stats call;  // ns per call, after IQR trimming
};

double per_sig(double ns_per_call, std::size_t n) {
    return ns_per_call / static_cast<double>(n);
}

double sigs_per_sec(double ns_per_call, std::size_t n) {
    return ns_per_call > 0.0 ? static_cast<double>(n) * 1.0e9 / ns_per_call : 0.0;
}

// Times `im` on `set`. Returns false, with no timing, if any call (calibration,
// warm-up or timed) rejected the all-valid set.
bool measure(Impl& im, const SetView& set, const Options& o, Row& row) {
    im.prepare(set);
    std::size_t calls = 0;
    std::size_t wrong = 0;
    auto once = [&]() {
        const bool verdict = im.run();
        bench::DoNotOptimize(verdict);
        ++calls;
        if (!verdict) ++wrong;
    };

    // Calls per pass: double a run of calls until it lasts a quarter of the
    // pass target, then scale to the target.
    const double target_ns = o.pass_ms * 1.0e6;
    double per_call_ns = 0.0;
    for (int k = 1;; k *= 2) {
        const std::uint64_t t0 = bench::Timer::now();
        for (int c = 0; c < k; ++c) once();
        const std::uint64_t t1 = bench::Timer::now();
        const double ns = bench::Timer::ticks_to_ns(t1 - t0);
        if (ns >= target_ns / 4.0 || k >= (1 << 20)) {
            per_call_ns = ns / static_cast<double>(k);
            break;
        }
    }
    // (std::min)/(std::max): immune to the min/max macros of a <windows.h>
    // that some header might include without NOMINMAX.
    const double want =
        (std::min)((std::max)(target_ns / (std::max)(per_call_ns, 1.0), 1.0), 1.0e8);
    const int iters = static_cast<int>(std::ceil(want));
    const int warm = (o.warmup >= 0) ? static_cast<int>(o.warmup) : iters;

    const bench::Harness harness(warm, static_cast<std::size_t>(o.passes));
    row.label = im.name();
    row.lib = im.lib();
    row.api = im.api();
    row.n = set.n;
    row.calls_per_pass = iters;
    row.warmup_calls = warm;
    row.passes = o.passes;
    row.call = harness.run_stats(iters, once);

    if (wrong != 0) {
        std::fprintf(stderr,
                     "[bench_bch_schnorr] %s, N=%zu: %zu of %zu calls rejected the all-valid set; "
                     "no timing reported%s\n",
                     im.name(), set.n, wrong, calls, im.detail().c_str());
        return false;
    }
    return true;
}

// CPU frequency warm-up before the first row (bench_unified.cpp
// cpu_frequency_warmup() runs 3 s of k*G; this runs verifies on the fixture).
void frequency_warmup(const Fixture& f, long ms) {
    if (ms <= 0 || f.n == 0) return;
    std::printf("CPU frequency warm-up: %ld ms of bch_schnorr_verify...", ms);
    std::fflush(stdout);
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    std::size_t i = 0;
    std::size_t done = 0;
    std::size_t valid = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count() < ms) {
        if (secp256k1::bch_schnorr_verify(f.sig.data() + i * kSigLen, f.msg.data() + i * kMsgLen,
                                          f.pub.data() + i * kPubLen)) {
            ++valid;
        }
        ++done;
        i = (i + 1) % f.n;
    }
    bench::DoNotOptimize(valid);
    std::printf(" done (%zu verifies)\n", done);
}

// ---- machine and build description -----------------------------------------

#if defined(__linux__)
std::string read_first_line(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "r");
    if (fp == nullptr) return std::string();
    char buf[256];
    std::string out;
    if (std::fgets(buf, static_cast<int>(sizeof(buf)), fp) != nullptr) out = trim(buf);
    std::fclose(fp);
    return out;
}

// "key: value" lookup in a /proc file.
std::string proc_field(const char* path, const char* key) {
    std::FILE* fp = std::fopen(path, "r");
    if (fp == nullptr) return std::string();
    const std::size_t klen = std::strlen(key);
    char line[512];
    std::string out;
    while (std::fgets(line, static_cast<int>(sizeof(line)), fp) != nullptr) {
        if (std::strncmp(line, key, klen) == 0) {
            const char* colon = std::strchr(line + klen, ':');
            if (colon != nullptr) {
                out = trim(colon + 1);
                break;
            }
        }
    }
    std::fclose(fp);
    return out;
}
#endif

std::string cpu_model() {
    // x86: CPUID brand string (already trimmed); empty on other architectures.
    const std::string brand = trim(secp256k1::cpu_features_raw().brand);
    if (!brand.empty()) return brand;
#if defined(__APPLE__)
    char buf[256] = {0};
    std::size_t len = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0 && buf[0] != '\0') {
        return trim(buf);
    }
#endif
#if defined(__linux__)
    const std::string model = proc_field("/proc/cpuinfo", "model name");
    if (!model.empty()) return model;
    const std::string hw = proc_field("/proc/cpuinfo", "Hardware");
    if (!hw.empty()) return hw;
#endif
    return "unknown";
}

std::string os_description() {
    std::string s;
#if defined(_WIN32)
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll != nullptr) {
        const FARPROC proc = GetProcAddress(ntdll, "RtlGetVersion");
        if (proc != nullptr) {
            using RtlGetVersionFn = LONG(WINAPI*)(OSVERSIONINFOW*);
            RtlGetVersionFn fn = nullptr;
            static_assert(sizeof(fn) == sizeof(proc), "function pointer size");
            std::memcpy(&fn, &proc, sizeof(fn));
            OSVERSIONINFOW vi;
            std::memset(&vi, 0, sizeof(vi));
            vi.dwOSVersionInfoSize = static_cast<DWORD>(sizeof(vi));
            if (fn(&vi) == 0) {
                char buf[96];
                std::snprintf(buf, sizeof(buf), "Windows %lu.%lu build %lu",
                              static_cast<unsigned long>(vi.dwMajorVersion),
                              static_cast<unsigned long>(vi.dwMinorVersion),
                              static_cast<unsigned long>(vi.dwBuildNumber));
                s = buf;
            }
        }
    }
    if (s.empty()) s = "Windows";
#elif defined(__unix__) || defined(__APPLE__)
    struct utsname u {};
    if (uname(&u) == 0) s = std::string(u.sysname) + " " + u.release + " " + u.machine;
#  if defined(__APPLE__)
    char ver[64] = {0};
    std::size_t len = sizeof(ver);
    if (sysctlbyname("kern.osproductversion", ver, &len, nullptr, 0) == 0 && ver[0] != '\0') {
        s = "macOS " + std::string(ver) + " (" + s + ")";
    }
#  endif
#endif
    return s.empty() ? std::string("unknown") : s;
}

const char* arch_name() {
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__riscv)
    return "riscv";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "unknown";
#endif
}

std::string compiler_name() {
    char buf[128];
#if defined(__clang__)
#  if defined(__apple_build_version__)
    std::snprintf(buf, sizeof(buf), "AppleClang %d.%d.%d", __clang_major__, __clang_minor__,
                  __clang_patchlevel__);
#  else
    std::snprintf(buf, sizeof(buf), "Clang %d.%d.%d", __clang_major__, __clang_minor__,
                  __clang_patchlevel__);
#  endif
    std::string s = buf;
#  if defined(_MSC_VER)
    s += " (clang-cl, MSVC " + std::to_string(_MSC_VER) + ")";
#  endif
    return s;
#elif defined(__GNUC__)
    std::snprintf(buf, sizeof(buf), "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    return buf;
#elif defined(_MSC_VER)
    std::snprintf(buf, sizeof(buf), "MSVC %d.%02d.%d", _MSC_VER / 100, _MSC_VER % 100,
                  _MSC_FULL_VER % 100000);
    return buf;
#else
    std::snprintf(buf, sizeof(buf), "unknown");
    return buf;
#endif
}

const char* optimized_state() {
#if defined(__OPTIMIZE__)
    return "yes";
#elif defined(__GNUC__) || defined(__clang__)
    return "no";
#else
    return "unknown";  // MSVC predefines nothing for /O2
#endif
}

bool ndebug_defined() {
#if defined(NDEBUG)
    return true;
#else
    return false;
#endif
}

std::string utc_timestamp() {
    const std::time_t t = std::time(nullptr);
    const std::tm* g = std::gmtime(&t);
    if (g == nullptr) return "unknown";
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", g) == 0) return "unknown";
    return buf;
}

std::string env_value(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
}

struct Machine {
    std::string cpu, os, affinity, turbo, turbo_source, governor, governor_cpu;
};

Machine describe_machine() {
    Machine m;
    m.cpu = cpu_model();
    m.os = os_description();
    m.affinity = "not available on this OS";
    m.turbo = "unknown";
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    std::size_t first_cpu = 0;
    bool have_first = false;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        m.affinity.clear();
        for (std::size_t c = 0; c < static_cast<std::size_t>(CPU_SETSIZE); ++c) {
            if (CPU_ISSET(c, &set)) {
                if (!m.affinity.empty()) m.affinity += ",";
                m.affinity += std::to_string(c);
                if (!have_first) {
                    first_cpu = c;
                    have_first = true;
                }
            }
        }
    }
    struct Knob {
        const char* path;
        const char* off_value;  // the value that means "turbo/boost disabled"
    };
    const Knob knobs[] = {
        {"/sys/devices/system/cpu/intel_pstate/no_turbo", "1"},
        {"/sys/devices/system/cpu/cpufreq/boost", "0"},
        {"/sys/devices/system/cpu/cpu0/cpufreq/boost", "0"},
    };
    for (const Knob& k : knobs) {
        const std::string val = read_first_line(k.path);
        if (val.empty()) continue;
        m.turbo = (val == k.off_value) ? "disabled" : "enabled";
        m.turbo_source = std::string(k.path) + " = " + val;
        break;
    }
    m.governor_cpu = std::to_string(first_cpu);
    m.governor = read_first_line("/sys/devices/system/cpu/cpu" + m.governor_cpu +
                                 "/cpufreq/scaling_governor");
#endif
    return m;
}

std::string process_threads() {
#if defined(__linux__)
    const std::string t = proc_field("/proc/self/status", "Threads");
    return t.empty() ? std::string("unknown") : t;
#else
    return "not queried on this OS";
#endif
}

std::string lane_list(unsigned mask) {
    namespace lanes = secp256k1::lanes;
    std::string s;
    for (unsigned b = 1; b < lanes::kLaneBackendCount; ++b) {
        const auto be = static_cast<lanes::LaneBackend>(b);
        if ((mask & lanes::lane_backend_bit(be)) != 0u) {
            if (!s.empty()) s += ",";
            s += lanes::lane_backend_name(be);
        }
    }
    return s.empty() ? std::string("none") : s;
}

// ---- JSON --------------------------------------------------------------------

std::string jstr(const std::string& s) {
    std::string r = "\"";
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"') {
            r += "\\\"";
        } else if (c == '\\') {
            r += "\\\\";
        } else if (c == '\n') {
            r += "\\n";
        } else if (c == '\t') {
            r += "\\t";
        } else if (c < 0x20) {
            char b[16];
            std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned>(c));
            r += b;
        } else {
            r += ch;
        }
    }
    r += "\"";
    return r;
}

std::string jnum(double v, int decimals) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof(b), "%.*f", decimals, v);
    return b;
}

std::string jopt(const std::string& s) {
    return s.empty() ? std::string("null") : jstr(s);
}

struct Report {
    std::string date, argv_line, note;
    std::string c_abi_version;
    Machine machine;
    std::string threads_after;
    std::string compiler;
    std::string lanes_compiled, lanes_available, lanes_selected, lanes_requested, lanes_reason;
    std::string lanes_verify;  // what verification calls get: "off" unless forced
    std::string env_lanes, env_cpu_disable;
    bool ref_present = false;
    std::string ref_pin_match;
    std::vector<std::string> impl_names;
    std::vector<Row> rows;
};

bool write_json(const std::string& path, const Options& o, const Report& r) {
    std::string j;
    j += "{\n";
    j += "  \"generated_by\": \"bench_bch_schnorr --json\",\n";
    j += "  \"schema\": \"ufsecp.bch_schnorr_bench.v1\",\n";
    j += "  \"issue\": \"https://github.com/shrec/UltrafastSecp256k1/issues/466\",\n";
    j += "  \"date_utc\": " + jstr(r.date) + ",\n";
    j += "  \"argv\": " + jstr(r.argv_line) + ",\n";
    j += "  \"note\": " + jopt(r.note) + ",\n";
    j += "  \"ufsecp\": {\n";
    j += "    \"version\": " + jopt(UFSECP_BCH_BENCH_VERSION) + ",\n";
    j += "    \"c_abi_version_string\": " + jopt(r.c_abi_version) + ",\n";
    j += "    \"git_describe_at_configure\": " + jopt(UFSECP_BCH_BENCH_GIT_DESCRIBE) + ",\n";
    j += "    \"git_commit_at_configure\": " + jopt(UFSECP_BCH_BENCH_GIT_COMMIT) + "\n";
    j += "  },\n";
    j += "  \"machine\": {\n";
    j += "    \"cpu\": " + jstr(r.machine.cpu) + ",\n";
    j += "    \"os\": " + jstr(r.machine.os) + ",\n";
    j += std::string("    \"arch\": ") + jstr(arch_name()) + ",\n";
    j += "    \"cpu_affinity\": " + jstr(r.machine.affinity) + ",\n";
    j += std::string("    \"pinned_by_harness\": ") + (o.pin ? "true" : "false") + ",\n";
    j += "    \"turbo\": " + jstr(r.machine.turbo) + ",\n";
    j += "    \"turbo_source\": " + jopt(r.machine.turbo_source) + ",\n";
    j += "    \"governor\": " + jopt(r.machine.governor) + ",\n";
    j += "    \"governor_of_cpu\": " + jopt(r.machine.governor_cpu) + ",\n";
    j += "    \"process_threads_after_run\": " + jstr(r.threads_after) + "\n";
    j += "  },\n";
    j += "  \"build\": {\n";
    j += "    \"compiler\": " + jstr(r.compiler) + ",\n";
    j += "    \"cmake_compiler\": " + jopt(UFSECP_BCH_BENCH_CMAKE_COMPILER) + ",\n";
    j += "    \"system\": " + jopt(UFSECP_BCH_BENCH_SYSTEM) + ",\n";
    j += "    \"config\": " + jopt(kBuildConfig[0] != '\0' ? kBuildConfig : UFSECP_BCH_BENCH_BUILD_TYPE) + ",\n";
    j += "    \"flags\": " + jopt(UFSECP_BCH_BENCH_FLAGS) + ",\n";
    j += std::string("    \"ndebug\": ") + (ndebug_defined() ? "true" : "false") + ",\n";
    j += std::string("    \"optimized\": ") + jstr(optimized_state()) + "\n";
    j += "  },\n";
    j += "  \"lanes\": {\n";
    j += "    \"compiled\": " + jstr(r.lanes_compiled) + ",\n";
    j += "    \"available\": " + jstr(r.lanes_available) + ",\n";
    j += "    \"selected\": " + jstr(r.lanes_selected) + ",\n";
    j += "    \"requested\": " + jstr(r.lanes_requested) + ",\n";
    j += "    \"reason\": " + jstr(r.lanes_reason) + ",\n";
    j += "    \"verify_backend\": " + jstr(r.lanes_verify) + ",\n";
    j += "    \"verify_backend_note\": \"lanes::lane_verify_backend(): the backend verification calls get (off unless UFSECP_LANES forces one); 'selected' is the field-kernel default pick\",\n";
    j += "    \"env_UFSECP_LANES\": " + jopt(r.env_lanes) + ",\n";
    j += "    \"env_UFSECP_CPU_DISABLE\": " + jopt(r.env_cpu_disable) + "\n";
    j += "  },\n";
    j += "  \"reference\": {\n";
    j += std::string("    \"present\": ") + (r.ref_present ? "true" : "false") + ",\n";
    j += "    \"pinned_commit\": " + jstr(kPinnedRefCommit) + ",\n";
    j += "    \"git_commit_at_configure\": " + jopt(UFSECP_BCH_BENCH_REF_GIT_COMMIT) + ",\n";
    j += "    \"matches_pin\": " + jstr(r.ref_pin_match) + ",\n";
    j += "    \"include_dir\": " + jopt(UFSECP_BCH_BENCH_REF_INCLUDE_DIR) + ",\n";
    j += "    \"header\": " + jopt(UFSECP_BCH_BENCH_REF_HEADER) + ",\n";
    j += "    \"library\": " + jopt(UFSECP_BCH_BENCH_REF_LIBRARY) + "\n";
    j += "  },\n";
    j += "  \"method\": {\n";
    j += std::string("    \"timer\": ") + jstr(bench::Timer::timer_name()) + ",\n";
    j += "    \"harness\": \"bench::Harness (src/cpu/include/secp256k1/benchmark_harness.hpp), as bench_unified\",\n";
    j += "    \"statistic\": \"per pass: mean ns over calls_per_pass calls; across passes: IQR outlier removal, then median; min/max are of the kept passes\",\n";
    j += "    \"passes\": " + std::to_string(o.passes) + ",\n";
    j += "    \"pass_target_ms\": " + jnum(o.pass_ms, 3) + ",\n";
    j += "    \"warmup_calls\": " + (o.warmup >= 0 ? std::to_string(o.warmup) : std::string("\"one pass worth\"")) + ",\n";
    j += "    \"freq_warmup_ms\": " + std::to_string(o.freq_warmup_ms) + ",\n";
    j += "    \"caller_threads\": 1,\n";
    j += "    \"ns_per_sig\": \"ns per call / N\"\n";
    j += "  },\n";
    j += "  \"cross_check\": {\n";
    j += "    \"passed\": true,\n";
    j += "    \"rule\": \"before timing, every implementation accepts every valid set and rejects the same set with signature N/2 corrupted (s := s + 1 mod n)\",\n";
    j += "    \"sizes\": [";
    for (std::size_t i = 0; i < o.sizes.size(); ++i) {
        j += (i ? ", " : "") + std::to_string(o.sizes[i]);
    }
    j += "],\n";
    j += "    \"implementations\": [";
    for (std::size_t i = 0; i < r.impl_names.size(); ++i) {
        j += (i ? ", " : "") + jstr(r.impl_names[i]);
    }
    j += "]\n";
    j += "  },\n";
    j += "  \"rows\": [\n";
    for (std::size_t i = 0; i < r.rows.size(); ++i) {
        const Row& w = r.rows[i];
        j += "    {\"impl\": " + jstr(w.label) + ", \"lib\": " + jstr(w.lib) + ", \"api\": " + jstr(w.api);
        j += ", \"n\": " + std::to_string(w.n) + ", \"status\": \"ok\"";
        j += ", \"calls_per_pass\": " + std::to_string(w.calls_per_pass);
        j += ", \"warmup_calls\": " + std::to_string(w.warmup_calls);
        j += ", \"passes\": " + std::to_string(w.passes);
        j += ", \"passes_kept\": " + std::to_string(w.call.samples);
        j += ", \"passes_trimmed\": " + std::to_string(w.call.outliers);
        j += ", \"ns_per_call\": {\"median\": " + jnum(w.call.median_ns, 1) + ", \"min\": " +
             jnum(w.call.min_ns, 1) + ", \"max\": " + jnum(w.call.max_ns, 1) + ", \"mean\": " +
             jnum(w.call.mean_ns, 1) + ", \"stddev\": " + jnum(w.call.stddev_ns, 1) + "}";
        j += ", \"ns_per_sig\": {\"median\": " + jnum(per_sig(w.call.median_ns, w.n), 3) + ", \"min\": " +
             jnum(per_sig(w.call.min_ns, w.n), 3) + ", \"max\": " + jnum(per_sig(w.call.max_ns, w.n), 3) + "}";
        j += ", \"sigs_per_sec_median\": " + jnum(sigs_per_sec(w.call.median_ns, w.n), 1) + "}";
        j += (i + 1 < r.rows.size()) ? ",\n" : "\n";
    }
    j += "  ]\n";
    j += "}\n";

    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        std::fprintf(stderr, "[bench_bch_schnorr] cannot open %s for writing\n", path.c_str());
        return false;
    }
    const bool ok = std::fwrite(j.data(), 1, j.size(), fp) == j.size();
    const bool closed = std::fclose(fp) == 0;
    if (!ok || !closed) {
        std::fprintf(stderr, "[bench_bch_schnorr] writing %s failed\n", path.c_str());
        return false;
    }
    return true;
}

void print_row(const Row& w) {
    std::printf("%7zu  %-19s %13.1f %12.1f %12.1f %13.0f   %d x %d\n", w.n, w.label,
                per_sig(w.call.median_ns, w.n), per_sig(w.call.min_ns, w.n),
                per_sig(w.call.max_ns, w.n), sigs_per_sec(w.call.median_ns, w.n), w.calls_per_pass,
                w.passes);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    const int parsed = parse_args(argc, argv, o);
    if (parsed == 1) return 0;
    if (parsed != 0) return 2;

    if (o.pin) bench::pin_thread_and_elevate();

    Report rep;
    rep.date = utc_timestamp();
    rep.note = o.note;
    for (int i = 0; i < argc; ++i) {
        if (i) rep.argv_line += " ";
        rep.argv_line += argv[i];
    }
    rep.machine = describe_machine();
    rep.compiler = compiler_name();
    {
        namespace lanes = secp256k1::lanes;
        const lanes::LaneSelection& sel = lanes::lane_selection();
        rep.lanes_compiled = lane_list(lanes::compiled_lane_backends());
        rep.lanes_available = lane_list(lanes::available_lane_backends());
        rep.lanes_selected = lanes::lane_backend_name(sel.backend);
        rep.lanes_requested = lanes::lane_backend_name(sel.requested);
        rep.lanes_reason = lanes::lane_selection_reason(sel.reason);
        // lanes_verify.cpp is compiled on every platform (None where nothing applies).
        rep.lanes_verify = lanes::lane_backend_name(lanes::lane_verify_backend());
    }
    rep.env_lanes = env_value("UFSECP_LANES");
    rep.env_cpu_disable = env_value("UFSECP_CPU_DISABLE");
#if defined(UFSECP_BCH_BENCH_HAVE_CABI)
    rep.c_abi_version = ufsecp_version_string();
#endif
#if defined(UFSECP_BCH_BENCH_HAVE_REF)
    rep.ref_present = true;
    {
        const std::string ref_git = UFSECP_BCH_BENCH_REF_GIT_COMMIT;
        rep.ref_pin_match = ref_git.empty()
                                ? "unknown (include dir is not src/secp256k1/include of a kth git checkout)"
                                : (ref_git == kPinnedRefCommit ? "yes" : "no");
    }
#else
    rep.ref_pin_match = "no reference";
#endif

    std::printf("bench_bch_schnorr -- BCH 2019 Schnorr verification, single and batch (issue #466)\n");
    std::printf("  CPU:       %s\n", rep.machine.cpu.c_str());
    std::printf("  OS:        %s (%s)\n", rep.machine.os.c_str(), arch_name());
    std::printf("  compiler:  %s, config %s\n", rep.compiler.c_str(),
                kBuildConfig[0] != '\0' ? kBuildConfig : UFSECP_BCH_BENCH_BUILD_TYPE);
    std::printf("  ufsecp:    %s (%s)\n", UFSECP_BCH_BENCH_VERSION, UFSECP_BCH_BENCH_GIT_DESCRIBE);
    std::printf("  affinity:  %s%s   turbo: %s\n", rep.machine.affinity.c_str(),
                o.pin ? " (--pin)" : "", rep.machine.turbo.c_str());
    std::printf("  lanes:     verify backend %s; field default pick %s (%s)\n", rep.lanes_verify.c_str(),
                rep.lanes_selected.c_str(), rep.lanes_reason.c_str());
#if defined(UFSECP_BCH_BENCH_HAVE_REF)
    std::printf("  reference: %s, header %s, commit %s\n", UFSECP_BCH_BENCH_REF_LIBRARY,
                UFSECP_BCH_BENCH_REF_HEADER,
                UFSECP_BCH_BENCH_REF_GIT_COMMIT[0] != '\0' ? UFSECP_BCH_BENCH_REF_GIT_COMMIT : "unknown");
    if (rep.ref_pin_match == "no") {
        std::printf("  WARNING: the reference is not the pinned commit %s\n", kPinnedRefCommit);
    }
#else
    std::printf("  reference: not configured -- no ref rows (set UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR\n"
                "             and UFSECP_BCH_REF_SECP256K1_LIBRARY, see docs/BCH_SCHNORR_BATCH_BENCH.md)\n");
#endif
#if !defined(UFSECP_BCH_BENCH_HAVE_CABI)
    std::printf("  C ABI:     ufsecp_static not built -- no \"ufsecp C ABI batch\" rows\n");
#endif
    std::printf("  timer:     %s, %d passes, ~%.1f ms per pass, IQR median\n", bench::Timer::timer_name(),
                o.passes, o.pass_ms);
    if (std::strcmp(optimized_state(), "no") == 0) {
        std::printf("  WARNING: built without optimization -- these numbers are not representative\n");
    }
    if (!ndebug_defined()) {
        std::printf("  WARNING: NDEBUG is not defined (assertions on) -- use a Release build\n");
    }
    std::fflush(stdout);

    const std::size_t n_max = o.sizes.back();
    std::fprintf(stderr, "[bench_bch_schnorr] signing %zu fixture entries...\n", n_max);
    Fixture f;
    if (!build_fixture(n_max, f)) {
        std::fprintf(stderr, "[bench_bch_schnorr] fixture signing failed\n");
        return 1;
    }

    std::vector<Impl*> impls;
    UfsecpSingle u_single;
    UfsecpBatch u_batch;
    impls.push_back(&u_single);
    impls.push_back(&u_batch);
#if defined(UFSECP_BCH_BENCH_HAVE_CABI)
    AbiContext abi;
    if (abi.ctx == nullptr) {
        std::fprintf(stderr, "[bench_bch_schnorr] ufsecp_ctx_create failed\n");
        return 1;
    }
    UfsecpAbiBatch u_abi(abi.ctx);
    impls.push_back(&u_abi);
#endif
#if defined(UFSECP_BCH_BENCH_HAVE_REF)
    RefContext ref;
    if (ref.ctx == nullptr) {
        std::fprintf(stderr, "[bench_bch_schnorr] secp256k1_context_create failed (reference)\n");
        return 1;
    }
    RefSingle r_single(ref.ctx);
    RefBatch r_batch(ref.ctx);
    impls.push_back(&r_single);
    impls.push_back(&r_batch);
#endif
    for (const Impl* im : impls) rep.impl_names.emplace_back(im->name());

    // Cross-library validity check, every size, before anything is timed.
    for (const std::size_t n : o.sizes) {
        const std::size_t bad = n / 2;
        const Fixture corrupted = corrupted_copy(f, n, bad);
        if (!cross_check(impls, view_of(f, n), view_of(corrupted, n), bad)) {
            std::fprintf(stderr, "[bench_bch_schnorr] implementations disagree; nothing was timed\n");
            return 1;
        }
    }
    std::printf("cross-check: %zu implementations x %zu sizes -- every valid set accepted, every set "
                "with one corrupted signature rejected\n",
                impls.size(), o.sizes.size());

    frequency_warmup(f, o.freq_warmup_ms);

    std::printf("\n%7s  %-19s %13s %12s %12s %13s   %s\n", "N", "implementation", "ns/sig median",
                "min", "max", "sig/s", "calls x passes");
    std::fflush(stdout);
    for (const std::size_t n : o.sizes) {
        for (Impl* im : impls) {
            Row w;
            if (!measure(*im, view_of(f, n), o, w)) return 1;
            print_row(w);
            rep.rows.push_back(w);
        }
    }
    rep.threads_after = process_threads();
    std::printf("(min/max: fastest and slowest kept pass after IQR trimming; sig/s from the median)\n");

    if (!o.json_path.empty()) {
        if (!write_json(o.json_path, o, rep)) return 1;
        std::printf("wrote %s\n", o.json_path.c_str());
    }
    return 0;
}
