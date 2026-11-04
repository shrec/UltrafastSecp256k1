/* ============================================================================
 * UltrafastSecp256k1 -- end-to-end GPU C ABI batch benchmark
 * ============================================================================
 * Measures what an application sees: one ufsecp_gpu_* call per timed sample,
 * including argument checks, host<->device transfers, device allocation (or
 * scratch reuse) and result download -- not just kernel time.
 *
 * Ops      : ecdsa_verify, schnorr_verify (BIP-340), hash160, generator_mul,
 *            ecdh, bch_schnorr_verify (Bitcoin Cash 2019 Schnorr)
 * Batches  : 1, 16, 256, 4096, 65536 (override with --sizes)
 * Backends : every available CUDA / OpenCL backend (select with --backend)
 *
 * Inputs are valid and made on the CPU through the public ufsecp_* API
 * (keys, ECDSA and BIP-340 signatures, peer keys). Every GPU result of every
 * warm-up and timed call is checked against a CPU reference; a row with any
 * wrong result is reported with status "refused" and carries no timing, and
 * the process then exits non-zero.
 *
 * Output: JSON (stdout, or --json PATH). This program writes measurements
 * only to its own output; it never edits docs or benchmark artifacts.
 *
 * Usage:
 *   bench_gpu_abi_batch [--backend cuda|opencl|all] [--sizes 1,16,256]
 *                       [--ops ecdsa_verify,ecdh] [--passes 11] [--warmup 2]
 *                       [--json out.json]
 * ============================================================================ */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"

namespace {

/* -- options ---------------------------------------------------------------- */

struct Options {
    std::string backend = "all";
    std::vector<size_t> sizes = {1, 16, 256, 4096, 65536};
    std::vector<std::string> ops = {"ecdsa_verify", "schnorr_verify", "hash160",
                                    "generator_mul", "ecdh", "bch_schnorr_verify"};
    int passes = 11;
    int warmup = 2;
    std::string json_path;
};

std::vector<std::string> split_csv(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s; ; ++p) {
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

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--backend") {
            const char* v = next("--backend"); if (!v) return false;
            o.backend = v;
            if (o.backend != "cuda" && o.backend != "opencl" && o.backend != "all") {
                std::fprintf(stderr, "--backend must be cuda, opencl or all\n");
                return false;
            }
        } else if (a == "--sizes") {
            const char* v = next("--sizes"); if (!v) return false;
            o.sizes.clear();
            for (const auto& t : split_csv(v)) {
                const unsigned long long n = std::strtoull(t.c_str(), nullptr, 10);
                if (n == 0) { std::fprintf(stderr, "bad batch size '%s'\n", t.c_str()); return false; }
                o.sizes.push_back(static_cast<size_t>(n));
            }
            if (o.sizes.empty()) return false;
        } else if (a == "--ops") {
            const char* v = next("--ops"); if (!v) return false;
            o.ops = split_csv(v);
            if (o.ops.empty()) return false;
        } else if (a == "--passes") {
            const char* v = next("--passes"); if (!v) return false;
            o.passes = std::atoi(v);
            if (o.passes < 1) return false;
        } else if (a == "--warmup") {
            const char* v = next("--warmup"); if (!v) return false;
            o.warmup = std::atoi(v);
            if (o.warmup < 0) return false;
        } else if (a == "--json") {
            const char* v = next("--json"); if (!v) return false;
            o.json_path = v;
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--backend cuda|opencl|all] [--sizes a,b,..] "
                        "[--ops op,..] [--passes N] [--warmup N] [--json PATH]\n", argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", a.c_str());
            return false;
        }
    }
    return true;
}

/* -- CPU-made inputs and references ------------------------------------------ */

void fill_deterministic(uint8_t* buf, size_t len, uint32_t seed) {
    uint32_t state = seed * 2654435761u + 0x9E3779B9u;
    for (size_t i = 0; i < len; ++i) {
        state = state * 1103515245u + 12345u;
        buf[i] = static_cast<uint8_t>((state >> 16) & 0xFF);
    }
}

struct Fixture {
    size_t n = 0;
    std::vector<uint8_t> msgs, sks, pubs, xonly, ecdsa_sigs, schnorr_sigs;
    std::vector<uint8_t> bch_sigs;                /* BCH 2019 Schnorr r || s */
    std::vector<uint8_t> peer_pubs;
    std::vector<uint8_t> ref_hash160, ref_ecdh;   /* CPU reference outputs */
};

bool make_valid_secret(uint8_t sk[32], uint32_t seed) {
    fill_deterministic(sk, 32, seed);
    sk[0] &= 0x7F;                               /* below the group order */
    sk[31] = static_cast<uint8_t>(sk[31] | 1u);  /* never zero */
    return true;
}

/* Bitcoin Cash 2019 Schnorr signature from public ufsecp_* primitives (the ABI
 * has no BCH signer): R = k*G, e = SHA256(r || P33 || m) mod n, s = e*x + k or
 * e*x - k, whichever ufsecp_bch_schnorr_verify accepts (exactly one: the R or
 * -R whose y is a quadratic residue). Bench fixture only. */
bool make_bch_sig(ufsecp_ctx* cpu, const uint8_t sk[32], const uint8_t msg[32],
                  const uint8_t pub33[33], uint32_t nonce_seed, uint8_t sig_out[64]) {
    uint8_t k[32];
    make_valid_secret(k, 0xBC000000u + nonce_seed);
    uint8_t R33[33];
    if (ufsecp_pubkey_create(cpu, k, R33) != UFSECP_OK) return false;
    uint8_t chal[97];
    std::memcpy(chal, R33 + 1, 32);
    std::memcpy(chal + 32, pub33, 33);
    std::memcpy(chal + 65, msg, 32);
    uint8_t e[32];
    if (ufsecp_sha256(chal, sizeof(chal), e) != UFSECP_OK) return false;
    uint8_t ex[32];
    std::memcpy(ex, sk, 32);
    if (ufsecp_seckey_tweak_mul(cpu, ex, e) != UFSECP_OK) return false;
    for (int c = 0; c < 2; ++c) {
        uint8_t kk[32];
        std::memcpy(kk, k, 32);
        if (c == 1 && ufsecp_seckey_negate(cpu, kk) != UFSECP_OK) return false;
        uint8_t s[32];
        std::memcpy(s, ex, 32);
        if (ufsecp_seckey_tweak_add(cpu, s, kk) != UFSECP_OK) return false;
        std::memcpy(sig_out, R33 + 1, 32);
        std::memcpy(sig_out + 32, s, 32);
        if (ufsecp_bch_schnorr_verify(cpu, msg, sig_out, pub33) == UFSECP_OK) return true;
    }
    return false;
}

bool build_fixture(ufsecp_ctx* cpu, size_t n, const Options& o, Fixture& f) {
    auto want = [&](const char* op) {
        return std::find(o.ops.begin(), o.ops.end(), op) != o.ops.end();
    };
    f.n = n;
    f.msgs.resize(n * 32);
    f.sks.resize(n * 32);
    f.pubs.resize(n * 33);
    f.xonly.resize(n * 32);
    f.ecdsa_sigs.resize(n * 64);
    f.schnorr_sigs.resize(n * 64);
    f.bch_sigs.resize(n * 64);
    f.peer_pubs.resize(n * 33);
    f.ref_hash160.resize(n * 20);
    f.ref_ecdh.resize(n * 32);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t s = static_cast<uint32_t>(i);
        uint8_t* sk  = f.sks.data() + i * 32;
        uint8_t* msg = f.msgs.data() + i * 32;
        make_valid_secret(sk, 3 * s + 1);
        fill_deterministic(msg, 32, 3 * s + 2);
        if (ufsecp_pubkey_create(cpu, sk, f.pubs.data() + i * 33) != UFSECP_OK) return false;
        if (want("schnorr_verify") &&
            ufsecp_pubkey_xonly(cpu, sk, f.xonly.data() + i * 32) != UFSECP_OK) return false;
        if (want("ecdsa_verify") &&
            ufsecp_ecdsa_sign(cpu, msg, sk, f.ecdsa_sigs.data() + i * 64) != UFSECP_OK) return false;
        if (want("schnorr_verify")) {
            uint8_t aux[32];
            fill_deterministic(aux, 32, 3 * s + 3);
            if (ufsecp_schnorr_sign(cpu, msg, sk, aux, f.schnorr_sigs.data() + i * 64) != UFSECP_OK)
                return false;
        }
        if (want("bch_schnorr_verify") &&
            !make_bch_sig(cpu, sk, msg, f.pubs.data() + i * 33, 3 * s + 4,
                          f.bch_sigs.data() + i * 64))
            return false;
        if (want("hash160") &&
            ufsecp_hash160(f.pubs.data() + i * 33, 33, f.ref_hash160.data() + i * 20) != UFSECP_OK)
            return false;
        if (want("ecdh")) {
            uint8_t peer_sk[32];
            make_valid_secret(peer_sk, 0x5EED0000u + s);
            if (ufsecp_pubkey_create(cpu, peer_sk, f.peer_pubs.data() + i * 33) != UFSECP_OK)
                return false;
            if (ufsecp_ecdh(cpu, sk, f.peer_pubs.data() + i * 33, f.ref_ecdh.data() + i * 32) != UFSECP_OK)
                return false;
        }
    }
    return true;
}

/* -- one measured row -------------------------------------------------------- */

struct Row {
    std::string backend, op, status, error;
    size_t batch = 0;
    int passes = 0;
    long long median_ns = 0, min_ns = 0, max_ns = 0;
    size_t bad_rows = 0;
};

/* Runs the op once on the first `n` fixture rows, writes into the output
 * buffers and returns the ABI status plus the number of wrong result rows. */
struct CallResult { ufsecp_error_t err = UFSECP_OK; size_t bad = 0; };

CallResult run_once(ufsecp_gpu_ctx* g, const std::string& op, const Fixture& f, size_t n,
                    std::vector<uint8_t>& out) {
    CallResult r;
    if (op == "ecdsa_verify") {
        out.assign(n, 0xEE);
        r.err = ufsecp_gpu_ecdsa_verify_batch(g, f.msgs.data(), f.pubs.data(),
                                              f.ecdsa_sigs.data(), n, out.data());
        if (r.err == UFSECP_OK)
            for (size_t i = 0; i < n; ++i) if (out[i] != 1) ++r.bad;
    } else if (op == "schnorr_verify") {
        out.assign(n, 0xEE);
        r.err = ufsecp_gpu_schnorr_verify_batch(g, f.msgs.data(), f.xonly.data(),
                                                f.schnorr_sigs.data(), n, out.data());
        if (r.err == UFSECP_OK)
            for (size_t i = 0; i < n; ++i) if (out[i] != 1) ++r.bad;
    } else if (op == "bch_schnorr_verify") {
        /* Every fixture row was accepted by ufsecp_bch_schnorr_verify at build
         * time, so the CPU reference verdict is 1 for every row. */
        out.assign(n, 0xEE);
        r.err = ufsecp_gpu_bch_schnorr_verify_batch(g, f.msgs.data(), f.pubs.data(),
                                                    f.bch_sigs.data(), n, out.data());
        if (r.err == UFSECP_OK)
            for (size_t i = 0; i < n; ++i) if (out[i] != 1) ++r.bad;
    } else if (op == "hash160") {
        out.assign(n * 20, 0xEE);
        r.err = ufsecp_gpu_hash160_pubkey_batch(g, f.pubs.data(), n, out.data());
        if (r.err == UFSECP_OK)
            for (size_t i = 0; i < n; ++i)
                if (std::memcmp(out.data() + i * 20, f.ref_hash160.data() + i * 20, 20) != 0) ++r.bad;
    } else if (op == "generator_mul") {
        out.assign(n * 33, 0xEE);
        r.err = ufsecp_gpu_generator_mul_batch(g, f.sks.data(), n, out.data());
        if (r.err == UFSECP_OK)
            for (size_t i = 0; i < n; ++i)
                if (std::memcmp(out.data() + i * 33, f.pubs.data() + i * 33, 33) != 0) ++r.bad;
    } else if (op == "ecdh") {
        out.assign(n * 32, 0xEE);
        r.err = ufsecp_gpu_ecdh_batch(g, f.sks.data(), f.peer_pubs.data(), n, out.data());
        if (r.err == UFSECP_OK)
            for (size_t i = 0; i < n; ++i)
                if (std::memcmp(out.data() + i * 32, f.ref_ecdh.data() + i * 32, 32) != 0) ++r.bad;
    } else {
        r.err = UFSECP_ERR_BAD_INPUT;
    }
    return r;
}

Row measure(ufsecp_gpu_ctx* g, const char* backend, const std::string& op,
            const Fixture& f, size_t n, const Options& o) {
    Row row;
    row.backend = backend;
    row.op = op;
    row.batch = n;
    row.passes = o.passes;
    std::vector<uint8_t> out;
    std::vector<long long> samples;
    samples.reserve(static_cast<size_t>(o.passes));

    for (int i = 0; i < o.warmup + o.passes; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const CallResult r = run_once(g, op, f, n, out);
        const auto t1 = std::chrono::steady_clock::now();
        if (r.err == UFSECP_ERR_GPU_UNSUPPORTED) {
            row.status = "unsupported";
            return row;
        }
        if (r.err != UFSECP_OK) {
            row.status = "error";
            row.error = ufsecp_gpu_error_str(r.err);
            return row;
        }
        if (r.bad != 0) {
            /* A wrong result disqualifies the whole row: no timing reported. */
            row.status = "refused";
            row.bad_rows = r.bad;
            return row;
        }
        if (i >= o.warmup)
            samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    row.min_ns = samples.front();
    row.max_ns = samples.back();
    const size_t m = samples.size();
    row.median_ns = (m % 2) ? samples[m / 2] : (samples[m / 2 - 1] + samples[m / 2]) / 2;
    row.status = "ok";
    return row;
}

/* -- JSON --------------------------------------------------------------------- */

std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') { r.push_back('\\'); r.push_back(c); }
        else if (static_cast<unsigned char>(c) < 0x20) r.push_back(' ');
        else r.push_back(c);
    }
    return r;
}

void write_json(std::FILE* fp, const Options& o, const std::vector<Row>& rows) {
    std::fprintf(fp, "{\n  \"generated_by\": \"bench_gpu_abi_batch\",\n");
    std::fprintf(fp, "  \"schema\": \"ufsecp.gpu_abi_batch_bench.v1\",\n");
    std::fprintf(fp, "  \"timing\": \"wall clock per ufsecp_gpu_* call, steady_clock\",\n");
    std::fprintf(fp, "  \"passes\": %d,\n  \"warmup\": %d,\n", o.passes, o.warmup);
    std::fprintf(fp, "  \"rows\": [\n");
    for (size_t i = 0; i < rows.size(); ++i) {
        const Row& r = rows[i];
        std::fprintf(fp, "    {\"backend\": \"%s\", \"op\": \"%s\", \"batch\": %zu, \"status\": \"%s\"",
                     json_escape(r.backend).c_str(), r.op.c_str(), r.batch, r.status.c_str());
        if (r.status == "ok") {
            const double per_item = static_cast<double>(r.median_ns) / static_cast<double>(r.batch);
            const double per_sec  = r.median_ns > 0
                ? static_cast<double>(r.batch) * 1e9 / static_cast<double>(r.median_ns) : 0.0;
            std::fprintf(fp, ", \"passes\": %d, \"median_ns_per_call\": %lld, \"min_ns_per_call\": %lld,"
                             " \"max_ns_per_call\": %lld, \"median_ns_per_item\": %.3f,"
                             " \"items_per_sec\": %.1f",
                         r.passes, r.median_ns, r.min_ns, r.max_ns, per_item, per_sec);
        } else if (r.status == "refused") {
            std::fprintf(fp, ", \"wrong_rows\": %zu", r.bad_rows);
        } else if (r.status == "error") {
            std::fprintf(fp, ", \"error\": \"%s\"", json_escape(r.error).c_str());
        }
        std::fprintf(fp, "}%s\n", (i + 1 < rows.size()) ? "," : "");
    }
    std::fprintf(fp, "  ]\n}\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 2;

    static const char* const kKnownOps[] = {"ecdsa_verify", "schnorr_verify", "hash160",
                                            "generator_mul", "ecdh", "bch_schnorr_verify"};
    for (const auto& op : o.ops) {
        bool known = false;
        for (const char* k : kKnownOps) known = known || (op == k);
        if (!known) { std::fprintf(stderr, "unknown op '%s'\n", op.c_str()); return 2; }
    }

    const size_t n_max = *std::max_element(o.sizes.begin(), o.sizes.end());

    ufsecp_ctx* cpu = nullptr;
    if (ufsecp_ctx_create(&cpu) != UFSECP_OK || !cpu) {
        std::fprintf(stderr, "CPU context creation failed\n");
        return 1;
    }
    std::fprintf(stderr, "[bench_gpu_abi_batch] building %zu CPU-made input rows...\n", n_max);
    Fixture f;
    const bool built = build_fixture(cpu, n_max, o, f);
    ufsecp_ctx_destroy(cpu);
    if (!built) {
        std::fprintf(stderr, "CPU fixture construction failed\n");
        return 1;
    }

    uint32_t ids[8] = {};
    const uint32_t nb = ufsecp_gpu_backend_count(ids, 8);
    std::vector<Row> rows;
    bool any_backend = false;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t id = ids[b];
        if (id != UFSECP_GPU_BACKEND_CUDA && id != UFSECP_GPU_BACKEND_OPENCL) continue;
        if (o.backend == "cuda"   && id != UFSECP_GPU_BACKEND_CUDA)   continue;
        if (o.backend == "opencl" && id != UFSECP_GPU_BACKEND_OPENCL) continue;
        if (!ufsecp_gpu_is_available(id)) continue;

        ufsecp_gpu_ctx* g = nullptr;
        const char* name = ufsecp_gpu_backend_name(id);
        if (ufsecp_gpu_ctx_create(&g, id, 0) != UFSECP_OK || !g) {
            std::fprintf(stderr, "[bench_gpu_abi_batch] %s: context creation failed\n", name);
            continue;
        }
        any_backend = true;
        for (const auto& op : o.ops) {
            for (size_t n : o.sizes) {
                std::fprintf(stderr, "[bench_gpu_abi_batch] %s %s n=%zu\n", name, op.c_str(), n);
                rows.push_back(measure(g, name, op, f, n, o));
            }
        }
        ufsecp_gpu_ctx_destroy(g);
    }

    if (!any_backend) {
        std::fprintf(stderr, "[bench_gpu_abi_batch] no matching GPU backend available\n");
        return 77;  /* skipped */
    }

    std::FILE* fp = stdout;
    if (!o.json_path.empty()) {
        fp = std::fopen(o.json_path.c_str(), "w");
        if (!fp) { std::fprintf(stderr, "cannot open %s\n", o.json_path.c_str()); return 1; }
    }
    write_json(fp, o, rows);
    if (fp != stdout) std::fclose(fp);

    int rc = 0;
    for (const Row& r : rows)
        if (r.status == "refused" || r.status == "error") rc = 1;
    return rc;
}
