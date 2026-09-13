// ============================================================================
// test_regression_metal_buffer_binding_order.cpp
// ============================================================================
// Regression: the Metal host's dispatch argument list must be in the same order
// as the kernel's [[buffer(N)]] parameters. `schnorr_verify_batch` was not.
//
// The host bound {pks, msgs, sigs, res, count} while the kernel declares
//
//     kernel void schnorr_verify_batch(
//         device const uchar *msg_hashes [[buffer(0)]],
//         device const uchar *pubkeys_x  [[buffer(1)]],
//         ...
//
// so every row was verified with the message and the x-only public key
// exchanged. BIP-340 verification of (pubkey, message) as (message, pubkey)
// fails for correct signatures, so Metal `schnorr_verify_batch` rejected every
// valid signature it was ever given. ECDSA was bound correctly and was fine.
//
// This is a false-NEGATIVE, not a false-accept: the swapped pair cannot make an
// invalid signature verify except with negligible probability, so no signature
// was ever wrongly accepted. What it did do is make GPU Schnorr batch verify
// unusable on Apple hardware.
//
// Why it survived
//   `audit/test_gpu_collect_verify_parity.cpp` cross-checks collect against
//   verify_batch per row and catches this exactly -- but only where a Metal
//   device exists, and `CI / macos (Release)` could not get that far: two audit
//   tests were leaking a fail-closed shader-path override (fixed in 5546119c)
//   and then the audit binary stopped building at all (fixed in 49d9c96c).
//   Once macOS actually ran the suite, the parity test failed on its first
//   green build with 3 of 24 checks red.
//
// What this module adds
//   The parity test needs a GPU. This one does not: it reads the shader sources
//   and the host backend and checks the two agree, on every platform, in every
//   build, with no Apple toolchain. A binding-order swap is a source-level
//   mistake and is catchable at the source level.
//
//   MBB-1  every kernel the host dispatches exists in the shader sources
//   MBB-2  the host passes exactly as many buffers as the kernel declares
//   MBB-3  the buffer at each position corresponds to the kernel parameter at
//          that position
//
// MBB-3 compares names, which are abbreviated on the host side (`buf_sigs` for
// `signatures`). Matching is prefix/token overlap first, and where the two
// vocabularies genuinely differ, an explicit alias below. That table is the
// point of review: adding an entry to it is a deliberate statement that this
// host buffer is that kernel parameter. It is not a wildcard -- `buf_pks` has
// no alias to `msg_hashes`, which is what makes the original bug fail here.
// ============================================================================

#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

namespace {

// Shader files that declare `kernel void`. Named rather than globbed: a new
// file with kernels in it must be added here, and MBB-1 fails loudly with the
// kernel name if one is missed.
const char* const kShaderFiles[] = {
    "src/metal/shaders/secp256k1_kernels.metal",
    "src/metal/shaders/secp256k1_extended.h",
    "src/metal/shaders/secp256k1_bp_gen_table.h",
};

// Every host that binds buffers to these kernels, with the pair of names it
// uses to do so.
//
// This started as a single file. `src/metal/src/metal_audit_runner.mm` was left
// out because it is "only" the diagnostic tool -- and it was the file that had
// the bug: mtl_schnorr_sign bound buffers 0..3 to a kernel that declares five,
// leaving `results [[buffer(4)]]` unbound, so every Schnorr module in the Metal
// audit failed on macOS CI while the signing path itself was fine. A checker
// that covers one caller of a shared shader is a checker that reports clean
// while a sibling is broken.
struct HostFile {
    const char* path;
    const char* get_pipeline;   // ...("KERNEL_NAME")
    const char* dispatch;       // ...(..., {a, b, c})
};

const HostFile kHostFiles[] = {
    { "src/gpu/src/gpu_backend_metal.mm",     "make_pipeline(\"", "dispatch_sync_checked" },
    { "src/metal/src/metal_audit_runner.mm",  "get_pipeline(\"",  "dispatch_sync"         },
};

bool is_ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

// The identifier ending immediately before `at` (skipping whitespace).
std::string ident_before(const std::string& s, std::size_t at) {
    std::size_t e = at;
    while (e > 0 && (s[e - 1] == ' ' || s[e - 1] == '\t' ||
                     s[e - 1] == '\n' || s[e - 1] == '\r')) --e;
    std::size_t b = e;
    while (b > 0 && is_ident_char(s[b - 1])) --b;
    return s.substr(b, e - b);
}

std::string ident_at(const std::string& s, std::size_t at) {
    std::size_t b = at;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t')) ++b;
    std::size_t e = b;
    while (e < s.size() && is_ident_char(s[e])) ++e;
    return s.substr(b, e - b);
}

struct Kernel {
    std::string name;
    std::vector<std::string> params;   // indexed by [[buffer(N)]]
};

// Parse `kernel void NAME( ... )` and pull out the identifier attached to each
// [[buffer(N)]], ordered by N. Parameter lists contain parentheses of their own
// (`[[buffer(0)]]`), so the closing paren is found by depth, not by first match.
std::vector<Kernel> parse_kernels(const std::string& src) {
    std::vector<Kernel> out;
    std::string const marker = "kernel void ";
    std::size_t pos = 0;
    while ((pos = src.find(marker, pos)) != std::string::npos) {
        std::size_t const name_at = pos + marker.size();
        std::string const name = ident_at(src, name_at);
        std::size_t const open = src.find('(', name_at);
        if (name.empty() || open == std::string::npos) { pos = name_at; continue; }

        int depth = 1;
        std::size_t i = open + 1;
        for (; i < src.size() && depth > 0; ++i) {
            if (src[i] == '(') ++depth;
            else if (src[i] == ')') --depth;
        }
        if (depth != 0) { pos = name_at; continue; }
        std::string const params = src.substr(open + 1, (i - 1) - (open + 1));

        // Collect (index, identifier) then order by index.
        std::vector<std::pair<int, std::string>> slots;
        std::string const btag = "[[buffer(";
        std::size_t bp = 0;
        while ((bp = params.find(btag, bp)) != std::string::npos) {
            std::size_t const num_at = bp + btag.size();
            std::size_t num_end = num_at;
            while (num_end < params.size() && params[num_end] >= '0' &&
                   params[num_end] <= '9') ++num_end;
            if (num_end > num_at) {
                int const idx = std::stoi(params.substr(num_at, num_end - num_at));
                slots.emplace_back(idx, ident_before(params, bp));
            }
            bp = num_at;
        }
        std::sort(slots.begin(), slots.end(),
                  [](auto const& a, auto const& b) { return a.first < b.first; });

        Kernel k;
        k.name = name;
        for (auto const& s : slots) k.params.push_back(s.second);
        if (!k.params.empty()) out.push_back(std::move(k));
        pos = i;
    }
    return out;
}

struct Site {
    std::string host;
    std::string kernel;
    std::vector<std::string> buffers;
};

// Parse `<getter>("NAME")` followed by the next `<dispatch>(..., {a, b, c})`
// before any further pipeline lookup. The two hosts spell both names
// differently (make_pipeline/dispatch_sync_checked vs
// get_pipeline/dispatch_sync), so they are parameters, not literals.
std::vector<Site> parse_dispatch_sites(const std::string& src,
                                       const std::string& getter,
                                       const std::string& dispatch) {
    std::vector<Site> out;
    std::string const mp = getter;
    std::string const ds = dispatch;
    std::size_t pos = 0;
    while ((pos = src.find(mp, pos)) != std::string::npos) {
        std::size_t const nb = pos + mp.size();
        std::size_t const ne = src.find('"', nb);
        if (ne == std::string::npos) break;
        std::string const kernel = src.substr(nb, ne - nb);

        std::size_t const next_mp = src.find(mp, ne);
        pos = ne;

        // The dispatch must be a CALL, i.e. the next non-space character after
        // the name is '('. Without this the parser also matched the function
        // name inside a comment and then swallowed an unrelated brace, turning
        // a prose mention into a bogus MBB-2 failure.
        std::size_t call = std::string::npos;
        for (std::size_t c = src.find(ds, ne); c != std::string::npos;
             c = src.find(ds, c + 1)) {
            std::size_t p = c + ds.size();
            while (p < src.size() && (src[p] == ' ' || src[p] == '\t' ||
                                      src[p] == '\n' || src[p] == '\r')) ++p;
            if (p < src.size() && src[p] == '(') { call = c; break; }
        }

        // No dispatch before the next pipeline is created. Either the pipeline
        // is used some other way, or this site encodes buffers by hand with
        // `[enc setBuffer:X offset:0 atIndex:N]` -- mtl_batch_field_inv does,
        // because the Montgomery trick needs dispatchThreadgroups. Parse that
        // form too rather than leaving the site uncovered.
        bool const no_call = (call == std::string::npos) ||
                             (next_mp != std::string::npos && call > next_mp);
        if (no_call) {
            std::size_t const limit = (next_mp == std::string::npos) ? src.size() : next_mp;
            Site s;
            s.kernel = kernel;
            std::string const sb = "setBuffer:";
            for (std::size_t b = src.find(sb, ne); b != std::string::npos && b < limit;
                 b = src.find(sb, b + 1)) {
                std::string const name = ident_at(src, b + sb.size());
                if (!name.empty()) s.buffers.push_back(name);
            }
            if (!s.buffers.empty()) out.push_back(std::move(s));
            continue;
        }

        std::size_t const brace = src.find('{', call);
        if (brace == std::string::npos) continue;
        std::size_t const close = src.find('}', brace);
        if (close == std::string::npos) continue;
        std::string const list = src.substr(brace + 1, close - brace - 1);

        Site s;
        s.kernel = kernel;
        std::size_t start = 0;
        while (start <= list.size()) {
            std::size_t const comma = list.find(',', start);
            std::string tok = list.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start);
            // "&pool_.buf_x" -> "buf_x"
            std::size_t const b = tok.find_first_not_of(" \t\n\r&");
            if (b != std::string::npos) {
                tok = tok.substr(b);
                std::size_t const e = tok.find_last_not_of(" \t\n\r");
                tok = tok.substr(0, e + 1);
                std::size_t const dot = tok.rfind('.');
                if (dot != std::string::npos) tok = tok.substr(dot + 1);
                if (!tok.empty()) s.buffers.push_back(tok);
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (!s.buffers.empty()) out.push_back(std::move(s));
    }
    return out;
}

// Lowercase, drop a leading "buf_", drop digits, drop a trailing '_'.
// `buf_pubs33` -> "pubs", `pubkeys_x32` -> "pubkeys_x".
std::string normalize(const std::string& in) {
    std::string s;
    for (char c : in) {
        if (c >= '0' && c <= '9') continue;
        s += static_cast<char>((c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c);
    }
    if (s.rfind("buf_", 0) == 0) s = s.substr(4);
    // The audit runner spells the same convention the other way round
    // (`msg_buf`, `key_buf`); strip either affix so one alias table serves both
    // hosts. Purely a naming convention -- it carries no meaning to compare.
    if (s.size() > 4 && s.compare(s.size() - 4, 4, "_buf") == 0)
        s = s.substr(0, s.size() - 4);
    while (!s.empty() && s.back() == '_') s.pop_back();
    return s;
}

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::vector<std::string> split_underscore(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '_') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Host abbreviation -> the kernel parameter names it is allowed to stand for.
// Every entry is a reviewed claim about one pairing. Deliberately narrow: no
// entry maps a key-shaped host buffer onto a message-shaped kernel parameter,
// which is what makes the schnorr_verify_batch swap fail MBB-3.
struct Alias { const char* host; const char* kernel_param; };
const Alias kAliases[] = {
    { "pubs",    "pubkeys"     },
    { "pubs",    "pubkeys_x"   },
    { "pk",      "pubkeys"     },
    { "pk",      "pubkeys_x"   },
    { "key",     "privkeys"    },
    { "cnt",     "count"       },
    { "pks",     "pubkeys"     },
    { "pks",     "pubkeys_x"   },
    { "bases",   "pubkeys"     },
    { "spend",   "pubkeys"     },
    { "sigs",    "signatures"  },
    { "msgs",    "messages"    },
    { "canary",  "ran_flag"    },
    { "inlen",   "input_len"   },
    { "nr",      "negate_r"    },
    { "nk",      "negate_key"  },
    { "partials","results"     },
    { "count",   "n"           },
    { "n",       "count"       },
    { "rx",      "proof_rx_in" },
    { "s",       "proof_s_in"  },
    { "e",       "proof_e_in"  },
    { "q",       "pubkeys_q"   },
    { "commits", "commitments" },
    { "hgen",    "h_gen"       },
    { "pt",      "plaintexts"  },
};

bool corresponds(const std::string& host_buf, const std::string& kernel_param) {
    std::string const h = normalize(host_buf);
    std::string const k = normalize(kernel_param);
    if (h.empty() || k.empty()) return false;
    if (starts_with(h, k) || starts_with(k, h)) return true;

    // Token overlap: "buf_msgs" vs "msg_hashes" share the "msg" stem.
    for (auto const& ht : split_underscore(h)) {
        for (auto const& kt : split_underscore(k)) {
            if (ht.size() < 3 && kt.size() < 3) continue;
            if (starts_with(ht, kt) || starts_with(kt, ht)) return true;
        }
    }
    for (auto const& a : kAliases)
        if (h == a.host && k == a.kernel_param) return true;
    return false;
}

const Kernel* find_kernel(const std::vector<Kernel>& ks, const std::string& name) {
    for (auto const& k : ks) if (k.name == name) return &k;
    return nullptr;
}

} // namespace

int test_regression_metal_buffer_binding_order_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: Metal dispatch order matches the kernel's [[buffer(N)]]\n");
    std::printf("======================================================================\n\n");

    std::string shaders;
    for (auto const* f : kShaderFiles) {
        std::string const s = audit_read_source_file(f);
        std::string msg = std::string("MBB-0: ") + f + " resolves from any CWD";
        CHECK(!s.empty(), msg);
        shaders += s;
        shaders += "\n";
    }

    std::vector<Site> sites;
    bool hosts_ok = true;
    for (auto const& h : kHostFiles) {
        std::string const src = audit_read_source_file(h.path);
        std::string const msg = std::string("MBB-0: ") + h.path + " resolves from any CWD";
        CHECK(!src.empty(), msg);
        if (src.empty()) { hosts_ok = false; continue; }
        auto found = parse_dispatch_sites(src, h.get_pipeline, h.dispatch);
        std::printf("  %-40s %zu dispatch site(s)\n", h.path, found.size());
        // A host that parses to nothing is a broken parser, not a clean file --
        // without this the whole file would drop out of the gate silently.
        std::string const nonempty =
            std::string("MBB-0: ") + h.path + " parsed to at least one dispatch site";
        CHECK(!found.empty(), nonempty);
        for (auto& s : found) { s.host = h.path; sites.push_back(std::move(s)); }
    }

    if (shaders.empty() || !hosts_ok) {
        std::printf("\n[regression_metal_buffer_binding_order] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    std::vector<Kernel> const kernels = parse_kernels(shaders);

    std::printf("\n  %zu kernel(s) with buffer parameters, %zu dispatch site(s) total\n\n",
                kernels.size(), sites.size());

    // A parser that silently found nothing would turn every check below into a
    // vacuous pass, which is the failure mode this module exists to prevent.
    CHECK(kernels.size() >= 20,
          "MBB-0: the shader parser found the kernels (a zero/near-zero count "
          "means the parser broke, not that the code is clean)");
    CHECK(sites.size() >= 40,
          "MBB-0: the host parsers found the dispatch sites across all hosts");

    for (auto const& s : sites) {
        const Kernel* k = find_kernel(kernels, s.kernel);
        std::string const where = " [" + s.host + "]";

        std::string msg = "MBB-1: dispatched kernel '" + s.kernel +
                          "' exists in the shader sources" + where;
        CHECK(k != nullptr, msg);
        if (!k) continue;

        msg = where + " MBB-2: '" + s.kernel + "' is dispatched with " +
              std::to_string(k->params.size()) + " buffer(s), matching its "
              "[[buffer(N)]] parameter count";
        bool const arity_ok = (s.buffers.size() == k->params.size());
        CHECK(arity_ok, msg);
        if (!arity_ok) {
            std::printf("      host passes %zu: ", s.buffers.size());
            for (auto const& b : s.buffers) std::printf("%s ", b.c_str());
            std::printf("\n      kernel wants %zu: ", k->params.size());
            for (auto const& p : k->params) std::printf("%s ", p.c_str());
            std::printf("\n");
            continue;
        }

        for (std::size_t i = 0; i < k->params.size(); ++i) {
            msg = where + " MBB-3: " + s.kernel + " buffer(" + std::to_string(i) +
                  ") -- host '" + s.buffers[i] + "' is the kernel's '" +
                  k->params[i] + "'";
            CHECK(corresponds(s.buffers[i], k->params[i]), msg);
        }
    }

    std::printf("\n[regression_metal_buffer_binding_order] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_metal_buffer_binding_order_run(); }
#endif
