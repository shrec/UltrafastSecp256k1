// ============================================================================
// test_regression_opencl_kernel_closure.cpp
// ============================================================================
// Regression: the OpenCL scan-only kernel embed must be self-contained.
//
// A consumer that wants only the scan half of this library's OpenCL kernels
// (a BIP-352 batch scanner, for instance) embeds six files --
//
//     secp256k1_field.cl  secp256k1_point.cl  secp256k1_gen_table_w8.cl
//     secp256k1_extended.cl  secp256k1_affine.cl  secp256k1_bip352.cl
//
// -- concatenates them, strips every #include line (clCreateProgramWithSource
// has no include path), and compiles the result at runtime. It never enqueues
// ecdsa_sign, schnorr_sign or ecdh.
//
// GitHub issue #415: at v4.5.0 that embed compiled. Then secp256k1_extended.cl
// grew four `#include "secp256k1_ct_*.cl"` lines and sign paths that call into
// them. Those four files are not in the embed set, so with includes stripped
// every ct_* symbol and CT* type is undefined -- and OpenCL does not dead-strip
// a function whose callees are unresolved, so the sign wrappers break the
// compile even though nothing calls them:
//
//     <kernel>:3348:5: error: unknown type name 'CTJacobianPoint'
//     <kernel>:3349:5: warning: implicit declaration of ct_generator_mul_impl
//
// This is issue #335 (Metal, SECP256K1_METAL_SCAN_ONLY) reproduced in OpenCL,
// and the remedy is the same: SECP256K1_OPENCL_SCAN_ONLY excludes the includes
// and everything that reaches them.
//
// What this module checks, on every platform, with no OpenCL device, no vendor
// compiler and no host preprocessor:
//
//   OKC-1  all six embed files resolve from any CWD
//   OKC-2  with the guard applied, the embed contains no ct_* identifier and
//          no CT* type name -- the property the consumer actually needs
//   OKC-3  WITHOUT the guard the embed does contain them, so OKC-2 cannot pass
//          vacuously if the guard were deleted
//   OKC-4  no #else sits at the top level of a SCAN_ONLY-guarded region, which
//          is the one shape the guard evaluator below does not model
//   OKC-5  the variable-time SafeGCD inverses (field_inv_var_impl,
//          scalar_inv_var_impl, modinv30_var_impl) are referenced by no
//          secp256k1_ct_*.cl file, no macro and no function whose name
//          contains sign, ecdh, keygen, derive or prove -- they take public
//          data only
//   OKC-6  that scan sees the real verifier call sites (positive control)
//   OKC-7  the scan rejects the inverse when it is put into the real signing
//          / ECDH code and into a real secp256k1_ct_*.cl file (negative
//          control), so OKC-5 cannot pass vacuously
//   OKC-8  Metal: the same rule for the Metal variable-time inverses
//          (field_inv_var, scalar_inv_var, modinv30_var_impl,
//          jacobian_to_affine_var) across every file in src/metal/shaders/:
//          no secp256k1_ct_*.h file, no macro and no sign / ecdh / keygen /
//          derive / prove function references them
//   OKC-9  that scan sees the real Metal verifier call sites (positive
//          control)
//   OKC-10 the Metal scan rejects the inverse when it is put into the real
//          Metal ECDH and recoverable-signing code and into a real
//          secp256k1_ct_*.h file (negative control), so OKC-8 cannot pass
//          vacuously
//
// The evaluator is deliberately not a C preprocessor. It copies every line
// except the regions opened by a directive that mentions
// SECP256K1_OPENCL_SCAN_ONLY in its excluding form, tracking #if nesting to
// find each matching #endif. That is exactly the transformation the vendor
// compiler performs on those regions, and it needs no toolchain to be present.
// ============================================================================

#include <cstdio>
#include <cstddef>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

namespace {

constexpr const char* kKernelDir = "src/opencl/kernels/";

// The consumer's embed set, in the consumer's order (GitHub issue #415).
const char* const kEmbedFiles[] = {
    "secp256k1_field.cl",
    "secp256k1_point.cl",
    "secp256k1_gen_table_w8.cl",
    "secp256k1_extended.cl",
    "secp256k1_affine.cl",
    "secp256k1_bip352.cl",
};
constexpr std::size_t kEmbedCount = sizeof(kEmbedFiles) / sizeof(kEmbedFiles[0]);

constexpr const char* kGuardMacro = "SECP256K1_OPENCL_SCAN_ONLY";

std::vector<std::string> split_lines(const std::string& src) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (pos <= src.size()) {
        std::size_t const eol = src.find('\n', pos);
        if (eol == std::string::npos) {
            if (pos < src.size()) out.push_back(src.substr(pos));
            break;
        }
        std::string line = src.substr(pos, eol - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
        pos = eol + 1;
    }
    return out;
}

// The directive word after '#', or "" when the line is not a directive.
std::string directive(const std::string& line) {
    std::size_t const h = line.find_first_not_of(" \t");
    if (h == std::string::npos || line[h] != '#') return "";
    std::size_t const b = line.find_first_not_of(" \t", h + 1);
    if (b == std::string::npos) return "";
    std::size_t const e = line.find_first_of(" \t(", b);
    return line.substr(b, (e == std::string::npos ? line.size() : e) - b);
}

bool opens_conditional(const std::string& d) {
    return d == "if" || d == "ifdef" || d == "ifndef";
}

// True for a directive that EXCLUDES its region when the guard is defined:
// "#ifndef SECP256K1_OPENCL_SCAN_ONLY", or an "#if" whose condition contains
// "!defined(SECP256K1_OPENCL_SCAN_ONLY)".
bool excludes_when_guarded(const std::string& line, const std::string& d) {
    if (line.find(kGuardMacro) == std::string::npos) return false;
    if (d == "ifndef") return true;
    if (d != "if") return false;
    return line.find("!defined(" + std::string(kGuardMacro) + ")") != std::string::npos
        || line.find("!defined (" + std::string(kGuardMacro) + ")") != std::string::npos;
}

bool is_include(const std::string& line) { return directive(line) == "include"; }

// The embed the consumer builds: six files concatenated, #include lines
// dropped. With apply_guard, regions excluded by SECP256K1_OPENCL_SCAN_ONLY
// are dropped too. stray_else counts #else directives at the top level of a
// dropped region -- a shape this evaluator does not model (OKC-4).
std::string build_embed(bool apply_guard,
                        std::vector<std::string>& unresolved,
                        int& stray_else) {
    std::string out;
    stray_else = 0;

    for (std::size_t f = 0; f < kEmbedCount; ++f) {
        std::string const src =
            audit_read_source_file((std::string(kKernelDir) + kEmbedFiles[f]).c_str());
        if (src.empty()) { unresolved.push_back(kEmbedFiles[f]); continue; }

        int skip_depth = 0;   // >0 while inside a dropped region; its nesting depth
        for (auto const& line : split_lines(src)) {
            std::string const d = directive(line);

            if (skip_depth > 0) {
                if (opens_conditional(d)) {
                    ++skip_depth;
                } else if (d == "endif") {
                    --skip_depth;
                } else if (d == "else" && skip_depth == 1) {
                    ++stray_else;
                }
                continue;                      // the whole region goes away
            }

            if (apply_guard && opens_conditional(d) && excludes_when_guarded(line, d)) {
                skip_depth = 1;
                continue;
            }
            if (is_include(line)) continue;    // the consumer strips these

            out += line;
            out += '\n';
        }
    }
    return out;
}

bool ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9') || c == '_';
}

// Whole-word occurrences of ct_<lower> identifiers and CT<Upper> type names.
// Comments count: a comment naming ct_generator_mul_impl is harmless to the
// compiler, so the scan skips // and /* */ to avoid false positives, but any
// surviving CODE reference is a hard failure.
std::vector<std::string> ct_references(const std::string& src) {
    std::vector<std::string> found;
    bool in_line_comment = false, in_block_comment = false;

    for (std::size_t i = 0; i < src.size(); ++i) {
        if (in_line_comment) { if (src[i] == '\n') in_line_comment = false; continue; }
        if (in_block_comment) {
            if (src[i] == '*' && i + 1 < src.size() && src[i + 1] == '/') {
                in_block_comment = false; ++i;
            }
            continue;
        }
        if (src[i] == '/' && i + 1 < src.size()) {
            if (src[i + 1] == '/') { in_line_comment = true; ++i; continue; }
            if (src[i + 1] == '*') { in_block_comment = true; ++i; continue; }
        }
        if (i > 0 && ident_char(src[i - 1])) continue;

        bool const lower = src.compare(i, 3, "ct_") == 0;
        bool const upper = src.compare(i, 2, "CT") == 0
                        && i + 2 < src.size() && src[i + 2] >= 'A' && src[i + 2] <= 'Z';
        if (!lower && !upper) continue;

        std::size_t j = i;
        while (j < src.size() && ident_char(src[j])) ++j;
        std::string const word = src.substr(i, j - i);
        if (word.size() > 3 || upper) {
            bool seen = false;
            for (auto const& w : found) if (w == word) { seen = true; break; }
            if (!seen) found.push_back(word);
        }
        i = j - 1;
    }
    return found;
}

// ---------------------------------------------------------------------------
// Helpers for the GH-436 static-inline guard (see the comment above
// test_regression_opencl_static_inline_link_run() below for full scope).
// ---------------------------------------------------------------------------

// True if `word` occurs in `line` as a standalone identifier -- neither
// neighbor is an identifier character. Case-sensitive, like the C/OpenCL
// keywords it is used to find.
bool contains_word(const std::string& line, const std::string& word) {
    std::size_t pos = 0;
    while ((pos = line.find(word, pos)) != std::string::npos) {
        bool const left_ok = (pos == 0) || !ident_char(line[pos - 1]);
        std::size_t const end = pos + word.size();
        bool const right_ok = (end >= line.size()) || !ident_char(line[end]);
        if (left_ok && right_ok) return true;
        pos = end;
    }
    return false;
}

// Same comment-skipping rule as ct_references() above, but blanks comment
// text instead of dropping it, so every newline (including ones inside a
// block comment) survives and split_lines() output still lines up with the
// original source's line numbers.
std::string strip_comments_keep_lines(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    bool in_line_comment = false, in_block_comment = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        char const c = src[i];
        if (in_line_comment) {
            if (c == '\n') { in_line_comment = false; out += c; }
            continue;
        }
        if (in_block_comment) {
            if (c == '*' && i + 1 < src.size() && src[i + 1] == '/') { in_block_comment = false; ++i; }
            else if (c == '\n') out += c;
            continue;
        }
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '/') { in_line_comment = true; ++i; continue; }
        if (c == '/' && i + 1 < src.size() && src[i + 1] == '*') { in_block_comment = true; ++i; continue; }
        out += c;
    }
    return out;
}

// Scans `src` (comments stripped first) line by line and appends one
// "label:line: text" entry to `offenders` for each:
//   - a non-#define line with a standalone "inline" and no standalone
//     "static" (a plain inline function declaration/definition), or
//   - a "#define FORCE_INLINE ..." line whose replacement text has no
//     standalone "static".
// Returns the number of offenders appended. This is a line-oriented text
// scan, not a preprocessor: declaration specifiers split across physical
// lines are not modeled (see the caller's comment for full scope).
int scan_bad_inline(const std::string& src, const std::string& label,
                     std::vector<std::string>& offenders) {
    std::vector<std::string> const lines = split_lines(strip_comments_keep_lines(src));
    int bad = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        std::string const& line = lines[i];
        bool is_offender;
        if (directive(line) == "define") {
            is_offender = contains_word(line, "FORCE_INLINE") && !contains_word(line, "static");
        } else {
            is_offender = contains_word(line, "inline") && !contains_word(line, "static");
        }
        if (is_offender) {
            offenders.push_back(label + ":" + std::to_string(i + 1) + ": " + line);
            ++bad;
        }
    }
    return bad;
}

// Extracts and concatenates every C++ raw-string body delimited by
// R"KERNEL( ... )KERNEL" found in `cpp_src` -- the embedded OpenCL C
// kernel_parts entries in opencl_context.cpp. A C++ raw string cannot
// contain its own closing delimiter, so the first )KERNEL" after an opening
// R"KERNEL( is always the matching terminator.
std::string extract_kernel_parts(const std::string& cpp_src) {
    static const std::string kOpen = "R\"KERNEL(";
    static const std::string kClose = ")KERNEL\"";
    std::string out;
    std::size_t pos = 0;
    while (true) {
        std::size_t const start = cpp_src.find(kOpen, pos);
        if (start == std::string::npos) break;
        std::size_t const body_start = start + kOpen.size();
        std::size_t const end = cpp_src.find(kClose, body_start);
        if (end == std::string::npos) break;
        out += cpp_src.substr(body_start, end - body_start);
        out += '\n';
        pos = end + kClose.size();
    }
    return out;
}

// Enumerates *.cl files directly under src/opencl/kernels/ (no recursion),
// resolved the same CWD-independent way audit_read_source_file() resolves
// individual files: prefer the compile-time UFSECP_SOURCE_ROOT the main
// CMake build bakes in, else fall back to a path relative to this process's
// CWD. A directory that fails to resolve simply yields no entries here --
// the caller treats an empty result as the hard failure (SIL-1).
std::vector<std::string> list_cl_kernel_files() {
    std::string const dir =
#ifdef UFSECP_SOURCE_ROOT
        std::string(UFSECP_SOURCE_ROOT) + "/" + kKernelDir;
#else
        std::string(kKernelDir);
#endif
    std::vector<std::string> names;
    std::error_code ec;
    for (auto const& entry : std::filesystem::directory_iterator(dir, ec)) {
        std::error_code file_ec;
        if (!entry.is_regular_file(file_ec)) continue;
        std::string const name = entry.path().filename().string();
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".cl") == 0) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

constexpr const char* kMetalShaderDir = "src/metal/shaders/";

// Enumerates *.h and *.metal files directly under src/metal/shaders/, resolved
// the same way as list_cl_kernel_files(). An empty result is treated by the
// caller as a hard failure (OKC-8).
std::vector<std::string> list_metal_shader_files() {
    std::string const dir =
#ifdef UFSECP_SOURCE_ROOT
        std::string(UFSECP_SOURCE_ROOT) + "/" + kMetalShaderDir;
#else
        std::string(kMetalShaderDir);
#endif
    std::vector<std::string> names;
    std::error_code ec;
    for (auto const& entry : std::filesystem::directory_iterator(dir, ec)) {
        std::error_code file_ec;
        if (!entry.is_regular_file(file_ec)) continue;
        std::string const name = entry.path().filename().string();
        bool const is_h = name.size() > 2 && name.compare(name.size() - 2, 2, ".h") == 0;
        bool const is_metal = name.size() > 6 && name.compare(name.size() - 6, 6, ".metal") == 0;
        if (is_h || is_metal) names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

// ---------------------------------------------------------------------------
// Public-only guard for the variable-time SafeGCD inverses (OKC-5..OKC-7).
//
// field_inv_var_impl / scalar_inv_var_impl / modinv30_var_impl (defined in
// secp256k1_field.cl and secp256k1_extended.cl) run in a time that depends on
// the value they invert, so only public data may reach them. CUDA fences its
// equivalents off with deleted declarations in namespace ct; OpenCL C has no
// namespaces, so this scan is the fence. With comments stripped it rejects
//   - any reference in a secp256k1_ct_*.cl file,
//   - any reference on a preprocessor line (a macro could hide a call),
//   - any reference inside a top-level function body whose name contains
//     sign, ecdh, keygen, derive or prove (case-insensitive).
// It is a direct-reference text scan, not a call graph: a secret path that
// reaches a verifier helper indirectly is not seen, which is why each helper
// that calls a _var inverse documents that its callers are all verifiers.
// The same scan guards the Metal port (OKC-8..OKC-10) with the Metal names:
// MSL has no deleted declarations to use as a fence either, and there the
// secret files are secp256k1_ct_*.h.
// ---------------------------------------------------------------------------
const std::vector<std::string> kVarInverseNames = {
    "field_inv_var_impl", "scalar_inv_var_impl", "modinv30_var_impl",
};
// Metal (src/metal/shaders/secp256k1_field.h and secp256k1_extended.h): the
// core, the two inverse wrappers and the public-point affine helper.
const std::vector<std::string> kMetalVarInverseNames = {
    "field_inv_var", "scalar_inv_var", "modinv30_var_impl", "jacobian_to_affine_var",
};
const char* const kSecretNameParts[] = { "sign", "ecdh", "keygen", "derive", "prove" };

bool references_var_inverse(const std::string& code,
                            const std::vector<std::string>& names) {
    for (auto const& n : names) {
        if (contains_word(code, n)) return true;
    }
    return false;
}

// Same line count as `src`; every preprocessor line (and its backslash
// continuations) becomes empty, so macro bodies do not count as braces.
std::string blank_directives_keep_lines(const std::string& src) {
    std::string out;
    out.reserve(src.size());
    bool continuation = false;
    for (auto const& line : split_lines(src)) {
        bool const is_dir = continuation || !directive(line).empty();
        continuation = is_dir && !line.empty() && line.back() == '\\';
        if (!is_dir) out += line;
        out += '\n';
    }
    return out;
}

// The function name in a top-level declaration header: the identifier just
// before the last '(' that opens at parenthesis depth 0 (so an
// __attribute__((...)) prefix is skipped). "" when there is no '(' -- a
// struct, an initialized array.
std::string function_name_from_header(const std::string& header) {
    int depth = 0;
    std::size_t last_open = std::string::npos;
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == '(') {
            if (depth == 0) last_open = i;
            ++depth;
        } else if (header[i] == ')' && depth > 0) {
            --depth;
        }
    }
    if (last_open == std::string::npos) return "";
    std::size_t e = last_open;
    while (e > 0 && (header[e - 1] == ' ' || header[e - 1] == '\t'
                     || header[e - 1] == '\n' || header[e - 1] == '\r')) --e;
    std::size_t b = e;
    while (b > 0 && ident_char(header[b - 1])) --b;
    return header.substr(b, e - b);
}

struct TopLevelBody {
    std::string name;   // "" for a non-function brace block
    std::string body;   // text between the outermost { and }
};

// Every brace block at file scope, with the name of the function it belongs
// to. Comments and preprocessor lines are removed first.
std::vector<TopLevelBody> top_level_bodies(const std::string& src) {
    std::string const code = blank_directives_keep_lines(strip_comments_keep_lines(src));
    std::vector<TopLevelBody> out;
    std::string header, name;
    int depth = 0;
    std::size_t body_start = 0;
    for (std::size_t i = 0; i < code.size(); ++i) {
        char const c = code[i];
        if (depth == 0) {
            if (c == '{') {
                name = function_name_from_header(header);
                depth = 1;
                body_start = i + 1;
            } else if (c == ';') {
                header.clear();
            } else {
                header += c;
            }
            continue;
        }
        if (c == '{') {
            ++depth;
        } else if (c == '}' && --depth == 0) {
            out.push_back({name, code.substr(body_start, i - body_start)});
            header.clear();
        }
    }
    return out;
}

bool is_secret_path_name(const std::string& name) {
    std::string lower = name;
    for (auto& ch : lower) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    for (auto const* part : kSecretNameParts) {
        if (lower.find(part) != std::string::npos) return true;
    }
    return false;
}

// Scans one OpenCL (or, with kMetalVarInverseNames, Metal) source. Appends
// "label: reason" per violation to `offenders` and the name of every
// non-secret function that calls a _var inverse to `public_callers`. Returns
// the number of violations.
int scan_var_inverse_use(const std::string& src, const std::string& file_name,
                         const std::string& label,
                         std::vector<std::string>& offenders,
                         std::vector<std::string>& public_callers,
                         const std::vector<std::string>& names = kVarInverseNames) {
    int bad = 0;
    std::string const code = strip_comments_keep_lines(src);
    if (file_name.rfind("secp256k1_ct_", 0) == 0 && references_var_inverse(code, names)) {
        offenders.push_back(label + ": a secp256k1_ct_* file references a "
                            "variable-time inverse");
        ++bad;
    }
    std::vector<std::string> const lines = split_lines(code);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (!directive(lines[i]).empty() && references_var_inverse(lines[i], names)) {
            offenders.push_back(label + ":" + std::to_string(i + 1)
                                + ": preprocessor line references a variable-time inverse");
            ++bad;
        }
    }
    for (auto const& fn : top_level_bodies(src)) {
        if (!references_var_inverse(fn.body, names)) continue;
        if (is_secret_path_name(fn.name)) {
            offenders.push_back(label + ": " + fn.name + "() calls a variable-time inverse");
            ++bad;
        } else {
            public_callers.push_back(fn.name);
        }
    }
    return bad;
}

bool has_entry(const std::vector<std::string>& v, const std::string& needle) {
    for (auto const& s : v) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

// Replaces every occurrence of `from` in `s` with `to`.
std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

} // namespace

int test_regression_opencl_kernel_closure_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: the OpenCL scan-only kernel embed is self-contained\n");
    std::printf("======================================================================\n\n");

    std::vector<std::string> unresolved;
    int guarded_stray_else = 0;
    std::string const guarded = build_embed(true, unresolved, guarded_stray_else);

    // OKC-1 -- the embed set exists where the consumer expects it.
    for (auto const& missing : unresolved) {
        std::printf("    UNRESOLVED: src/opencl/kernels/%s\n", missing.c_str());
    }
    CHECK(unresolved.empty(),
          "OKC-1: all six scan-only embed files resolve from any CWD");
    if (!unresolved.empty()) {
        std::printf("\n[regression_opencl_kernel_closure] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    // OKC-4 -- the evaluator models the guard shapes actually in the tree.
    CHECK(guarded_stray_else == 0,
          "OKC-4: no #else at the top level of a SECP256K1_OPENCL_SCAN_ONLY region "
          "(the guard is exclude-only; an #else arm would need a real preprocessor)");

    // OKC-2 -- the property the consumer needs. Each leaked name is printed
    // because the name is the diagnosis: it says which secp256k1_ct_*.cl file
    // the embed now depends on and does not carry.
    std::vector<std::string> const leaked = ct_references(guarded);
    std::printf("  guarded embed: %zu lines\n", split_lines(guarded).size());
    for (auto const& w : leaked) {
        std::printf("    LEAKED: %s  (defined in a secp256k1_ct_*.cl the consumer "
                    "does not embed -- this is issue #415)\n", w.c_str());
    }
    CHECK(leaked.empty(),
          "OKC-2: the -D" + std::string(kGuardMacro)
              + " embed references no ct_* symbol and no CT* type");

    // OKC-3 -- negative control: delete the guard and this must go red.
    std::vector<std::string> unresolved_plain;
    int plain_stray_else = 0;
    std::string const plain = build_embed(false, unresolved_plain, plain_stray_else);
    std::vector<std::string> const present = ct_references(plain);
    std::printf("  unguarded embed: %zu lines, %zu ct reference(s)\n",
                split_lines(plain).size(), present.size());
    for (auto const& w : present) std::printf("    %s\n", w.c_str());
    CHECK(!present.empty(),
          "OKC-3: without the guard the embed DOES reference ct_* / CT* -- if this "
          "fails, OKC-2 is passing vacuously and proves nothing");

    // The guard must remove lines, not merely exist.
    CHECK(guarded.size() < plain.size(),
          "OKC-3: the guard removes source from the embed");

    // OKC-5 -- the variable-time SafeGCD inverses see public data only: no
    // secp256k1_ct_*.cl file, no macro and no sign / ecdh / keygen / derive /
    // prove function references them, in any kernel file or in the OpenCL C
    // embedded in opencl_context.cpp.
    std::vector<std::string> var_offenders, var_public_callers;
    std::vector<std::string> const all_cl = list_cl_kernel_files();
    bool all_cl_read = !all_cl.empty();
    for (auto const& name : all_cl) {
        std::string const rel = std::string(kKernelDir) + name;
        std::string const src = audit_read_source_file(rel.c_str());
        if (src.empty()) { all_cl_read = false; continue; }
        scan_var_inverse_use(src, name, rel, var_offenders, var_public_callers);
    }
    {
        std::string const ctx_src = audit_read_source_file("src/opencl/src/opencl_context.cpp");
        if (ctx_src.empty()) all_cl_read = false;
        scan_var_inverse_use(extract_kernel_parts(ctx_src), "opencl_context.cpp",
                             "src/opencl/src/opencl_context.cpp (kernel_parts)",
                             var_offenders, var_public_callers);
    }
    for (auto const& off : var_offenders) std::printf("    VAR-INVERSE OFFENDER: %s\n", off.c_str());
    std::printf("  variable-time inverse callers (public paths): %zu\n", var_public_callers.size());
    for (auto const& c : var_public_callers) std::printf("    %s\n", c.c_str());
    CHECK(all_cl_read, "OKC-5: every OpenCL kernel file and opencl_context.cpp is readable");
    CHECK(var_offenders.empty(),
          "OKC-5: field_inv_var_impl / scalar_inv_var_impl / modinv30_var_impl appear in "
          "no secp256k1_ct_*.cl file, no macro and no sign/ecdh/keygen/derive/prove function");

    // OKC-6 -- live positive control: the scan sees the real verifier call
    // sites, so OKC-5 cannot pass because the body extractor found nothing.
    CHECK(has_entry(var_public_callers, "ecdsa_verify_impl")
              && has_entry(var_public_callers, "schnorr_verify_impl")
              && has_entry(var_public_callers, "ecdsa_recover_impl"),
          "OKC-6: the scan finds the variable-time inverse in ecdsa_verify_impl, "
          "schnorr_verify_impl and ecdsa_recover_impl");

    // OKC-7 -- live negative control: put the variable-time inverse into the
    // real signing and ECDH code and into a real secp256k1_ct_*.cl file; the
    // scan must reject each. A synthetic verifier must stay accepted.
    {
        std::string const ext =
            audit_read_source_file((std::string(kKernelDir) + "secp256k1_extended.cl").c_str());
        std::vector<std::string> mut_off, mut_pub;
        scan_var_inverse_use(replace_all(ext, "field_inv_impl(", "field_inv_var_impl("),
                             "secp256k1_extended.cl", "<mutated extended.cl>", mut_off, mut_pub);
        for (auto const& off : mut_off) std::printf("    mutation caught: %s\n", off.c_str());
        CHECK(!ext.empty() && has_entry(mut_off, "ecdsa_sign_impl()")
                  && has_entry(mut_off, "schnorr_sign_impl()")
                  && has_entry(mut_off, "ecdh_compute_raw_impl()"),
              "OKC-7: field_inv_impl -> field_inv_var_impl in the real extended.cl is "
              "rejected in ecdsa_sign_impl, schnorr_sign_impl and ecdh_compute_raw_impl");

        std::string const ct_src =
            audit_read_source_file((std::string(kKernelDir) + "secp256k1_ct_field.cl").c_str());
        std::vector<std::string> ct_off, ct_pub;
        scan_var_inverse_use(ct_src + "\nstatic inline void probe(FieldElement* r, "
                                      "const FieldElement* a) { field_inv_var_impl(r, a); }\n",
                             "secp256k1_ct_field.cl", "<mutated secp256k1_ct_field.cl>",
                             ct_off, ct_pub);
        CHECK(!ct_src.empty() && !ct_off.empty(),
              "OKC-7: a variable-time inverse call added to the real secp256k1_ct_field.cl "
              "is rejected");

        std::vector<std::string> mac_off, mac_pub;
        scan_var_inverse_use("#define INV(r, a) field_inv_var_impl(r, a)\n",
                             "secp256k1_demo.cl", "<macro>", mac_off, mac_pub);
        std::vector<std::string> ok_off, ok_pub;
        scan_var_inverse_use("static inline int demo_verify(const Scalar* s, Scalar* r) {\n"
                             "    // field_inv_var_impl in a comment does not count\n"
                             "    scalar_inv_var_impl(s, r); return 1;\n}\n"
                             "static inline int demo_sign(const Scalar* k, Scalar* r) {\n"
                             "    /* scalar_inv_var_impl(k, r); */ scalar_inverse_impl(k, r);"
                             " return 1;\n}\n",
                             "secp256k1_demo.cl", "<synthetic>", ok_off, ok_pub);
        CHECK(!mac_off.empty(), "OKC-7: a macro wrapping a variable-time inverse is rejected");
        CHECK(ok_off.empty() && ok_pub.size() == 1 && ok_pub[0] == "demo_verify",
              "OKC-7: a verifier calling scalar_inv_var_impl is accepted and a commented-out "
              "call in a signer is ignored");
    }

    // OKC-8 -- Metal: the variable-time SafeGCD inverses (field_inv_var,
    // scalar_inv_var, modinv30_var_impl, jacobian_to_affine_var) see public
    // data only: no secp256k1_ct_*.h file, no macro and no sign / ecdh /
    // keygen / derive / prove function references them, in any *.h or *.metal
    // file under src/metal/shaders/.
    {
        std::vector<std::string> m_off, m_pub;
        std::vector<std::string> const shaders = list_metal_shader_files();
        bool all_read = !shaders.empty();
        for (auto const& name : shaders) {
            std::string const rel = std::string(kMetalShaderDir) + name;
            std::string const src = audit_read_source_file(rel.c_str());
            if (src.empty()) { all_read = false; continue; }
            scan_var_inverse_use(src, name, rel, m_off, m_pub, kMetalVarInverseNames);
        }
        for (auto const& off : m_off) std::printf("    METAL VAR-INVERSE OFFENDER: %s\n", off.c_str());
        std::printf("  Metal variable-time inverse callers (public paths): %zu\n", m_pub.size());
        for (auto const& c : m_pub) std::printf("    %s\n", c.c_str());
        CHECK(all_read, "OKC-8: every Metal shader file (*.h, *.metal) is readable");
        CHECK(m_off.empty(),
              "OKC-8: field_inv_var / scalar_inv_var / modinv30_var_impl / jacobian_to_affine_var "
              "appear in no secp256k1_ct_*.h file, no macro and no sign/ecdh/keygen/derive/prove "
              "function of the Metal shaders");

        // OKC-9 -- live positive control: the real Metal verifier call sites.
        CHECK(has_entry(m_pub, "ecdsa_verify") && has_entry(m_pub, "schnorr_verify")
                  && has_entry(m_pub, "ecdsa_recover") && has_entry(m_pub, "ecdsa_recover_metal")
                  && has_entry(m_pub, "ecdsa_snark_witness_batch")
                  && has_entry(m_pub, "lbtc_commitment_verify")
                  && has_entry(m_pub, "frost_jac_equal")
                  && has_entry(m_pub, "field_inv_var") && has_entry(m_pub, "scalar_inv_var"),
              "OKC-9: the Metal scan finds the variable-time inverse in ecdsa_verify, "
              "schnorr_verify, ecdsa_recover, ecdsa_recover_metal, ecdsa_snark_witness_batch, "
              "lbtc_commitment_verify, frost_jac_equal and the two wrappers");
    }

    // OKC-10 -- live negative control for the Metal scan: put the
    // variable-time inverse into the real Metal ECDH code, into the real
    // recoverable signer and into a real secp256k1_ct_*.h file; the scan must
    // reject each. A synthetic Metal verifier must stay accepted.
    {
        std::string const ext =
            audit_read_source_file((std::string(kMetalShaderDir) + "secp256k1_extended.h").c_str());
        std::string const mut_ext =
            replace_all(replace_all(ext, "jacobian_to_affine(", "jacobian_to_affine_var("),
                        "field_inv(", "field_inv_var(");
        std::vector<std::string> mut_off, mut_pub;
        scan_var_inverse_use(mut_ext, "secp256k1_extended.h", "<mutated extended.h>",
                             mut_off, mut_pub, kMetalVarInverseNames);
        for (auto const& off : mut_off) std::printf("    Metal mutation caught: %s\n", off.c_str());
        CHECK(!ext.empty() && has_entry(mut_off, "ecdh_compute_raw()")
                  && has_entry(mut_off, "ecdh_compute()")
                  && has_entry(mut_off, "ecdh_shared_secret_xonly()"),
              "OKC-10: jacobian_to_affine / field_inv -> the _var forms in the real Metal "
              "extended.h is rejected in ecdh_compute_raw, ecdh_compute and "
              "ecdh_shared_secret_xonly");

        std::string const rec =
            audit_read_source_file((std::string(kMetalShaderDir) + "secp256k1_recovery.h").c_str());
        std::vector<std::string> rec_off, rec_pub;
        scan_var_inverse_use(replace_all(rec, "scalar_inverse(", "scalar_inv_var("),
                             "secp256k1_recovery.h", "<mutated recovery.h>",
                             rec_off, rec_pub, kMetalVarInverseNames);
        for (auto const& off : rec_off) std::printf("    Metal mutation caught: %s\n", off.c_str());
        CHECK(!rec.empty() && has_entry(rec_off, "ecdsa_sign_recoverable_metal()"),
              "OKC-10: scalar_inverse -> scalar_inv_var in the real Metal recoverable signer "
              "(the nonce inverse k^-1) is rejected");

        std::string const ct_src =
            audit_read_source_file((std::string(kMetalShaderDir) + "secp256k1_ct_point.h").c_str());
        std::vector<std::string> ct_off, ct_pub;
        scan_var_inverse_use(ct_src + "\ninline FieldElement probe(thread const FieldElement &a) "
                                      "{ return field_inv_var(a); }\n",
                             "secp256k1_ct_point.h", "<mutated secp256k1_ct_point.h>",
                             ct_off, ct_pub, kMetalVarInverseNames);
        CHECK(!ct_src.empty() && !ct_off.empty(),
              "OKC-10: a variable-time inverse call added to the real Metal "
              "secp256k1_ct_point.h is rejected");

        std::vector<std::string> mac_off, mac_pub;
        scan_var_inverse_use("#define INV(a) field_inv_var(a)\n",
                             "secp256k1_demo.h", "<metal macro>", mac_off, mac_pub,
                             kMetalVarInverseNames);
        std::vector<std::string> ok_off, ok_pub;
        scan_var_inverse_use("inline bool demo_verify(thread const Scalar256 &s) {\n"
                             "    // field_inv_var in a comment does not count\n"
                             "    Scalar256 r = scalar_inv_var(s); return r.limbs[0] != 0;\n}\n"
                             "inline bool demo_sign(thread const Scalar256 &k) {\n"
                             "    /* scalar_inv_var(k); */ Scalar256 r = scalar_inverse(k);"
                             " return r.limbs[0] != 0;\n}\n",
                             "secp256k1_demo.h", "<metal synthetic>", ok_off, ok_pub,
                             kMetalVarInverseNames);
        CHECK(!mac_off.empty(), "OKC-10: a Metal macro wrapping a variable-time inverse is rejected");
        CHECK(ok_off.empty() && ok_pub.size() == 1 && ok_pub[0] == "demo_verify",
              "OKC-10: a Metal verifier calling scalar_inv_var is accepted and a commented-out "
              "call in a signer is ignored");
    }

    std::printf("\n[regression_opencl_kernel_closure] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

// GH-436 regression guard: every OpenCL helper must be static inline (or a
// FORCE_INLINE macro whose own expansion contains static) so AMD/ROCm (and
// any C99-inline vendor) always has an external definition even when the
// compiler chooses not to inline a large helper (field_inv_impl,
// field_sqr_impl, sha streaming bodies, bip chacha, keccak etc). Bare
// "inline" was the root cause of "undefined hidden symbol" at
// clBuildProgram time.
//
// What this checks: every *.cl file in src/opencl/kernels/ (enumerated from
// disk, not a hardcoded list -- 26 files today) plus the OpenCL C embedded
// as kernel_parts raw strings in src/opencl/src/opencl_context.cpp, scanned
// (comments stripped) for a non-static "inline" function declaration or
// definition, and every "#define FORCE_INLINE ..." whose replacement text
// has no "static". A missing/unreadable input is a hard failure, not a
// skip (SIL-1/SIL-2 below).
// What this does NOT check: it is a source-text scan, not a preprocessor or
// a vendor build -- it does not expand macros, does not model declaration
// specifiers split across physical lines, and proves nothing about whether
// a given vendor's OpenCL compiler actually accepts the kernel (that is the
// GPU CI matrix's job). src/bch/opencl and src/ltc/opencl are separate
// kernel trees and are out of scope for this guard.
int test_regression_opencl_static_inline_link_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: OpenCL static-inline link hygiene (GH-436)\n");
    std::printf("======================================================================\n\n");

    std::vector<std::string> offenders;
    std::vector<std::string> unresolved;

    std::vector<std::string> const kernel_files = list_cl_kernel_files();
    std::printf("  kernel files found: %zu\n", kernel_files.size());
    for (auto const& name : kernel_files) {
        std::string const rel = std::string(kKernelDir) + name;
        std::string const src = audit_read_source_file(rel.c_str());
        if (src.empty()) { unresolved.push_back(rel); continue; }
        scan_bad_inline(src, rel, offenders);
    }
    for (auto const& missing : unresolved) {
        std::printf("    UNRESOLVED: %s\n", missing.c_str());
    }
    CHECK(!kernel_files.empty(),
          "SIL-1: src/opencl/kernels/ resolves from this CWD and contains at least one .cl file");
    CHECK(unresolved.empty(), "SIL-1: every enumerated kernel file is readable");
    if (kernel_files.empty() || !unresolved.empty()) {
        std::printf("\n[regression_opencl_static_inline_link] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    std::string const ctx_path = "src/opencl/src/opencl_context.cpp";
    std::string const ctx_src = audit_read_source_file(ctx_path.c_str());
    CHECK(!ctx_src.empty(), "SIL-2: src/opencl/src/opencl_context.cpp resolves from this CWD");
    if (ctx_src.empty()) {
        std::printf("\n[regression_opencl_static_inline_link] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    std::string const kernel_parts_src = extract_kernel_parts(ctx_src);
    CHECK(!kernel_parts_src.empty(),
          "SIL-2: opencl_context.cpp contains at least one R\"KERNEL( ... )KERNEL\" kernel_parts entry");
    if (!kernel_parts_src.empty()) {
        scan_bad_inline(kernel_parts_src, ctx_path + " (kernel_parts)", offenders);
    }

    for (auto const& off : offenders) {
        std::printf("    OFFENDER: %s\n", off.c_str());
    }
    CHECK(offenders.empty(),
          "SIL-3: no non-static \"inline\" function decl/def and no non-static "
          "FORCE_INLINE define in any scanned OpenCL source");

    // SIL-4 -- negative control: the scanner itself must catch the bad
    // shapes and leave the good ones alone, or SIL-3 could be passing
    // vacuously (same idea as OKC-3 above).
    std::vector<std::string> neg_bad;
    std::vector<std::string> neg_good;
    scan_bad_inline("inline void f(void) {}\n", "<neg>", neg_bad);
    scan_bad_inline("#define FORCE_INLINE inline\n", "<neg>", neg_bad);
    scan_bad_inline("static inline void f(void) {}\n", "<pos>", neg_good);
    scan_bad_inline("#define FORCE_INLINE static inline\n", "<pos>", neg_good);
    scan_bad_inline("FORCE_INLINE void f(void) {}\n", "<pos>", neg_good);
    scan_bad_inline("// inline void f(void) {}\n", "<pos>", neg_good);
    CHECK(neg_bad.size() == 2,
          "SIL-4: the scanner flags a bare `inline void f()` and a bare "
          "`#define FORCE_INLINE inline` -- if this fails, SIL-3 proves nothing");
    CHECK(neg_good.empty(),
          "SIL-4: the scanner does not flag static inline, FORCE_INLINE use, "
          "a static FORCE_INLINE define, or a commented-out inline");

    std::printf("\n[regression_opencl_static_inline_link] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() {
    int const closure_rc = test_regression_opencl_kernel_closure_run();
    int const static_inline_rc = test_regression_opencl_static_inline_link_run();
    return (closure_rc != 0 || static_inline_rc != 0) ? 1 : 0;
}
#endif
