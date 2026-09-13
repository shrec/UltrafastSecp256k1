// ============================================================================
// test_regression_metal_shader_closure.cpp
// ============================================================================
// Regression: the Metal runtime-shader-compile fallback must be able to build a
// translation unit. It could not, and had not been able to for as long as the
// header list existed.
//
// gpu_backend_metal.mm's ensure_library() has two ways to get a Metal library:
// load the prebuilt secp256k1_kernels.metallib, or -- when that fails -- compile
// the shader sources at runtime with newLibraryWithSource(). The second path was
// dead:
//
//   * metal_load_combined_source() concatenated a hardcoded list that named
//     `secp256k1_bloom.h`. No such file exists -- not in src/metal/shaders, not
//     anywhere in the tree. One missing header made the whole candidate
//     directory fail, so the function returned an empty string for every
//     directory, every time.
//   * The same list named 4 of the 11 headers secp256k1_kernels.metal actually
//     includes, and ignored the nested includes (point -> field,
//     extended -> point, zk -> extended). newLibraryWithSource has no include
//     path, so a concatenation that leaves `#include "..."` lines in place is a
//     hard error even when every file is present.
//   * src/metal/CMakeLists.txt's SHADER_FILES copied the same 4 headers, so a
//     copied shaders/ directory was incomplete regardless.
//
// This surfaced on `CI / macos (Release)`, where the audit binaries run from
// <build>/audit: the metallib did not load, the fallback had nothing to fall
// back to, and gpu_abi_gate / gpu_collect_verify_parity / unified_audit all
// reported GPU failures.
//
// The loader now expands the entry file's own includes recursively, so there is
// no list in C++ that can drift. What remains checkable -- and is checked here,
// on every platform, with no Metal device and no Apple toolchain -- is that the
// shader tree and the build system agree with that entry file:
//
//   MSC-1  every `#include "..."` in the closure of secp256k1_kernels.metal
//          resolves to a file in src/metal/shaders (this is what the
//          `secp256k1_bloom.h` bug looked like)
//   MSC-2  src/metal/CMakeLists.txt's SHADER_FILES covers the whole closure,
//          so a copied shaders/ directory is complete
//   MSC-3  the loader has no hardcoded header list left to drift, and still
//          names the entry file
//   MSC-4  the closure terminates: no include cycle, and the entry file is
//          reachable
// ============================================================================

#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

namespace {

// Every quoted include on its own line, in source order.
std::vector<std::string> quoted_includes(const std::string& src) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    while (pos < src.size()) {
        std::size_t const eol = src.find('\n', pos);
        std::string const line =
            src.substr(pos, (eol == std::string::npos ? src.size() : eol) - pos);
        pos = (eol == std::string::npos) ? src.size() : eol + 1;

        std::size_t const h = line.find_first_not_of(" \t");
        if (h == std::string::npos || line[h] != '#') continue;
        std::size_t const k = line.find("include", h);
        if (k == std::string::npos) continue;
        std::size_t const q1 = line.find('"', k);
        if (q1 == std::string::npos) continue;
        std::size_t const q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) continue;
        std::string name = line.substr(q1 + 1, q2 - q1 - 1);
        // A CRLF checkout puts '\r' before the closing quote of nothing here,
        // but the surrounding line handling elsewhere splits on '\n' only --
        // strip defensively so a name never carries a stray carriage return.
        name.erase(std::remove(name.begin(), name.end(), '\r'), name.end());
        out.push_back(name);
    }
    return out;
}

constexpr const char* kShaderDir = "src/metal/shaders/";
constexpr const char* kEntry = "secp256k1_kernels.metal";

// Walk the include closure exactly the way metal_expand_includes() does.
// `missing` collects names that do not resolve -- the secp256k1_bloom.h shape.
bool build_closure(const std::string& file,
                   std::vector<std::string>& seen,
                   std::vector<std::string>& missing,
                   int depth) {
    if (depth > 16) return false;                 // cycle / runaway guard
    std::string const src = audit_read_source_file((std::string(kShaderDir) + file).c_str());
    if (src.empty()) { missing.push_back(file); return true; }

    for (auto const& name : quoted_includes(src)) {
        if (std::find(seen.begin(), seen.end(), name) != seen.end()) continue;
        seen.push_back(name);
        if (!build_closure(name, seen, missing, depth + 1)) return false;
    }
    return true;
}

} // namespace

int test_regression_metal_shader_closure_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: the Metal shader include closure is complete and copied\n");
    std::printf("======================================================================\n\n");

    std::string const entry_src = audit_read_source_file((std::string(kShaderDir) + kEntry).c_str());
    CHECK(!entry_src.empty(),
          "MSC-4: src/metal/shaders/secp256k1_kernels.metal resolves from any CWD");
    if (entry_src.empty()) {
        std::printf("\n[regression_metal_shader_closure] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }

    std::vector<std::string> seen;
    std::vector<std::string> missing;
    bool const terminated = build_closure(kEntry, seen, missing, 0);
    CHECK(terminated,
          "MSC-4: the include closure terminates (no cycle, depth <= 16)");

    std::printf("  closure: %zu header(s) reachable from %s\n", seen.size(), kEntry);
    for (auto const& h : seen) std::printf("    %s\n", h.c_str());

    // MSC-1 -- the bug that killed the fallback.
    if (!missing.empty()) {
        for (auto const& m : missing) {
            std::string const msg =
                "MSC-1: '" + m + "' is included from the Metal shader closure but does not "
                "exist in src/metal/shaders -- the runtime-compile fallback cannot build a "
                "translation unit without it";
            CHECK(false, msg);
        }
    } else {
        CHECK(true,
              "MSC-1: every quoted include in the closure resolves inside src/metal/shaders");
    }

    // MSC-2 -- CMake copies the whole closure, not a subset.
    std::string const metal_cmake = audit_read_source_file("src/metal/CMakeLists.txt");
    CHECK(!metal_cmake.empty(), "MSC-2: src/metal/CMakeLists.txt resolves from any CWD");
    if (!metal_cmake.empty()) {
        std::size_t const list_begin = metal_cmake.find("set(SHADER_FILES");
        CHECK(list_begin != std::string::npos,
              "MSC-2: src/metal/CMakeLists.txt still declares SHADER_FILES");
        if (list_begin != std::string::npos) {
            std::size_t const list_end = metal_cmake.find(')', list_begin);
            std::string const list =
                metal_cmake.substr(list_begin,
                                   (list_end == std::string::npos ? std::string::npos
                                                                  : list_end - list_begin));
            for (auto const& h : seen) {
                std::string const msg =
                    "MSC-2: SHADER_FILES copies '" + h + "', which the kernel includes";
                CHECK(list.find(h) != std::string::npos, msg);
            }
            std::string const entry_msg =
                std::string("MSC-2: SHADER_FILES copies the entry file '") + kEntry + "'";
            CHECK(list.find(kEntry) != std::string::npos, entry_msg);
        }
    }

    // MSC-3 -- the loader expands from the entry file and keeps no header list.
    std::string const loader = audit_read_source_file("src/gpu/src/gpu_backend_metal.mm");
    CHECK(!loader.empty(), "MSC-3: src/gpu/src/gpu_backend_metal.mm resolves from any CWD");
    if (!loader.empty()) {
        CHECK(loader.find("metal_expand_includes") != std::string::npos,
              "MSC-3: the shader loader expands the entry file's own includes");
        CHECK(loader.find(kEntry) != std::string::npos,
              "MSC-3: the shader loader still names secp256k1_kernels.metal as its entry");
        // As a STRING LITERAL -- the comment above the loader names the file
        // deliberately, to record why the fallback was dead.
        CHECK(loader.find("\"secp256k1_bloom.h\"") == std::string::npos,
              "MSC-3: the loader has no \"secp256k1_bloom.h\" string literal left "
              "(the header has never existed)");
        // A reintroduced kHeaders[]-style list is the drift this module exists
        // to stop; the expansion needs no such array.
        CHECK(loader.find("kHeaders") == std::string::npos,
              "MSC-3: no hardcoded shader-header array remains in the loader");
    }

    std::printf("\n[regression_metal_shader_closure] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_metal_shader_closure_run(); }
#endif
