// ============================================================================
// test_regression_opencl_generator_w4.cpp
// ============================================================================
// Regression: OpenCL scalar_mul_generator_windowed_impl (window w=4
// precomputed generator table {0*G..15*G}) used to rebuild a 16-entry
// AffinePoint table as a LOCAL/private array via 128 individual per-limb
// literal assignments on EVERY kernel-thread invocation
// (src/opencl/kernels/secp256k1_extended.cl). Measured
// CL_KERNEL_PRIVATE_MEM_SIZE_bytes on the RTX 5060 Ti / NVIDIA OpenCL driver
// 580.173.02 was ~1056 bytes for that shape. The table is now declared ONCE
// at OpenCL program scope in __constant address space
// (GENERATOR_TABLE_W4[16]) and read by index instead of rebuilt -- measured
// private memory on the same hardware/driver dropped to ~32 bytes. This is a
// storage-only change: scalar-nibble extraction, doubling-and-add control
// flow, and field/point arithmetic are unchanged (task
// opencl-generator-w4-production-claude-v4).
//
// Evidence:
//   data/tasking/artifacts/opencl_generator_w4_production_claude_v4.json
//     (clGetKernelWorkGroupInfo before/after + 5-run x 7-pass x 3-batch
//      before/after benchmark + zero-mismatch differential correctness
//      against an independent CPU oracle for zero/one/n-1/n/n+1/max256 and
//      4090 deterministic random scalars, including a direct build+launch of
//      the REAL production kernel, not just a benchmark copy)
//   docs/BACKEND_ASSURANCE_MATRIX.md
//
// Guard: THREE layers, mirroring test_regression_gpu_beta_constants.cpp:
//   (Layer 1) Value verification -- the 16 canonical AffinePoint entries
//             {0*G..15*G} in 4x64-bit little-endian limb form.
//   (Layer 2) Pre-fix pattern absence -- the OLD per-call local-array
//             rebuild signature ("AffinePoint table[16]") must NOT appear
//             inside scalar_mul_generator_windowed_impl's body.
//   (Layer 3) Source-coupled scan -- reads the ACTUAL production
//             secp256k1_extended.cl file and asserts: exactly one
//             __constant GENERATOR_TABLE_W4[16] declaration; the function
//             reads GENERATOR_TABLE_W4[idx] (not a rebuilt local table);
//             and all 16 parsed table entries match the canonical values
//             above byte-for-byte (parsed programmatically, not just
//             substring-checked).
//
// This test does NOT require a GPU/OpenCL runtime -- it is a static source
// scan plus arithmetic-free value comparison, so it always runs (advisory
//=false in unified_audit_runner.cpp's ALL_MODULES[]).
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

// ============================================================================
// Layer 1: Canonical window-4 generator table {0*G..15*G}
// ============================================================================
// 4x64-bit little-endian limbs, standard secp256k1 domain parameters (SEC2).
// Independently transcribed for this test (not #included from the production
// .cl file) so a corrupted production table is actually caught, not just
// echoed back.
struct CanonicalAffine {
    uint64_t x[4];
    uint64_t y[4];
};

static const CanonicalAffine CANONICAL_TABLE_W4[16] = {
    { { 0x0UL, 0x0UL, 0x0UL, 0x0UL }, { 0x0UL, 0x0UL, 0x0UL, 0x0UL } },  // 0 (unused sentinel)
    { { 0x59F2815B16F81798UL, 0x029BFCDB2DCE28D9UL, 0x55A06295CE870B07UL, 0x79BE667EF9DCBBACUL },
      { 0x9C47D08FFB10D4B8UL, 0xFD17B448A6855419UL, 0x5DA4FBFC0E1108A8UL, 0x483ADA7726A3C465UL } },  // 1*G
    { { 0xABAC09B95C709EE5UL, 0x5C778E4B8CEF3CA7UL, 0x3045406E95C07CD8UL, 0xC6047F9441ED7D6DUL },
      { 0x236431A950CFE52AUL, 0xF7F632653266D0E1UL, 0xA3C58419466CEAEEUL, 0x1AE168FEA63DC339UL } },  // 2*G
    { { 0x8601F113BCE036F9UL, 0xB531C845836F99B0UL, 0x49344F85F89D5229UL, 0xF9308A019258C310UL },
      { 0x6CB9FD7584B8E672UL, 0x6500A99934C2231BUL, 0x0FE337E62A37F356UL, 0x388F7B0F632DE814UL } },  // 3*G
    { { 0x74FA94ABE8C4CD13UL, 0xCC6C13900EE07584UL, 0x581E4904930B1404UL, 0xE493DBF1C10D80F3UL },
      { 0xCFE97BDC47739922UL, 0xD967AE33BFBDFE40UL, 0x5642E2098EA51448UL, 0x51ED993EA0D455B7UL } },  // 4*G
    { { 0xCBA8D569B240EFE4UL, 0xE88B84BDDC619AB7UL, 0x55B4A7250A5C5128UL, 0x2F8BDE4D1A072093UL },
      { 0xDCA87D3AA6AC62D6UL, 0xF788271BAB0D6840UL, 0xD4DBA9DDA6C9C426UL, 0xD8AC222636E5E3D6UL } },  // 5*G
    { { 0x2F057A1460297556UL, 0x82F6472F8568A18BUL, 0x20453A14355235D3UL, 0xFFF97BD5755EEEA4UL },
      { 0x3C870C36B075F297UL, 0xDE80F0F6518FE4A0UL, 0xF3BE96017F45C560UL, 0xAE12777AACFBB620UL } },  // 6*G
    { { 0xE92BDDEDCAC4F9BCUL, 0x3D419B7E0330E39CUL, 0xA398F365F2EA7A0EUL, 0x5CBDF0646E5DB4EAUL },
      { 0xA5082628087264DAUL, 0xA813D0B813FDE7B5UL, 0xA3178D6D861A54DBUL, 0x6AEBCA40BA255960UL } },  // 7*G
    { { 0x67784EF3E10A2A01UL, 0x0A1BDD05E5AF888AUL, 0xAFF3843FB70F3C2FUL, 0x2F01E5E15CCA351DUL },
      { 0xB5DA2CB76CBDE904UL, 0xC2E213D6BA5B7617UL, 0x293D082A132D13B4UL, 0x5C4DA8A741539949UL } },  // 8*G
    { { 0xC35F110DFC27CCBEUL, 0xE09796974C57E714UL, 0x09AD178A9F559ABDUL, 0xACD484E2F0C7F653UL },
      { 0x05CC262AC64F9C37UL, 0xADD888A4375F8E0FUL, 0x64380971763B61E9UL, 0xCC338921B0A7D9FDUL } },  // 9*G
    { { 0x52A68E2A47E247C7UL, 0x3442D49B1943C2B7UL, 0x35477C7B1AE6AE5DUL, 0xA0434D9E47F3C862UL },
      { 0x3CBEE53B037368D7UL, 0x6F794C2ED877A159UL, 0xA3B6C7E693A24C69UL, 0x893ABA425419BC27UL } },  // 10*G
    { { 0xBBEC17895DA008CBUL, 0x5649980BE5C17891UL, 0x5EF4246B70C65AACUL, 0x774AE7F858A9411EUL },
      { 0x301D74C9C953C61BUL, 0x372DB1E2DFF9D6A8UL, 0x0243DD56D7B7B365UL, 0xD984A032EB6B5E19UL } },  // 11*G
    { { 0xC5B0F47070AFE85AUL, 0x687CF4419620095BUL, 0x15C38F004D734633UL, 0xD01115D548E7561BUL },
      { 0x6B051B13F4062327UL, 0x79238C5DD9A86D52UL, 0xA8B64537E17BD815UL, 0xA9F34FFDC815E0D7UL } },  // 12*G
    { { 0xDEEDDF8F19405AA8UL, 0xB075FBC6610E58CDUL, 0xC7D1D205C3748651UL, 0xF28773C2D975288BUL },
      { 0x29B5CB52DB03ED81UL, 0x3A1A06DA521FA91FUL, 0x758212EB65CDAF47UL, 0x0AB0902E8D880A89UL } },  // 13*G
    { { 0xE49B241A60E823E4UL, 0x26AA7B63678949E6UL, 0xFD64E67F07D38E32UL, 0x499FDF9E895E719CUL },
      { 0xC65F40D403A13F5BUL, 0x464279C27A3F95BCUL, 0x90F044E4A7B3D464UL, 0xCAC2F6C4B54E8551UL } },  // 14*G
    { { 0x44ADBCF8E27E080EUL, 0x31E5946F3C85F79EUL, 0x5A465AE3095FF411UL, 0xD7924D4F7D43EA96UL },
      { 0xC504DC9FF6A26B58UL, 0xEA40AF2BD896D3A5UL, 0x83842EC228CC6DEFUL, 0x581E2872A86C72A6UL } },  // 15*G
};

// ============================================================================
// Helpers -- file reading
// ============================================================================
// Repair (issue #335 acceptance repair, round 5): the previous bounded
// CWD-relative prefix list ("", "../", "../../", ...) never resolved when
// unified_audit_runner was invoked from a CWD unrelated to the repo (e.g.
// /tmp) -- unlike the other round-5 fixes, this one did NOT silently pass
// with 0 checks (every FAIL branch below already increments g_fail), but it
// DID make this module's real check count diverge between "run from the
// repo root" and "run from an unrelated CWD" (8/8 vs a hard, blocking
// failure), which the same "source resolution must be CWD-independent"
// finding also covers. Route through the shared, UFSECP_SOURCE_ROOT-aware
// audit_read_source_file() (audit_check.hpp) -- same mechanism now used by
// every other repaired source-reading module in this file set -- keeping
// the bounded CWD-relative walk-up only as its own internal fallback.
static std::string read_repo_file(const char* const* candidates, int n_candidates) {
    for (int c = 0; c < n_candidates; ++c) {
        std::string src = audit_read_source_file(candidates[c]);
        if (!src.empty()) return src;
    }
    return {};
}

static const char* EXTENDED_CL_CANDIDATES[] = {
    "src/opencl/kernels/secp256k1_extended.cl",                          // standalone UltrafastSecp256k1 repo root
    "libs/UltrafastSecp256k1/src/opencl/kernels/secp256k1_extended.cl",  // Secp256K1fast monorepo
};
static const int EXTENDED_CL_N_CANDIDATES = 2;

static const char* BIP352_CL_CANDIDATES[] = {
    "src/opencl/kernels/secp256k1_bip352.cl",
    "libs/UltrafastSecp256k1/src/opencl/kernels/secp256k1_bip352.cl",
};
static const int BIP352_CL_N_CANDIDATES = 2;

// ============================================================================
// Tiny hex-literal parser: "0x1234ABULu" style tokens -> uint64_t
// ============================================================================
static uint64_t parse_hex_literal(std::string tok) {
    while (!tok.empty()) {
        char c = tok.back();
        if (c == 'u' || c == 'U' || c == 'l' || c == 'L') { tok.pop_back(); continue; }
        break;
    }
    // strip leading/trailing whitespace
    size_t b = tok.find_first_not_of(" \t\r\n");
    size_t e = tok.find_last_not_of(" \t\r\n");
    if (b == std::string::npos) return 0;
    tok = tok.substr(b, e - b + 1);
    return std::strtoull(tok.c_str(), nullptr, 0);
}

// Find "needle" starting at/after `from`; returns npos if not found.
static size_t find_all_count(const std::string& hay, const std::string& needle) {
    size_t count = 0, pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

// Extract the balanced-brace initializer body of
// "__constant AffinePoint GENERATOR_TABLE_W4[16] = { ... };"
static bool extract_table_initializer(const std::string& src, std::string& out_body) {
    size_t decl = src.find("__constant AffinePoint GENERATOR_TABLE_W4[16]");
    if (decl == std::string::npos) return false;
    size_t brace_start = src.find('{', decl);
    if (brace_start == std::string::npos) return false;
    int depth = 0;
    for (size_t i = brace_start; i < src.size(); ++i) {
        if (src[i] == '{') ++depth;
        else if (src[i] == '}') {
            --depth;
            if (depth == 0) {
                out_body = src.substr(brace_start, i - brace_start + 1);
                return true;
            }
        }
    }
    return false;
}

// Extract the balanced-brace body of scalar_mul_generator_windowed_impl(...).
static bool extract_function_body(const std::string& src, std::string& out_body) {
    size_t sig = src.find("inline void scalar_mul_generator_windowed_impl");
    if (sig == std::string::npos) return false;
    size_t brace_start = src.find('{', sig);
    if (brace_start == std::string::npos) return false;
    int depth = 0;
    for (size_t i = brace_start; i < src.size(); ++i) {
        if (src[i] == '{') ++depth;
        else if (src[i] == '}') {
            --depth;
            if (depth == 0) {
                out_body = src.substr(sig, i - sig + 1);
                return true;
            }
        }
    }
    return false;
}

// Parse the 16 "{ {{x0,x1,x2,x3}}, {{y0,y1,y2,y3}} }" entries out of the
// initializer body, in order.
static bool parse_table_entries(const std::string& body, CanonicalAffine out[16]) {
    size_t pos = 0;
    for (int i = 0; i < 16; ++i) {
        size_t open1 = body.find("{{", pos);
        if (open1 == std::string::npos) return false;
        size_t close1 = body.find("}}", open1);
        if (close1 == std::string::npos) return false;
        std::string xs = body.substr(open1 + 2, close1 - open1 - 2);

        size_t open2 = body.find("{{", close1);
        if (open2 == std::string::npos) return false;
        size_t close2 = body.find("}}", open2);
        if (close2 == std::string::npos) return false;
        std::string ys = body.substr(open2 + 2, close2 - open2 - 2);

        // split xs/ys on commas into 4 limbs each
        uint64_t xl[4] = {}, yl[4] = {};
        {
            size_t start = 0;
            for (int j = 0; j < 4; ++j) {
                size_t comma = xs.find(',', start);
                std::string tok = (comma == std::string::npos) ? xs.substr(start) : xs.substr(start, comma - start);
                xl[j] = parse_hex_literal(tok);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            start = 0;
            for (int j = 0; j < 4; ++j) {
                size_t comma = ys.find(',', start);
                std::string tok = (comma == std::string::npos) ? ys.substr(start) : ys.substr(start, comma - start);
                yl[j] = parse_hex_literal(tok);
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        }
        std::memcpy(out[i].x, xl, sizeof(xl));
        std::memcpy(out[i].y, yl, sizeof(yl));
        pos = close2 + 2;
    }
    return true;
}

// ============================================================================
// Layer 1: Value verification -- canonical table is internally sane
// ============================================================================
static void test_canonical_table_sentinel_is_zero() {
    const CanonicalAffine& z = CANONICAL_TABLE_W4[0];
    bool all_zero = true;
    for (int i = 0; i < 4; ++i) if (z.x[i] != 0 || z.y[i] != 0) all_zero = false;
    if (!all_zero) {
        std::printf("[opencl-gen-w4] FAIL: CANONICAL_TABLE_W4[0] (unused sentinel) is not all-zero\n");
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: CANONICAL_TABLE_W4[0] sentinel is all-zero\n");
        ++g_pass;
    }
}

static void test_canonical_table_entries_distinct() {
    // 1*G .. 15*G must all be pairwise distinct (a real precomputed table,
    // not 16 copies of the same point by some construction bug).
    bool all_distinct = true;
    for (int i = 1; i <= 15 && all_distinct; ++i) {
        for (int j = i + 1; j <= 15; ++j) {
            if (std::memcmp(CANONICAL_TABLE_W4[i].x, CANONICAL_TABLE_W4[j].x, sizeof(CANONICAL_TABLE_W4[i].x)) == 0 &&
                std::memcmp(CANONICAL_TABLE_W4[i].y, CANONICAL_TABLE_W4[j].y, sizeof(CANONICAL_TABLE_W4[i].y)) == 0) {
                all_distinct = false;
                break;
            }
        }
    }
    if (!all_distinct) {
        std::printf("[opencl-gen-w4] FAIL: CANONICAL_TABLE_W4 entries 1..15 are not pairwise distinct\n");
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: CANONICAL_TABLE_W4 entries 1..15 are pairwise distinct\n");
        ++g_pass;
    }
}

// ============================================================================
// Layer 2 + 3: Source-coupled scan of the ACTUAL production kernel file
// ============================================================================
static void test_source_generator_table_declared_exactly_once() {
    std::string src = read_repo_file(EXTENDED_CL_CANDIDATES, EXTENDED_CL_N_CANDIDATES);
    if (src.empty()) {
        std::printf("[opencl-gen-w4] FAIL: cannot read secp256k1_extended.cl (tried standalone + monorepo paths)\n");
        ++g_fail;
        return;
    }
    size_t count = find_all_count(src, "__constant AffinePoint GENERATOR_TABLE_W4[16]");
    if (count != 1) {
        std::printf("[opencl-gen-w4] FAIL: expected exactly 1 GENERATOR_TABLE_W4[16] declaration, found %zu\n", count);
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: GENERATOR_TABLE_W4[16] declared exactly once\n");
        ++g_pass;
    }
}

static void test_source_local_rebuild_absent_from_function() {
    std::string src = read_repo_file(EXTENDED_CL_CANDIDATES, EXTENDED_CL_N_CANDIDATES);
    if (src.empty()) {
        std::printf("[opencl-gen-w4] FAIL: cannot read secp256k1_extended.cl\n");
        ++g_fail;
        return;
    }
    std::string func_body;
    if (!extract_function_body(src, func_body)) {
        std::printf("[opencl-gen-w4] FAIL: cannot locate scalar_mul_generator_windowed_impl body\n");
        ++g_fail;
        return;
    }
    // Layer 2 (pre-fix pattern absence): the old per-call rebuild declared a
    // LOCAL "AffinePoint table[16]" and assigned it via 128 literal
    // statements. That pattern must be gone from the function body.
    if (func_body.find("AffinePoint table[16]") != std::string::npos) {
        std::printf("[opencl-gen-w4] FAIL: scalar_mul_generator_windowed_impl still rebuilds a "
                     "local 'AffinePoint table[16]' -- per-call table reconstruction was not removed\n");
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: local 'AffinePoint table[16]' rebuild absent from function body\n");
        ++g_pass;
    }
    if (func_body.find("GENERATOR_TABLE_W4[") == std::string::npos) {
        std::printf("[opencl-gen-w4] FAIL: scalar_mul_generator_windowed_impl does not read GENERATOR_TABLE_W4[...]\n");
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: function reads from GENERATOR_TABLE_W4[...]\n");
        ++g_pass;
    }
    // Storage-only change: nibble extraction and helper calls must survive verbatim.
    bool ctrl_flow_ok = func_body.find("(w >> (nib * 4)) & 0xFUL") != std::string::npos &&
                        func_body.find("point_double_unchecked(r, r)") != std::string::npos &&
                        func_body.find("point_from_affine(r,") != std::string::npos &&
                        func_body.find("point_add_mixed_unchecked(r, r,") != std::string::npos;
    if (!ctrl_flow_ok) {
        std::printf("[opencl-gen-w4] FAIL: scalar-nibble extraction / point helper calls changed -- "
                     "this optimization must be storage-only\n");
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: scalar-nibble extraction and point helper calls unchanged\n");
        ++g_pass;
    }
}

static void test_source_table_values_match_canonical() {
    std::string src = read_repo_file(EXTENDED_CL_CANDIDATES, EXTENDED_CL_N_CANDIDATES);
    if (src.empty()) {
        std::printf("[opencl-gen-w4] FAIL: cannot read secp256k1_extended.cl\n");
        ++g_fail;
        return;
    }
    std::string init_body;
    if (!extract_table_initializer(src, init_body)) {
        std::printf("[opencl-gen-w4] FAIL: cannot locate GENERATOR_TABLE_W4 initializer body\n");
        ++g_fail;
        return;
    }
    CanonicalAffine parsed[16] = {};
    if (!parse_table_entries(init_body, parsed)) {
        std::printf("[opencl-gen-w4] FAIL: could not parse 16 table entries from GENERATOR_TABLE_W4 initializer\n");
        ++g_fail;
        return;
    }
    int mismatches = 0;
    for (int i = 0; i < 16; ++i) {
        bool eq = std::memcmp(parsed[i].x, CANONICAL_TABLE_W4[i].x, sizeof(parsed[i].x)) == 0 &&
                  std::memcmp(parsed[i].y, CANONICAL_TABLE_W4[i].y, sizeof(parsed[i].y)) == 0;
        if (!eq) {
            ++mismatches;
            std::printf("[opencl-gen-w4] FAIL: GENERATOR_TABLE_W4[%d] does not match canonical %d*G\n", i, i);
        }
    }
    if (mismatches != 0) {
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: all 16 GENERATOR_TABLE_W4 entries match canonical {0*G..15*G}\n");
        ++g_pass;
    }
}

static void test_source_bip352_inherits_table_via_include() {
    std::string src = read_repo_file(BIP352_CL_CANDIDATES, BIP352_CL_N_CANDIDATES);
    if (src.empty()) {
        std::printf("[opencl-gen-w4] FAIL: cannot read secp256k1_bip352.cl\n");
        ++g_fail;
        return;
    }
    // secp256k1_bip352.cl must NOT redeclare its own GENERATOR_TABLE_W4 --
    // it must inherit the single declaration via #include, so this file
    // never duplicates the table across translation units.
    size_t own_decl = find_all_count(src, "__constant AffinePoint GENERATOR_TABLE_W4[16]");
    bool includes_extended = src.find("#include \"secp256k1_extended.cl\"") != std::string::npos;
    if (own_decl != 0) {
        std::printf("[opencl-gen-w4] FAIL: secp256k1_bip352.cl redeclares GENERATOR_TABLE_W4 "
                     "(found %zu) -- table must not be duplicated across files\n", own_decl);
        ++g_fail;
    } else if (!includes_extended) {
        std::printf("[opencl-gen-w4] FAIL: secp256k1_bip352.cl no longer #includes "
                     "secp256k1_extended.cl -- it would lose access to GENERATOR_TABLE_W4\n");
        ++g_fail;
    } else {
        std::printf("[opencl-gen-w4] PASS: secp256k1_bip352.cl inherits GENERATOR_TABLE_W4 via "
                     "#include, no duplicate declaration\n");
        ++g_pass;
    }
}

// ============================================================================
// Entry point
// ============================================================================
int test_regression_opencl_generator_w4_run() {
    std::printf("\n=== OpenCL Generator W4 Constant-Table Regression ===\n");
    std::printf("=== Layer 1: Value verification (canonical table sanity)\n");
    std::printf("=== Layer 2: Pre-fix pattern absence (local rebuild removed)\n");
    std::printf("=== Layer 3: Source-coupled scan (production secp256k1_extended.cl)\n\n");

    g_pass = 0;
    g_fail = 0;

    test_canonical_table_sentinel_is_zero();
    test_canonical_table_entries_distinct();
    test_source_generator_table_declared_exactly_once();
    test_source_local_rebuild_absent_from_function();
    test_source_table_values_match_canonical();
    test_source_bip352_inherits_table_via_include();

    std::printf("\n  %d passed  %d failed  (total %d)\n", g_pass, g_fail, g_pass + g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_opencl_generator_w4_run();
}
#endif
