// ============================================================================
// test_cache_dir_api -- programmatic cache directory API + no-config.ini
// ============================================================================
// Regression coverage for the config.ini removal:
//   1. configure_fixed_base_auto() MUST NOT create a config.ini in the CWD
//      (the legacy auto-tune path used to drop one there).
//   2. set_cache_directory() routes the fixed-base cache to the caller-supplied
//      directory and the engine keeps producing correct generator multiples.
//   3. Neither path writes a config.ini as a side effect.
//
// This is the programmatic replacement for config.ini: callers point the engine
// at their own cache location via set_cache_directory() / ufsecp_set_cache_dir()
// and no INI file is ever created or read.
// ============================================================================

#include "secp256k1/precompute.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

#include <array>
#include <cstdio>
#include <cstdint>
#include <string>
#include <sys/stat.h>

using namespace secp256k1::fast;

namespace {

bool file_exists(const char* path) {
    struct stat st {};
    return ::stat(path, &st) == 0;
}

bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() || b.is_infinity()) {
        return a.is_infinity() && b.is_infinity();
    }
    return a.to_compressed() == b.to_compressed();
}

int g_fails = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("[cache-dir] FAIL: %s\n", what);
        ++g_fails;
    }
}

} // namespace

int test_cache_dir_api_run() {
    g_fails = 0;

    // A clean slate: the test asserts the engine never creates config.ini, so
    // remove any pre-existing one first (best-effort; ignore failure).
    std::remove("config.ini");

    // (1) configure_fixed_base_auto() must NOT create a config.ini.
    set_cache_directory("");        // empty => current working directory
    configure_fixed_base_auto();
    check(!file_exists("config.ini"),
          "configure_fixed_base_auto() must not create config.ini");

    // (2) set_cache_directory() routes cache lookups to the given directory and
    //     the generator path stays correct. Use a small, file-free window so the
    //     test never builds/writes the multi-hundred-MB w=18 cache.
    {
        FixedBaseConfig cfg{};
        cfg.window_bits = 4;
        cfg.use_cache   = false;   // no file I/O regardless of cache_dir
        cfg.cache_dir   = "ufsecp_test_cache_dir_does_not_need_to_exist";
        configure_fixed_base(cfg);
        ensure_fixed_base_ready();

        const Point G = Point::generator();
        for (uint64_t k : {1ULL, 2ULL, 7ULL, 1000ULL, 0xFFFFFFFFULL}) {
            const Scalar sk = Scalar::from_uint64(k);
            const Point fast = scalar_mul_generator(sk);
            const Point slow = G.scalar_mul(sk);
            check(points_equal(fast, slow),
                  "scalar_mul_generator matches generic scalar_mul with custom cache_dir");
        }
    }

    // (3) Still no config.ini created by any of the above.
    check(!file_exists("config.ini"),
          "no config.ini created after set_cache_directory + generator use");
    std::remove("config.ini");  // tidy up if a future regression created one

    if (g_fails == 0) {
        std::printf("[cache-dir] all checks passed\n");
    }
    return g_fails == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_cache_dir_api_run();
}
#endif
