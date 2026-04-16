#pragma once

#include <iostream>
#include <cstdlib>
#include <mutex>

namespace secp256k1::fast {

// External selftest function from library
extern bool Selftest(bool verbose);

// Auto-run selftest on first library use
// Call this at the start of every application's main()
inline bool ensure_library_integrity(bool verbose = false) {
    static std::once_flag flag;
    static bool result = true;

    std::call_once(flag, [verbose]() {
        if (verbose) {
            std::cout << "[*] Running library integrity check...\n" << std::flush;
        }

        result = Selftest(verbose);

        if (!result) {
            std::cerr << "\n[FAIL] CRITICAL: Library integrity check FAILED!\n";
            std::cerr << "   The secp256k1 library has failed self-validation.\n";
            std::cerr << "   This application cannot continue safely.\n" << std::flush;
            std::abort();
        }

        if (verbose) {
            std::cout << "[OK] Library integrity verified\n\n" << std::flush;
        }
    });

    return result;
}

// Macro to automatically run selftest at program startup
// Usage: Add SECP256K1_INIT(); as the first line in main()
#define SECP256K1_INIT() \
    secp256k1::fast::ensure_library_integrity(false)

#define SECP256K1_INIT_VERBOSE() \
    secp256k1::fast::ensure_library_integrity(true)

} // namespace secp256k1::fast
