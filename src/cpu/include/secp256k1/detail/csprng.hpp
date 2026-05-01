#ifndef SECP256K1_DETAIL_CSPRNG_HPP
#define SECP256K1_DETAIL_CSPRNG_HPP

// -- OS-level cryptographic random number generation, fail-closed ------------
// Single canonical implementation used by ecies, ellswift, bip324, musig2.
// All callers must #include this header; do NOT define a local csprng_fill().

#include <cstddef>
#include <cstdlib>

#if defined(_WIN32)
#  include <windows.h>
#  include <bcrypt.h>
#  pragma comment(lib, "bcrypt.lib")
#elif defined(__APPLE__)
#  include <Security/SecRandom.h>
#elif defined(__ANDROID__)
#  include <stdlib.h>  // arc4random_buf (Android API 12+)
#elif defined(ESP_PLATFORM)
#  include <esp_random.h>
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#  include <sys/random.h>
#else
#  include <cstdio>
#endif

namespace secp256k1::detail {

inline void csprng_fill(unsigned char* buf, std::size_t len) noexcept {
    if (len == 0) return;
#if defined(_WIN32)
    NTSTATUS const status = BCryptGenRandom(
        nullptr, buf, static_cast<ULONG>(len), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status != 0) std::abort();
#elif defined(__APPLE__)
    if (SecRandomCopyBytes(kSecRandomDefault, len, buf) != errSecSuccess)
        std::abort();
#elif defined(__ANDROID__)
    arc4random_buf(buf, len);
#elif defined(ESP_PLATFORM)
    esp_fill_random(buf, len);
#elif defined(__linux__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    std::size_t filled = 0;
    while (filled < len) {
        ssize_t const r = getrandom(buf + filled, len - filled, 0);
        if (r <= 0) std::abort();
        filled += static_cast<std::size_t>(r);
    }
#else
    FILE* f = std::fopen("/dev/urandom", "rb");
    if (!f) std::abort();
    if (std::fread(buf, 1, len, f) != len) { std::fclose(f); std::abort(); }
    std::fclose(f);
#endif
}

} // namespace secp256k1::detail

#endif // SECP256K1_DETAIL_CSPRNG_HPP
