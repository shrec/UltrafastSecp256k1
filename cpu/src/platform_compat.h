#ifndef SECP256K1_PLATFORM_COMPAT_H
#define SECP256K1_PLATFORM_COMPAT_H

// Platform compatibility layer for Windows types and intrinsics on Linux/Unix

// ESP32 platform detection
#if defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_ESP32)
#define SECP256K1_ESP32_BUILD 1
#endif

// STM32 platform detection (Cortex-M3, bare-metal)
#if defined(SECP256K1_PLATFORM_STM32)
#define SECP256K1_STM32_BUILD 1
#endif

#if defined(SECP256K1_ESP32_BUILD) || defined(SECP256K1_STM32_BUILD)
// ESP32 platform - minimal definitions, no POSIX mmap or Windows APIs
#include <cstdint>
#include <cstring>

// Stub types for compatibility (not used on ESP32)
typedef int HANDLE;
#define INVALID_HANDLE_VALUE (-1)

#elif !defined(_WIN32)
// Linux/Unix platform - define Windows types and map to POSIX equivalents

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>

// Windows handle types
typedef int HANDLE;
#define INVALID_HANDLE_VALUE (-1)
#ifndef NULL
#define NULL 0
#endif

// Windows file constants
#define GENERIC_READ    0x80000000
#define FILE_SHARE_READ 0x00000001
#define OPEN_EXISTING   3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define PAGE_READONLY   0x02
#define FILE_MAP_READ   0x0004

// Stub structures for compatibility
struct LARGE_INTEGER {
    int64_t QuadPart;
};

// No-op functions for Windows API on Linux (actual implementation uses POSIX)
inline HANDLE CreateFileA(const char*, uint32_t, uint32_t, void*, uint32_t, uint32_t, HANDLE) { return INVALID_HANDLE_VALUE; }
inline bool GetFileSizeEx(HANDLE, LARGE_INTEGER*) { return false; }
inline HANDLE CreateFileMappingA(HANDLE, void*, uint32_t, uint32_t, uint32_t, const char*) { return 0; }
inline void* MapViewOfFile(HANDLE, uint32_t, uint32_t, uint32_t, size_t) { return nullptr; }
inline bool UnmapViewOfFile(const void*) { return true; }
inline bool CloseHandle(HANDLE) { return true; }
inline uint32_t GetLastError() { return 0; }

#endif // !_WIN32 && !ESP32

// Cross-platform intrinsics compatibility
#if defined(__x86_64__) || defined(_M_X64)
    // x86-64: Use BMI2 intrinsics
    #if defined(__GNUC__) && !defined(_MSC_VER)
        #include <x86intrin.h>
        #define COMPAT_ADDCARRY_U64(carry, a, b, out) \
            _addcarry_u64(carry, a, b, reinterpret_cast<unsigned long long*>(out))
        #define COMPAT_SUBBORROW_U64(borrow, a, b, out) \
            _subborrow_u64(borrow, a, b, reinterpret_cast<unsigned long long*>(out))
    #else
        // MSVC
        #define COMPAT_ADDCARRY_U64(carry, a, b, out) _addcarry_u64(carry, a, b, out)
        #define COMPAT_SUBBORROW_U64(borrow, a, b, out) _subborrow_u64(borrow, a, b, out)
    #endif
#else
    // RISC-V, ARM, etc: Portable implementation matching field.cpp/scalar.cpp logic
    inline unsigned char compat_addcarry_u64_impl(unsigned char carry, uint64_t a, uint64_t b, uint64_t* out) {
        uint64_t sum = a + b;
        unsigned char carry1 = (sum < a);
        uint64_t result = sum + carry;
        unsigned char carry2 = (result < sum);
        *out = result;
        return carry1 | carry2;
    }
    
    inline unsigned char compat_subborrow_u64_impl(unsigned char borrow, uint64_t a, uint64_t b, uint64_t* out) {
        uint64_t temp = a - borrow;
        unsigned char borrow1 = (a < borrow);
        uint64_t result = temp - b;
        unsigned char borrow2 = (temp < b);
        *out = result;
        return borrow1 | borrow2;
    }
    
    #define COMPAT_ADDCARRY_U64(carry, a, b, out) compat_addcarry_u64_impl(carry, a, b, out)
    #define COMPAT_SUBBORROW_U64(borrow, a, b, out) compat_subborrow_u64_impl(borrow, a, b, out)
#endif

#endif // SECP256K1_PLATFORM_COMPAT_H
