// ============================================================================
// Test: libatomic transitive link closure regression
// ============================================================================
// Deliberately performs atomic operations on a type wide enough that GCC/
// Clang commonly lower them to __atomic_* library calls (from libatomic)
// instead of inline lock-free instructions:
//   1. std::atomic<T>::is_lock_free() / compare_exchange_strong() on a
//      16-byte struct (SecpWideAtomicPayload) -- the same shape of call
//      (__atomic_is_lock_free, __atomic_compare_exchange) that regresses to
//      an undefined reference when libatomic is required but not linked.
//   2. std::atomic<uint64_t>::is_lock_free() -- the exact symbol name from
//      the original "lost fix" bug report (__atomic_is_lock_free).
//
// This target links ONLY against ${SECP256K1_LIB_NAME} in CMake (never
// `atomic` directly). If the PUBLIC libatomic propagation in
// src/cpu/CMakeLists.txt regresses to PRIVATE, is dropped, or a future
// change wrongly attempts to force `atomic` on MSVC/Apple, this executable
// fails at LINK time (undefined `__atomic_is_lock_free`, or -- on
// MSVC/Apple -- "cannot find -latomic") before it ever runs.
// ============================================================================

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace {

struct SecpWideAtomicPayload {
    std::uint64_t lo;
    std::uint64_t hi;
};

bool test_wide_atomic_roundtrip() {
    std::atomic<SecpWideAtomicPayload> wide{SecpWideAtomicPayload{0, 0}};
    SecpWideAtomicPayload expected{0, 0};
    SecpWideAtomicPayload desired{0x1111111111111111ULL, 0x2222222222222222ULL};
    if (!wide.compare_exchange_strong(expected, desired)) {
        return false;
    }
    SecpWideAtomicPayload loaded = wide.load();
    return loaded.lo == desired.lo && loaded.hi == desired.hi;
}

bool test_narrow_atomic_is_lock_free_query() {
    std::atomic<std::uint64_t> narrow{42};
    // is_lock_free() is exactly the symbol (__atomic_is_lock_free) that
    // regresses to an undefined reference when libatomic is required but
    // not linked transitively into the consumer.
    (void)narrow.is_lock_free();
    narrow.fetch_add(1);
    return narrow.load() == 43;
}

}  // namespace

#ifdef STANDALONE_TEST
int main() {
    bool ok = true;
    ok = test_wide_atomic_roundtrip() && ok;
    ok = test_narrow_atomic_is_lock_free_query() && ok;
    if (!ok) {
        std::fprintf(stderr, "test_atomic_link_closure: FAILED\n");
        return 1;
    }
    std::printf("test_atomic_link_closure: PASSED (linked without direct -latomic; transitive closure verified)\n");
    return 0;
}
#endif
