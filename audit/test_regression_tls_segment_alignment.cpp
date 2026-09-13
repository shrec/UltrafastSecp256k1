// ============================================================================
// test_regression_tls_segment_alignment.cpp
// ============================================================================
// Regression: an executable linking this library must have a PT_TLS segment
// aligned to at least 64 bytes, or Android arm64 refuses to load it.
//
// Bionic's loader rejects the binary outright:
//
//     error: "<binary>": executable's TLS segment is underaligned:
//            alignment is 8, needs to be at least 64 for ARM64 Bionic
//
// Every thread_local in this library is naturally 8- or 16-aligned
// (tl_context_owner, ct::g_blinding, the statics inside
// scalar_mul_generator_with_context), so the linker emitted p_align = 8 and no
// Android arm64 executable that linked libfastsecp256k1 could start at all.
// The fix is alignas(64) on precompute.cpp's tl_context_owner, which raises the
// segment alignment for the whole library.
//
// Why this test reads ELF rather than the source
//   The property that matters is a property of the LINKED IMAGE, not of any one
//   declaration. A future thread_local added anywhere, or an alignas removed
//   during a refactor, changes it — and a source scan for "alignas(64)" would
//   keep passing while the segment silently dropped back to 8. So this parses
//   the program headers of its own running binary and asserts on p_align. It is
//   the same question the loader asks.
//
// Why it was never caught
//   The CI android(arm64-v8a) job cross-compiles and never executes anything on
//   a device. This module runs inside every audit binary on every platform that
//   has ELF program headers, so the check now travels with the ordinary suite.
//
// Found on a Rockchip RK3588 (Cortex-A76) over adb while measuring issue #336.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

#include "audit_check.hpp"

// Pull in the library's thread_local objects.
//
// PT_TLS is a link-time property: it exists only if the linker actually included
// an object file that defines a thread_local. Without a reference to something
// in precompute.cpp, the standalone build of this module links none of them, has
// no PT_TLS segment at all, and the check below has nothing to assert -- which
// is exactly the vacuous pass this module must not become. scalar_mul_generator
// lives beside tl_context_owner, so referencing it brings that TLS in.
#include "secp256k1/precompute.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

static int g_pass = 0, g_fail = 0;

// ELF is the only container with a PT_TLS segment to inspect. Mach-O and PE
// express thread-local storage differently and Bionic's rule does not apply to
// them, so there is nothing to assert there.
#if defined(__ELF__) && (defined(__linux__) || defined(__ANDROID__))
#  define UFSECP_TLS_ALIGN_TEST_SUPPORTED 1
#  include <elf.h>
#  include <link.h>
#  include <sys/auxv.h>
#else
#  define UFSECP_TLS_ALIGN_TEST_SUPPORTED 0
#endif

namespace {

#if UFSECP_TLS_ALIGN_TEST_SUPPORTED

struct TlsScan {
    bool  found = false;
    std::uint64_t align = 0;
    std::uint64_t memsz = 0;
};

// Read the MAIN EXECUTABLE's program headers straight from the auxiliary
// vector, which is where the kernel put them and what the loader itself reads.
//
// The obvious alternative, dl_iterate_phdr with the callback stopping at the
// first object, is wrong: it assumes the first object enumerated is the main
// executable. That holds on glibc and does NOT hold on Bionic, where this
// module reported "no PT_TLS segment" on an Android arm64 binary whose
// readelf output plainly showed `TLS ... R 0x40`. A silent skip on the one
// platform the check exists for is worse than no check, so it reads AT_PHDR.
TlsScan scan_own_program_headers() {
    TlsScan r;
    const auto phdr  = static_cast<unsigned long>(getauxval(AT_PHDR));
    const auto phnum = static_cast<unsigned long>(getauxval(AT_PHNUM));
    const auto phent = static_cast<unsigned long>(getauxval(AT_PHENT));
    if (phdr == 0 || phnum == 0 || phent < sizeof(ElfW(Phdr))) return r;

    const auto* base = reinterpret_cast<const unsigned char*>(phdr);
    for (unsigned long i = 0; i < phnum; ++i) {
        const auto* ph = reinterpret_cast<const ElfW(Phdr)*>(base + i * phent);
        if (ph->p_type == PT_TLS) {
            r.found = true;
            r.align = static_cast<std::uint64_t>(ph->p_align);
            r.memsz = static_cast<std::uint64_t>(ph->p_memsz);
        }
    }
    return r;
}

#endif  // UFSECP_TLS_ALIGN_TEST_SUPPORTED

// Bionic's arm64 requirement. Applied on every ELF platform, not just Android:
// the alignment is a property of this library's thread_local objects, so a
// regression introduced on a Linux developer machine would ship broken to
// Android. Catching it here is the point.
constexpr std::uint64_t kBionicArm64MinTlsAlign = 64;

} // namespace

int test_regression_tls_segment_alignment_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: PT_TLS alignment >= 64 (Android arm64 loadability)\n");
    std::printf("======================================================================\n\n");

#if !UFSECP_TLS_ALIGN_TEST_SUPPORTED
    std::printf("  [skip] no ELF program headers on this platform -- PT_TLS alignment is\n");
    std::printf("         an ELF/Bionic property and does not apply here\n");
    std::printf("\n[regression_tls_segment_alignment] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return 0;
#else
    // Force the library's TLS into this image before inspecting it (see the
    // include note above). The result is discarded; only the linkage matters.
    {
        auto const p = secp256k1::fast::scalar_mul_generator(
            secp256k1::fast::Scalar::from_uint64(1));
        volatile bool sink = p.is_infinity();
        (void)sink;
    }

    TlsScan const scan = scan_own_program_headers();

    // No PT_TLS at all is a legitimate outcome for a binary that pulled in none
    // of this library's thread_local objects; there is nothing for the loader to
    // reject. Report it rather than passing silently.
    if (!scan.found) {
        std::printf("  [skip] this binary has no PT_TLS segment -- none of the library's\n");
        std::printf("         thread_local objects were linked in, so there is nothing to align\n");
        std::printf("\n[regression_tls_segment_alignment] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 0;
    }

    std::printf("  PT_TLS: p_align = %llu, p_memsz = %llu\n",
                static_cast<unsigned long long>(scan.align),
                static_cast<unsigned long long>(scan.memsz));

    char msg[256];
    std::snprintf(msg, sizeof(msg),
        "TLS-ALIGN-1: PT_TLS p_align is %llu, must be >= %llu -- below this "
        "Android arm64 Bionic refuses to load the executable "
        "(\"TLS segment is underaligned\")",
        static_cast<unsigned long long>(scan.align),
        static_cast<unsigned long long>(kBionicArm64MinTlsAlign));
    CHECK(scan.align >= kBionicArm64MinTlsAlign, msg);

    // A power of two is what the loader assumes when it lays the block out;
    // anything else would be a malformed segment rather than a policy failure.
    CHECK(scan.align != 0 && (scan.align & (scan.align - 1)) == 0,
          "TLS-ALIGN-2: PT_TLS p_align is a power of two");

    std::printf("\n[regression_tls_segment_alignment] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
#endif
}

#ifdef STANDALONE_TEST
int main() { return test_regression_tls_segment_alignment_run(); }
#endif
