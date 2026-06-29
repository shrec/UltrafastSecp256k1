cmake_minimum_required(VERSION 3.18)

# (Trivial touch: re-trigger Benchmark Dashboard with the workflow fix in 6896a181.)
# Reject in-source builds and builds that land outside out/ in the source tree.
# All build directories should be under out/ (enforced by CMakePresets.json).
# Use: cmake --preset cpu-release  OR  cmake -S . -B out/<name>
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed.\n"
        "Run: cmake -S . -B out/release\n"
        "Or use a preset: cmake --preset cpu-release")
endif()
get_filename_component(_src_real "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_bin_real "${CMAKE_BINARY_DIR}" REALPATH)
string(FIND "${_bin_real}" "${_src_real}/" _bin_inside_src)
if(_bin_inside_src EQUAL 0)
    # Build is inside the source tree — warn unless it's under out/
    string(FIND "${_bin_real}" "${_src_real}/out/" _bin_in_out)
    if(NOT _bin_in_out EQUAL 0)
        message(WARNING
            "Build directory '${CMAKE_BINARY_DIR}' is inside the source tree but not under out/.\n"
            "Prefer: cmake -S . -B out/<name>  (e.g. out/release)\n"
            "Or use a preset: cmake --preset cpu-release")
    endif()
endif()

# Read version from VERSION.txt (single source of truth)
# NOTE: named VERSION.txt (not VERSION) to avoid clashing with the
#       C++20 <version> header on case-insensitive filesystems (macOS).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" _version_raw)
string(STRIP "${_version_raw}" UFSECP_VERSION)

# Ensure 3-component semver (CMake needs MAJOR.MINOR.PATCH)
if(NOT UFSECP_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+")
    string(APPEND UFSECP_VERSION ".0")
endif()

project(UltrafastSecp256k1
    VERSION ${UFSECP_VERSION}
    DESCRIPTION "Ultra high-performance secp256k1 elliptic curve cryptography library"
    LANGUAGES CXX C
)

# Project metadata
set(PROJECT_HOMEPAGE_URL "https://github.com/shrec/UltrafastSecp256k1")
set(PROJECT_LICENSE "MIT")

# C++20 used where available; the project builds on GCC 10+, Clang 12+,
# MSVC 2019+, ESP32-IDF 5.x, STM32 arm-none-eabi toolchains, and WASM.
# CMAKE_CXX_STANDARD_REQUIRED is OFF so CMake falls back gracefully on
# toolchains that only support C++17.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate version header from template
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/secp256k1/version.hpp"
    @ONLY
)

# Build options
option(SECP256K1_BUILD_CPU "Build CPU implementation" ON)
option(SECP256K1_BUILD_CUDA "Build CUDA GPU support" OFF)
option(SECP256K1_BUILD_ROCM "Build ROCm/HIP GPU support (AMD)" OFF)
option(SECP256K1_BUILD_OPENCL "Build OpenCL support" OFF)
option(SECP256K1_BUILD_METAL "Build Apple Metal GPU support" OFF)

# ── Independent GPU feature modules ───────────────────────────────────────────
# Toggle individual GPU operations on/off SEPARATELY from the CPU feature
# modules. The GPU op surface is independent: a consumer may want, for example,
# BIP-324 compiled on the CPU but stripped from the GPU build (or vice-versa).
# Default ON (full GPU op surface, no behaviour change). Turning one OFF removes
# its kernels + backend dispatch from every GPU backend (CUDA/OpenCL/Metal); the
# stable C ABI entry point remains and returns UFSECP_ERR_GPU_UNSUPPORTED.
option(SECP256K1_GPU_BUILD_ZK        "GPU ZK ops (knowledge / DLEQ / bulletproof / SNARK witness)" ON)
option(SECP256K1_GPU_BUILD_BIP324    "GPU BIP-324 AEAD encrypt/decrypt ops"                        ON)
option(SECP256K1_GPU_BUILD_FROST     "GPU FROST partial-signature verify op"                       ON)
option(SECP256K1_GPU_BUILD_BIP352    "GPU BIP-352 silent-payment scan op"                          ON)
option(SECP256K1_GPU_BUILD_ECDH      "GPU ECDH batch op (secret-bearing)"                          ON)
option(SECP256K1_GPU_BUILD_MSM       "GPU multi-scalar multiplication op"                          ON)
option(SECP256K1_GPU_BUILD_HASH160   "GPU Hash160(pubkey) batch op"                                ON)
option(SECP256K1_GPU_BUILD_ECRECOVER "GPU ECDSA public-key recovery batch op"                      ON)
# A minimal node GPU build (script-signature verification only) can turn ALL of
# the above OFF — only generator-mul + ECDSA/Schnorr batch verify remain.
foreach(_gpu_mod ZK BIP324 FROST BIP352 ECDH MSM HASH160 ECRECOVER)
    if(SECP256K1_GPU_BUILD_${_gpu_mod})
        add_compile_definitions(SECP256K1_GPU_HAS_${_gpu_mod}=1)
    else()
        add_compile_definitions(SECP256K1_GPU_HAS_${_gpu_mod}=0)
        message(STATUS "secp256k1-fast: GPU module ${_gpu_mod} = OFF (stripped from GPU build)")
    endif()
endforeach()
option(SECP256K1_BUILD_TESTS "Build test suite" ON)
option(SECP256K1_BUILD_BENCH "Build benchmarks" ON)
option(SECP256K1_BUILD_EXAMPLES "Build example programs" ON)
option(SECP256K1_BUILD_JAVA "Build Java JNI bindings" ON)
option(SECP256K1_USE_ASM "Enable assembly optimizations (x64/RISC-V)" ON)

# Installation options
option(SECP256K1_INSTALL "Generate install target" ON)
option(SECP256K1_INSTALL_PKGCONFIG "Install pkg-config file" ON)

# Performance/safety tradeoffs
option(SECP256K1_SPEED_FIRST "Reserved: SECP256K1_FAST_NO_SECURITY_CHECKS compile definition (currently has no effect — CT guards are always on)" OFF)
option(SECP256K1_BUILD_SHARED "Build shared library" OFF)
option(SECP256K1_BUILD_SHIM "Build libsecp256k1-compatible shim + tests" OFF)

# Bitcoin strict mode (default ON): public API rejects non-canonical inputs
option(UFSECP_BITCOIN_STRICT "Enforce BIP-340 strict encoding in public API (reject r>=p, s>=n)" ON)

# Ethereum module (default ON): Keccak-256, EIP-55, EIP-191, EIP-155, ecrecover
# Set OFF to exclude Ethereum functionality (Bitcoin-only build)
option(SECP256K1_BUILD_ETHEREUM "Build Ethereum module (Keccak, EIP-55/155/191, ecrecover)" ON)

# CT enforcement: deprecate non-CT signing functions at compile time
option(SECP256K1_REQUIRE_CT "Deprecate non-CT sign functions (compile warnings on fast:: signing)" OFF)

# Bitcoin Core backend mode: bundles CT enforcement + strict ABI for backend integration.
# Enables SECP256K1_REQUIRE_CT and UFSECP_BITCOIN_STRICT; suitable for a deterministic,
# RFC 6979-only signing configuration matching libsecp256k1's external contract.
option(SECP256K1_CORE_BACKEND_MODE
       "Bitcoin Core alternative backend: CT enforced, strict ABI, RFC 6979 deterministic signing" OFF)
if(SECP256K1_CORE_BACKEND_MODE)
    set(SECP256K1_REQUIRE_CT ON CACHE BOOL "Enabled by SECP256K1_CORE_BACKEND_MODE" FORCE)
    set(UFSECP_BITCOIN_STRICT ON CACHE BOOL "Enabled by SECP256K1_CORE_BACKEND_MODE" FORCE)
    add_compile_definitions(SECP256K1_CORE_BACKEND_MODE=1)
    message(STATUS "secp256k1-fast: Bitcoin Core backend mode (CT enforce + strict ABI + RFC 6979)")
endif()

# -- Bitcoin Core single-flag entry point ---------------------------------------
# This is THE flag that Bitcoin Core's CMakeLists.txt will set.
# It activates the shim library + core backend mode together, so that a single
# `-DSECP256K1_USE_ULTRAFAST=ON` is all Bitcoin Core needs to switch backends.
option(SECP256K1_USE_ULTRAFAST
       "[Bitcoin Core] Use UltrafastSecp256k1 as the secp256k1 backend (enables shim + CT + strict ABI)" OFF)
if(SECP256K1_USE_ULTRAFAST)
    set(SECP256K1_BUILD_SHIM ON CACHE BOOL "Enabled by SECP256K1_USE_ULTRAFAST" FORCE)
    set(SECP256K1_CORE_BACKEND_MODE ON CACHE BOOL "Enabled by SECP256K1_USE_ULTRAFAST" FORCE)
    message(STATUS "secp256k1-fast: ✅ SECP256K1_USE_ULTRAFAST=ON — Bitcoin Core secondary backend mode activated")
endif()

# -- libbitcoin single-flag entry point -----------------------------------------
# Canonical libbitcoin integration is the BRIDGE-FREE, header-only C++ surface
# <ufsecp/libbitcoin.hpp> (ufsecp::lbtc::* inline functions) linked against the
# engine. ONE flag (SECP256K1_BUILD_LIBBITCOIN) builds that minimal surface: the
# engine + the direct-header interface target `secp256k1::fastsecp256k1_libbitcoin`,
# which carries <ufsecp/libbitcoin.hpp>, C++20 and HAVE_ULTRAFAST as usage
# requirements. NO libsecp256k1 shim, NO ufsecp C ABI, NO ufsecp_lbtc bridge are
# pulled in by default. The legacy shim + C ABI + ufsecp_lbtc batch bridge remain
# available for COMPATIBILITY/TESTING only behind the explicit opt-in
# SECP256K1_BUILD_LIBBITCOIN_BRIDGE.
option(SECP256K1_BUILD_LIBBITCOIN
       "[libbitcoin] Canonical bridge-free profile: engine + direct C++ header (ufsecp/libbitcoin.hpp); extras off" OFF)
# Compatibility-only opt-in: re-enable the libsecp256k1 shim + C ABI + ufsecp_lbtc
# batch bridge (legacy/reference integration surface). OFF for the canonical
# direct profile.
option(SECP256K1_BUILD_LIBBITCOIN_BRIDGE
       "[libbitcoin] Compatibility opt-in: libsecp256k1 shim + C ABI + ufsecp_lbtc batch bridge (legacy)" OFF)
# libbitcoin integration tests/bench. TESTS builds the canonical direct verify
# CTest (lbtc_direct_verify); with SECP256K1_BUILD_LIBBITCOIN_BRIDGE it also
# builds the legacy bridge consensus tests. BENCH builds the canonical direct
# C++ benchmark; with BRIDGE=ON it also builds the legacy C-ABI bridge bench.
option(SECP256K1_BUILD_LIBBITCOIN_TESTS
       "[libbitcoin] Build libbitcoin integration tests (direct verify; + bridge consensus when BRIDGE=ON)" OFF)
option(SECP256K1_BUILD_LIBBITCOIN_BENCH
       "[libbitcoin] Build libbitcoin direct C++ batch bench; + legacy bridge bench when BRIDGE=ON" OFF)
if(SECP256K1_BUILD_LIBBITCOIN)
    # libsecp256k1-compatible contract (engine-level; CT enforced, strict ABI).
    set(SECP256K1_REQUIRE_CT   ON)
    set(UFSECP_BITCOIN_STRICT  ON)
    # Package-friendly static engine target. The direct libbitcoin target is an
    # INTERFACE library that links this static engine target transitively.
    set(SECP256K1_BUILD_AS_OBJECT OFF)
    # Minimal surface — drop everything not needed by libbitcoin direct verify.
    # KEEP PIPPENGER/MSM: batch_verify.cpp depends on msm()/Pippenger for large
    # ECDSA/Schnorr batches. Everything else below is unused by the canonical
    # header-only surface.
    set(SECP256K1_BUILD_MUSIG2   OFF)
    set(SECP256K1_BUILD_FROST    OFF)
    set(SECP256K1_BUILD_ZK       OFF)
    set(SECP256K1_BUILD_ECIES    OFF)
    set(SECP256K1_BUILD_BIP352   OFF)
    set(SECP256K1_BUILD_LTC_SP   OFF)
    set(SECP256K1_BUILD_ADAPTOR  OFF)
    set(SECP256K1_BUILD_WALLET   OFF)
    set(SECP256K1_BUILD_BCH      OFF)
    set(SECP256K1_BUILD_BIP324   OFF)
    set(SECP256K1_BUILD_PIPPENGER ON)
    set(SECP256K1_BUILD_TESTS    OFF)
    set(SECP256K1_BUILD_BENCH    OFF)
    set(SECP256K1_BUILD_EXAMPLES OFF)
    set(SECP256K1_BUILD_JAVA     OFF)
    set(SECP256K1_BUILD_ETHEREUM OFF)
    # Prevent a stale BUILD_TESTING=ON cache from building the full audit/CPU
    # standalone test suite after this profile strips wallet/ZK/etc. The direct
    # lbtc_direct_verify CTest is enabled explicitly after audit/ below.
    set(BUILD_TESTING OFF CACHE BOOL "Disabled by the minimal libbitcoin profile" FORCE)
    # Compatibility opt-in: legacy libsecp256k1 shim + C ABI + ufsecp_lbtc bridge.
    if(SECP256K1_BUILD_LIBBITCOIN_BRIDGE)
        set(SECP256K1_BUILD_SHIM ON)
        set(SECP256K1_BUILD_CABI ON)
        message(STATUS "secp256k1-fast: libbitcoin profile — direct C++ header + COMPATIBILITY bridge (shim + C ABI + ufsecp_lbtc)")
    else()
        set(SECP256K1_BUILD_SHIM OFF)
        set(SECP256K1_BUILD_CABI OFF)
        set(SECP256K1_INSTALL_CABI OFF)
        message(STATUS "secp256k1-fast: ✅ libbitcoin profile — bridge-free direct C++ header (ufsecp/libbitcoin.hpp), static engine, minimal surface")
    endif()
endif()

# -- Bitcoin Knots single-flag entry point --------------------------------------
# Minimal libsecp256k1 drop-in for Bitcoin Knots (Bitcoin Core 29 base, libsecp256k1
# 0.6.0). Knots enables exactly: ECDSA core + recovery + schnorrsig + extrakeys +
# ellswift (ecdh/musig present-but-unused). This profile compiles ONLY that subset
# and strips every other crypto module + all GPU/bench/test/binding surface.
#
# MODULE build (not a separate library): SECP256K1_BUILD_AS_OBJECT makes the engine
# an OBJECT library, so its objects splice directly into Knots' own bitcoind link —
# a continuation of the node's source, NOT a static/shared boundary. With
# -ffunction-sections + Knots' --gc-sections, any module Knots never calls is
# dropped from bitcoind at the final link. Keep -O3 (Release): the ~1 MB ecmult
# precompute is the size floor, not code — -Os/-Oz would slow the hot path.
# Mirrors SECP256K1_BUILD_LIBBITCOIN above.
option(SECP256K1_BUILD_KNOTS
       "[Bitcoin Knots] Minimal libsecp256k1 backend: ecdsa+recovery+schnorr+extrakeys+ellswift; everything else off" OFF)
if(SECP256K1_BUILD_KNOTS)
    # libsecp256k1-compatible contract: CT signing + strict libsecp ABI (same as the
    # libbitcoin profile). We deliberately do NOT set SECP256K1_CORE_BACKEND_MODE: that
    # mode bakes a static w=8 ecmult table into the binary (~0.85 MB of .rodata) for
    # deterministic / shared-RAM startup, which INFLATES the file (2.27 MB vs 1.26 MB).
    # Knots wants the small binary, so use the runtime-built table instead (measured:
    # CT+strict module-stripped = 1.26 MB, stock-libsecp parity, sigs byte-identical).
    # Opt into the static table explicitly with -DSECP256K1_CORE_BACKEND_MODE=ON if
    # cold-start / shared-RAM matters more than file size for your deployment.
    set(SECP256K1_BUILD_SHIM   ON)
    set(SECP256K1_REQUIRE_CT   ON)
    set(UFSECP_BITCOIN_STRICT  ON)
    # Module/subtree splice into the consumer's link (same as the libbitcoin profile):
    # the engine becomes an OBJECT library whose objects link directly into bitcoind.
    # This is the DEFAULT Knots path. It is mutually exclusive with the standalone
    # whole-archive .so (SECP256K1_SHIM_BUILD_SHARED) — an OBJECT library cannot be
    # WHOLE_ARCHIVE'd into a shared lib — so only force AS_OBJECT for the module path;
    # if a consumer explicitly asks for the .so, leave AS_OBJECT off so it whole-
    # archives a static engine instead.
    if(NOT SECP256K1_SHIM_BUILD_SHARED)
        set(SECP256K1_BUILD_AS_OBJECT ON)
    endif()
    # Strip every crypto module Knots does not call. KEEP: core ecdsa/recovery/
    # schnorr/extrakeys (always compiled), PIPPENGER (ecmult_gen comb — sign/verify
    # core; gc-sections drops its unused batch/MSM code from bitcoind), and BIP-324
    # ellswift (required by Knots v2 P2P).
    set(SECP256K1_BUILD_MUSIG2   OFF)
    set(SECP256K1_BUILD_FROST    OFF)
    set(SECP256K1_BUILD_ZK       OFF)
    set(SECP256K1_BUILD_ECIES    OFF)
    set(SECP256K1_BUILD_BIP352   OFF)
    set(SECP256K1_BUILD_LTC_SP   OFF)   # MUST be OFF when BIP352 OFF (FATAL_ERROR else)
    set(SECP256K1_BUILD_ADAPTOR  OFF)
    set(SECP256K1_BUILD_WALLET   OFF)
    set(SECP256K1_BUILD_ETHEREUM OFF)
    set(SECP256K1_BUILD_BCH      OFF)
    set(SECP256K1_BUILD_BIP324   ON)    # ellswift — required by Knots
    # Strip harness / bindings.
    set(SECP256K1_BUILD_TESTS    OFF)
    set(SECP256K1_BUILD_BENCH    OFF)
    set(SECP256K1_BUILD_EXAMPLES OFF)
    set(SECP256K1_BUILD_JAVA     OFF)
    # Per-TU section splitting so the consumer's --gc-sections drops any module
    # with no live caller. NOTE: we deliberately do NOT add -fvisibility=hidden
    # here — the shim's secp256k1_* C ABI macro does not (yet) expand to
    # visibility("default"), so hiding everything would strip the ABI from a
    # standalone .so (export table collapses, symbols become undefined). The export
    # table is a .so-only cost (irrelevant to the bundled-tree object/static path);
    # shrinking it is a follow-up that needs the ABI macro to mark exports.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
            $<$<COMPILE_LANGUAGE:C,CXX>:-ffunction-sections>
            $<$<COMPILE_LANGUAGE:C,CXX>:-fdata-sections>)
    endif()
    message(STATUS "secp256k1-fast: ✅ SECP256K1_BUILD_KNOTS — minimal bundled-tree libsecp256k1 backend (ecdsa+recovery+schnorr+extrakeys+ellswift)")
endif()

# x86-64 -march override: use "x86-64-v3" in CI with ccache to avoid SIGILL
# when cached objects from a different runner CPU are reused.  Empty = native.
set(SECP256K1_MARCH "" CACHE STRING "x86-64 -march override (empty = auto-detect)")

# Warning policy: promote warnings to errors (recommended for CI)
option(SECP256K1_WERROR "Treat compiler warnings as errors (-Werror / /WX)" OFF)
option(UFSECP_REFRESH_SOURCE_GRAPH "Refresh the repo source graph during builds (mutates source tree — OFF by default for reproducibility)" OFF)

# ── BCH module (Reusable Payment Addresses, CashAddr, EC grinding) ────────────
# Default OFF — zero impact on Bitcoin Core / standard builds.
# Enable: cmake -DSECP256K1_BUILD_BCH=ON
option(SECP256K1_BUILD_BCH
    "Build BCH-specific modules: RPA (Reusable Payment Addresses), CashAddr, EC grinding pipeline"
    OFF)

# Reproducible build mode (G-7 gate).
# Forces deterministic compiler output: strips source paths, fixes march, disables
# non-deterministic LTO passes, and enables SOURCE_DATE_EPOCH awareness.
# Enable with -DUFSECP_REPRODUCIBLE=ON. Implied by SECP256K1_CORE_BACKEND_MODE.
option(UFSECP_REPRODUCIBLE "Enable reproducible-build flags (path stripping, fixed march)" OFF)
if(SECP256K1_CORE_BACKEND_MODE)
    set(UFSECP_REPRODUCIBLE ON CACHE BOOL "Enabled by SECP256K1_CORE_BACKEND_MODE" FORCE)
endif()
if(UFSECP_REPRODUCIBLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Strip the build directory prefix so debug info paths are relative.
    # cmake_current_binary_dir → "." makes two builds from different checkout
    # directories produce bit-identical DWARF.
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.>
        $<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}=.>
        # Warn if source uses __DATE__ / __TIME__ (non-reproducible).
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wdate-time>
    )
    # Force a stable march (not -march=native) so the artifact is identical
    # regardless of which specific CPU the runner uses.
    set(SECP256K1_MARCH "x86-64-v3" CACHE STRING "Forced to x86-64-v3 by UFSECP_REPRODUCIBLE" FORCE)
    message(STATUS "secp256k1-fast: Reproducible build enabled (path normalization, march=x86-64-v3, Wdate-time)")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

# Global compile definitions
if(SECP256K1_SPEED_FIRST)
    add_compile_definitions(SECP256K1_FAST_NO_SECURITY_CHECKS=1)
endif()

if(SECP256K1_REQUIRE_CT)
    add_compile_definitions(SECP256K1_REQUIRE_CT=1)
    message(STATUS "secp256k1-fast: CT enforcement enabled -- non-CT sign functions are deprecated")
endif()

if(UFSECP_BITCOIN_STRICT)
    add_compile_definitions(UFSECP_BITCOIN_STRICT=1)
endif()

if(SECP256K1_BUILD_ETHEREUM)
    add_compile_definitions(SECP256K1_BUILD_ETHEREUM=1)
    message(STATUS "secp256k1-fast: Ethereum module enabled (Keccak, EIP-55/155/191, ecrecover)")
else()
    message(STATUS "secp256k1-fast: Ethereum module disabled (Bitcoin-only build)")
endif()

# GLV window width override (4-7). Empty = platform default (5 on x86/ARM/RISC-V, 4 on ESP32/WASM).
set(SECP256K1_GLV_WINDOW_WIDTH "" CACHE STRING "GLV window width (4-7, empty = platform default)")
if(SECP256K1_GLV_WINDOW_WIDTH)
    if(NOT SECP256K1_GLV_WINDOW_WIDTH MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "SECP256K1_GLV_WINDOW_WIDTH must be an integer in [4,7]. Got: '${SECP256K1_GLV_WINDOW_WIDTH}'")
    endif()
    if(SECP256K1_GLV_WINDOW_WIDTH LESS 4 OR SECP256K1_GLV_WINDOW_WIDTH GREATER 7)
        message(FATAL_ERROR
            "SECP256K1_GLV_WINDOW_WIDTH out of range [4,7]. Got: ${SECP256K1_GLV_WINDOW_WIDTH}")
    endif()
    add_compile_definitions(SECP256K1_GLV_WINDOW_WIDTH=${SECP256K1_GLV_WINDOW_WIDTH})
    message(STATUS "secp256k1-fast: GLV window width override: ${SECP256K1_GLV_WINDOW_WIDTH}")
endif()

# Resolve effective target architecture (handles macOS cross-compilation).
# On macOS, CMAKE_SYSTEM_PROCESSOR reflects the HOST, not the TARGET when
# cross-compiling via CMAKE_OSX_ARCHITECTURES (e.g. ARM64 host -> x86_64 target).
# See: https://github.com/shrec/UltrafastSecp256k1/issues/101
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        set(_SECP_TARGET_ARCH "x86_64")
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(_SECP_TARGET_ARCH "arm64")
    else()
        set(_SECP_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
else()
    set(_SECP_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Platform detection
if(_SECP_TARGET_ARCH MATCHES "x86_64|AMD64|X64")
    set(SECP256K1_PLATFORM "x86_64")
elseif(_SECP_TARGET_ARCH MATCHES "riscv64|RISCV64")
    set(SECP256K1_PLATFORM "riscv64")
elseif(_SECP_TARGET_ARCH MATCHES "aarch64|ARM64|arm64")
    set(SECP256K1_PLATFORM "aarch64")
else()
    set(SECP256K1_PLATFORM "generic")
endif()

message(STATUS "secp256k1-fast: Detected platform: ${SECP256K1_PLATFORM}")

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Platform-specific -march flag
    if(SECP256K1_PLATFORM STREQUAL "riscv64")
        set(MARCH_FLAG "-march=rv64gc_zba_zbb")
    elseif(SECP256K1_PLATFORM STREQUAL "x86_64")
        if(SECP256K1_MARCH)
            set(MARCH_FLAG "-march=${SECP256K1_MARCH}")
        elseif(APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
            # macOS cross-compilation (ARM64 host -> single-arch x86_64 target)
            set(MARCH_FLAG "-march=x86-64-v3")
        else()
            set(MARCH_FLAG "-march=native")
        endif()
    else()
        set(MARCH_FLAG "")
    endif()
    
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wshadow>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wformat=2>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wundef>
        $<$<COMPILE_LANGUAGE:C,CXX>:${MARCH_FLAG}>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-O3>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-DNDEBUG>
        # Build hardening (supply-chain / CAAS gate requirement)
        # -fstack-protector-strong: stack canaries on vulnerable frames
        $<$<COMPILE_LANGUAGE:C,CXX>:-fstack-protector-strong>
        # -D_FORTIFY_SOURCE=2: fortified glibc wrappers (Release only)
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-D_FORTIFY_SOURCE=2>
    )
    # Position-independent code: -fPIC for shared libs, -fPIE for executables.
    # CMake applies the right variant per target type via this global variable.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if(SECP256K1_WERROR)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
    endif()
    # GCC warns about unsigned __int128 under -Wpedantic (ISO C++ extension).
    # This project fundamentally requires __int128 for 64-bit ECC arithmetic.
    # Clang treats __int128 as a built-in type and does not warn.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-pedantic>)
        # Suppress GCC false-positive -Wstringop-overflow in LTO mode.
        # GCC's LTO analysis cannot trace the buf_len_ < 64 invariant in SHA256::update
        # across the LTO boundary, producing spurious "writing N bytes into region of M"
        # diagnostics for code that is statically correct.
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-stringop-overflow>)
        add_link_options(-Wno-stringop-overflow)
    endif()
    # macOS SDK headers (CFBase.h) trigger -Welaborated-enum-base with -Wpedantic
    if(APPLE)
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wno-elaborated-enum-base>)
    endif()
elseif(MSVC)
    # ------------------------------------------------------------------
    # MSVC optimization (Windows).
    #
    # On Windows x64, MSVC defaults to the SSE2 ABI baseline. Production
    # libbitcoin targets can rely on Intel/AMD SSE4-class machines, but the
    # benchmark sweep showed that forcing /arch:AVX2 or /arch:AVX512 regresses
    # important bigint kernels on cl, so the portable default keeps SSE2 and
    # lets tuned intrinsics/asm handle hot arithmetic. On Windows ARM64, NEON is
    # architectural baseline and there is no x64 /arch switch to apply.
    # Independently-toggleable MSVC perf knobs (defaults chosen empirically by
    # the MSVC-vs-Clang Windows benchmark sweep — see docs/BENCHMARKS / commit).
    set(SECP256K1_MSVC_ARCH "SSE2" CACHE STRING
        "MSVC x64 /arch: ISA level (SSE2|AVX|AVX2|AVX512). ARM64 uses the NEON architectural baseline and ignores this unless misconfigured.")
    set_property(CACHE SECP256K1_MSVC_ARCH PROPERTY STRINGS SSE2 AVX AVX2 AVX512)
    if(NOT SECP256K1_PLATFORM STREQUAL "x86_64" AND NOT SECP256K1_MSVC_ARCH STREQUAL "SSE2")
        message(FATAL_ERROR
            "SECP256K1_MSVC_ARCH=${SECP256K1_MSVC_ARCH} is x86/x64-only; "
            "${SECP256K1_PLATFORM} Windows builds use the platform baseline (ARM64 NEON).")
    endif()
    # /Ob3 (most-aggressive inlining) is a consistent ~8-15% win on the Windows
    # MSVC-vs-Clang sweep (field_mul 20->17ns, field_sqr 15->13ns = Clang parity,
    # scalar_mul 100->92us, field add 6.3->2.2ns) with no measurable regression,
    # so it is ON by default. /arch:AVX2 was neutral-to-negative for the
    # non-vectorizable bigint kernels, hence ARCH stays SSE2 above.
    #
    # SECP256K1_MSVC_WPO (/GL whole-program + /LTCG) is OFF by default and is a
    # CONSUMER opt-in, NOT a forced library policy. Reason: /GL emits whole-program
    # IL into every object, so a *static* lib compiled with /GL forces EVERY
    # downstream consumer's linker into /LTCG for its whole program — even consumers
    # who do not want it. Some consumers (e.g. libbitcoin) measure LTCG as a net
    # runtime REGRESSION for their workload, so the library must not impose it.
    # Our own standalone microbenchmarks see a small WPO gain; build with
    # -DSECP256K1_MSVC_WPO=ON to opt in for a self-contained release/benchmark
    # build. When OFF, no /GL is emitted, so the produced .lib never dictates the
    # consumer's link-time optimization. See docs/BUILD_OPTIONS.md.
    option(SECP256K1_MSVC_OB3 "MSVC: /Ob3 most-aggressive inlining" ON)
    option(SECP256K1_MSVC_WPO "MSVC: /GL whole-program + /LTCG + /OPT:REF,ICF (opt-in; a /GL static lib forces consumer /LTCG)" OFF)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
        $<$<COMPILE_LANGUAGE:C,CXX>:/permissive->
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/O2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Oi>   # intrinsics
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Gy>   # COMDAT funcs -> /OPT:REF,ICF
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Gw>   # COMDAT globals
    )
    if(SECP256K1_MSVC_WPO)
        add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/GL>)
        add_link_options($<$<CONFIG:Release>:/LTCG> $<$<CONFIG:Release>:/OPT:REF> $<$<CONFIG:Release>:/OPT:ICF>)
    endif()
    if(SECP256K1_MSVC_OB3)
        add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Ob3>)
    endif()
    if(SECP256K1_PLATFORM STREQUAL "x86_64" AND NOT SECP256K1_MSVC_ARCH STREQUAL "SSE2")
        add_compile_options(
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/arch:${SECP256K1_MSVC_ARCH}>)
    endif()
    message(STATUS "secp256k1-fast: MSVC opt: /O2 /Oi /Gy /Gw arch=${SECP256K1_MSVC_ARCH} Ob3=${SECP256K1_MSVC_OB3} WPO=${SECP256K1_MSVC_WPO}")
    if(SECP256K1_WERROR)
        add_compile_options(/WX)
    endif()
endif()

# ---------------------------------------------------------------------------
# Clang on Windows (clang++ or clang-cl, MSVC ABI).
#
# RECOMMENDED FAST PATH ON WINDOWS x86_64: build with clang-cl. It is a drop-in,
# MSVC-ABI-compatible Clang driver (link.exe / MSVC CRT / .pdb / Visual Studio
# toolset) whose codegen runs the inline MULX/ADCX/ADOX field kernels that MSVC
# cl cannot (no inline asm, no __int128). Measured ~3.3-3.6x faster scalar_mul /
# batch_verify / FROST than cl, i.e. near clang++ parity, while staying in the
# MSVC toolchain. See benchmarks/comparison/windows_msvc_vs_clang_20260614.md.
#   cmake --preset windows-clang-cl     (or -DCMAKE_CXX_COMPILER=clang-cl)
#
# Windows ARM64 is supported by the same clang-cl path: AArch64 has mandatory
# NEON, and compiler-rt must be selected as clang_rt.builtins-aarch64.lib.
#
# The native __int128 field code emits 128-bit runtime calls (__modti3 /
# __umodti3 / __multi3) that the MSVC CRT does not provide, so auto-locate and
# link the matching compiler-rt builtins archive — otherwise the link fails with
# LNK2019 unresolved __umodti3. (Harmless no-op on the cl build.)
# ---------------------------------------------------------------------------
if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
        OUTPUT_VARIABLE _ufsecp_clang_res
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _ufsecp_res_rc)
    if(_ufsecp_res_rc EQUAL 0 AND _ufsecp_clang_res)
        file(TO_CMAKE_PATH "${_ufsecp_clang_res}" _ufsecp_clang_res)
        if(SECP256K1_PLATFORM STREQUAL "x86_64")
            set(_ufsecp_clang_rt_arch "x86_64")
        elseif(SECP256K1_PLATFORM STREQUAL "aarch64")
            set(_ufsecp_clang_rt_arch "aarch64")
        else()
            set(_ufsecp_clang_rt_arch "")
        endif()
        if(_ufsecp_clang_rt_arch)
            set(_ufsecp_builtins "${_ufsecp_clang_res}/lib/windows/clang_rt.builtins-${_ufsecp_clang_rt_arch}.lib")
            if(EXISTS "${_ufsecp_builtins}")
                add_link_options("${_ufsecp_builtins}")
                message(STATUS "secp256k1-fast: Clang/Windows ${SECP256K1_PLATFORM} — linking compiler-rt builtins: ${_ufsecp_builtins}")
            else()
                message(WARNING "secp256k1-fast: Clang/Windows ${SECP256K1_PLATFORM} — clang_rt.builtins not found at "
                                "${_ufsecp_builtins}; 128-bit runtime calls (__umodti3) may be unresolved.")
            endif()
        else()
            message(WARNING "secp256k1-fast: Clang/Windows — no compiler-rt builtins mapping for platform "
                            "${SECP256K1_PLATFORM}; 128-bit runtime calls (__umodti3) may be unresolved.")
        endif()
    endif()
endif()

# Subdirectories
# Honor explicit `-DBUILD_TESTING=ON` even when the project-specific
# SECP256K1_BUILD_TESTS/SECP256K1_BUILD_BENCH toggles are OFF. CI shim gates use
# that profile: audit/ registers standalone CTest targets, but CTest must be
# enabled at the top level for `ctest --test-dir out/...` to see them.
if(BUILD_TESTING)
    include(CTest)
    enable_testing()
endif()
if(SECP256K1_BUILD_CPU)
    # Enable testing so src/cpu/CMakeLists.txt BUILD_TESTING guard works
    if(SECP256K1_BUILD_TESTS OR SECP256K1_BUILD_BENCH)
        set(BUILD_TESTING ON CACHE BOOL "Enable testing" FORCE)
        include(CTest)   # generates DartConfiguration.tcl for -T MemCheck
    endif()
    add_subdirectory(src/cpu)
endif()

# BCH module — added after CPU lib so fastsecp256k1 target already exists
if(SECP256K1_BUILD_BCH)
    add_subdirectory(src/bch)
endif()


if(SECP256K1_BUILD_CUDA)
    # On some platforms (notably Windows), nvcc requires an MSVC host compiler (cl.exe).
    # If it's not available, we'd rather fall back to CPU-only than fail configuration.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY OR CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "")
            set(CMAKE_CUDA_RUNTIME_LIBRARY Static CACHE STRING
                "CUDA runtime linkage (Static by default so engine binaries do not depend on libcudart.so/cudart64*.dll)"
                FORCE)
        endif()
        set_property(CACHE CMAKE_CUDA_RUNTIME_LIBRARY PROPERTY STRINGS Static Shared None)
        enable_language(CUDA)
        add_subdirectory(src/cuda)
    else()
        message(WARNING "SECP256K1_BUILD_CUDA=ON but CUDA toolchain was not found or failed to configure (e.g. nvcc cannot find cl.exe). Falling back to CPU-only build.")
    endif()
endif()

if(SECP256K1_BUILD_OPENCL)
    find_package(OpenCL QUIET)
    if(NOT OpenCL_FOUND AND WIN32)
        # Fallback: look for system OpenCL.lib (installed by GPU drivers)
        find_library(_OCL_FALLBACK OpenCL
            PATHS "$ENV{SYSTEMROOT}/System32" "C:/Windows/System32"
        )
        if(_OCL_FALLBACK)
            set(OpenCL_FOUND TRUE)
            message(STATUS "OpenCL: using system fallback (${_OCL_FALLBACK})")
        endif()
    endif()
    if(OpenCL_FOUND OR (WIN32 AND _OCL_FALLBACK))
        add_subdirectory(src/opencl)
    else()
        message(WARNING "SECP256K1_BUILD_OPENCL=ON but OpenCL not found. Skipping OpenCL build.")
    endif()
endif()

# ROCm/HIP build -- reuses cuda/ sources with portable math fallbacks
if(SECP256K1_BUILD_ROCM)
    # CMake 3.21+ has native HIP language support
    cmake_minimum_required(VERSION 3.21)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "ROCm/HIP found: ${hip_VERSION}")
        enable_language(HIP)
        add_subdirectory(src/cuda cuda_rocm)  # reuse sources, separate build dir
    else()
        message(WARNING "SECP256K1_BUILD_ROCM=ON but HIP SDK not found. Skipping ROCm build.")
    endif()
endif()

# Apple Metal backend -- macOS / iOS / visionOS
# Host-side type tests always build; GPU runtime only on Apple
if(SECP256K1_BUILD_METAL)
    if(APPLE)
        enable_language(OBJCXX)
        find_library(_METAL_FW Metal)
        find_library(_FOUNDATION_FW Foundation)
        if(_METAL_FW AND _FOUNDATION_FW)
            message(STATUS "Metal framework found -- building Metal backend (GPU + host tests)")
            add_subdirectory(src/metal)
        else()
            message(WARNING "SECP256K1_BUILD_METAL=ON but Metal.framework not found. Building host tests only.")
            add_subdirectory(src/metal)
        endif()
    else()
        message(STATUS "SECP256K1_BUILD_METAL=ON on non-Apple platform -- building host tests only")
        add_subdirectory(src/metal)
    endif()
endif()

if(SECP256K1_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -- GPU Host Ops Layer (Layer 2) -------------------------------------------
# Must be after cuda/, opencl/, metal/ so backend targets exist.
if(SECP256K1_BUILD_CUDA OR SECP256K1_BUILD_OPENCL OR SECP256K1_BUILD_METAL)
    add_subdirectory(src/gpu)
endif()

# -- Optional stable C ABI layer (ufsecp_*) --------------------------------
# Built before audit/ so that ufsecp_static is available for fuzz tests.
# Native C++ / libsecp256k1-shim integrations should link secp256k1::fast (or
# secp256k1_shim). libufsecp is the optional C ABI / binding / bridge package.
option(SECP256K1_BUILD_CABI
       "Build optional libufsecp C ABI package (ufsecp_* FFI/bridge surface; native engine consumers link secp256k1::fast)"
       ON)
option(SECP256K1_INSTALL_CABI
       "Install optional libufsecp C ABI package during the top-level install"
       OFF)
if(SECP256K1_BUILD_CABI AND SECP256K1_BUILD_CPU)
    add_subdirectory(include/ufsecp)
    message(STATUS "  C ABI (ufsecp):   ON (optional; native integrations link secp256k1::fast)")
endif()

if(SECP256K1_BUILD_JAVA AND SECP256K1_BUILD_CABI AND SECP256K1_BUILD_CPU)
    find_package(JNI QUIET)
    if(JNI_FOUND)
        add_subdirectory(bindings/java)
        message(STATUS "  Java JNI:         ON")
    else()
        message(STATUS "  Java JNI:         OFF (JNI not found)")
    endif()
endif()

# -- Audit infrastructure (standalone CTest targets + unified runner) -------
# All audit-specific targets live in audit/ to keep the library source clean.
#
# Subdirectory ordering: audit/ is intentionally added BEFORE
# compat/libsecp256k1_shim/. This gives the shim layer two complementary
# gating surfaces:
#
#   * unified_audit_runner — single binary, shim-independent by design.
#     Built without the shim target visible, so any shim regression test
#     that needs the libsecp256k1 ABI is wired via the ADVISORY_SKIP_CODE(77)
#     stubs in audit/shim_run_stubs_unified.cpp. The runner stays buildable
#     in environments that intentionally exclude the shim (Bitcoin Core
#     integration smoke, sanitizer matrices, cross-compile targets) and the
#     report cleanly labels these modules `advisory_skipped` rather than
#     forcing them mandatory in scopes where the libsecp256k1 ABI is not
#     in the loop. The shim is exercised authoritatively elsewhere.
#
#   * Standalone CTest targets — authoritative shim-dependent gate.
#     audit/CMakeLists.txt creates one standalone executable per shim-
#     dependent regression test (test_regression_shim_pubkey_sort,
#     test_regression_musig2_session_token, etc.) under
#     `if(TARGET secp256k1_shim)`, after the shim subdirectory is processed.
#     These targets compile and run the real test bodies against the live
#     shim ABI and are the canonical gate for shim regressions. Invoke them
#     via `ctest -L shim` (or any subset matching the relevant label) after
#     a build with `-DSECP256K1_BUILD_SHIM=ON`.
#
# This split is the intended architecture, not a transient workaround: it
# lets the unified runner remain a self-contained artifact that consumers
# can ship without pulling the shim, while the standalone targets keep
# providing strict regression coverage of the shim ABI surface.
if(SECP256K1_BUILD_CPU AND BUILD_TESTING)
    add_subdirectory(audit)
endif()

# -- libsecp256k1 compatibility shim (Bitcoin Core integration) ---------------
# Add the shim subdirectory when EITHER (a) BUILD_SHIM mode is on, OR (b) the shim
# conformance tests are explicitly requested with BUILD_SHIM=OFF
# (-DSECP256K1_SHIM_BUILD_TESTS=ON). Case (b) is required because the shim test block
# is guarded by `NOT SECP256K1_BUILD_SHIM`: without it that block is unreachable and the
# official BIP-340/341/327 vector tests (which exercise the shim ABI — the libsecp drop-in
# surface integrators actually use) never build, so they never run in CI.
if((SECP256K1_BUILD_SHIM OR SECP256K1_SHIM_BUILD_TESTS) AND SECP256K1_BUILD_CPU)
    # Default ON for standalone builds; parent project can override with FORCE before
    # calling add_subdirectory (e.g. Bitcoin Core sets SECP256K1_SHIM_BUILD_TESTS OFF FORCE).
    set(SECP256K1_SHIM_BUILD_TESTS ON CACHE BOOL "Build shim compatibility tests")
    add_subdirectory(compat/libsecp256k1_shim)
endif()

option(SECP256K1_BCHN_SHIM_BUILD_TESTS "Build BCHN libsecp256k1 compatibility shim tests" OFF)
if(SECP256K1_BCHN_SHIM_BUILD_TESTS AND SECP256K1_BUILD_CPU)
    add_subdirectory(compat/libsecp256k1_bchn_shim)
endif()

# -- libbitcoin direct integration (canonical, bridge-free) ------------------
# Header-only C++ surface <ufsecp/libbitcoin.hpp> + the engine. This is the
# canonical libbitcoin integration target (secp256k1::fastsecp256k1_libbitcoin);
# the shim / C ABI / ufsecp_lbtc bridge are compatibility-only and gated behind
# SECP256K1_BUILD_LIBBITCOIN_BRIDGE. Added after src/cpu so the engine target
# fastsecp256k1 already exists.
if(SECP256K1_BUILD_LIBBITCOIN AND SECP256K1_BUILD_CPU)
    if(SECP256K1_BUILD_LIBBITCOIN_TESTS)
        # The libbitcoin profile forces SECP256K1_BUILD_TESTS OFF, so top-level
        # CTest is not enabled above; enable it here so `ctest --test-dir` finds
        # the direct verify test registered in the subdirectory.
        include(CTest)
        enable_testing()
    endif()
    add_subdirectory(compat/libbitcoin_direct)
endif()

# NOTE: bench_compare removed -- use src/src/cpu/bench/bench_unified instead
#       (bench_unified has full apple-to-apple in a single binary, no FetchContent)

# -- Cross-library differential test -----------------------------------------
# Moved to audit/CMakeLists.txt -- enable with -DSECP256K1_BUILD_CROSS_TESTS=ON

# -- Parser fuzz tests ------------------------------------------------------
# Moved to audit/CMakeLists.txt -- enable with -DSECP256K1_BUILD_FUZZ_TESTS=ON

# -- MuSig2 + FROST protocol tests -----------------------------------------
# Moved to audit/CMakeLists.txt -- enable with -DSECP256K1_BUILD_PROTOCOL_TESTS=ON

# Export targets
if(SECP256K1_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install public headers (exclude BCH when the module is OFF;
    # libbitcoin profile: only headers needed by <ufsecp/libbitcoin.hpp>)
    if(SECP256K1_BUILD_LIBBITCOIN)
        install(DIRECTORY src/cpu/include/secp256k1
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.hpp"
            PATTERN "bch" EXCLUDE
            PATTERN "coins" EXCLUDE
            PATTERN "ltc" EXCLUDE
            PATTERN "ecdh.hpp" EXCLUDE
            PATTERN "ellswift.hpp" EXCLUDE
            PATTERN "frost*" EXCLUDE
            PATTERN "musig2.hpp" EXCLUDE
            PATTERN "ecies*" EXCLUDE
            PATTERN "bip32*" EXCLUDE
            PATTERN "bip39*" EXCLUDE
            PATTERN "bip143*" EXCLUDE
            PATTERN "bip144*" EXCLUDE
            PATTERN "bip324*" EXCLUDE
            PATTERN "bip352*" EXCLUDE
            PATTERN "chacha20*" EXCLUDE
            PATTERN "hkdf*" EXCLUDE
            PATTERN "sha256*" EXCLUDE
            PATTERN "sha512*" EXCLUDE
            PATTERN "unicode*" EXCLUDE
            PATTERN "segwit*" EXCLUDE
            PATTERN "address*" EXCLUDE
            PATTERN "adaptor*" EXCLUDE
            PATTERN "benchmark_*" EXCLUDE
            PATTERN "test_framework*" EXCLUDE
            PATTERN "debug_invariants*" EXCLUDE
            PATTERN "context*" EXCLUDE
            PATTERN "fast.hpp" EXCLUDE
            PATTERN "field_26.hpp" EXCLUDE
            PATTERN "field_branchless.hpp" EXCLUDE
            PATTERN "field_asm.hpp" EXCLUDE
            PATTERN "zk.hpp" EXCLUDE
            PATTERN "pedersen.hpp" EXCLUDE
        )
    elseif(SECP256K1_BUILD_BCH)
        install(DIRECTORY src/cpu/include/secp256k1
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.hpp"
        )
    else()
        install(DIRECTORY src/cpu/include/secp256k1
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.hpp"
            PATTERN "bch" EXCLUDE
        )
    endif()

    # Install generated version header
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/secp256k1/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/secp256k1
    )

    # libbitcoin direct integration: canonical bridge-free C++ surface. Install
    # the header <ufsecp/libbitcoin.hpp> and add the interface target to the
    # secp256k1-fast-targets export set so the package exposes
    # secp256k1::fastsecp256k1_libbitcoin alongside the engine.
    if(SECP256K1_BUILD_LIBBITCOIN AND TARGET fastsecp256k1_libbitcoin)
        install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/compat/libbitcoin_direct/include/ufsecp/libbitcoin.hpp"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ufsecp
        )
        install(TARGETS fastsecp256k1_libbitcoin
            EXPORT secp256k1-fast-targets
        )
    endif()

    # Generate package config
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/secp256k1-fast-config-version.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/secp256k1-fast-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/secp256k1-fast-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/secp256k1-fast
    )
    
    # Install targets
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/secp256k1-fast-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/secp256k1-fast-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/secp256k1-fast
    )
    
    # Pkg-config
    if(SECP256K1_INSTALL_PKGCONFIG)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/secp256k1-fast.pc.in"
            "${CMAKE_CURRENT_BINARY_DIR}/secp256k1-fast.pc"
            @ONLY
        )
        install(
            FILES "${CMAKE_CURRENT_BINARY_DIR}/secp256k1-fast.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        )
    endif()

    if(SECP256K1_INSTALL_CABI)
        if(NOT SECP256K1_BUILD_CABI)
            message(FATAL_ERROR
                "SECP256K1_INSTALL_CABI=ON requires SECP256K1_BUILD_CABI=ON")
        endif()

        install(FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ufsecp/ufsecp.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ufsecp/ufsecp256k1.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ufsecp/ufsecp_error.h"
            "${CMAKE_CURRENT_BINARY_DIR}/include/ufsecp/ufsecp_version.h"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ufsecp
        )

        if(TARGET secp256k1_gpu_host)
            install(FILES
                "${CMAKE_CURRENT_SOURCE_DIR}/include/ufsecp/ufsecp_gpu.h"
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ufsecp
            )
        endif()

        # Legacy bridge header — compatibility-only (ufsecp_lbtc_* C batch API).
        # Installed only when the compatibility bridge is explicitly opted in.
        if(SECP256K1_BUILD_LIBBITCOIN_BRIDGE)
            install(FILES
                "${CMAKE_CURRENT_SOURCE_DIR}/compat/libbitcoin_bridge/include/ufsecp_libbitcoin.h"
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ufsecp
            )
        endif()

        if(TARGET ufsecp_shared)
            install(TARGETS ufsecp_shared
                EXPORT ufsecpTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
        endif()
        if(TARGET ufsecp_static)
            install(TARGETS ufsecp_static
                EXPORT ufsecpTargets
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
        endif()
        install(EXPORT ufsecpTargets
            FILE ufsecpTargets.cmake
            NAMESPACE ufsecp::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ufsecp
        )

        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ufsecp/ufsecp.pc.in"
            "${CMAKE_CURRENT_BINARY_DIR}/ufsecp.pc"
            @ONLY
        )
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ufsecp.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        )

        message(STATUS
            "  C ABI install:    ON (libufsecp; optional FFI/bridge package)")
    endif()
endif()

# -- Code quality targets (local dev + CI) -----------------------------------
set(UFSECP_CODE_QUALITY_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/ci/run_code_quality.py")
if(Python3_Interpreter_FOUND AND EXISTS "${UFSECP_CODE_QUALITY_RUNNER}")
    # Human-readable report (cmake --build build --target code-quality)
    add_custom_target(code-quality
        COMMAND "${Python3_EXECUTABLE}" "${UFSECP_CODE_QUALITY_RUNNER}"
                --fail-on-regression
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running code-quality scanners (regression gate)"
        USES_TERMINAL
        VERBATIM
    )
    # JSON report variant (cmake --build build --target code-quality-report)
    add_custom_target(code-quality-report
        COMMAND "${Python3_EXECUTABLE}" "${UFSECP_CODE_QUALITY_RUNNER}"
                --json --fail-on-regression
                -o "${CMAKE_BINARY_DIR}/code_quality_report.json"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Generating code-quality JSON report"
        USES_TERMINAL
        VERBATIM
    )
endif()

# -- CPack packaging ---------------------------------------------------------
set(UFSECP_SOURCE_GRAPH_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/tools/source_graph_kit/source_graph.py")
if(UFSECP_REFRESH_SOURCE_GRAPH)
    if(Python3_Interpreter_FOUND AND EXISTS "${UFSECP_SOURCE_GRAPH_TOOL}")
        add_custom_target(ufsecp_source_graph_refresh ALL
            COMMAND "${Python3_EXECUTABLE}" "${UFSECP_SOURCE_GRAPH_TOOL}" build -i --best-effort
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Refreshing UltrafastSecp256k1 source graph incrementally"
            USES_TERMINAL
            VERBATIM
        )
    else()
        message(STATUS "secp256k1-fast: source graph refresh disabled (missing Python3 interpreter or source_graph.py)")
    endif()
endif()

set(CPACK_PACKAGE_NAME "UltrafastSecp256k1")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "shrec")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "High-performance secp256k1 ECC library with stable C ABI (ufsecp)")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/shrec/UltrafastSecp256k1")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${_SECP_TARGET_ARCH}")
set(CPACK_STRIP_FILES ON)

# Generate ZIP on all platforms, plus TGZ + DEB + RPM on Unix
if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ;ZIP;DEB;RPM")
endif()

# DEB-specific
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "shrec <shrec@users.noreply.github.com>")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.17)")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

# Map target arch -> DEB architecture (critical for cross-compilation where
# dpkg --print-architecture returns the HOST arch, not the TARGET arch).
if(_SECP_TARGET_ARCH MATCHES "aarch64|ARM64|arm64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
elseif(_SECP_TARGET_ARCH MATCHES "x86_64|AMD64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
endif()

# RPM-specific
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_GROUP "System Environment/Libraries")
set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.17")

# Source package
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "/build[^/]*/" "/[.]git/" "/[.]github/" "/[.]vscode/" "[.]DS_Store$" "[.]user$")

include(CPack)

# Summary
message(STATUS "")
message(STATUS "+===========================================================+")
message(STATUS "|   UltrafastSecp256k1 Configuration                        |")
message(STATUS "+===========================================================+")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Platform:         ${SECP256K1_PLATFORM}")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "  Components:")
message(STATUS "    CPU:            ${SECP256K1_BUILD_CPU}")
message(STATUS "    CUDA:           ${SECP256K1_BUILD_CUDA}")
message(STATUS "    ROCm/HIP:       ${SECP256K1_BUILD_ROCM}")
message(STATUS "    OpenCL:         ${SECP256K1_BUILD_OPENCL}")
message(STATUS "    Metal:          ${SECP256K1_BUILD_METAL}")
message(STATUS "    Tests:          ${SECP256K1_BUILD_TESTS}")
message(STATUS "    Benchmarks:     ${SECP256K1_BUILD_BENCH}")

message(STATUS "    Examples:       ${SECP256K1_BUILD_EXAMPLES}")
message(STATUS "    Java JNI:       ${SECP256K1_BUILD_JAVA}")
message(STATUS "")
message(STATUS "  Optimizations:")
message(STATUS "    Assembly:       ${SECP256K1_USE_ASM}")
message(STATUS "    Speed First:    ${SECP256K1_SPEED_FIRST}")
message(STATUS "    Bitcoin Strict: ${UFSECP_BITCOIN_STRICT}")
message(STATUS "")
message(STATUS "===========================================================")
message(STATUS "")
