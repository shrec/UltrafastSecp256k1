cmake_minimum_required(VERSION 3.18)

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

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
option(SECP256K1_BUILD_TESTS "Build test suite" ON)
option(SECP256K1_BUILD_BENCH "Build benchmarks" ON)
option(SECP256K1_BUILD_EXAMPLES "Build example programs" ON)
option(SECP256K1_BUILD_JAVA "Build Java JNI bindings" ON)
option(SECP256K1_USE_ASM "Enable assembly optimizations (x64/RISC-V)" ON)

# Installation options
option(SECP256K1_INSTALL "Generate install target" ON)
option(SECP256K1_INSTALL_PKGCONFIG "Install pkg-config file" ON)

# Performance/safety tradeoffs
option(SECP256K1_SPEED_FIRST "Prioritize speed over safety checks" OFF)
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

# x86-64 -march override: use "x86-64-v3" in CI with ccache to avoid SIGILL
# when cached objects from a different runner CPU are reused.  Empty = native.
set(SECP256K1_MARCH "" CACHE STRING "x86-64 -march override (empty = auto-detect)")

# Warning policy: promote warnings to errors (recommended for CI)
option(SECP256K1_WERROR "Treat compiler warnings as errors (-Werror / /WX)" OFF)
option(UFSECP_REFRESH_SOURCE_GRAPH "Refresh the repo source graph during builds" ON)

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
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
        $<$<COMPILE_LANGUAGE:C,CXX>:/permissive->
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/O2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/GL>
    )
    if(SECP256K1_WERROR)
        add_compile_options(/WX)
    endif()
endif()

# Subdirectories
if(SECP256K1_BUILD_CPU)
    # Enable testing so cpu/CMakeLists.txt BUILD_TESTING guard works
    if(SECP256K1_BUILD_TESTS OR SECP256K1_BUILD_BENCH)
        set(BUILD_TESTING ON CACHE BOOL "Enable testing" FORCE)
        include(CTest)   # generates DartConfiguration.tcl for -T MemCheck
    endif()
    add_subdirectory(cpu)
endif()

if(SECP256K1_BUILD_CUDA)
    # On some platforms (notably Windows), nvcc requires an MSVC host compiler (cl.exe).
    # If it's not available, we'd rather fall back to CPU-only than fail configuration.
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_subdirectory(cuda)
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
        add_subdirectory(opencl)
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
        add_subdirectory(cuda cuda_rocm)  # reuse sources, separate build dir
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
            add_subdirectory(metal)
        else()
            message(WARNING "SECP256K1_BUILD_METAL=ON but Metal.framework not found. Building host tests only.")
            add_subdirectory(metal)
        endif()
    else()
        message(STATUS "SECP256K1_BUILD_METAL=ON on non-Apple platform -- building host tests only")
        add_subdirectory(metal)
    endif()
endif()

if(SECP256K1_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -- GPU Host Ops Layer (Layer 2) -------------------------------------------
# Must be after cuda/, opencl/, metal/ so backend targets exist.
if(SECP256K1_BUILD_CUDA OR SECP256K1_BUILD_OPENCL OR SECP256K1_BUILD_METAL)
    add_subdirectory(gpu)
endif()

# -- Stable C ABI layer (ufsecp_*) -----------------------------------------
# Built before audit/ so that ufsecp_static is available for fuzz tests.
option(SECP256K1_BUILD_CABI "Build the stable ufsecp_* C ABI library" ON)
if(SECP256K1_BUILD_CABI AND SECP256K1_BUILD_CPU)
    add_subdirectory(include/ufsecp)
    message(STATUS "  C ABI (ufsecp):   ON")
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
if(SECP256K1_BUILD_CPU AND BUILD_TESTING)
    add_subdirectory(audit)
endif()

# -- libsecp256k1 compatibility shim (Bitcoin Core integration) ---------------
if(SECP256K1_BUILD_SHIM AND SECP256K1_BUILD_CPU)
    set(SECP256K1_SHIM_BUILD_TESTS ON CACHE BOOL "" FORCE)
    add_subdirectory(compat/libsecp256k1_shim)
endif()

# NOTE: bench_compare removed -- use cpu/bench/bench_unified instead
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

    # Install public headers
    install(DIRECTORY cpu/include/secp256k1
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
    )

    # Install generated version header
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/secp256k1/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/secp256k1
    )

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
endif()

# -- Code quality targets (local dev + CI) -----------------------------------
set(UFSECP_CODE_QUALITY_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_code_quality.py")
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
