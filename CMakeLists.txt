cmake_minimum_required(VERSION 3.18)

# Read version from VERSION.txt (single source of truth)
# NOTE: named VERSION.txt (not VERSION) to avoid clashing with the
#       C++20 <version> header on case-insensitive filesystems (macOS).
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" _version_raw)
string(STRIP "${_version_raw}" UFSECP_VERSION)

project(UltrafastSecp256k1
    VERSION ${UFSECP_VERSION}
    DESCRIPTION "Ultra high-performance secp256k1 elliptic curve cryptography library"
    LANGUAGES CXX
)

# Project metadata
set(PROJECT_HOMEPAGE_URL "https://github.com/shrec/UltrafastSecp256k1")
set(PROJECT_LICENSE "AGPL-3.0")

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
option(SECP256K1_USE_ASM "Enable assembly optimizations (x64/RISC-V)" ON)

# Installation options
option(SECP256K1_INSTALL "Generate install target" ON)
option(SECP256K1_INSTALL_PKGCONFIG "Install pkg-config file" ON)

# Performance/safety tradeoffs
option(SECP256K1_SPEED_FIRST "Prioritize speed over safety checks" OFF)
option(SECP256K1_BUILD_SHARED "Build shared library" OFF)

# Global compile definitions
if(SECP256K1_SPEED_FIRST)
    add_compile_definitions(SECP256K1_FAST_NO_SECURITY_CHECKS=1)
endif()

# Platform detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
    set(SECP256K1_PLATFORM "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(SECP256K1_PLATFORM "riscv64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
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
        set(MARCH_FLAG "-march=native")
    else()
        set(MARCH_FLAG "")
    endif()
    
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:C,CXX>:${MARCH_FLAG}>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-O3>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-DNDEBUG>
    )
elseif(MSVC)
    add_compile_options(
        /W4
        /permissive-
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
    )
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

# ROCm/HIP build — reuses cuda/ sources with portable math fallbacks
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

# Apple Metal backend — macOS / iOS / visionOS
# Host-side type tests always build; GPU runtime only on Apple
if(SECP256K1_BUILD_METAL)
    if(APPLE)
        find_library(_METAL_FW Metal)
        find_library(_FOUNDATION_FW Foundation)
        if(_METAL_FW AND _FOUNDATION_FW)
            message(STATUS "Metal framework found — building Metal backend (GPU + host tests)")
            add_subdirectory(metal)
        else()
            message(WARNING "SECP256K1_BUILD_METAL=ON but Metal.framework not found. Building host tests only.")
            add_subdirectory(metal)
        endif()
    else()
        message(STATUS "SECP256K1_BUILD_METAL=ON on non-Apple platform — building host tests only")
        add_subdirectory(metal)
    endif()
endif()

if(SECP256K1_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Stable C ABI layer (ufsecp_*) ─────────────────────────────────────────
option(SECP256K1_BUILD_CABI "Build the stable ufsecp_* C ABI library" ON)
if(SECP256K1_BUILD_CABI AND SECP256K1_BUILD_CPU)
    add_subdirectory(include/ufsecp)
    message(STATUS "  C ABI (ufsecp):   ON")
endif()

# ── Cross-library differential test (Phase I, Task 1.1.4) ─────────────────
# Links both UltrafastSecp256k1 AND bitcoin-core/libsecp256k1 in one process
# and compares outputs (pubkey derivation, ECDSA, Schnorr, point arithmetic).
# OFF by default — requires internet for FetchContent download.
option(SECP256K1_BUILD_CROSS_TESTS
       "Build in-process differential tests against bitcoin-core/libsecp256k1" OFF)

if(SECP256K1_BUILD_CROSS_TESTS AND SECP256K1_BUILD_CPU AND BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        libsecp256k1_ref
        GIT_REPOSITORY https://github.com/bitcoin-core/secp256k1.git
        GIT_TAG        v0.6.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(libsecp256k1_ref)
    if(NOT libsecp256k1_ref_POPULATED)
        FetchContent_Populate(libsecp256k1_ref)
        # Configure reference lib: static only, disable its tests/bench, enable modules we test
        set(BUILD_SHARED_LIBS                  OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_SHARED             OFF CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_SCHNORRSIG ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_EXTRAKEYS  ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_RECOVERY   ON  CACHE BOOL "" FORCE)
        set(SECP256K1_ENABLE_MODULE_ECDH       ON  CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_BENCHMARK          OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_TESTS              OFF CACHE BOOL "" FORCE)
        set(SECP256K1_BUILD_EXHAUSTIVE_TESTS   OFF CACHE BOOL "" FORCE)
        add_subdirectory(
            ${libsecp256k1_ref_SOURCE_DIR}
            ${libsecp256k1_ref_BINARY_DIR}
            EXCLUDE_FROM_ALL
        )
    endif()

    add_executable(test_cross_libsecp256k1
        tests/test_cross_libsecp256k1.cpp
    )
    target_link_libraries(test_cross_libsecp256k1
        PRIVATE fastsecp256k1 secp256k1
    )
    target_include_directories(test_cross_libsecp256k1
        PRIVATE ${libsecp256k1_ref_SOURCE_DIR}/include
    )
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_cross_libsecp256k1 PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME cross_libsecp256k1 COMMAND test_cross_libsecp256k1)
    set_tests_properties(cross_libsecp256k1 PROPERTIES TIMEOUT 300)
    message(STATUS "  Cross-test vs libsecp256k1: ON (ref: v0.6.0)")
endif()

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

# ── CPack packaging ─────────────────────────────────────────────────────────
set(CPACK_PACKAGE_NAME "UltrafastSecp256k1")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "shrec")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "High-performance secp256k1 ECC library with stable C ABI (ufsecp)")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/shrec/UltrafastSecp256k1")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
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

# Map target arch → DEB architecture (critical for cross-compilation where
# dpkg --print-architecture returns the HOST arch, not the TARGET arch).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
endif()

# RPM-specific
set(CPACK_RPM_PACKAGE_LICENSE "AGPL-3.0-or-later")
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
message(STATUS "╔═══════════════════════════════════════════════════════════╗")
message(STATUS "║   UltrafastSecp256k1 Configuration                        ║")
message(STATUS "╚═══════════════════════════════════════════════════════════╝")
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
message(STATUS "")
message(STATUS "  Optimizations:")
message(STATUS "    Assembly:       ${SECP256K1_USE_ASM}")
message(STATUS "    Speed First:    ${SECP256K1_SPEED_FIRST}")
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "")
