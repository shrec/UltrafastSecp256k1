cmake_minimum_required(VERSION 3.18)

project(UltrafastSecp256k1
    VERSION 1.0.0
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

# Build options
option(SECP256K1_BUILD_CPU "Build CPU implementation" ON)
option(SECP256K1_BUILD_CUDA "Build CUDA GPU support" OFF)
option(SECP256K1_BUILD_OPENCL "Build OpenCL support" OFF)
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
        message(WARNING "SECP256K1_BUILD_CUDA=ON მაგრამ CUDA toolchain ვერ მოიძებნა/არ იკონფიგურება (მაგ: nvcc ვერ პოულობს cl.exe). CPU-only ბილდზე გადავდივართ.")
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

if(SECP256K1_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Export targets
if(SECP256K1_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    
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
message(STATUS "    OpenCL:         ${SECP256K1_BUILD_OPENCL}")
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
