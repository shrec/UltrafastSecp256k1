# ============================================================================
# UltrafastSecp256k1 -- ufsecp stable C ABI library
# ============================================================================
# Builds libufsecp.so / ufsecp.dll / libufsecp.dylib
#
# Two modes:
#
#   1) Sub-project (preferred) -- add_subdirectory() from the top-level CMake.
#      This links against the already-compiled `fastsecp256k1` CPU target.
#
#   2) Standalone -- configure directly:
#        cmake -S include/ufsecp -B out/ufsecp -DCMAKE_BUILD_TYPE=Release
#      Gathers CPU sources automatically.
# ============================================================================

cmake_minimum_required(VERSION 3.18)

# Read version from VERSION.txt (single source of truth)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION.txt" _version_raw)
string(STRIP "${_version_raw}" UFSECP_VERSION)

project(ufsecp
    VERSION ${UFSECP_VERSION}
    DESCRIPTION "UltrafastSecp256k1 -- Stable C ABI"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -- Generate ufsecp_version.h from template --------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ufsecp_version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ufsecp_version.h"
    @ONLY
)

# -- Paths ------------------------------------------------------------------
set(ULTRAFAST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")  # UltrafastSecp256k1/
set(CPU_INCLUDE_DIR "${ULTRAFAST_ROOT}/src/cpu/include")
set(SHARED_INCLUDE_DIR "${ULTRAFAST_ROOT}/include")

# -- Library target ---------------------------------------------------------

option(UFSECP_BUILD_SHARED "Build shared library" ON)
option(UFSECP_BUILD_STATIC "Build static library" ON)

set(UFSECP_IMPL_SRC "${ULTRAFAST_ROOT}/src/cpu/src/ufsecp_impl.cpp")

# GPU C ABI implementation (only when gpu host layer is built)
set(UFSECP_GPU_IMPL_SRC "")
if(TARGET secp256k1_gpu_host)
    set(UFSECP_GPU_IMPL_SRC "${ULTRAFAST_ROOT}/src/cpu/src/ufsecp_gpu_impl.cpp")
endif()

# libbitcoin acceleration bridge — compiled INTO this C ABI library (central
# source, not a separate lib) under the libbitcoin profile. Provides the
# ufsecp_lbtc_* batch script-signature verify + BIP-352 scan entry points,
# dispatching to GPU (when the host layer is present) with CPU fallback.
set(UFSECP_LBTC_SRC "")
if(SECP256K1_BUILD_LIBBITCOIN)
    set(UFSECP_LBTC_SRC "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/src/ufsecp_libbitcoin.cpp")
endif()

# Helper: configure a target with the right includes/definitions
function(ufsecp_configure tgt)
    target_include_directories(${tgt}
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
                $<INSTALL_INTERFACE:include>
        PRIVATE ${CPU_INCLUDE_DIR}
                ${SHARED_INCLUDE_DIR}
    )
    target_compile_definitions(${tgt} PRIVATE UFSECP_BUILDING)

    # Link against parent-project CPU static lib if available
    if(TARGET fastsecp256k1)
        target_link_libraries(${tgt} PRIVATE fastsecp256k1)
    elseif(TARGET UltrafastSecp256k1_cpu)
        target_link_libraries(${tgt} PRIVATE UltrafastSecp256k1_cpu)
    elseif(TARGET secp256k1_cpu)
        target_link_libraries(${tgt} PRIVATE secp256k1_cpu)
    else()
        # Standalone: gather all CPU .cpp sources
        file(GLOB _CPU_SOURCES "${ULTRAFAST_ROOT}/src/cpu/src/*.cpp")
        # Exclude MSVC-incompatible 128-bit intrinsic path when needed
        if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(FILTER _CPU_SOURCES EXCLUDE REGEX "field_52\\.cpp$")
            target_compile_definitions(${tgt} PRIVATE SECP256K1_NO_INT128)
        endif()
        target_sources(${tgt} PRIVATE ${_CPU_SOURCES})
    endif()

    set_target_properties(${tgt} PROPERTIES
        VERSION   ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(${tgt} PROPERTIES
            CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
    endif()

    # Link GPU host ops layer if available
    if(TARGET secp256k1_gpu_host)
        target_link_libraries(${tgt} PRIVATE secp256k1_gpu_host)
        target_include_directories(${tgt} PRIVATE
            ${ULTRAFAST_ROOT}/src/gpu/include
        )
    endif()

    # libbitcoin bridge: expose its header (PUBLIC so consumers can include it)
    # and enable GPU dispatch only when the GPU host layer is in the build.
    if(SECP256K1_BUILD_LIBBITCOIN)
        target_include_directories(${tgt} PUBLIC
            $<BUILD_INTERFACE:${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/include>
            $<INSTALL_INTERFACE:include>
        )
        if(TARGET secp256k1_gpu_host)
            target_compile_definitions(${tgt} PRIVATE UFSECP_LBTC_WITH_GPU)
        endif()
    endif()
endfunction()

# Shared library
if(UFSECP_BUILD_SHARED)
    add_library(ufsecp_shared SHARED ${UFSECP_IMPL_SRC} ${UFSECP_GPU_IMPL_SRC} ${UFSECP_LBTC_SRC})
    set_target_properties(ufsecp_shared PROPERTIES OUTPUT_NAME ufsecp)
    ufsecp_configure(ufsecp_shared)
    # VER-008: Explicit SOVERSION/VERSION ensures Linux ldconfig symlinks work
    # (libufsecp.so -> libufsecp.so.${MAJOR} -> libufsecp.so.${FULL}). Use
    # PROJECT_VERSION_MAJOR/PROJECT_VERSION sourced from VERSION.txt so the
    # SONAME stays in lockstep with VERSION.txt, RPM (libufsecp.spec %global
    # soversion) and Debian (control: libufsecp${MAJOR}). Hardcoded 1/1.0.0
    # previously drifted from VERSION.txt and broke versioned symlinks.
    set_target_properties(ufsecp_shared PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION   ${PROJECT_VERSION}
    )

    # Disable LTO when CUDA is linked (nvcc LTO version differs from host compiler)
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(ufsecp_shared PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    endif()

    if(WIN32)
        set_target_properties(ufsecp_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    elseif(APPLE)
        set_target_properties(ufsecp_shared PROPERTIES
            MACOSX_RPATH ON
            INSTALL_RPATH "@loader_path"
        )
    else()
        set_target_properties(ufsecp_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
endif()

# Static library (always useful for embedding)
if(UFSECP_BUILD_STATIC)
    add_library(ufsecp_static STATIC ${UFSECP_IMPL_SRC} ${UFSECP_GPU_IMPL_SRC} ${UFSECP_LBTC_SRC})
    # On Windows avoid output collision with shared import lib
    if(WIN32 AND UFSECP_BUILD_SHARED)
        set_target_properties(ufsecp_static PROPERTIES OUTPUT_NAME ufsecp_s)
    else()
        set_target_properties(ufsecp_static PROPERTIES OUTPUT_NAME ufsecp)
    endif()
    ufsecp_configure(ufsecp_static)

    # Consumers of the static library must NOT use __declspec(dllimport) on Windows.
    # Propagate UFSECP_STATIC_LIB so the header switches UFSECP_API to empty.
    target_compile_definitions(ufsecp_static PUBLIC UFSECP_STATIC_LIB)

    if(SECP256K1_BUILD_CUDA)
        set_target_properties(ufsecp_static PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    endif()
endif()

# Alias for parent project: `ufsecp::ufsecp`
if(TARGET ufsecp_static)
    add_library(ufsecp::ufsecp ALIAS ufsecp_static)
elseif(TARGET ufsecp_shared)
    add_library(ufsecp::ufsecp ALIAS ufsecp_shared)
endif()

# libbitcoin batch sig-verify benchmark — built as part of the central source
# (static link to ufsecp_static, which already pulls in fastsecp256k1 + the GPU
# host layer + the ufsecp_lbtc batch bridge). OFF in the minimal ship profile.
option(SECP256K1_BUILD_LIBBITCOIN_BENCH
       "Build the libbitcoin batch sig-verify benchmark (bench_lbtc_batch)" OFF)
if(SECP256K1_BUILD_LIBBITCOIN AND SECP256K1_BUILD_LIBBITCOIN_BENCH)
    # Built straight from the central source: the C ABI, GPU C ABI and the
    # ufsecp_lbtc bridge are compiled into the benchmark target itself, which
    # links the in-tree engine targets (fastsecp256k1 [+ GPU host layer]). No
    # separate ufsecp static/shared library is involved.
    add_executable(bench_lbtc_batch
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/bench/bench_lbtc_batch.cpp"
        ${UFSECP_IMPL_SRC} ${UFSECP_GPU_IMPL_SRC} ${UFSECP_LBTC_SRC})
    target_compile_definitions(bench_lbtc_batch PRIVATE UFSECP_BUILDING UFSECP_STATIC_LIB)
    target_include_directories(bench_lbtc_batch PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
        ${CPU_INCLUDE_DIR} ${SHARED_INCLUDE_DIR}
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/include")
    target_link_libraries(bench_lbtc_batch PRIVATE fastsecp256k1)
    if(TARGET secp256k1_gpu_host)
        target_link_libraries(bench_lbtc_batch PRIVATE secp256k1_gpu_host)
        target_include_directories(bench_lbtc_batch PRIVATE ${ULTRAFAST_ROOT}/src/gpu/include)
        target_compile_definitions(bench_lbtc_batch PRIVATE UFSECP_LBTC_WITH_GPU)
    endif()
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(bench_lbtc_batch PROPERTIES
            CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
    endif()
    message(STATUS "  libbitcoin batch bench: ON (bench_lbtc_batch, central source)")
endif()

# libbitcoin GPU-vs-CPU consensus differential — a standalone runnable that lets
# integrators (and our local-only GPU CI) prove the GPU verdict matches the CPU
# reference bit-for-bit. GPU is local-only, so this never runs in GitHub CI; it
# advisory-skips (exit 77) when no GPU backend is present. Built from the central
# source exactly like the bench (no separate ufsecp library involved).
option(SECP256K1_BUILD_LIBBITCOIN_TESTS
       "Build the libbitcoin GPU/CPU consensus differential (test_lbtc_consensus_diff)" OFF)
if(SECP256K1_BUILD_LIBBITCOIN AND SECP256K1_BUILD_LIBBITCOIN_TESTS)
    add_executable(test_lbtc_bridge
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/tests/test_lbtc_bridge.cpp"
        ${UFSECP_IMPL_SRC} ${UFSECP_GPU_IMPL_SRC} ${UFSECP_LBTC_SRC})
    target_compile_features(test_lbtc_bridge PRIVATE cxx_std_20)
    target_compile_definitions(test_lbtc_bridge PRIVATE UFSECP_BUILDING UFSECP_STATIC_LIB)
    target_include_directories(test_lbtc_bridge PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
        ${CPU_INCLUDE_DIR} ${SHARED_INCLUDE_DIR}
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/include")
    target_link_libraries(test_lbtc_bridge PRIVATE fastsecp256k1)
    if(TARGET secp256k1_gpu_host)
        target_link_libraries(test_lbtc_bridge PRIVATE secp256k1_gpu_host)
        target_include_directories(test_lbtc_bridge PRIVATE ${ULTRAFAST_ROOT}/src/gpu/include)
        target_compile_definitions(test_lbtc_bridge PRIVATE UFSECP_LBTC_WITH_GPU)
    endif()
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(test_lbtc_bridge PROPERTIES
            CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
    endif()
    include(CTest)
    enable_testing()
    add_test(NAME lbtc_bridge COMMAND test_lbtc_bridge)
    set_tests_properties(lbtc_bridge PROPERTIES LABELS "libbitcoin;batch;compat")

    add_executable(test_lbtc_consensus_diff
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/tests/test_lbtc_consensus_diff.cpp"
        ${UFSECP_IMPL_SRC} ${UFSECP_GPU_IMPL_SRC} ${UFSECP_LBTC_SRC})
    target_compile_definitions(test_lbtc_consensus_diff
        PRIVATE UFSECP_BUILDING UFSECP_STATIC_LIB STANDALONE_TEST)
    target_include_directories(test_lbtc_consensus_diff PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
        ${CPU_INCLUDE_DIR} ${SHARED_INCLUDE_DIR}
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/include")
    target_link_libraries(test_lbtc_consensus_diff PRIVATE fastsecp256k1)
    if(TARGET secp256k1_gpu_host)
        target_link_libraries(test_lbtc_consensus_diff PRIVATE secp256k1_gpu_host)
        target_include_directories(test_lbtc_consensus_diff PRIVATE ${ULTRAFAST_ROOT}/src/gpu/include)
        target_compile_definitions(test_lbtc_consensus_diff PRIVATE UFSECP_LBTC_WITH_GPU)
    endif()
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(test_lbtc_consensus_diff PROPERTIES
            CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
    endif()
    # Direct libsecp256k1 leg: link the drop-in shim so the gate proves
    # GPU==CPU==libsecp256k1 directly, not just transitively. Gate on the build
    # flags (not if(TARGET ...)) because compat/libsecp256k1_shim is added AFTER
    # this subdirectory, so the target does not exist yet at configure time here;
    # target_link_libraries resolves the dependency at generate time. The libbitcoin
    # profile turns SECP256K1_BUILD_SHIM on, so the shim is always present here.
    if(SECP256K1_BUILD_SHIM AND SECP256K1_BUILD_CPU)
        target_link_libraries(test_lbtc_consensus_diff PRIVATE secp256k1_shim)
        target_include_directories(test_lbtc_consensus_diff PRIVATE
            ${ULTRAFAST_ROOT}/compat/libsecp256k1_shim/include)
        target_compile_definitions(test_lbtc_consensus_diff PRIVATE UFSECP_LBTC_HAVE_LIBSECP)
    endif()
    # Registered as a CTest so local CI (which has a GPU) gates on it. A 77 exit
    # (no GPU) is treated as "skipped", not a failure.
    include(CTest)
    enable_testing()
    add_test(NAME lbtc_consensus_diff COMMAND test_lbtc_consensus_diff)
    set_tests_properties(lbtc_consensus_diff PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;local-ci;consensus")

    add_executable(test_lbtc_multisig_threshold
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/tests/test_lbtc_multisig_threshold.cpp"
        ${UFSECP_IMPL_SRC} ${UFSECP_GPU_IMPL_SRC} ${UFSECP_LBTC_SRC})
    target_compile_features(test_lbtc_multisig_threshold PRIVATE cxx_std_20)
    target_compile_definitions(test_lbtc_multisig_threshold PRIVATE UFSECP_BUILDING UFSECP_STATIC_LIB)
    target_include_directories(test_lbtc_multisig_threshold PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
        ${CPU_INCLUDE_DIR} ${SHARED_INCLUDE_DIR}
        "${ULTRAFAST_ROOT}/compat/libbitcoin_bridge/include")
    target_link_libraries(test_lbtc_multisig_threshold PRIVATE fastsecp256k1)
    if(TARGET secp256k1_gpu_host)
        target_link_libraries(test_lbtc_multisig_threshold PRIVATE secp256k1_gpu_host)
        target_include_directories(test_lbtc_multisig_threshold PRIVATE ${ULTRAFAST_ROOT}/src/gpu/include)
        target_compile_definitions(test_lbtc_multisig_threshold PRIVATE UFSECP_LBTC_WITH_GPU)
    endif()
    if(SECP256K1_BUILD_CUDA)
        set_target_properties(test_lbtc_multisig_threshold PROPERTIES
            CUDA_RUNTIME_LIBRARY "${CMAKE_CUDA_RUNTIME_LIBRARY}")
    endif()
    add_test(NAME lbtc_multisig_threshold COMMAND test_lbtc_multisig_threshold)
    set_tests_properties(lbtc_multisig_threshold PROPERTIES LABELS "libbitcoin;batch;compat")
    message(STATUS "  libbitcoin tests: ON (bridge + consensus diff + multisig/threshold, central source)")
endif()

# -- Install (standalone only -- skip when used as sub-project) --------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(GNUInstallDirs)

    # Headers
    install(FILES
        ufsecp.h
        ${CMAKE_CURRENT_BINARY_DIR}/ufsecp_version.h
        ufsecp_error.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ufsecp
    )

    # Libraries
    if(TARGET ufsecp_shared)
        install(TARGETS ufsecp_shared
            EXPORT ufsecpTargets
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
    if(TARGET ufsecp_static)
        install(TARGETS ufsecp_static
            EXPORT ufsecpTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

    # CMake package config
    install(EXPORT ufsecpTargets
        FILE ufsecpTargets.cmake
        NAMESPACE ufsecp::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ufsecp
    )

    # -- pkg-config ---------------------------------------------------------
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/ufsecp.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/ufsecp.pc"
        @ONLY
    )
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/ufsecp.pc")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ufsecp.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        )
    endif()
endif()
