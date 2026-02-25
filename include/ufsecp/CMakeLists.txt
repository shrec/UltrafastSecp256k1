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
#        cmake -S include/ufsecp -B build-ufsecp -DCMAKE_BUILD_TYPE=Release
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
set(CPU_INCLUDE_DIR "${ULTRAFAST_ROOT}/cpu/include")
set(SHARED_INCLUDE_DIR "${ULTRAFAST_ROOT}/include")

# -- Library target ---------------------------------------------------------

option(UFSECP_BUILD_SHARED "Build shared library" ON)
option(UFSECP_BUILD_STATIC "Build static library" ON)

set(UFSECP_IMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ufsecp_impl.cpp")

# Helper: configure a target with the right includes/definitions
function(ufsecp_configure tgt)
    target_include_directories(${tgt}
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
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
        file(GLOB _CPU_SOURCES "${ULTRAFAST_ROOT}/cpu/src/*.cpp")
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
endfunction()

# Shared library
if(UFSECP_BUILD_SHARED)
    add_library(ufsecp_shared SHARED ${UFSECP_IMPL_SRC})
    set_target_properties(ufsecp_shared PROPERTIES OUTPUT_NAME ufsecp)
    ufsecp_configure(ufsecp_shared)

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
    add_library(ufsecp_static STATIC ${UFSECP_IMPL_SRC})
    # On Windows avoid output collision with shared import lib
    if(WIN32 AND UFSECP_BUILD_SHARED)
        set_target_properties(ufsecp_static PROPERTIES OUTPUT_NAME ufsecp_s)
    else()
        set_target_properties(ufsecp_static PROPERTIES OUTPUT_NAME ufsecp)
    endif()
    ufsecp_configure(ufsecp_static)
endif()

# Alias for parent project: `ufsecp::ufsecp`
if(TARGET ufsecp_static)
    add_library(ufsecp::ufsecp ALIAS ufsecp_static)
elseif(TARGET ufsecp_shared)
    add_library(ufsecp::ufsecp ALIAS ufsecp_shared)
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
