# ============================================================================
# UltrafastSecp256k1 -- C API Shared Library
# ============================================================================
# Builds libultrafast_secp256k1.so / .dll / .dylib
# Usage:
#   cmake -S bindings/c_api -B bindings/c_api/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build bindings/c_api/build
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(ultrafast_secp256k1_capi
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "UltrafastSecp256k1 C API shared library")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -- Find the CPU library ---------------------------------------------------
# The CPU library is built by the parent CMake project.
# We locate its include dirs and link against it.

# Paths relative to this CMakeLists.txt
set(ULTRAFAST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(CPU_INCLUDE_DIR "${ULTRAFAST_ROOT}/cpu/include")
set(SHARED_INCLUDE_DIR "${ULTRAFAST_ROOT}/include")

# Check that includes exist
if(NOT EXISTS "${CPU_INCLUDE_DIR}/UltrafastSecp256k1.hpp")
    message(FATAL_ERROR "Cannot find UltrafastSecp256k1.hpp at ${CPU_INCLUDE_DIR}")
endif()

# -- Shared library target -------------------------------------------------

add_library(ultrafast_secp256k1 SHARED
    ultrafast_secp256k1.cpp
)

target_include_directories(ultrafast_secp256k1 PRIVATE
    ${CPU_INCLUDE_DIR}
    ${SHARED_INCLUDE_DIR}
)

target_compile_definitions(ultrafast_secp256k1 PRIVATE
    ULTRAFAST_SECP256K1_BUILDING
)

# Link against the CPU library (as static archive or object files)
# When built as part of the parent project, the cpu target is available.
# Otherwise, we need to find or build it.
if(TARGET UltrafastSecp256k1_cpu)
    target_link_libraries(ultrafast_secp256k1 PRIVATE UltrafastSecp256k1_cpu)
elseif(TARGET secp256k1_cpu)
    target_link_libraries(ultrafast_secp256k1 PRIVATE secp256k1_cpu)
else()
    # Standalone build: gather all CPU .cpp sources
    file(GLOB CPU_SOURCES "${ULTRAFAST_ROOT}/cpu/src/*.cpp")

    # MSVC (non clang-cl): no __uint128_t / __int128 support
    if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # field_52.cpp has a hard #error without __uint128_t
        list(FILTER CPU_SOURCES EXCLUDE REGEX "field_52\\.cpp$")
        # scalar.cpp uses __int128 in the 64-bit path; enable 32-bit fallback
        target_compile_definitions(ultrafast_secp256k1 PRIVATE SECP256K1_NO_INT128)
    endif()

    target_sources(ultrafast_secp256k1 PRIVATE ${CPU_SOURCES})
endif()

# -- Platform-specific flags -----------------------------------------------

if(WIN32)
    # Windows: export all symbols through the SECP256K1_API macro
    set_target_properties(ultrafast_secp256k1 PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
elseif(APPLE)
    set_target_properties(ultrafast_secp256k1 PROPERTIES
        MACOSX_RPATH ON
        INSTALL_RPATH "@loader_path"
    )
else()
    set_target_properties(ultrafast_secp256k1 PROPERTIES
        INSTALL_RPATH "$ORIGIN"
    )
    target_link_options(ultrafast_secp256k1 PRIVATE
        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map
    )
endif()

set_target_properties(ultrafast_secp256k1 PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER ultrafast_secp256k1.h
)

# -- Install ---------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS ultrafast_secp256k1
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
