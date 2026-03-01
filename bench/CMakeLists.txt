# bench/CMakeLists.txt -- Apples-to-apples benchmark: UF vs libsecp256k1
#
# Enable with: cmake -DSECP256K1_BUILD_BENCH_COMPARE=ON ...
# Build:       cmake --build <dir> --target bench_compare -j
# Run:         ./bench_compare [OPTIONS]

# Guard: only build if explicitly enabled
if(NOT SECP256K1_BUILD_BENCH_COMPARE)
    return()
endif()

message(STATUS "  bench_compare: ON")

# --- FetchContent: bitcoin-core/libsecp256k1 v0.6.0 ---
include(FetchContent)
FetchContent_Declare(
    libsecp256k1_bench
    GIT_REPOSITORY https://github.com/bitcoin-core/secp256k1.git
    GIT_TAG        v0.6.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(libsecp256k1_bench)
if(NOT libsecp256k1_bench_POPULATED)
    FetchContent_Populate(libsecp256k1_bench)
    # Configure libsecp256k1 with all modules we need
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
        ${libsecp256k1_bench_SOURCE_DIR}
        ${libsecp256k1_bench_BINARY_DIR}
        EXCLUDE_FROM_ALL
    )
endif()

# --- bench_compare executable ---
add_executable(bench_compare
    src/main.cpp
    src/bench_config.cpp
    src/providers/provider_uf.cpp
    src/providers/provider_libsecp.cpp
)

target_include_directories(bench_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${libsecp256k1_bench_SOURCE_DIR}/include
)

target_link_libraries(bench_compare PRIVATE
    fastsecp256k1   # UltrafastSecp256k1 CPU library
    secp256k1       # bitcoin-core/libsecp256k1
)

target_compile_features(bench_compare PRIVATE cxx_std_20)

# Platform-specific: larger stack for Windows (precompute tables can be large)
if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
    target_link_options(bench_compare PRIVATE "LINKER:/STACK:8388608")
endif()

# Optimization: match library build type
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(bench_compare PRIVATE -O3 -DNDEBUG)
    endif()
endif()

message(STATUS "  bench_compare: target ready (UF + libsecp256k1 v0.6.0)")
