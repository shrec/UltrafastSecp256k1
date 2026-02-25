cmake_minimum_required(VERSION 3.18)
project(secp256k1_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -- Shim library --------------------------------------------------------------

add_library(secp256k1_shim STATIC
    src/shim_context.cpp
    src/shim_pubkey.cpp
    src/shim_ecdsa.cpp
    src/shim_schnorr.cpp
    src/shim_extrakeys.cpp
    src/shim_seckey.cpp
    src/shim_tagged_hash.cpp
)

# Public includes -- exposes libsecp256k1-compatible headers
target_include_directories(secp256k1_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against UltrafastSecp256k1 CPU library
# When used via add_subdirectory from the main project, secp256k1_fast target exists.
# Otherwise, find it via find_package or provide path.
if(TARGET secp256k1_fast)
    target_link_libraries(secp256k1_shim PRIVATE secp256k1_fast)
else()
    # Fallback: expect the main library's include path
    message(WARNING "secp256k1_fast target not found -- add UltrafastSecp256k1 via add_subdirectory first")
endif()

# -- Optional: test that the shim compiles -------------------------------------

option(SECP256K1_SHIM_BUILD_TESTS "Build shim tests" OFF)

if(SECP256K1_SHIM_BUILD_TESTS)
    enable_testing()
    add_executable(shim_test tests/shim_test.cpp)
    target_link_libraries(shim_test PRIVATE secp256k1_shim)
    add_test(NAME secp256k1_shim_test COMMAND shim_test)
endif()
