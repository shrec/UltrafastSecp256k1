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
    src/shim_recovery.cpp
    src/shim_ellswift.cpp
    src/shim_ecdh.cpp
)

# Public includes -- exposes libsecp256k1-compatible headers
target_include_directories(secp256k1_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against UltrafastSecp256k1 CPU library.
# The internal target is 'fastsecp256k1'; 'secp256k1_fast' is an older alias
# kept for backward compatibility. Try both names.
if(TARGET fastsecp256k1)
    target_link_libraries(secp256k1_shim PRIVATE fastsecp256k1)
elseif(TARGET secp256k1_fast)
    target_link_libraries(secp256k1_shim PRIVATE secp256k1_fast)
elseif(TARGET secp256k1::fast)
    target_link_libraries(secp256k1_shim PRIVATE secp256k1::fast)
else()
    message(FATAL_ERROR
        "UltrafastSecp256k1 library target not found.\n"
        "Add UltrafastSecp256k1 via add_subdirectory before this shim.")
endif()

# -- Optional: shim API test ---------------------------------------------------

option(SECP256K1_SHIM_BUILD_TESTS "Build shim compatibility test" OFF)

if(SECP256K1_SHIM_BUILD_TESTS)
    enable_testing()
    add_executable(shim_test tests/shim_test.cpp)
    target_link_libraries(shim_test PRIVATE secp256k1_shim)
    add_test(NAME secp256k1_shim_test COMMAND shim_test)
    set_tests_properties(secp256k1_shim_test PROPERTIES LABELS "shim")
endif()
